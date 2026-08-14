"""Turn attested routes into the decisions our agent actually faces.

`route_corpus` mines paths through MetaNetX. A path is not training data: the agent is
never asked "what is the precursor of X", it is asked "which of these frontier molecules
should be expanded next". This replays each route through the real engine so that the
training state is byte-for-byte the state at inference -- same expansion, same
depth-stratified quota, same per-seed shuffle.

The loop, per route
-------------------
Expand the current on-route molecule, attach every validated neighbour, compute the
frontier, and look for the next molecule of the route among the candidates *shown*. If it
is there, emit the pair and step onto it. If not, the trajectory stops and we keep the
prefix.

Two ways a step is lost, counted separately because they have different fixes
----------------------------------------------------------------------------
**Not reachable.** No template produces the attested precursor from the current molecule.
Measured at 50% per step at radius 2 on the curated pathways, and it is the binding
constraint on this whole plan: a lower radius raises it to 61% (r1) or 73% (r0) at the
cost of a frontier nobody can rank.

**Truncated away.** The precursor is in the frontier but not among the 20 candidates the
quota shows. Then the correct answer is not visible and the pair would teach the model to
pick something else -- worse than emitting nothing. The frontier audit found classical
policies' choices always visible, but that was an aggregate over all their picks, not a
specific molecule we need, so this is measured rather than assumed.

The state carries no columns, and that is a result rather than a default: sink closeness
accounts for the entire plateau we measured (+5 points from N=10 to N=50 with it, +18
without), so a trained policy must learn value from attested routes instead of being
handed a heuristic that makes it greedy.

    python -m morganbiopilot.data_processing.replay_routes --limit 2000
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from morganbiopilot.agents.policy import SYSTEM_PROMPT
from morganbiopilot.agents.state import build_frontier_view
from morganbiopilot.core.building_blocks import skeleton as _skeleton
from morganbiopilot.core.chem import sanitize
from morganbiopilot.data_processing.route_corpus import is_available
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.graph import SearchGraph
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import prefilter_from_rules


def render_user(graph: SearchGraph, view, target: str) -> str:
    """The user message, identical to what `LLMPolicy` sends at inference."""
    reachable = sum(1 for m in graph.molecules.values() if m.available)
    return (f"Target molecule: {target}\n"
            f"Search graph so far: {len(graph.molecules)} molecules, "
            f"{len(graph.reactions)} reactions, {reachable} already reachable "
            f"from the chassis.\n\n"
            f"Frontier candidates:\n{view.render()}\n\n"
            f"Which candidate should be expanded next?")


def replay(target: str, tree: Dict[str, int], rules, prefilter, top_k: int,
           seed: int, max_depth: int) -> tuple:
    """(pairs, outcome) for one route tree.

    A route is a tree, not a chain: one step can leave two precursors to make and both
    must be. So at any point *several* frontier molecules may be on the route, and each is
    a correct answer. The trajectory ends when every non-available molecule of the tree has
    been expanded -- not at the first molecule we fail to find, which is what the chain
    version did and what made it stop after a single decision.

    Where several on-route molecules are visible, the label is the one of **highest
    remaining cost**: the hardest branch first. That is a convention and it is stated as
    such, chosen because deferring hard branches is precisely the greedy behaviour the
    measured plateau consists of, and a corpus should not teach it.
    """
    on_route = {}
    for molecule, cost in tree.items():
        key = _skeleton(molecule)
        if key and not is_available(molecule):
            on_route[key] = cost

    graph = SearchGraph(target)
    pairs = []
    current, outcome = graph.root, "complete"

    for _ in range(len(tree) + 2):
        node = graph.molecules[current]
        if node.depth >= max_depth:
            outcome = "depth cap"
            break
        try:
            report = expand(node.smiles, rules, prefilter)
        except Exception:                                        # noqa: BLE001
            outcome = "expansion failed"
            break
        if not report.neighbours:
            outcome = "no candidate"
            break
        for neighbour in report.neighbours:
            graph.add_neighbour(current, neighbour)
        node.expanded = True

        frontier = [i for i, m in graph.molecules.items()
                    if not m.solved and not m.expanded and not m.available]
        if not frontier:
            break

        # Seeded per step so the correct answer does not sit at a fixed position across a
        # trajectory, which the model would learn instead of the chemistry.
        view = build_frontier_view(graph, frontier, top_k=top_k, seed=seed + len(pairs))
        visible = [(k, on_route[_skeleton(c.smiles)])
                   for k, c in enumerate(view.candidates)
                   if _skeleton(c.smiles) in on_route]
        if not visible:
            hidden = any(_skeleton(graph.molecules[i].smiles) in on_route
                         for i in frontier)
            if hidden:
                outcome = "truncated away"
            else:
                # Nothing on-route left in the frontier has two opposite meanings, and
                # collapsing them hid a success rate: either every molecule of the tree
                # has been expanded, which is the trajectory finishing, or the thread was
                # lost. Distinguish by checking what the graph actually contains.
                seen = {_skeleton(m.smiles) for m in graph.molecules.values()
                        if m.expanded or m.available}
                outcome = ("complete" if on_route.keys() <= seen else "thread lost")
            break

        index = max(visible, key=lambda pair: pair[1])[0]
        pairs.append({
            "system": SYSTEM_PROMPT,
            "user": render_user(graph, view, target),
            "assistant": json.dumps({"choice": index, "reason": ""}),
            "depth": graph.molecules[view.candidates[index].node_id].depth,
            "n_on_route": len(visible),
            "n_shown": len(view.candidates),
            "n_frontier": len(frontier),
        })
        current = view.candidates[index].node_id

    return pairs, outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--routes", default="results/route_corpus.jsonl")
    parser.add_argument("--out", default="results/sft_pairs.jsonl")
    parser.add_argument("--radius", type=int, default=2)
    # Must match the sink the corpus was mined with. Mining against BioNavi (38
    # skeletons) and replaying against the E. coli chassis (753) makes most of a
    # route tree "available" at replay time, so it never reaches the frontier and
    # every trajectory reads as exhausted after one step.
    parser.add_argument("--sink", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="0 = every route")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.sink:
        from morganbiopilot.core.building_blocks import use_sink
        use_sink(args.sink)
        print(f"sink overridden: {args.sink}")

    routes = []
    with open(args.routes, encoding="utf-8") as fh:
        for line in fh:
            routes.append(json.loads(line))
    if args.limit and args.limit < len(routes):
        # Not `routes[:limit]`: the corpus is written level by level outwards from the
        # sink, so its head is its shallow end and a prefix is not a sample of it.
        random.Random(args.seed).shuffle(routes)
        routes = routes[:args.limit]
    print(f"{len(routes)} routes | radius r{args.radius} | top_k {args.top_k}")

    print("loading rules ...")
    rules = load_rules(radius=args.radius)
    prefilter = prefilter_from_rules(rules)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    outcomes: Counter = Counter()
    by_depth: Counter = Counter()
    emitted = 0

    with open(out, "w", encoding="utf-8") as fh:
        for i, entry in enumerate(routes):
            pairs, outcome = replay(entry["target"], entry["tree"], rules, prefilter,
                                    args.top_k, args.seed + i, args.max_depth)
            outcomes[outcome] += 1
            for pair in pairs:
                by_depth[pair["depth"]] += 1
                fh.write(json.dumps(pair) + "\n")
                emitted += 1
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(routes)} routes -> {emitted} pairs", flush=True)

    print()
    print(f"{emitted} pairs from {len(routes)} routes "
          f"({emitted / max(len(routes), 1):.2f} per route)")
    print("  where trajectories stopped:")
    for reason, count in outcomes.most_common():
        print(f"    {reason:18s} {count:6d}  ({100 * count / len(routes):.0f}%)")
    print("  pairs by depth of the chosen molecule:")
    for depth in sorted(by_depth):
        print(f"    depth {depth}: {by_depth[depth]}")

    if emitted < 2000:
        print()
        print("NOTE: below the 2,000-pair floor we set for fine-tuning a 7B.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
