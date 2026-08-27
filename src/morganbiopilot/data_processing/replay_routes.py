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
import time
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
           seed: int, max_depth: int, ranker=None, top_n=None, rule_ec=None,
           show_ec: bool = False) -> tuple:
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
            report = expand(node.smiles, rules, prefilter,
                            ranker=ranker, top_n=top_n)
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
        # The same view the agent will meet, ranker included. Building the corpus with
        # a different frontier order than inference uses is the deeper form of the
        # distribution mismatch this file exists to avoid: the model would be taught to
        # choose among twenty molecules selected one way and then asked to choose among
        # twenty selected another.
        # `rule_ec=None`, deliberately, and this was a real defect until it was measured.
        # Passing it made every corpus line carry `| EC=1,2`, while the arms the paper
        # reports run `--tooling untooled`, whose ToolSurface has `rule_ec=None` and
        # renders no such field. The model was therefore trained to read a column it
        # never sees at evaluation -- the exact train/inference mismatch this replay
        # exists to prevent.
        #
        # Only the rendering is affected: `rule_ec` also feeds the portfolio's
        # `precedent` member, but over 183 frontier states of eight curated targets the
        # selected top-20 was identical with and without it, so the candidates shown do
        # not change. Pass `--show-ec` to restore the column for a tooled corpus.
        view = build_frontier_view(graph, frontier, top_k=top_k, seed=seed + len(pairs),
                                   ranker=ranker, prefilter=prefilter,
                                   rule_ec=rule_ec, show_ec=show_ec)
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
            "assistant": json.dumps({"choice": index}),
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
    # Threads buy nothing here and the measurement says so: 60 routes took the same
    # wall time on 8 workers as on 1, because `ranker.order` is a Python loop and the
    # GIL serialises it. Processes would work but each holds its own copy of the rule
    # set. So the unit of parallelism is the *job*: slice the corpus with --offset,
    # run one job per slice, concatenate. Kept anyway for the case where the expensive
    # call releases the GIL.
    parser.add_argument("--workers", type=int, default=1,
                        help="routes replayed concurrently within one process. Threads, "
                             "so expect little gain; prefer --offset across jobs")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many routes before --limit. Slicing happens "
                             "after the shuffle and each route keeps its own seed, so "
                             "concatenated slices equal one undivided run")
    # The cap must match the one the search will run under, or every training state is
    # drawn from a graph the model will never see again. It belongs to the environment,
    # so mining (`route_corpus`) stays uncapped -- that builds the oracle -- while the
    # replay that turns routes into decisions is capped exactly like inference.
    parser.add_argument("--ranker", default=None,
                        help="'native_similarity' to order rules before validation")
    parser.add_argument("--show-ec", action="store_true",
                        help="render the EC column, matching --tooling ec_only or "
                             "tooled. Off by default: the reported arms are untooled "
                             "and show no EC, and a corpus that does would train the "
                             "model on a column it never sees")
    parser.add_argument("--top-n", type=int, default=None,
                        help="neighbours kept per expansion; needs --ranker")
    args = parser.parse_args()

    if args.sink:
        from morganbiopilot.core.building_blocks import use_sink
        use_sink(args.sink)
        print(f"sink overridden: {args.sink}")

    routes = []
    with open(args.routes, encoding="utf-8") as fh:
        for line in fh:
            routes.append(json.loads(line))
    # The seed each route is replayed under is its index in this list, so it must be
    # fixed *before* slicing -- otherwise slice 2 would replay its routes under the
    # seeds slice 1 used, and concatenating the slices would not equal one whole run.
    if (args.limit and args.limit < len(routes)) or args.offset:
        # Not `routes[:limit]`: the corpus is written level by level outwards from the
        # sink, so its head is its shallow end and a prefix is not a sample of it.
        random.Random(args.seed).shuffle(routes)
    indexed = list(enumerate(routes))
    if args.offset:
        indexed = indexed[args.offset:]
    if args.limit:
        indexed = indexed[:args.limit]
    routes = indexed
    if args.offset or args.limit:
        print(f"slice: routes {routes[0][0]}..{routes[-1][0]} of {len(indexed)}")
    print(f"{len(routes)} routes | radius r{args.radius} | top_k {args.top_k}")

    print("loading rules ...")
    rules = load_rules(radius=args.radius)
    prefilter = prefilter_from_rules(rules)
    from morganbiopilot.one_step.ranking import make_ranker
    ranker = make_ranker(args.ranker, rules)
    if ranker is not None:
        print(f"expansions ranked by {args.ranker}, capped at {args.top_n}")

    # Needed by the portfolio frontier order, which counts a molecule's best-attested
    # rule among its members. Without it that member is flat and the view degrades to
    # three of four -- quietly, and differently from the view inference will build.
    from morganbiopilot.core.ec import annotate_rules
    rule_ec = annotate_rules(rules)
    print(f"frontier view: portfolio (depth / similarity / precedent / size), "
          f"top_k {args.top_k}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    outcomes: Counter = Counter()
    by_depth: Counter = Counter()
    emitted = 0

    def one(job):
        """One route, replayed. Pure apart from the shared per-molecule score cache,
        whose worst case under concurrency is computing an entry twice."""
        i, entry = job
        return replay(entry["target"], entry["tree"], rules, prefilter,
                      args.top_k, args.seed + i, args.max_depth,
                      ranker, args.top_n, rule_ec, args.show_ec)

    # Already (original_index, entry): the index is the route's seed, and it must
    # survive slicing so that concatenated slices equal one undivided run.
    jobs = routes
    t0 = time.perf_counter()

    # LF everywhere: see route_corpus for why this is not cosmetic.
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        def take(pairs, outcome, done):
            nonlocal emitted
            outcomes[outcome] += 1
            for pair in pairs:
                by_depth[pair["depth"]] += 1
                fh.write(json.dumps(pair) + "\n")
                emitted += 1
            if done % 250 == 0:
                rate = done / max(time.perf_counter() - t0, 1e-9)
                print(f"  {done}/{len(jobs)} routes -> {emitted} pairs "
                      f"({rate * 60:.0f}/min, eta {(len(jobs) - done) / rate / 60:.0f} min)",
                      flush=True)

        if args.workers > 1:
            # Routes are independent, and the portfolio frontier order made this loop
            # 22x more expensive than the plain stratified one: 29 h serial against the
            # 1.3 h it used to take. Threads, not processes, so the score cache is
            # shared -- and RDKit releases the GIL in the calls that dominate here.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(one, j) for j in jobs]
                for done, fut in enumerate(as_completed(futures), 1):
                    pairs, outcome = fut.result()
                    take(pairs, outcome, done)
        else:
            for done, job in enumerate(jobs, 1):
                pairs, outcome = one(job)
                take(pairs, outcome, done)

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
