"""Attested biosynthetic routes, mined from MetaNetX, for training a search policy.

The agent we measured is prompted, and what we inject into its prompt is what harms it:
sink closeness accounts for the entire early advantage and the entire plateau (+5 points
from N=10 to N=50 with it, +18 without). Meanwhile the systems that do work in organic
retrosynthesis are not prompted at all -- RetroAgent trains a 4B model, RETRO-R1 a 7B,
the same size as ours. They build their corpus by *searching* a reaction graph, not by
curating routes: 299,202 multi-step routes over USPTO, of which they keep the 11,366
longer than four steps.

This is the retrobiosynthetic analogue. Molecules are nodes and a route is a path from
a target down to the chassis -- but read `build_edges_from_rules` before assuming how
the edges are made, because it is not what an earlier version of this file did and not
what "mining MetaNetX" would suggest.

An edge is **a reaction rule applied to the substrate it was extracted from**, not a
MetaNetX reaction read off the database. The two sound equivalent and are not. Building
edges from MetaNetX reactions directly gave a graph whose steps our own engine could
reproduce only 11% of the time, because a reaction and the template extracted from it
do not always agree on what a molecule decomposes into. Going through the template
instead makes every edge something `expand` can produce by construction, which is the
property the replay depends on -- and the chemistry stays attested, since every
template was extracted from a balanced MetaNetX reaction. What is recovered is which
substrate-product pair each template encodes, not a new transformation.

Why distances are computed from the sink outwards
------------------------------------------------
Mining one route per target by searching from each target would repeat the same traversal
thousands of times. A single breadth-first sweep *from* the sink, following edges
backwards, labels every reachable molecule with its distance to the chassis and a parent
on a shortest route -- one O(V+E) pass for the whole corpus. Shortest routes are also the
right target: they are what a metabolic engineer would prefer, and route length is one of
the quantities the search is scored on.

What this corpus is not
-----------------------
It is **attested chemistry, not our engine's chemistry**. That distinction is the point.
Mining routes from our own template search would teach the policy our sink-closeness
heuristic, which is the thing we are trying to get away from; mining them from MetaNetX
reactions teaches enzymology the heuristic does not contain -- that to make a diol you
work the carbon backbone rather than esterify it with a fatty acid, to take a case where
closeness actively prefers the wrong branch.

The consequence is that a mined route is not guaranteed to be reproducible by our
templates. Measured on the 20 curated pathways, the attested disconnection is reachable
at radius 2 for only 50% of steps, so the replay step that turns these routes into
frontier decisions will lose ground. That attrition is the main risk of the whole plan and
is measured separately.

    python -m morganbiopilot.data_processing.route_corpus --out results/routes.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from morganbiopilot.core.building_blocks import is_building_block, is_cofactor
from morganbiopilot.core.chem import sanitize


def is_available(smiles: str) -> bool:
    """What the engine calls available: a chassis metabolite or a cofactor.

    `MoleculeNode.available` is exactly this disjunction, and both terminate a branch for
    the same reason -- the cell already has them. Route mining has to use the same
    predicate or the corpus disagrees with inference about what counts as finished.
    """
    return is_building_block(smiles) or is_cofactor(smiles)


def build_edges_from_rules(radius: int) -> Dict[str, List[frozenset]]:
    """`substrate -> [precursor sets]`, every option one the engine can reproduce.

    The rule set already records the pairing we were trying to recover: `smi_sub` is the
    substrate a template was extracted to act on, and running that template on it yields
    the other side of the mono-component reaction. So an edge built this way is a template
    application by *identity*, not by approximation -- the replay calls the same
    `apply_reaction` on the same molecule.

    Two properties follow, and they are why this is the right construction. Reachability
    is ~100% instead of the 11% we measured, since the answer is by definition something
    `expand` can produce. And the edge is still attested chemistry, because every rule was
    extracted from a balanced MetaNetX reaction -- we are not inventing transformations,
    only recovering which substrate-product pair each one encodes.

    Availability follows the engine exactly -- `MoleculeNode.available` is
    `in_sink or is_cofactor`, so a cofactor ends a branch just as a chassis metabolite
    does. Available products are therefore kept as edge targets, which is what lets a
    route legitimately terminate on one, and they are seeded at distance 0 rather than
    expanded further.
    """
    from morganrxn.core.reaction_utils import apply_reaction

    from morganbiopilot.core.chem import split_components
    from morganbiopilot.core.rules import load_rules

    rules = load_rules(radius=radius)
    edges: Dict[str, List[frozenset]] = {}
    failed = 0
    for index in range(len(rules.smi_sub)):
        substrate = sanitize(str(rules.smi_sub[index]))
        if not substrate or is_available(substrate):
            continue
        try:
            products = apply_reaction(str(rules.template_reaction[index]), substrate)
        except Exception:                                        # noqa: BLE001
            failed += 1
            continue
        if not products:
            failed += 1
            continue
        # One entry per rule application, precursor set intact: the AND structure is
        # the whole point, and flattening it is what made the first corpus optimistic.
        for product in products:
            kept = {f for f in (sanitize(r) for r in split_components(product))
                    if f and f != substrate}
            if kept:
                edges.setdefault(substrate, [])
                if frozenset(kept) not in edges[substrate]:
                    edges[substrate].append(frozenset(kept))
    print(f"  {failed} rules did not run on their own substrate")
    return edges


def and_or_costs(reactions: Dict[str, List[frozenset]]):
    """Route depth under the engine's own semantics, and the reaction that achieves it.

        cost(m) = 0                                          if m is available
                = 1 + min over reactions ( max over precursors cost(p) )

    The `max` is what makes this an AND-OR quantity: a reaction is only as finished as its
    least finished precursor, exactly as `SearchGraph` propagates solved status. It is also
    the quantity our search bounds with `max_depth`, so a cost computed this way is
    directly comparable to the depths the engine reports.

    Computed level by level rather than by relaxation to a fixpoint. Costs are
    non-negative integers and a molecule reaches cost `k` precisely when some reaction has
    all its precursors at cost `<= k-1`, so one sweep per level assigns every molecule
    exactly once and the loop ends when a level adds nothing.

    Every reaction achieving the minimum is kept, not just the first found. Those are
    genuinely equivalent routes rather than detours, so sampling among them multiplies the
    corpus without teaching the policy to take the long way round -- which is the one thing
    a route corpus must not do, since route length is scored.
    """
    molecules = set(reactions)
    for options in reactions.values():
        for precursors in options:
            molecules.update(precursors)

    cost: Dict[str, int] = {m: 0 for m in molecules if is_available(m)}
    chosen: Dict[str, List[frozenset]] = {}

    level = 0
    while True:
        level += 1
        added = 0
        for molecule, options in reactions.items():
            if molecule in cost:
                continue
            best = [p for p in options
                    if all(q in cost and cost[q] < level for q in p)]
            if best:
                cost[molecule] = level
                chosen[molecule] = best
                added += 1
        if not added:
            break
    return cost, chosen


def route_tree(target: str, chosen: Dict[str, List[frozenset]], cost: Dict[str, int],
               rng: Optional[random.Random] = None) -> Dict[str, int]:
    """One optimal route, as {molecule: cost}. A tree, not a chain.

    Under AND semantics a route branches: one step can leave two precursors to make, and
    both must be. The training trajectory is therefore a traversal of this tree, and at any
    point several frontier molecules may be on it -- all of them correct answers.

    Where several reactions tie at the optimum, one is drawn. Repeated draws give distinct
    optimal routes for the same target, which is how the corpus grows without lengthening.
    """
    out: Dict[str, int] = {}
    stack = [target]
    while stack:
        molecule = stack.pop()
        if molecule in out:
            continue
        out[molecule] = cost.get(molecule, -1)
        options = chosen.get(molecule)
        if not options:
            continue
        precursors = options[0] if rng is None else rng.choice(options)
        for precursor in precursors:
            if not is_available(precursor):
                stack.append(precursor)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/route_corpus.jsonl")
    parser.add_argument("--radius", type=int, default=2)
    # A stricter sink pushes every molecule further from "available", so routes
    # lengthen and deep decisions appear without touching the radius. The BioNavi-NP
    # list holds 38 skeletons against the 753 of the E. coli chassis, and it is the
    # sink our external benchmark scores against -- mining and evaluating against the
    # same sink is the coherent choice.
    parser.add_argument("--sink", default=None,
                        help="override the chassis sink (see core.building_blocks)")
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--per-target", type=int, default=3,
                        help="distinct optimal routes per target, drawn where reactions tie")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.sink:
        from morganbiopilot.core.building_blocks import use_sink
        use_sink(args.sink)
        print(f"sink overridden: {args.sink}")

    print(f"building the molecule graph from the r{args.radius} rule set ...")
    edges = build_edges_from_rules(args.radius)
    n_edges = sum(len(v) for v in edges.values())
    print(f"  {len(edges)} molecules with precursors, {n_edges} reaction options")

    print("solving route depth under AND-OR semantics ...")
    depth, chosen = and_or_costs(edges)
    sink = sum(1 for d in depth.values() if d == 0)
    print(f"  {sink} available molecules in the graph, "
          f"{len(depth) - sink} molecules reachable from them")

    lengths = Counter(d for d in depth.values() if d > 0)
    print("  routes available by depth: " + ", ".join(
        f"{k}:{lengths[k]}" for k in sorted(lengths) if k <= 12))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, sizes = 0, []
    rng = random.Random(args.seed)
    # newline="\n" so the file is byte-identical wherever it is produced. Python's text
    # mode translates to CRLF on Windows, which left a corpus mined here and the same
    # corpus mined on the cluster differing by exactly one byte per line -- identical
    # content, different checksum, and a reproducibility check that fails for a reason
    # that has nothing to do with chemistry.
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for molecule, d in depth.items():
            if not (args.min_length <= d <= args.max_length):
                continue
            # A tree, not a chain: one step can leave two precursors to make and both
            # must be, so several molecules may be on-route at once. Where reactions tie
            # at the optimum, draw several distinct trees -- alternative optimal routes,
            # not detours, so the corpus grows without teaching longer chemistry.
            seen = set()
            for _ in range(args.per_target):
                tree = route_tree(molecule, chosen, depth, rng)
                key = frozenset(tree)
                if key in seen:
                    continue
                seen.add(key)
                sizes.append(len(tree))
                fh.write(json.dumps({"target": molecule, "depth": d,
                                     "tree": tree}) + "\n")
                kept += 1
    if sizes:
        sizes.sort()
        print(f"  molecules per route tree: median {sizes[len(sizes) // 2]}, "
              f"max {sizes[-1]}")

    print()
    print(f"wrote {kept} routes of length {args.min_length}-{args.max_length} to {out}")
    if kept < 2000:
        print("NOTE: fewer routes than the 2,000 we set as the floor for fine-tuning.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
