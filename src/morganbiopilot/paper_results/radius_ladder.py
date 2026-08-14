"""A radius ladder: high precision where it exists, low radius only where it must.

`paper_results.rank_disconnections` swept the promiscuity radius one value at a time and
found no good choice. Recall of the attested disconnection falls from 73% at r0 to 34% at
r5, while rankability moves the other way -- sink closeness puts the reference in the top
5 for 1% of steps at r0 and 89% at r5. The product peaks at r2 around 0.31, which is the
ceiling of a fixed radius: about a third of attested steps are both reachable and well
ranked.

One column of that sweep suggests the fixed choice is the mistake. "No candidate at all"
rises from 0 steps at r0 and r1 to 57 at r5. A high radius is not imprecise, it is
*silent*: when it fires it offers a dozen candidates and ranks them well, and the reason
its recall looks bad is that it often produces nothing.

So try radii in descending order and keep the first non-empty set. Each step is then
answered at the most conservative radius that has anything to say, and a low radius is
paid for only where nothing else applies -- instead of every step paying the average cost
of one compromise setting.

This measures whether that works, per reference step, against the fixed radii it is
built from. Rules are loaded one radius at a time and released before the next: six rule
sets at 82k x 1024 int32 apiece do not fit in memory together.

    python -m morganbiopilot.paper_results.radius_ladder --radii 5,4,3,2,1
"""

import argparse
import gc
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.golden_dataset import load_golden_dataset
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.heuristics import SinkCloseness
from morganbiopilot.one_step.prefilter import prefilter_from_rules
from morganbiopilot.paper_results.rank_disconnections import (
    _skeleton, candidates, rank_of, reference_steps)


def measure(steps, radius: int, closeness, rng) -> Dict[int, Tuple[int, Optional[int]]]:
    """{step index: (candidate count, 1-based rank of the reference or None)}."""
    rules = load_rules(radius=radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)

    out: Dict[int, Tuple[int, Optional[int]]] = {}
    for i, (_, product, precursor) in enumerate(steps):
        cands = candidates(product, rules, prefilter, rule_ec)
        if not cands:
            out[i] = (0, None)
            continue
        wanted = _skeleton(precursor)
        hit = next((k for k, (mols, _) in enumerate(cands)
                    if any(_skeleton(m) == wanted for m in mols)), None)
        if hit is None:
            out[i] = (len(cands), None)
            continue
        near = [max((closeness.closeness(m) for m in mols), default=0.0)
                for mols, _ in cands]
        out[i] = (len(cands), rank_of(near, hit, rng))

    del rules, prefilter, rule_ec
    gc.collect()
    return out


def report(label: str, records: List[Tuple[int, Optional[int]]], total: int) -> None:
    sizes = [n for n, _ in records if n > 0]
    found = [r for _, r in records if r is not None]
    if not sizes:
        print(f"{label:22s} no candidates anywhere")
        return
    print(f"{label:22s} {100 * len(found) / total:6.0f}% "
          f"{int(np.median(sizes)):8d} {max(sizes):7d} "
          f"{100 * np.mean([r <= 1 for r in found]) if found else 0:7.0f}% "
          f"{100 * np.mean([r <= 5 for r in found]) if found else 0:7.0f}% "
          f"{len(found) / total * (np.mean([r <= 5 for r in found]) if found else 0):8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--radii", default="5,4,3,2,1",
                        help="descending; r0 is excluded by default because its "
                             "median 2095 candidates are unrankable (1%% top-5)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    radii = [int(r) for r in args.radii.split(",")]
    if radii != sorted(radii, reverse=True):
        print("--radii must be descending: the ladder tries the strictest first.",
              file=sys.stderr)
        return 1

    steps = reference_steps(load_golden_dataset(variant="experimental"))
    print(f"{len(steps)} reference steps | ladder {radii}")
    closeness = SinkCloseness(radii[0])
    rng = random.Random(args.seed)

    per_radius: Dict[int, Dict[int, Tuple[int, Optional[int]]]] = {}
    for radius in radii:
        print(f"  measuring r{radius} ...", flush=True)
        per_radius[radius] = measure(steps, radius, closeness, rng)

    print()
    print(f"{'':22s} {'recall':>6s} {'median':>8s} {'max':>7s} "
          f"{'top-1':>8s} {'top-5':>8s} {'product':>8s}")
    print("-" * 72)
    for radius in radii:
        report(f"fixed r{radius}", list(per_radius[radius].values()), len(steps))

    # The ladder: first radius, strictest first, that produced anything at all.
    ladder, used = [], []
    for i in range(len(steps)):
        for radius in radii:
            count, rank = per_radius[radius][i]
            if count > 0:
                ladder.append((count, rank))
                used.append(radius)
                break
        else:
            ladder.append((0, None))
            used.append(None)
    print("-" * 72)
    report("ladder", ladder, len(steps))

    print()
    print("  radius the ladder ended up using:")
    for radius in radii:
        n = sum(1 for u in used if u == radius)
        if n:
            print(f"    r{radius}: {n} steps ({100 * n / len(steps):.0f}%)")
    silent = sum(1 for u in used if u is None)
    if silent:
        print(f"    none: {silent} steps ({100 * silent / len(steps):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
