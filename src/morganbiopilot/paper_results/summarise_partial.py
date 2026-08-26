"""Rebuild the solve-rate tables from a `.partial.tsv`, without the job.

`compare_policies` streams one fully-formed `Row` per (policy, target, seed) as it
completes and prints its tables only at the end. A job killed at the wall -- or one
stuck on a single pathological target -- therefore holds complete measurements it
never reports. This reads the stream back and prints the same tables.

The number that must be read first is COVERAGE. Runs finish in completion order and
solved runs finish fastest, so a partial file over-states the solve rate: on one
measured campaign the fastest 20% of runs were 96% solved against 68% overall. A
partial at 99% of its targets is a result; one at 60% is not, and the header says
which one you are looking at.

Budgets are not stored per row -- a run is executed once at the largest budget and
the curve is obtained by thresholding `solved_at` -- so they must be supplied, and
must match the ones the job was launched with or the columns are mislabelled.

    python -m morganbiopilot.paper_results.summarise_partial \
        results/compare_policies/base_laser_1452298.partial.tsv \
        --budgets 10,25,50,100,200
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

from morganbiopilot.paper_results.tables import Row, summarise


def _opt_int(text: str) -> Optional[int]:
    return int(text) if text not in ("", "None") else None


def read_rows(path: Path) -> List[Row]:
    """Parse a partial (or final) TSV back into `Row` objects.

    csv gives strings for everything, and `summarise` compares `solved_at` with
    `<=` and sums `expandable` -- so the coercion below is not cosmetic.
    """
    names = {f.name for f in fields(Row)}
    rows: List[Row] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            if rec.get("policy") in (None, "", "policy"):
                continue  # a header re-emitted mid-file, or a torn last line
            missing = names - rec.keys()
            if missing:
                raise SystemExit(f"{path.name}: missing columns {sorted(missing)}")
            rows.append(Row(
                policy=rec["policy"],
                target=rec["target"],
                repeat=int(rec["repeat"]),
                seed=int(rec["seed"]),
                expandable=rec["expandable"] == "True",
                solved=rec["solved"] == "True",
                n_expansions=int(rec["n_expansions"]),
                solved_at=_opt_int(rec["solved_at"]),
                stopped_because=rec["stopped_because"],
                n_molecules=int(rec["n_molecules"]),
                n_reactions=int(rec["n_reactions"]),
                n_routes=int(rec["n_routes"]),
                shortest_route=_opt_int(rec["shortest_route"]),
                reference_route=int(rec["reference_route"]),
                ec_fraction=float(rec["ec_fraction"]),
                elapsed_s=float(rec["elapsed_s"]),
                llm_calls=int(rec["llm_calls"] or 0),
                llm_input_tokens=int(rec["llm_input_tokens"] or 0),
                llm_output_tokens=int(rec["llm_output_tokens"] or 0),
                llm_fallbacks=int(rec["llm_fallbacks"] or 0),
            ))
    if not rows:
        raise SystemExit(f"{path.name}: no rows")
    return rows


def fake_pathways(rows: List[Row]) -> Dict:
    """`summarise` reads `pathways` only to get the set of reference lengths.

    Benchmarks without reference routes store 0 there; dropping it keeps the
    stratified table from collapsing into a single meaningless '0 steps' column.
    """
    return {n: [None] * n for n in {r.reference_route for r in rows} if n > 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--budgets", default="10,25,50,100,200",
                    help="must match the job's --budgets, or the columns lie")
    ap.add_argument("--expected-targets", type=int, default=0,
                    help="targets the job was given; enables the coverage warning")
    args = ap.parse_args()

    rows = read_rows(args.tsv)
    budgets = [int(b) for b in args.budgets.split(",") if b]
    seeds = sorted({r.seed for r in rows})
    policies = sorted({r.policy for r in rows})

    print(f"{args.tsv.name}: {len(rows)} runs | "
          f"{len(policies)} polic{'y' if len(policies) == 1 else 'ies'} | "
          f"seeds {seeds}")

    for policy in policies:
        mine = [r for r in rows if r.policy == policy]
        done = {r.target for r in mine}
        if not args.expected_targets:
            print(f"  {policy:32s} {len(done):4d} targets, {len(mine)} runs")
            continue
        # Targets AND runs. Counting only distinct targets hides a missing
        # (seed, target) whenever another seed covers that target -- which is
        # exactly what a multi-seed partial looks like near the end, and it would
        # report 100% while a run is still outstanding.
        want_runs = args.expected_targets * len(seeds)
        pct_t = 100.0 * len(done) / args.expected_targets
        pct_r = 100.0 * len(mine) / want_runs
        flag = "" if min(pct_t, pct_r) >= 95 else "   <-- too partial to read"
        print(f"  {policy:32s} {len(done):3d}/{args.expected_targets} targets "
              f"({pct_t:.0f}%), {len(mine):3d}/{want_runs} runs ({pct_r:.0f}%){flag}")

    if not args.expected_targets:
        print("\nCOVERAGE unknown: pass --expected-targets. Solved runs finish first,\n"
              "so a partial file over-states the solve rate.")

    summarise(rows, budgets, seeds, fake_pathways(rows), require_ec=False)


if __name__ == "__main__":
    main()
