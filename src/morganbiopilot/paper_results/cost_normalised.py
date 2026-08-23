"""Solve rate against compute, not against expansions.

Every planning paper we borrow metrics from -- Retro*, EG-MCTS, RetroGraph, and the
Syntheseus standardisation on top of them -- reports solve rate against the number of
one-step model calls. That normalisation rests on an assumption: the one-step model is a
neural network, so it dominates the cost and the search itself is free.

**That assumption is inverted here.** Our one-step model is a counted-ECFP prefilter
followed by RDKit validation: microseconds, no GPU. The expensive component is the search
policy, because an LLM `select` costs ~2 s and a few thousand tokens while UCT's costs
microseconds. Reporting solve rate per expansion therefore flatters the LLM arms by
several orders of magnitude, and the flattery is invisible in the standard table.

So this module re-reads the TSVs `compare_policies` writes and reports the same solve
rates against wall-clock and tokens. Nothing is re-run; no GPU is needed.

Why the time curve is measured rather than estimated
----------------------------------------------------
`compare_policies` runs with `stop_on_first_pathway`, so a solved run halts at the moment
it solves. For those runs `n_expansions == solved_at` and `elapsed_s` *is* the time to
solution -- not a rate multiplied by a count. The share of runs with `elapsed_s <= T` is
therefore an exact solve-rate-at-T-seconds.

It is exact only up to a censoring point. An unsolved run stopped because it exhausted its
*expansion* budget, not its time, so we do not know whether it would have solved in more
seconds. Past `min(elapsed_s)` over the unsolved runs of a policy, the time curve is a
lower bound. The table marks those cells and prints the censoring point per policy, which
is the honest way to show a censored measurement rather than quietly truncating the axis.

Reading the output
------------------
Three tables, and the third is the one to look at first:

  BY EXPANSION BUDGET   the familiar curve, reproduced here so the two sit side by side
  BY TIME BUDGET        the same runs against seconds
  COST PER SOLVED       seconds, expansions, LLM calls and tokens per target solved

A policy that wins the first table and loses the third has bought expansions with compute,
which is a result worth stating, not a flaw worth hiding.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from morganbiopilot.core.paths import RESULTS_DIR


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One row of a compare_policies TSV, typed."""

    policy: str
    target: str
    seed: int
    expandable: bool
    solved: bool
    n_expansions: int
    solved_at: Optional[int]
    stopped_because: str
    shortest_route: Optional[int]
    reference_route: int
    ec_fraction: float
    elapsed_s: float
    llm_calls: int
    llm_input_tokens: int
    llm_output_tokens: int
    source: str


def _int(value: str) -> Optional[int]:
    # csv.DictWriter wrote `None` as the empty string; every other absence is a bug
    # we would rather see than paper over with a zero.
    return int(value) if value not in ("", "None") else None


def _float(value: str) -> float:
    return float(value) if value not in ("", "None") else float("nan")


def load(patterns: Sequence[str]) -> List[Run]:
    """Read every TSV matching the patterns into one flat list.

    Several files are the normal case: baselines come from a CPU job and the LLM arms
    from a GPU job, and the whole point is to put them in one table. The `source` field
    keeps track of which file a row came from, because a policy name colliding across
    two jobs run on different target samples would otherwise pool silently.
    """
    paths: List[str] = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if not hits:
            raise SystemExit(f"no file matches {pattern!r}")
        paths.extend(hits)

    runs: List[Run] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh, delimiter="\t"):
                runs.append(Run(
                    policy=rec["policy"],
                    target=rec["target"],
                    seed=int(rec["seed"]),
                    expandable=rec["expandable"] == "True",
                    solved=rec["solved"] == "True",
                    n_expansions=int(rec["n_expansions"]),
                    solved_at=_int(rec["solved_at"]),
                    stopped_because=rec["stopped_because"],
                    shortest_route=_int(rec["shortest_route"]),
                    reference_route=int(rec["reference_route"]),
                    ec_fraction=_float(rec["ec_fraction"]),
                    elapsed_s=_float(rec["elapsed_s"]),
                    llm_calls=int(rec.get("llm_calls") or 0),
                    llm_input_tokens=int(rec.get("llm_input_tokens") or 0),
                    llm_output_tokens=int(rec.get("llm_output_tokens") or 0),
                    source=path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
                ))
    return runs


def by_policy(runs: Sequence[Run]) -> Dict[str, List[Run]]:
    out: Dict[str, List[Run]] = {}
    for run in runs:
        out.setdefault(run.policy, []).append(run)
    return out


def _mean_sd(values: Sequence[float]) -> tuple:
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def _rate_over_seeds(group: Sequence[Run], hit) -> tuple:
    """Mean +/- sd of a per-seed solve rate.

    Deterministic policies contribute a single seed even when three were requested, so
    the spread is reported only where there is one -- printing `+-0` next to rows that
    genuinely varied would suggest a precision we did not measure.
    """
    seeds = sorted({r.seed for r in group})
    per_seed = []
    for seed in seeds:
        runs = [r for r in group if r.seed == seed]
        if runs:
            per_seed.append(100.0 * sum(1 for r in runs if hit(r)) / len(runs))
    mean, sd = _mean_sd(per_seed)
    return mean, sd, len(per_seed)


def _cell(mean: float, sd: float, n_seeds: int) -> str:
    return f"{mean:8.0f}+-{sd:<3.0f}" if n_seeds > 1 else f"{mean:12.0f}%"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _width(groups: Dict[str, List[Run]]) -> int:
    """Policy column wide enough for the longest name actually present.

    A fixed width silently misaligns every table as soon as one arm is called
    `llm[anthropic:claude-sonnet-5|medium|tools=none]`, which is 47 characters.
    """
    return max([len(p) for p in groups] + [20]) + 2


def table_by_expansions(groups: Dict[str, List[Run]], budgets: Sequence[int]) -> None:
    """The familiar curve. Reproduced so the reader can put the two side by side."""
    w = _width(groups)
    head = "".join(f"{'N=' + str(b):>13s}" for b in budgets)
    print("\nSOLVE RATE BY EXPANSION BUDGET")
    print(f"{'policy':{w}s}{head}")
    print("-" * (w + 13 * len(budgets)))
    for policy, group in sorted(groups.items()):
        line = f"{policy:{w}s}"
        for budget in budgets:
            line += _cell(*_rate_over_seeds(
                group, lambda r, b=budget: r.solved_at is not None and r.solved_at <= b))
        print(line)


def table_by_time(groups: Dict[str, List[Run]], times: Sequence[float]) -> None:
    """The same runs against seconds, with the censoring made explicit.

    `elapsed_s` on a solved run is the time to solution (see the module docstring). On an
    unsolved run it is the time the run spent before its *expansion* budget ran out, which
    says nothing about whether more seconds would have solved it. So for each policy the
    curve is exact below the smallest such `elapsed_s` and a lower bound above it; the
    marked cells are the ones a reader must not read as an endpoint.

    Only runs that stopped on `budget` censor the curve. A run that stopped on
    `frontier_empty` explored everything reachable and failed: more seconds would not have
    saved it, so it is a genuine failure at every time budget. Ignoring that distinction
    put the censoring point at 0 s -- 38 of these targets exhaust in under a second -- and
    starred every cell in the table, which is the same as reporting nothing.
    """
    w = _width(groups)
    head = "".join(f"{_fmt_time(t):>13s}" for t in times)
    print("\nSOLVE RATE BY TIME BUDGET (seconds of wall-clock per target)")
    print(f"{'policy':{w}s}{head}{'censored>':>12s}")
    print("-" * (w + 13 * len(times) + 12))
    for policy, group in sorted(groups.items()):
        unsolved = [r.elapsed_s for r in group
                    if not r.solved and r.stopped_because in ("budget", "time")]
        censor = min(unsolved) if unsolved else float("inf")
        line = f"{policy:{w}s}"
        for t in times:
            mean, sd, n = _rate_over_seeds(
                group, lambda r, t=t: r.solved and r.elapsed_s <= t)
            cell = _cell(mean, sd, n)
            # A trailing marker rather than a footnote per row: the eye needs to see
            # which numbers are bounds while scanning the row.
            line += (cell[:-1] + "*") if t > censor else cell
        line += (f"{_fmt_time(censor):>12s}" if unsolved else f"{'--':>12s}")
        print(line)
    print("  * lower bound: past this point some runs were stopped by the expansion "
          "budget, not by time.")


def _fmt_time(seconds: float) -> str:
    if seconds == float("inf"):
        return "inf"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def table_cost_per_solved(groups: Dict[str, List[Run]]) -> None:
    """What a solved target actually cost.

    `s/exp` is the number that makes the expansion normalisation indefensible here: it is
    the per-decision price of the policy, and it differs between arms by orders of
    magnitude while the expansion axis treats them as equal units.

    Total time is charged over *all* runs, solved or not, because a planner does not get
    to spend nothing on the targets it fails. Dividing that by the number solved is the
    cost of a success, which is the quantity a user of the system pays.
    """
    w = _width(groups)
    print("\nCOST PER SOLVED TARGET (all runs charged, solved or not)")
    print(f"{'policy':{w}s}{'solved':>8s}{'s/exp':>9s}{'s/solve':>10s}"
          f"{'exp/solve':>11s}{'calls':>8s}{'ktok':>8s}{'ktok/solve':>12s}")
    print("-" * (w + 66))
    for policy, group in sorted(groups.items()):
        n_solved = sum(1 for r in group if r.solved)
        total_s = sum(r.elapsed_s for r in group)
        total_exp = sum(r.n_expansions for r in group)
        total_calls = sum(r.llm_calls for r in group)
        total_tok = sum(r.llm_input_tokens + r.llm_output_tokens for r in group)
        per = (lambda x: x / n_solved) if n_solved else (lambda x: float("nan"))
        print(
            f"{policy:{w}s}{n_solved:8d}"
            f"{(total_s / total_exp if total_exp else float('nan')):9.3f}"
            f"{per(total_s):10.1f}{per(total_exp):11.1f}"
            f"{total_calls:8d}{total_tok / 1000:8.0f}{per(total_tok / 1000):12.1f}"
        )
    print("  s/exp = seconds per expansion, i.e. the price of one `select` decision.\n"
          "  Expansion-normalised solve rates treat these as equal units. They are not.")


def table_routes(groups: Dict[str, List[Run]]) -> None:
    """Route quality, and its biological reading.

    With an E. coli chassis as the sink, the length of a solved route is the number of
    heterologous enzymes the strain would have to acquire. That makes route length a
    engineering cost rather than an aesthetic preference, which is not true of route
    length in a purely chemical retrosynthesis paper.
    """
    w = _width(groups)
    # The BioNavi targets carry no reference route, so `reference_route` is 0 on every
    # row and the delta column would read as "+2.9 steps too long" against nothing.
    # Drop both columns rather than print a comparison to a missing reference.
    has_ref = any(r.reference_route > 0 for g in groups.values() for r in g)
    print("\nROUTE QUALITY ON SOLVED RUNS")
    ref_head = f"{'ref':>7s}{'delta':>8s}" if has_ref else ""
    print(f"{'policy':{w}s}{'route':>8s}{ref_head}{'EC%':>8s}")
    print("-" * (w + 16 + (15 if has_ref else 0)))
    for policy, group in sorted(groups.items()):
        solved = [r for r in group if r.shortest_route is not None]
        if not solved:
            print(f"{policy:{w}s}{'--':>8s}")
            continue
        route, _ = _mean_sd([r.shortest_route for r in solved])
        ecs = [r.ec_fraction for r in solved if r.ec_fraction == r.ec_fraction]
        ec = 100 * sum(ecs) / len(ecs) if ecs else float("nan")
        cells = f"{policy:{w}s}{route:8.1f}"
        if has_ref:
            ref, _ = _mean_sd([r.reference_route for r in solved])
            cells += f"{ref:7.1f}{route - ref:+8.1f}"
        print(cells + f"{ec:8.1f}")
    print("  route = mean shortest solved route = heterologous enzymes to introduce.")
    if has_ref:
        print("  ref = mean reference route length on the same runs (curated set only).")
    else:
        print("  no reference routes in this sample, so ref and delta are omitted.")


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------

def _two_sided_binomial(k: int, n: int) -> float:
    """Exact two-sided sign test, 2 * min(tail), capped at 1.

    Written out rather than pulled from scipy: this is the whole test, and a reader
    checking whether a 3-point difference means anything should be able to see it.
    """
    if n == 0:
        return float("nan")
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(lower, upper))


def paired(runs: Sequence[Run], a: str, b: str, budget: int) -> None:
    """Discordant-target counts between two policies at one budget.

    The design is paired -- same targets, same seeds -- so the comparison that carries
    information is the count of targets one policy solves and the other does not. On 60
    targets a 3-point difference in solve rate is under two discordant targets, which no
    test will call a difference; printing the discordant counts makes that visible where
    two overlapping error bars only hint at it.
    """
    index = {}
    for run in runs:
        if run.policy in (a, b):
            index.setdefault((run.target, run.seed), {})[run.policy] = run

    both = [pair for pair in index.values() if a in pair and b in pair]
    if not both:
        raise SystemExit(f"no (target, seed) cell has both {a!r} and {b!r}")

    def hit(run: Run) -> bool:
        return run.solved_at is not None and run.solved_at <= budget

    only_a = sum(1 for p in both if hit(p[a]) and not hit(p[b]))
    only_b = sum(1 for p in both if hit(p[b]) and not hit(p[a]))
    n_disc = only_a + only_b
    print(f"\nPAIRED COMPARISON AT N={budget} over {len(both)} (target, seed) cells")
    print(f"  {a}")
    print(f"  {b}")
    print(f"  both solve      {sum(1 for p in both if hit(p[a]) and hit(p[b])):4d}")
    print(f"  neither         {sum(1 for p in both if not hit(p[a]) and not hit(p[b])):4d}")
    print(f"  only the first  {only_a:4d}")
    print(f"  only the second {only_b:4d}")
    print(f"  discordant      {n_disc:4d}   two-sided sign test "
          f"p = {_two_sided_binomial(min(only_a, only_b), n_disc):.3f}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-read compare_policies TSVs and report solve rate against "
                    "compute rather than against expansions.")
    p.add_argument("--tsv", nargs="+", required=True,
                   help="TSV paths or globs, e.g. 'results/compare_policies/base_*.tsv'")
    p.add_argument("--budgets", default="10,25,50",
                   help="expansion budgets for the reference curve")
    p.add_argument("--times", default="1,10,60,300,1800",
                   help="wall-clock budgets in seconds per target")
    p.add_argument("--expandable-only", action="store_true",
                   help="drop targets no rule could expand at all; a solve rate is only "
                        "interpretable against that ceiling")
    p.add_argument("--paired", nargs=2, metavar=("A", "B"),
                   help="two policy names to compare target by target")
    p.add_argument("--paired-budget", type=int, default=50)
    p.add_argument("--out", default=None,
                   help="optional TSV of the per-policy cost summary")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    budgets = [int(b) for b in args.budgets.split(",") if b]
    times = [float(t) for t in args.times.split(",") if t]

    runs = load(args.tsv)
    if args.expandable_only:
        before = len(runs)
        runs = [r for r in runs if r.expandable]
        print(f"kept {len(runs)}/{before} runs whose target was expandable")

    sources = sorted({r.source for r in runs})
    targets = sorted({r.target for r in runs})
    print("=" * 78)
    print(f"{len(runs)} runs | {len(by_policy(runs))} policies | {len(targets)} targets")
    print(f"files: {', '.join(sources)}")
    # Two jobs run on different target samples would pool into a table comparing
    # policies on different problems. Cheap to check, expensive to miss.
    per_source = {s: {r.target for r in runs if r.source == s} for s in sources}
    if len({frozenset(t) for t in per_source.values()}) > 1:
        print("WARNING: the files do not cover the same targets. Cross-file rows are "
              "not comparable; check --max-targets and --sample-seed matched.")
        for s, t in per_source.items():
            print(f"  {s}: {len(t)} targets")
    print("=" * 78)

    groups = by_policy(runs)
    table_by_expansions(groups, budgets)
    table_by_time(groups, times)
    table_cost_per_solved(groups)
    table_routes(groups)

    if args.paired:
        paired(runs, args.paired[0], args.paired[1], args.paired_budget)

    if args.out:
        out = RESULTS_DIR / args.out if not args.out.startswith("/") else args.out
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["policy", "n_runs", "n_solved", "s_per_expansion",
                             "s_per_solve", "exp_per_solve", "llm_calls",
                             "tokens", "tokens_per_solve", "mean_route"])
            for policy, group in sorted(groups.items()):
                n_solved = sum(1 for r in group if r.solved)
                total_s = sum(r.elapsed_s for r in group)
                total_exp = sum(r.n_expansions for r in group)
                total_tok = sum(r.llm_input_tokens + r.llm_output_tokens for r in group)
                lengths = [r.shortest_route for r in group if r.shortest_route is not None]
                writer.writerow([
                    policy, len(group), n_solved,
                    f"{total_s / total_exp:.4f}" if total_exp else "",
                    f"{total_s / n_solved:.1f}" if n_solved else "",
                    f"{total_exp / n_solved:.1f}" if n_solved else "",
                    sum(r.llm_calls for r in group), total_tok,
                    f"{total_tok / n_solved:.0f}" if n_solved else "",
                    f"{sum(lengths) / len(lengths):.2f}" if lengths else "",
                ])
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
