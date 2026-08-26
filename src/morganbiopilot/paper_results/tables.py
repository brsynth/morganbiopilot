"""The result row and the tables printed from it. Standard library only.

Split out of `compare_policies` so that reading results does not require the
machinery that produced them. `compare_policies` pulls in RDKit, pandas and the
whole search stack at import time; re-tabulating a finished TSV needs none of it,
and a laptop without the chemistry environment can still read the numbers.

`compare_policies` re-exports `Row` and `summarise`, so existing imports of
`compare_policies.Row` keep working -- this is a move, not an interface change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Row:
    """One (policy, target, repeat) run.

    Note what is *not* here: the budget. A run is executed once at the largest
    budget requested and the whole solve-rate curve is obtained by thresholding
    `solved_at`, so one row carries every column of the curve.
    """

    policy: str
    target: str
    repeat: int
    seed: int
    expandable: bool
    solved: bool
    n_expansions: int
    solved_at: Optional[int]    # expansions used when solved; None otherwise
    stopped_because: str
    n_molecules: int
    n_reactions: int
    n_routes: int
    shortest_route: Optional[int]
    reference_route: int
    ec_fraction: float          # share of route steps with a native EC
    elapsed_s: float
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_fallbacks: int = 0


def _mean_std(values) -> tuple:
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def summarise(rows: List[Row], budgets: List[int], seeds: List[int],
              pathways: Dict, require_ec: bool) -> None:
    by_policy: Dict[str, List[Row]] = {}
    for row in rows:
        by_policy.setdefault(row.policy, []).append(row)

    # ---- solve-rate curve over budgets, mean +/- std across seeds -------------
    head = "".join(f"{'N=' + str(b):>13s}" for b in budgets)
    print(f"\nSOLVE RATE BY BUDGET (mean +/- sd over {len(seeds)} seed(s))")
    print(f"{'policy':44s}{head}{'exp2sol':>10s}{'ceiling':>9s}")
    print("-" * (44 + 13 * len(budgets) + 19))

    for policy, group in by_policy.items():
        line = f"{policy:44s}"
        for budget in budgets:
            per_seed = []
            for seed in seeds:
                runs = [r for r in group if r.seed == seed]
                if not runs:
                    continue
                hit = sum(1 for r in runs if r.solved_at is not None and r.solved_at <= budget)
                per_seed.append(100.0 * hit / len(runs))
            mean, sd = _mean_std(per_seed)
            # A deterministic policy contributes one seed even when several were
            # requested, so the +-sd column would read a spurious 0 next to rows
            # that genuinely varied. Report the spread only where there is one.
            line += (f"{mean:8.0f}+-{sd:<3.0f}" if len(per_seed) > 1
                     else f"{mean:12.0f}%")
        solved_at = [r.solved_at for r in group if r.solved_at is not None]
        exp2sol, _ = _mean_std(solved_at)
        ceiling = 100.0 * sum(r.expandable for r in group) / len(group)
        line += f"{exp2sol:10.1f}{ceiling:8.0f}%"
        print(line)

    # ---- the rest, at the largest budget --------------------------------------
    print(f"\nAT N={budgets[-1]}")
    print(f"{'policy':44s}{'route':>7s}{'ref':>6s}{'EC%':>7s}{'time':>9s}"
          f"{'calls':>7s}{'fallbk':>8s}")
    print("-" * 88)
    for policy, group in by_policy.items():
        lengths = [r.shortest_route for r in group if r.shortest_route is not None]
        refs = [r.reference_route for r in group if r.shortest_route is not None]
        ecs = [r.ec_fraction for r in group if r.ec_fraction == r.ec_fraction]
        print(
            f"{policy:44s}"
            f"{(sum(lengths)/len(lengths) if lengths else float('nan')):7.1f}"
            f"{(sum(refs)/len(refs) if refs else float('nan')):6.1f}"
            f"{(100*sum(ecs)/len(ecs) if ecs else float('nan')):7.1f}"
            f"{sum(r.elapsed_s for r in group):8.0f}s"
            f"{sum(r.llm_calls for r in group):7d}"
            f"{sum(r.llm_fallbacks for r in group):8d}"
        )

    # ---- stratified by reference route length ---------------------------------
    # AOT* stratifies by SC-score quartile; the natural analogue here is how many
    # steps the curated route takes, which is what the search has to rediscover.
    strata = sorted({len(p) for p in pathways.values()})
    # LASER and BioNavi-NP ship targets without attested routes, so there is nothing
    # to stratify by and the table would print as a bare header over blank rows --
    # which reads like a policy that solved nothing rather than a benchmark that
    # cannot ask the question.
    if not strata:
        print("\n(no stratified table: this benchmark carries no reference routes)")
    else:
        print(f"\nSOLVE RATE AT N={budgets[-1]} BY REFERENCE ROUTE LENGTH")
        print(f"{'policy':44s}" + "".join(f"{str(s) + ' steps':>11s}" for s in strata))
        print("-" * (44 + 11 * len(strata)))
        for policy, group in by_policy.items():
            line = f"{policy:44s}"
            for steps in strata:
                runs = [r for r in group if r.reference_route == steps]
                if not runs:
                    line += f"{'-':>11s}"
                    continue
                hit = sum(1 for r in runs if r.solved)
                line += f"{hit:>6d}/{len(runs):<4d}"
            print(line)

    print(
        "\nexp2sol = mean expansions used on runs that solved (search efficiency);\n"
        "ceiling = share of runs whose target was expandable at all — a solve rate\n"
        "is only interpretable against it. ref = mean reference route length on the\n"
        "same runs. fallbk = agent decisions the model could not supply."
    )
    if require_ec:
        print(
            "\nNOTE: --require-ec is on, so EC% is 100% by construction and says nothing\n"
            "about the policies. Re-run without it to make that column informative."
        )
