"""Policy comparison on the golden set — the paper's main table.

Runs every requested policy on the same targets, the same rule set, and the same
expansion budget, and reports the metrics of section 8 of the project note:

- solve rate **as a curve over budgets**, not at a single point;
- expansions-to-solution among solved targets (search efficiency);
- route length (shortest solved route), against the reference route length;
- enzymatic plausibility: the share of route steps whose rule carries a native EC;
- cost: expansions, wall-clock, and for LLM policies the call and token counts.

Why a curve, and how it is obtained for free
--------------------------------------------
Reporting one solve rate at one budget cannot support an efficiency claim. AOT*
(Song et al., ACL 2026) reports solve rates at N=100/300/500 and at N=20..100, and
its central result — "3-5x fewer iterations" — exists only because of that curve.

The curve costs nothing extra here. With `stop_on_first_pathway`, a run halts the
moment the target is solved, so `n_expansions` *is* the expansions-to-solution.
Solve rate at budget B is then the share of runs with `solved_at <= B`. One run per
(policy, target, seed) at the largest budget yields every smaller budget's number.

Seeds and error bars
--------------------
LLM runs cannot be pinned: sampling parameters are rejected on the Claude
generation used here. RetroAgent (Zhu et al., COLM 2026) faces the same problem
and solves it by running 10 experiments per target whose only stochasticity is the
shuffled candidate ordering, reporting mean and standard deviation. `--seeds` does
the same: the seed drives the frontier presentation shuffle, so repeats differ even
under greedy decoding. A single seed gives a draw, not an estimate.

Only the policy changes between rows. That is the point of the table, and it is
enforced by construction: every policy is handed the same `RuleSet`, the same
prefilter, and the same budget.

Two things this script reports that are easy to omit and change how the numbers
read:

- **Unexpandable targets.** A target with zero applicable rules is unsolvable
  before any policy runs, so it caps the achievable solve rate. The `ceiling`
  column states that cap; a solve rate quoted without it is misleading.
- **Agent fallbacks.** When a model's answer is unusable (refusal, unparseable,
  out of range) the policy takes the first candidate and records it. A run with
  many fallbacks is partly BFS, and the table says so.

Usage:

    python -m morganbiopilot.paper_results.compare_policies --budget 100
    python -m morganbiopilot.paper_results.compare_policies \\
        --budget 100 --llm --models claude-opus-5 --efforts low,medium

LLM policies cost money and are therefore opt-in: without `--llm` only the
deterministic baselines run.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional

# The row and the tables it prints live in a standard-library-only module, so a
# finished TSV can be re-tabulated without RDKit, pandas or the search stack.
# Re-exported here: `compare_policies.Row` is still a valid import.
from morganbiopilot.paper_results.tables import Row, summarise  # noqa: F401

# `agents.state` is stdlib-only apart from the search graph, so importing the
# frontier defaults here costs nothing; the LLM backends stay lazily imported in
# `build_policies` so a classical run needs no API client installed.
from morganbiopilot.agents.state import DEFAULT_TOP_K
from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.building_blocks import use_sink
from morganbiopilot.core.golden_dataset import load_golden_dataset
from morganbiopilot.core.paths import RESULTS_DIR
from morganbiopilot.core.rules import load_rules
from morganbiopilot.core.target_list import load_target_list
from morganbiopilot.multi_step.heuristics import SinkCloseness
from morganbiopilot.multi_step.mcts import MCTS
from morganbiopilot.multi_step.policy import GreedySimilarity  # noqa: F401
from morganbiopilot.multi_step.policy import (
    BreadthFirst, DepthFirst, GreedyECFP, RandomPolicy,
)
from morganbiopilot.multi_step.routes import extract_routes, save_routes
from morganbiopilot.multi_step.search import search
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import prefilter_from_rules


def first_solved_step(result) -> Optional[int]:
    """Expansions used when the graph first became solved; None if never.

    Read from the decision trace rather than from the final expansion count, so it
    means the same thing whether or not the run stopped at the first route.
    """
    step = next((d["step"] for d in result.decisions if d["solved_after"]), None)
    if step is None and result.solved:
        # Solved before any expansion: the target was already in the sink.
        return 0
    return step


def enzymatic_fraction(result, rule_ec) -> float:
    """Share of the shortest solved route's steps whose rule carries an EC.

    Section 8 asks for enzymatic plausibility of routes. This is its cheap,
    fully deterministic half: does a real enzyme back each step. Cofactor balance
    is the other half and is not computed here.

    **Degenerate under `--require-ec`.** The enzymatic reality filter already
    guarantees every rule in the graph carries an EC, so this returns 1.0 for every
    route and every policy — it measures the filter, not the policy. It is only
    informative with the filter off, which is the configuration in which enzymatic
    plausibility is a genuine differentiator between policies. Run both and report
    the unfiltered one for this metric.
    """
    routes = result.pathways()
    if not routes or rule_ec is None:
        return float("nan")

    shortest = min(routes, key=len)
    if not shortest:
        return float("nan")

    # The neighbour's merged annotation, not the representative rule's: `expand` folds
    # templates reaching the same molecule set into one node and unions their EC.
    with_ec = sum(
        1 for rxn_id in shortest
        if (result.graph.reactions[rxn_id].neighbour.ec_numbers
            or rule_ec.ec[result.graph.reactions[rxn_id].rule_idx])
    )
    return with_ec / len(shortest)


def build_policies(args, radius: int, rule_ec, seeds, rules=None, prefilter=None,
                   ranker=None) -> List:
    """Factories taking a seed and returning a fresh policy. LLM policies opt-in."""
    closeness = SinkCloseness(radius)

    def greedy_similarity():
        if ranker is None:
            raise SystemExit("greedy_similarity needs --ranker native_similarity")
        return GreedySimilarity(rules, prefilter, ranker)

    available = {
        "greedy_similarity": lambda seed: greedy_similarity(),
        "bfs": lambda seed: BreadthFirst(),
        "dfs": lambda seed: DepthFirst(),
        "random": lambda seed: RandomPolicy(seed=seed),
        "greedy": lambda seed: GreedyECFP(closeness),
        # UCT with static evaluation -- see MCTS for why rollouts are absent.
        "mcts": lambda seed: MCTS(closeness),
    }

    policies = []
    for name in args.policies.split(","):
        name = name.strip()
        # `--policies ""` runs the agent arms alone. Not a convenience: against a paid
        # hosted model, re-running baselines that are already measured on the same 60
        # targets spends money to reproduce a number we have.
        if not name:
            continue
        # A results row records the policy under its `name` attribute, which is not
        # always the key used here: GreedyECFP writes "greedy_ecfp". Reading an arm off
        # a TSV and rerunning it with the name shown there must work -- it did not, and
        # cost a job.
        name = {"greedy_ecfp": "greedy"}.get(name, name)
        if name not in available:
            raise SystemExit(f"unknown policy {name!r}; choose from {sorted(available)}")
        policies.append(available[name])

    if args.llm:
        from morganbiopilot.agents.backends import make_backend
        from morganbiopilot.agents.policy import LLMPolicy
        # `SinkCloseness` is already imported at module level and used above for the
        # classical policies; re-importing it here would make the name local to this
        # function and break that earlier use.
        from morganbiopilot.agents.tools import ToolSurface, tooled, untooled
        from morganbiopilot.multi_step.plausibility import RoutePlausibility

        # One scorer for the whole grid. Its cache is keyed on reaction content, so
        # sharing it across targets and arms is safe and warms it; the model file is
        # loaded once instead of once per run.
        scorer = RoutePlausibility() if any(
            t.strip() in ("tooled_enz", "enz_only")
            for t in args.tooling.split(",")) else None

        for spec in args.models.split(","):
            spec = spec.strip()
            # `effort` is Anthropic-only. Sweeping it against a provider that
            # ignores it would fill the grid with duplicate runs presented as
            # different conditions, so those cells are skipped and announced.
            probe = make_backend(spec, effort="medium")
            efforts = args.efforts.split(",") if probe.supports_effort else ["n/a"]
            if not probe.supports_effort and args.efforts != "medium":
                print(f"  note: {spec} ignores effort; running one cell instead of "
                      f"{len(args.efforts.split(','))}")

            for effort in efforts:
                for tooling in args.tooling.split(","):
                    tooling = tooling.strip()
                    # Three conditions, not two. `tooled_plain` shows the same
                    # engine-computed columns as `tooled` but leaves the system
                    # prompt silent about what they mean -- the ablation that
                    # separates "grounding does not help" from "grounding was never
                    # explained". See `agents.policy.build_system_prompt`.
                    #
                    # `enz_only` and `tooled_enz` add the route-plausibility column
                    # (`multi_step.plausibility`). Two arms rather than one because they
                    # answer different questions: alone, whether a weak per-edge
                    # enzymatic score aggregated along the route carries anything at
                    # all; on top of `tooled`, whether it adds to grounding that is
                    # already there. `tooled` itself is left untouched, so every number
                    # already in the tables stays comparable.
                    #
                    # `ec_only` and `closeness_only` split `tooled` into its two
                    # columns, and they are the arms that test a mechanism rather than
                    # an effect. On BioNavi-NP the tooled agent gains +5 points between
                    # N=10 and N=50 where the untooled one gains +20, and solves in 6.7
                    # expansions against 16.5 -- the signature of a policy that has
                    # turned greedy. Closeness to the sink is the plausible culprit: it
                    # is a good first move and a bad exploration rule. If that is right,
                    # `ec_only` keeps the early advantage without the plateau and
                    # `closeness_only` reproduces the plateau on its own.
                    known = ("tooled", "tooled_plain", "untooled", "ec_only",
                             "closeness_only", "tooled_enz", "enz_only")
                    if tooling not in known:
                        raise SystemExit(f"unknown tooling {tooling!r}; choose from "
                                         f"{', '.join(known)}")
                    if tooling == "untooled":
                        surface = untooled()
                    elif tooling == "ec_only":
                        surface = ToolSurface(rule_ec=rule_ec)
                    elif tooling == "closeness_only":
                        surface = ToolSurface(closeness=SinkCloseness(radius))
                    elif tooling == "enz_only":
                        surface = ToolSurface(plausibility=scorer)
                    elif tooling == "tooled_enz":
                        surface = tooled(radius, rule_ec, plausibility=scorer)
                    else:
                        surface = tooled(radius, rule_ec)
                    policies.append(
                        lambda seed, sp=spec, e=effort.strip(), s=surface,
                        x=(tooling != "tooled_plain"): LLMPolicy(
                            tools=s,
                            backend=make_backend(sp, effort=(e if e != "n/a" else "medium")),
                            top_k=args.top_k, seed=seed, explain=x,
                            # The frontier order is part of the environment; without
                            # these the view silently falls back to `_stratify`.
                            ranker=ranker, prefilter=prefilter,
                        )
                    )
    return policies


def output_path(args) -> Path:
    """Where the table goes. Computed before the run so the crash log can name it."""
    if args.out:
        return Path(args.out)
    return (RESULTS_DIR / "compare_policies" /
            f"r{args.radius}_b{max(int(b) for b in args.budgets.split(','))}"
            f"{'_ec' if args.require_ec else ''}.tsv")


def run(args, partial_path: Optional[Path] = None):
    # The sink defines what "solved" means, so it must be swapped before anything
    # caches it -- before the rules, before SinkCloseness, before any policy is
    # built. An external benchmark scores against its own available set: running
    # BioNavi-NP's targets against our 753-metabolite E. coli table would answer an
    # easier question than theirs and the numbers would not be comparable.
    if args.sink:
        use_sink(args.sink)
        print(f"sink overridden: {args.sink}")

    rules = load_rules(args.radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)
    if args.targets_file:
        listed = load_target_list(args.targets_file)
        if args.max_targets and args.max_targets < len(listed):
            # A uniform draw, fixed by seed. Selecting targets a baseline fails on
            # would score that baseline at zero by construction and flatter whatever
            # is compared against it; the sample must not know the results.
            import random as _random
            listed = _random.Random(args.sample_seed).sample(listed, args.max_targets)
            listed.sort(key=lambda t: t.name)
            print(f"sampled {len(listed)} targets (seed {args.sample_seed})")
        pathways = {p.name: p for p in listed}
        print(f"{len(pathways)} targets from {args.targets_file} "
              "(no reference routes: the stratified table collapses to one bucket)")
    else:
        pathways = {p.name: p for p in load_golden_dataset(args.variant)}

    names = args.targets.split(",") if args.targets else list(pathways)
    print(f"{len(rules)} rules at r{args.radius} | EC coverage {100*rule_ec.coverage:.1f}% "
          f"| require_ec={args.require_ec}")

    # A target with no applicable rule is unsolvable before any policy runs.
    expandable: Dict[str, bool] = {}
    for name in names:
        report = expand(pathways[name].target, rules, prefilter,
                        rule_ec=rule_ec, require_ec=args.require_ec)
        expandable[name] = bool(report.neighbours)
    n_expandable = sum(expandable.values())
    print(f"expandable targets: {n_expandable}/{len(names)} "
          f"-> solve-rate ceiling {100*n_expandable/len(names):.0f}%\n")

    budgets = sorted(int(b) for b in args.budgets.split(","))
    max_budget = budgets[-1]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"budgets reported: {budgets} (runs execute at {max_budget}) | seeds: {seeds}\n")

    if args.workers > 1:
        # Warm the sink caches on the main thread. They are `lru_cache`d, so concurrent
        # first calls are safe but would each read and fingerprint the sink before one
        # of them won -- several seconds and several hundred MB of duplicated work at
        # the exact moment every worker starts.
        from morganbiopilot.core.building_blocks import is_building_block, is_cofactor
        is_building_block("CCO")
        is_cofactor("O")
        print(f"running {args.workers} concurrent (seed, target) runs\n")

    rows: List[Row] = []
    # Fieldnames come from the dataclass, not from the first row, so the header is on
    # disk before any run finishes -- a job killed during its first target still leaves
    # a readable file.
    partial_fh = open(partial_path, "w", newline="", encoding="utf-8") if partial_path else None
    partial = None
    if partial_fh is not None:
        partial = csv.DictWriter(partial_fh, fieldnames=[f.name for f in fields(Row)],
                                 delimiter="\t")
        partial.writeheader()
        partial_fh.flush()
        print(f"streaming rows to {partial_path} as they complete\n")

    decisions_fh = None
    if partial_path is not None and args.save_decisions:
        decisions_path = partial_path.with_name(
            partial_path.name.replace(".partial.tsv", ".decisions.jsonl"))
        # LF everywhere: see route_corpus for why this is not cosmetic.
        decisions_fh = open(decisions_path, "w", encoding="utf-8", newline="\n")
        print(f"streaming the decision trace to {decisions_path}\n")

    try:
        # One ranker for every arm: the cap is a property of the environment, so a
        # policy comparison where the arms saw different graphs would measure nothing.
        from morganbiopilot.one_step.ranking import make_ranker
        ranker = make_ranker(args.ranker, rules)
        if ranker is not None:
            print(f"expansions ranked by {args.ranker}, capped at {args.top_n}\n")
        rows = _run_policies(args, rules, rule_ec, prefilter, pathways, names, seeds,
                             max_budget, expandable, rows, partial, partial_fh, ranker,
                             prefilter, decisions_fh)
    finally:
        if partial_fh is not None:
            partial_fh.close()
        if decisions_fh is not None:
            decisions_fh.close()

    # Completion order is nondeterministic under concurrency, so the table would
    # otherwise depend on scheduling. Sorting here makes the TSV byte-identical to the
    # sequential run.
    rows.sort(key=lambda r: (r.policy, r.seed, r.target))
    return rows, pathways


def _run_policies(args, rules, rule_ec, prefilter, pathways, names, seeds,
                  max_budget, expandable, rows, partial, partial_fh, ranker=None,
                  pf=None, decisions_fh=None):
    """The policy loop, split out of `run` only so the partial file can be closed
    in a `finally` without indenting the whole body."""
    for make_policy in build_policies(args, args.radius, rule_ec, seeds,
                                      rules, pf or prefilter, ranker):
        # A seed only changes two things: the frontier presentation shuffle, which
        # exists solely for the LLM, and `RandomPolicy`'s draw. Breadth-first,
        # depth-first, greedy and MCTS are deterministic functions of the graph, so
        # repeating them across seeds recomputes an identical answer. Measured, not
        # assumed: on 379 BioNavi-NP targets, zero changed result between seeds for
        # either bfs or dfs, and every classical row on the golden set reported
        # +-0. That triple cost is what pushed the first BioNavi job past its 12 h
        # limit before MCTS had run at all.
        probe = make_policy(seeds[0])
        stochastic = getattr(probe, "n_decisions", None) is not None \
            or getattr(probe, "name", "") == "random"
        policy_seeds = seeds if stochastic else seeds[:1]
        if policy_seeds is not seeds:
            print(f"  note: {getattr(probe, 'name', '?')} is deterministic; "
                  f"running 1 seed instead of {len(seeds)}")

        def run_one(job):
            """One (seed, target) run. Pure apart from its own policy instance."""
            repeat, seed, name = job
            policy = make_policy(seed)
            gold = pathways[name]
            t0 = time.perf_counter()
            result = search(
                gold.target, rules, prefilter, policy,
                budget=max_budget, max_depth=args.max_depth,
                rule_ec=rule_ec, require_ec=args.require_ec,
                stop_on_first_pathway=not args.exhaustive,
                ranker=ranker, top_n=args.top_n,
                max_seconds=args.max_seconds,
            )
            row = Row(
                policy=result.policy, target=name, repeat=repeat, seed=seed,
                expandable=expandable[name], solved=result.solved,
                n_expansions=result.n_expansions,
                # The step at which the graph first became solved, read off the
                # decision trace. Equal to `n_expansions` under stop_on_first_pathway,
                # which is the default -- but `--exhaustive` keeps spending the budget
                # after the first route, and taking the expansion count there would
                # silently report every solved run as having needed the full budget.
                # Solve rate at budget B is `solved_at <= B`, so that one substitution
                # would corrupt every number in the paper without failing anything.
                solved_at=first_solved_step(result),
                stopped_because=result.stopped_because,
                n_molecules=result.n_molecules, n_reactions=result.n_reactions,
                n_routes=len(result.pathways()),
                shortest_route=result.shortest_pathway_length,
                reference_route=len(gold),
                ec_fraction=enzymatic_fraction(result, rule_ec),
                elapsed_s=time.perf_counter() - t0,
                llm_calls=getattr(policy, "n_calls", 0),
                llm_input_tokens=getattr(policy, "total_input_tokens", 0),
                llm_output_tokens=getattr(policy, "total_output_tokens", 0),
                llm_fallbacks=getattr(policy, "n_fallbacks", 0),
            )
            line = (f"  {result.policy:44s} {name:26s} "
                    f"solved={str(result.solved):5s} exp={result.n_expansions:4d} "
                    f"mol={result.n_molecules:5d} {result.elapsed_s:6.1f}s")
            # Extracted inside the worker (it walks the graph, which is thread-local);
            # written on the main thread, so concurrent runs never race on a path.
            routes = (extract_routes(result, rule_ec)
                      if result.solved and args.save_routes else None)
            return row, line, routes, result

        def finish(row, line, routes, result):
            rows.append(row)
            print(line)
            # Append-and-flush before anything else can fail. Job 1308521 spent 10 h of
            # L40S on 358 of 360 runs, was cancelled at the wall, and wrote nothing at
            # all -- the final TSV is written once, after every run returns. The table
            # only survived because each run had also printed a line, and reconstructing
            # it from the log lost the seeds. One flushed row per run costs nothing and
            # makes a killed job a truncated result instead of no result.
            if partial is not None:
                partial.writerow(asdict(row))
                partial_fh.flush()
            if routes is not None:
                slug = "".join(ch if ch.isalnum() else "_" for ch in result.policy)
                # The environment belongs in the name, not only in the metadata. Without
                # it, a second run of the same policy and seed at another radius -- or
                # at the same radius with a different ranker -- silently overwrote the
                # first, and neither the name nor `meta` could tell the two apart. The
                # suffix goes last so that the target prefix and policy substring that
                # `visualize_routes` filters on still match.
                env = f"r{args.radius}__" + (f"{args.ranker}{args.top_n}"
                                             if args.ranker else "raw")
                save_routes(
                    routes,
                    RESULTS_DIR / "routes"
                    / f"{row.target}__{slug}__seed{row.seed}__{env}.json",
                    meta={"policy": result.policy, "target": row.target,
                          "seed": row.seed, "radius": args.radius,
                          "ranker": args.ranker, "top_n": args.top_n,
                          "require_ec": args.require_ec,
                          "expansions": result.n_expansions},
                )
                if args.print_routes and routes:
                    print(routes[0].render())

            # The decision trace, one line per expansion. `search` builds it and the
            # docstring calls a run "reconstructible without re-running" -- but nothing
            # persisted it, so every run recomputed `frontier_size` and threw it away.
            # That is the quantity the frontier-visibility argument rests on, and it
            # had to be approximated by graph size for want of this file.
            if decisions_fh is not None:
                for d in result.decisions:
                    decisions_fh.write(json.dumps({
                        "policy": result.policy, "target": row.target,
                        "seed": row.seed, "repeat": row.repeat,
                        "radius": args.radius, "ranker": args.ranker,
                        "top_n": args.top_n, "top_k": args.top_k, **d,
                    }) + "\n")
                decisions_fh.flush()

        jobs = [(repeat, seed, name)
                for repeat, seed in enumerate(policy_seeds) for name in names]

        # Runs are independent, so concurrency changes throughput and nothing else --
        # each carries its own seed and its own graph. It matters because the agent arms
        # are latency-bound, not compute-bound: the engine charges one expansion per
        # `select`, so decisions are strictly sequential within a run and the GPU sat at
        # `Running: 0-1 reqs` and 40 tok/s through a 3 h job whose server advertised a
        # concurrency ceiling of 57x. Overlapping targets is what fills it.
        if args.workers > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_one, job) for job in jobs]
                for future in as_completed(futures):
                    finish(*future.result())
        else:
            for job in jobs:
                finish(*run_one(job))

    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--radius", type=int, default=2, help="ECFP radius of the rule set")
    p.add_argument("--budgets", default="20,40,60,80,100",
                   help="comma-separated budgets to report; runs execute at the largest")
    p.add_argument("--max-depth", type=int, default=6, help="0 = unlimited")
    p.add_argument("--require-ec", action="store_true",
                   help="enzymatic reality filter: drop rules with no EC")
    p.add_argument("--policies", default="bfs,random,greedy")
    p.add_argument("--targets", default="", help="comma-separated names; default all 20")
    p.add_argument("--targets-file", default="",
                   help="name<TAB>SMILES list instead of the golden set "
                        "(e.g. data/bionavinp/testset.txt)")
    p.add_argument("--max-targets", type=int, default=0,
                   help="uniform subsample of --targets-file (0 = all); the agent "
                        "costs ~2 s per decision, so 379 targets is a day per condition")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="seed of that subsample, so the same targets recur across jobs")
    p.add_argument("--sink", default="",
                   help="alternative building-block table; use the benchmark's own "
                        "(e.g. data/bionavinp/bionavi-np_building_blocks.txt)")
    p.add_argument("--variant", default="experimental", choices=["experimental", "core"])
    p.add_argument("--seeds", default="0",
                   help="comma-separated seeds; drives the frontier presentation "
                        "shuffle, so >1 gives error bars on LLM policies")
    p.add_argument("--exhaustive", action="store_true",
                   help="keep spending the budget after the first route")
    p.add_argument("--ranker", default=None,
                   help="'native_similarity' orders rules before RDKit validation so "
                        "--top-n keeps the most promising. Applied to every arm.")
    p.add_argument("--top-n", type=int, default=None,
                   help="neighbours kept per expansion; needs --ranker")
    # Off by default, because a bound that fires turns a solve into a censored
    # measurement and every published rate here was measured unbounded. But without
    # it one target can hold a whole job: three jobs sat on a single unfinished
    # target for seven hours each, at 140/141, 19/20 and 58/60 runs done, and the
    # tables they had already earned were never printed. `route_recovery` has had
    # this bound from the start; the comparison path never did. Runs that hit it
    # report stopped_because=time and are visible as such in the TSV.
    p.add_argument("--max-seconds", type=float, default=None,
                   help="wall-clock bound on ONE search; unset means unbounded")
    p.add_argument("--llm", action="store_true", help="also run LLM policies (costs money)")
    p.add_argument("--models", default="claude-opus-5")
    p.add_argument("--efforts", default="medium")
    p.add_argument("--tooling", default="tooled,untooled",
                   help="comma-separated: tooled, tooled_plain, untooled, ec_only, "
                        "closeness_only, tooled_enz, enz_only")
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent (seed, target) runs. Results are unchanged -- runs "
                        "are independent and the table is sorted -- only throughput "
                        "moves. Match it to the server's --max-num-seqs for LLM arms; "
                        "leave at 1 for classical policies, which are CPU-bound.")
    # Imported, not restated: a hard-coded default here would silently override
    # the one in `agents.state`, which is exactly how the frontier-order default
    # survived being fixed. Sweep it with --top-k 10,20,40 for the ablation.
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help="frontier candidates shown (default matches RetroAgent's k=20)")
    p.add_argument("--out", default="", help="TSV output path")
    p.add_argument("--save-routes", action="store_true", default=True,
                   help="write solved pathways to results/routes/*.json")
    p.add_argument("--no-save-routes", dest="save_routes", action="store_false")
    p.add_argument("--save-decisions", action="store_true", default=True,
                   help="write one JSONL line per expansion: which node the policy "
                        "picked, out of how large a frontier, and what it bought")
    p.add_argument("--no-save-decisions", dest="save_decisions",
                   action="store_false")
    p.add_argument("--print-routes", action="store_true",
                   help="also print the first route of each solved run")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out.with_suffix(".partial.tsv")

    rows, pathways = run(args, partial_path=partial_path)
    budgets = sorted(int(b) for b in args.budgets.split(","))
    seeds = [int(x) for x in args.seeds.split(",")]
    summarise(rows, budgets, seeds, pathways, args.require_ec)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nwrote {len(rows)} rows to {out}")
    # Removed only on success, so a leftover .partial.tsv is exactly the signal that a
    # job died -- and it holds every run that had completed.
    partial_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
