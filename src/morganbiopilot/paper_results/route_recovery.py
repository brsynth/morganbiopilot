"""Is the attested route among the ones we return, and at what rank?

Our solve rate answers "did the search reach the chassis". The retrobiosynthesis
literature asks a harder question, and both of the team's own tools ask it:

* **RetroPath2.0** (Delepine et al. 2018) never stops on a success. It iterates until a
  fixed iteration count or an empty source set, computes the *scope*, and a separate
  utility (RP2paths) enumerates pathways from it as elementary flux modes. The authors
  state plainly that the workflow neither extracts nor ranks pathways -- ranking is
  downstream and a posteriori.
* **RetroPath RL** (Koch et al. 2020) runs MCTS "until a resource budget (time or number
  of iterations) has been exhausted". Solving does not halt the search; it backpropagates
  as a reward. The output is a scope holding many pathways.

Koch et al. then score with three criteria, and the second is the one that matters here:
the returned results must *include* the literature-described experimental pathway, exact
intermediates found. Their headline -- 75% strict, 95% with tolerant settings -- is a
recall over the whole returned set, not "a route was found". We have never measured it.

This module does, on the same 20 curated pathways Koch et al. published, which is the
only benchmark here carrying reference routes. It runs exhaustively
(`stop_on_first_pathway=False`), enumerates every route the graph yields, and asks where
the attested one sits under several ranking functions.

Matching, and why it is defined against *our* sink
--------------------------------------------------
A route is compared to the reference by its set of intermediates, keyed on InChIKey
skeleton (protonation disagrees between the search and MetaNetX; see
`core.building_blocks.skeleton`). "Intermediate" means: a molecule of the route other
than the target, excluding anything available in the chassis.

Availability is evaluated with **our** E. coli sink on both sides, never with the
dataset's own `inSink` flag. The two sinks are not nested -- that is measured, not
assumed -- so using each side's own definition would compare two different questions.
Applying ours to the reference too can make a reference terminal compound count as an
intermediate, and that is the honest outcome: it means our chassis does not supply what
theirs did, and the search really would have to keep going.

Five outcomes, following Koch et al.'s own categories:

    exact      the intermediate sets are equal
    covers     the reference is a strict subset -- our route detours through it
    shortcut   our route is a strict subset -- it skipped attested steps
    partial    they intersect, neither contains the other
    miss       disjoint

Only `exact` counts as recovery. `covers` is reported beside it because a route passing
through every attested intermediate is a defensible weaker claim, and hiding it would
overstate how binary the result is.

The diagnostic that explains a failure
--------------------------------------
`ref_in_graph` counts how many reference intermediates the search *built at all*, whether
or not a route used them. It separates two very different failures: the one-step model
never proposed the molecule, or it did and no solved route ran through it. The first is a
rule-set limit, the second a search limit, and they call for opposite fixes.

A prediction, recorded before running
-------------------------------------
Recall of attested single-step disconnections is 50% at r2. A full route needs every step,
so at three steps and independence the exact-recovery rate would be near 12%. Nothing
close to 75% should be expected, and the gap is the radius trade-off we already
characterised rather than a bug.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from morganbiopilot.core.building_blocks import (is_building_block, is_cofactor,
                                                 skeleton, use_sink)
from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.golden_dataset import load_golden_dataset
from morganbiopilot.core.paths import RESULTS_DIR
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.heuristics import SinkCloseness
from morganbiopilot.multi_step.mcts import MCTS
from morganbiopilot.multi_step.policy import (BreadthFirst, DepthFirst, GreedyECFP,
                                              GreedySimilarity, RandomPolicy)
from morganbiopilot.multi_step.routes import Route, extract_routes
from morganbiopilot.multi_step.search import search
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import prefilter_from_rules

OUTCOMES = ("exact", "covers", "shortcut", "partial", "miss")


# --------------------------------------------------------------------------- matching

def _intermediates(smiles_iter, target_skeleton: Optional[str]) -> FrozenSet[str]:
    """Skeletons of molecules that are neither the target nor chassis-available."""
    out = set()
    for smi in smiles_iter:
        sk = skeleton(smi)
        if sk is None or sk == target_skeleton:
            continue
        if is_building_block(smi) or is_cofactor(smi):
            continue
        out.add(sk)
    return frozenset(out)


def reference_intermediates(gold) -> Dict[str, str]:
    """Attested intermediates as skeleton -> a representative SMILES.

    The SMILES is kept, not discarded, so `built_reference` can try the graph's own
    canonical-SMILES key before paying for InChIKey computation.
    """
    target_sk = skeleton(gold.target)
    out: Dict[str, str] = {}
    for smi in gold.compounds.values():
        sk = skeleton(smi)
        if sk is None or sk == target_sk:
            continue
        if is_building_block(smi) or is_cofactor(smi):
            continue
        out.setdefault(sk, smi)
    return out


def built_reference(graph, reference: Dict[str, str]) -> int:
    """How many attested intermediates the search built, route or no route.

    Two phases on purpose. The canonical-SMILES lookup is free -- it is the graph's own
    key -- and catches every match where protonation agrees. Only if something is still
    missing do we fingerprint the whole graph, and that is what has to be avoided by
    default: InChIKey costs ~17 ms per molecule, so the 878-molecule graph at budget 30
    spent 15 s of a 22 s run on this one diagnostic.
    """
    missing = {}
    built = 0
    for sk, smi in reference.items():
        if graph.molecule_id(smi) is not None:
            built += 1
        else:
            missing[sk] = smi
    if not missing:
        return built

    graph_skeletons = {skeleton(m.smiles) for m in graph.molecules.values()}
    return built + sum(1 for sk in missing if sk in graph_skeletons)


def route_intermediates(route: Route) -> FrozenSet[str]:
    smiles = []
    for step in route.steps:
        smiles.append(step.substrate)
        smiles.extend(step.precursors)
    return _intermediates(smiles, skeleton(route.target))


def classify(found: FrozenSet[str], reference: FrozenSet[str]) -> str:
    """Set relation between a found route and the attested one.

    An empty reference is possible and not degenerate: a one-step attested route whose
    precursor our chassis already supplies has no intermediate at all. Then `exact`
    means our route is also direct, which is the right reading.
    """
    if found == reference:
        return "exact"
    if reference < found:
        return "covers"
    if found < reference:
        return "shortcut"
    return "partial" if found & reference else "miss"


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# --------------------------------------------------------------------------- ranking

def enzymatic_per_step(routes: Sequence[Route], model_path=None) -> List[float]:
    """Geometric mean of the edge scores of each route.

    One batched call for every edge of every route of the target: `score_reactions`
    fingerprints in bulk and a per-edge call would pay that overhead one molecule at a
    time. Unscorable edges are dropped from the mean rather than replaced by 0.5, so a
    route is never rewarded for an edge the model could not read; a route with no
    scorable edge gets NaN and sorts last.
    """
    from morganbiopilot.data_processing.enzymatic_model import score_reactions

    pairs: List[Tuple[str, List[str]]] = []
    spans: List[Tuple[int, int]] = []
    for route in routes:
        start = len(pairs)
        for step in route.steps:
            pairs.append((step.substrate, list(step.precursors)))
        spans.append((start, len(pairs)))

    if not pairs:
        return [float("nan")] * len(routes)

    scores = score_reactions(pairs, path=model_path)
    out = []
    for start, end in spans:
        usable = [s for s in scores[start:end] if s == s]
        out.append(math.exp(sum(math.log(max(s, 1e-4)) for s in usable) / len(usable))
                   if usable else float("nan"))
    return out


def _nan_last(value: float) -> float:
    return -1.0 if value != value else value


def rankings(routes: Sequence[Route], enz: Sequence[float]) -> Dict[str, List[int]]:
    """Route indices ordered best-first, one order per ranking function.

    `enumeration` is the order the graph produced and is the control: any ranker that
    does not beat it is not ranking, it is reordering. `stable` sorts keep ties in that
    same enumeration order so no ranker gets a lucky tie-break the control lacks.
    """
    idx = list(range(len(routes)))
    return {
        "enumeration": idx,
        "shortest": sorted(idx, key=lambda i: (len(routes[i]),
                                               -routes[i].ec_coverage)),
        "enzymatic": sorted(idx, key=lambda i: -_nan_last(enz[i])),
        "ec_coverage": sorted(idx, key=lambda i: (-routes[i].ec_coverage,
                                                  len(routes[i]))),
    }


# --------------------------------------------------------------------------- one run

@dataclass
class Recovery:
    """One (policy, target) exhaustive run, scored against the attested route."""

    policy: str
    target: str
    solved: bool
    stopped_because: str
    n_expansions: int
    first_solved_at: Optional[int]
    n_molecules: int
    n_routes: int
    shortest_route: Optional[int]
    reference_route: int
    n_ref_intermediates: int
    ref_in_graph: int
    best_outcome: str
    best_jaccard: float
    elapsed_s: float
    ranks: Dict[str, Optional[int]] = field(default_factory=dict)
    ranks_covered: Dict[str, Optional[int]] = field(default_factory=dict)


def _first_solved_at(result) -> Optional[int]:
    """The expansion at which the graph first became solved.

    Without this the exhaustive mode would silently destroy the budget axis: the loop
    no longer breaks on success, so `n_expansions` is the whole budget and every solved
    target would look like it solved exactly at the cap. The per-decision trace records
    `solved_after`, so the real first-solve step is recoverable.
    """
    return next((d["step"] for d in result.decisions if d["solved_after"]), None)


def score_run(result, gold, rule_ec, max_pathways: int, model_path,
              want_enzymatic: bool) -> Tuple[Recovery, List[Route], List[str]]:
    ref_map = reference_intermediates(gold)
    reference = frozenset(ref_map)
    ref_in_graph = built_reference(result.graph, ref_map)

    routes = (extract_routes(result, rule_ec, max_routes=max_pathways,
                             max_pathways=max_pathways)
              if result.solved else [])
    # One skeleton pass per route, reused by both the class and the overlap: computing
    # it twice doubled the InChIKey bill for nothing.
    found = [route_intermediates(r) for r in routes]
    outcomes = [classify(f, reference) for f in found]
    jaccards = [jaccard(f, reference) for f in found]

    enz = (enzymatic_per_step(routes, model_path)
           if want_enzymatic and routes else [float("nan")] * len(routes))

    ranks: Dict[str, Optional[int]] = {}
    ranks_covered: Dict[str, Optional[int]] = {}
    for name, order in rankings(routes, enz).items():
        ranks[name] = next((pos for pos, i in enumerate(order, 1)
                            if outcomes[i] == "exact"), None)
        ranks_covered[name] = next((pos for pos, i in enumerate(order, 1)
                                    if outcomes[i] in ("exact", "covers")), None)

    best = min(outcomes, key=OUTCOMES.index) if outcomes else "miss"
    rec = Recovery(
        policy=result.policy, target=gold.name, solved=result.solved,
        stopped_because=result.stopped_because, n_expansions=result.n_expansions,
        first_solved_at=_first_solved_at(result),
        n_molecules=result.n_molecules, n_routes=len(routes),
        shortest_route=min((len(r) for r in routes), default=None),
        reference_route=len(gold), n_ref_intermediates=len(reference),
        ref_in_graph=ref_in_graph, best_outcome=best,
        best_jaccard=max(jaccards, default=0.0), elapsed_s=result.elapsed_s,
        ranks=ranks, ranks_covered=ranks_covered,
    )
    return rec, routes, outcomes


# --------------------------------------------------------------------------- reporting

def report(records: Sequence[Recovery], ks: Sequence[int]) -> None:
    by_policy: Dict[str, List[Recovery]] = {}
    for rec in records:
        by_policy.setdefault(rec.policy, []).append(rec)
    w = max([len(p) for p in by_policy] + [10]) + 2

    print("\nEXACT RECOVERY OF THE ATTESTED ROUTE (top-k over the enumerated set)")
    rankers = list(records[0].ranks) if records else []
    head = "".join(f"{r[:11] + '@' + str(k):>15s}" for r in rankers for k in ks)
    print(f"{'policy':{w}s}{'solved':>8s}{head}")
    print("-" * (w + 8 + 15 * len(rankers) * len(ks)))
    for policy, group in sorted(by_policy.items()):
        line = f"{policy:{w}s}{sum(1 for r in group if r.solved):4d}/{len(group):<3d}"
        for ranker in rankers:
            for k in ks:
                hit = sum(1 for r in group
                          if r.ranks.get(ranker) is not None and r.ranks[ranker] <= k)
                line += f"{hit:>10d}/{len(group):<4d}"
        print(line)

    print("\nWEAKER CLAIM: a route passing through every attested intermediate")
    print(f"{'policy':{w}s}" + "".join(f"{r[:11] + '@' + str(max(ks)):>15s}"
                                       for r in rankers))
    print("-" * (w + 15 * len(rankers)))
    for policy, group in sorted(by_policy.items()):
        line = f"{policy:{w}s}"
        for ranker in rankers:
            hit = sum(1 for r in group if r.ranks_covered.get(ranker) is not None
                      and r.ranks_covered[ranker] <= max(ks))
            line += f"{hit:>10d}/{len(group):<4d}"
        print(line)

    print("\nOUTCOME OF THE BEST ROUTE PER TARGET")
    print(f"{'policy':{w}s}" + "".join(f"{o:>10s}" for o in OUTCOMES)
          + f"{'jaccard':>10s}{'routes':>8s}")
    print("-" * (w + 10 * len(OUTCOMES) + 18))
    for policy, group in sorted(by_policy.items()):
        line = f"{policy:{w}s}"
        for outcome in OUTCOMES:
            line += f"{sum(1 for r in group if r.best_outcome == outcome):>10d}"
        js = [r.best_jaccard for r in group]
        n_routes = [r.n_routes for r in group]
        line += f"{sum(js) / len(js):>10.2f}{sum(n_routes) / len(n_routes):>8.0f}"
        print(line)

    print("\nWHERE A FAILURE COMES FROM: were the attested intermediates even built?")
    print(f"{'policy':{w}s}{'ref inter':>11s}{'in graph':>10s}{'share':>8s}")
    print("-" * (w + 29))
    for policy, group in sorted(by_policy.items()):
        total = sum(r.n_ref_intermediates for r in group)
        built = sum(r.ref_in_graph for r in group)
        print(f"{policy:{w}s}{total:>11d}{built:>10d}"
              f"{(100 * built / total if total else float('nan')):>7.0f}%")
    print("  A low share is a one-step limit: the rule set never proposed the molecule.\n"
          "  A high share with few exact hits is a search limit: it was built but no\n"
          "  solved route ran through it.")


# --------------------------------------------------------------------------- entry

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Top-k recovery of the attested route on the 20 curated pathways, "
                    "the Koch et al. (2020) criterion.")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--budget", type=int, default=200,
                   help="expansions per run; the search never stops early here")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-seconds", type=float, default=180.0,
                   help="wall-clock cap per run; 0 disables it. Exhaustive expansion "
                        "cost is wildly non-uniform across targets, so an uncapped job "
                        "cannot be scheduled")
    p.add_argument("--policies", default="bfs,greedy,mcts")
    p.add_argument("--models", default="",
                   help="comma-separated backend specs to add as agent arms, e.g. "
                        "'openai:policy,openai:Qwen/Qwen2.5-7B-Instruct'. Each becomes "
                        "a policy named llm:<spec>; add it to --policies to run it")
    p.add_argument("--top-k", type=int, default=20,
                   help="frontier candidates shown to an agent arm")
    p.add_argument("--seed", type=int, default=0, help="for `random` only")
    p.add_argument("--max-pathways", type=int, default=256,
                   help="cap on the route enumeration; it is a cartesian product")
    p.add_argument("--variant", default="experimental", choices=("experimental", "core"))
    p.add_argument("--targets", default=None,
                   help="comma-separated target names; default is all 20")
    p.add_argument("--max-targets", type=int, default=0,
                   help="0 = all; a prefix of the name-sorted list, for smoke tests")
    p.add_argument("--sink", default=None, help="override the chassis sink")
    p.add_argument("--require-ec", action="store_true")
    # The question a capped expansion has to answer is not whether the attested route
    # survives in the rule set -- `rule_coverage` answers that -- but whether a search
    # under the cap actually returns it. Those are different, because coverage is a
    # per-step property and a route is a conjunction the search has to walk.
    p.add_argument("--ranker", default=None,
                   help="'native_similarity' to order rules before validation")
    p.add_argument("--top-n", type=int, default=None,
                   help="neighbours kept per expansion; needs --ranker")
    p.add_argument("--ks", default="1,5,10", help="top-k cut-offs to report")
    p.add_argument("--no-enzymatic", action="store_true",
                   help="skip the enzymatic ranker (it needs models/enzymatic_score.joblib)")
    p.add_argument("--model", default=None, help="path to the enzymatic score artifact")
    p.add_argument("--print-best", action="store_true",
                   help="render the best-matching route of each target")
    p.add_argument("--out", default="route_recovery.tsv")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ks = [int(k) for k in args.ks.split(",") if k]

    # Before the rules and before SinkCloseness: the sink defines "solved" and both
    # cache it.
    if args.sink:
        use_sink(args.sink)
        print(f"sink overridden: {args.sink}")

    rules = load_rules(args.radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)
    closeness = SinkCloseness(args.radius)
    from morganbiopilot.one_step.ranking import make_ranker
    ranker = make_ranker(args.ranker, rules)

    factories = {
        "bfs": lambda: BreadthFirst(),
        "dfs": lambda: DepthFirst(),
        "random": lambda: RandomPolicy(seed=args.seed),
        "greedy": lambda: GreedyECFP(closeness),
        "mcts": lambda: MCTS(closeness),
    }
    if ranker is not None:
        # The strongest policy measured on LASER, and it was missing here: recovery
        # could not be reported for the one baseline that beats the fine-tuned model.
        # It needs the ranker, so it only exists when one was asked for.
        factories["greedy_similarity"] = lambda: GreedySimilarity(rules, prefilter,
                                                                  ranker)

    # The agent arms. Imported lazily so a classical run needs no API client, and
    # named after the model they serve so a TSV row says which one it was. The tool
    # surface is `untooled`, matching every reported search; `rule_ec` still goes in
    # because it feeds the portfolio's precedent member, which is environment rather
    # than grounding.
    for spec in [m.strip() for m in (args.models or "").split(",") if m.strip()]:
        def _make(sp=spec):
            from morganbiopilot.agents.backends import make_backend
            from morganbiopilot.agents.policy import LLMPolicy
            from morganbiopilot.agents.tools import untooled
            return LLMPolicy(tools=untooled(), backend=make_backend(sp),
                             top_k=args.top_k, seed=args.seed,
                             ranker=ranker, prefilter=prefilter, rule_ec=rule_ec)
        factories[f"llm:{spec}"] = _make

    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    # A results row records GreedyECFP under the name "greedy_ecfp"; accept what the
    # TSV shows so that rerunning an arm read off a table works.
    names = [{"greedy_ecfp": "greedy"}.get(n, n) for n in names]
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise SystemExit(f"unknown policies: {unknown}; have {sorted(factories)}")

    golden = load_golden_dataset(args.variant)
    if args.targets:
        wanted = {t.strip() for t in args.targets.split(",") if t.strip()}
        missing = wanted - {g.name for g in golden}
        if missing:
            raise SystemExit(f"unknown targets: {sorted(missing)}")
        golden = tuple(g for g in golden if g.name in wanted)
    if args.max_targets:
        golden = golden[:args.max_targets]

    print("=" * 78)
    print(f"{len(golden)} curated pathways ({args.variant}) | {len(rules)} rules at "
          f"r{args.radius} | EC coverage {100 * rule_ec.coverage:.1f}%")
    cap = f" | ranked by {args.ranker}, top {args.top_n}" if ranker else ""
    print(f"budget {args.budget} expansions, exhaustive (no early stop) | "
          f"policies {','.join(names)}{cap}")
    # Sequential on purpose: at --workers 8 a single MCTS run measured 14x slower than
    # at 1, from memory-bandwidth contention on the closeness matrix product.
    print("runs are sequential; MCTS cost grows with the graph it has already built")
    print("=" * 78)

    want_enz = not args.no_enzymatic
    if want_enz:
        from morganbiopilot.core.paths import ENZYMATIC_SCORE_MODEL
        path = args.model or ENZYMATIC_SCORE_MODEL
        if not Path(path).exists():
            print(f"note: {path} is missing, the enzymatic ranker is disabled")
            want_enz = False

    records: List[Recovery] = []
    for name in names:
        for gold in golden:
            t0 = time.perf_counter()
            report_one = expand(gold.target, rules, prefilter, rule_ec=rule_ec,
                                require_ec=args.require_ec)
            if not report_one.neighbours:
                print(f"{name:8s} {gold.name:28s} not expandable, skipped")
                continue
            result = search(
                gold.target, rules, prefilter, factories[name](),
                budget=args.budget, max_depth=args.max_depth,
                rule_ec=rule_ec, require_ec=args.require_ec,
                stop_on_first_pathway=False,
                max_seconds=args.max_seconds or None,
                ranker=ranker, top_n=args.top_n,
            )
            rec, routes, outcomes = score_run(
                result, gold, rule_ec, args.max_pathways,
                args.model, want_enz)
            records.append(rec)
            print(f"{name:8s} {gold.name:28s} solved={str(rec.solved):5s} "
                  f"exp={rec.n_expansions:4d} stop={rec.stopped_because:9s} "
                  f"routes={rec.n_routes:4d} best={rec.best_outcome:9s} "
                  f"J={rec.best_jaccard:.2f} ref={rec.reference_route} "
                  f"built={rec.ref_in_graph}/{rec.n_ref_intermediates} "
                  f"{time.perf_counter() - t0:6.1f}s")
            if args.print_best and routes:
                best = min(range(len(routes)), key=lambda i: OUTCOMES.index(outcomes[i]))
                print(routes[best].render())

    if not records:
        print("\nno run produced a record")
        return 1

    report(records, ks)

    out = RESULTS_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rankers = list(records[0].ranks)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        base = [f for f in asdict(records[0]) if f not in ("ranks", "ranks_covered")]
        writer.writerow(base + [f"rank_{r}" for r in rankers]
                        + [f"covered_{r}" for r in rankers])
        for rec in records:
            row = [getattr(rec, f) for f in base]
            row += [rec.ranks.get(r) for r in rankers]
            row += [rec.ranks_covered.get(r) for r in rankers]
            writer.writerow(["" if v is None else v for v in row])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
