"""Which metric should order the frontier subset shown to the agent?

`candidate_ranking` asks where the attested disconnection lands among the *rules* that
apply to one molecule. This asks the other half of the question the introduction
separates: given a frontier of molecules, where does the one that lies on the attested
route land under a given ordering?

Why it matters, in one measurement. The production view is `_stratify`: round-robin
across depths, discovery order within a depth -- diversity, and no quality signal at
all. Replaying mined routes through it, 36% of trajectories end in `truncated away`,
meaning the on-route molecule fell outside the twenty shown and the corpus builder gave
up. At inference the same thing happens: the graph holds ~20 molecules at the first
expansion and ~150 by the sixth, so the agent sees everything at first and an eighth of
it later -- which is exactly the window where the fine-tuned policy loses ground to a
greedy similarity heuristic (2 solves against 10).

So the quantity to optimise is not MRR for its own sake but **coverage@k**: how often a
top-k view keeps the answer visible at all. A metric that ranks the answer 3rd instead
of 7th changes little; one that moves it from 40th into the top 20 changes whether the
agent can answer correctly even in principle.

The walk follows the attested route regardless of what any metric says. Scoring only
the states a good metric reaches would measure the easy early frontiers and hide the
deep ones, which are the whole problem.

    python -m morganbiopilot.paper_results.frontier_ranking --limit 300
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

from morganbiopilot.core.building_blocks import is_building_block, is_cofactor, skeleton
from morganbiopilot.core.paths import RESULTS_DIR, ROOT_DIR
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.graph import SearchGraph
from morganbiopilot.multi_step.heuristics import SinkCloseness
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import prefilter_from_rules
from morganbiopilot.one_step.ranking import make_ranker


def is_available(smi: str) -> bool:
    return is_building_block(smi) or is_cofactor(smi)


def known_skeletons() -> set:
    """Connectivity skeletons of every compound MetaNetX catalogues.

    The one prior in this file that is biological rather than structural: a frontier
    molecule that exists in MetaNetX is a compound someone has observed, while template
    application invents molecules freely. Being binary, it splits the frontier in two
    rather than concentrating the twenty slots in one region -- which is the shape the
    benchmark rewards.

    chem_prop.tsv is 810 MB, so the first block of each InChIKey is cached beside it.
    """
    from morganbiopilot.core.paths import METANETX_DIR

    cache = METANETX_DIR / "processed" / "known_skeletons.txt"
    if cache.exists():
        return set(cache.read_text(encoding="utf-8").split())

    src = METANETX_DIR / "chem_prop.tsv"
    if not src.exists():
        print(f"  no {src}; the known-metabolite metric will be all-False",
              file=sys.stderr)
        return set()
    out = set()
    with open(src, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 7 and parts[7]:
                key = parts[7].split("-")[0]
                if len(key) == 14:
                    out.add(key)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(out)), encoding="utf-8")
    print(f"  cached {len(out)} MetaNetX skeletons -> {cache}")
    return out


# --------------------------------------------------------------------- scorers
class Scorers:
    """Per-molecule quantities, cached: a frontier molecule recurs at every step."""

    def __init__(self, rules, prefilter, ranker, closeness, rule_ec=None, known=(),
                 profile_n: int = 20):
        self.rules, self.prefilter = rules, prefilter
        self.ranker, self._close = ranker, closeness
        self.rule_ec, self.known = rule_ec, known
        self.profile_n = profile_n
        self._sim: Dict[str, List[float]] = {}
        self._nrules: Dict[str, int] = {}
        self._clo: Dict[str, float] = {}
        self._heavy: Dict[str, int] = {}
        self._prec: Dict[str, int] = {}
        self._idxs: Dict[str, list] = {}
        self._to_target: Dict[str, float] = {}
        self._target_fp = None

    def _prefiltered(self, smi: str):
        import numpy as np

        from morganbiopilot.core.chem import mol_ecfp

        ecfp = np.asarray(mol_ecfp(smi, self.rules.radius), dtype=np.int32)
        _vecs, idxs = self.prefilter.one_step(ecfp)
        return idxs

    def _profile(self, smi: str) -> List[float]:
        """Per-rule Tanimoto for the rules the environment would actually keep.

        Aggregating over *all* applicable rules would average over hundreds the search
        never validates. `top_n` of them survive an expansion, so the profile stops
        there: the question is what the agent's own environment offers, not what the
        rule set contains.
        """
        if smi not in self._sim:
            try:
                idxs = self._prefiltered(smi)
                self._nrules[smi] = len(idxs)
                self._idxs[smi] = list(idxs)
                ordered = self.ranker.order(smi, idxs)[:self.profile_n]
                self._sim[smi] = [self.ranker.similarity(smi, int(i)) for i in ordered]
            except Exception:                                    # noqa: BLE001
                self._sim[smi], self._nrules[smi], self._idxs[smi] = [], 0, []
        return self._sim[smi]

    def sim_max(self, smi: str) -> float:
        """Best Tanimoto to the native substrate of any rule that applies.

        The quantity `GreedySimilarity` follows, and the only scorer measured to carry
        a search: 72% on LASER against 56% for sink closeness.
        """
        p = self._profile(smi)
        return p[0] if p else 0.0

    def sim_mean3(self, smi: str) -> float:
        """Mean of the three best. One excellent precedent is not the same claim as
        three good ones, and a max cannot tell them apart."""
        p = self._profile(smi)[:3]
        return sum(p) / len(p) if p else 0.0

    def sim_breadth(self, smi: str) -> float:
        """How many kept rules sit within 10% of the best -- the breadth of precedent
        rather than its height."""
        p = self._profile(smi)
        if not p:
            return 0.0
        return float(sum(1 for s in p if s >= 0.9 * p[0]))

    def sim_to_target(self, smi: str) -> float:
        """Tanimoto to the molecule the search is making.

        A different question from every scorer above: those ask how well precedented a
        disconnection is, this asks how far the frontier has travelled. High means the
        node still looks like the target, so little has been taken apart yet.
        """
        import numpy as np

        from morganbiopilot.core.chem import mol_ecfp
        from morganbiopilot.multi_step.heuristics import counted_tanimoto

        if smi not in self._to_target:
            if self._target_fp is None:
                self._to_target[smi] = 0.0
            else:
                try:
                    v = np.asarray(mol_ecfp(smi, self.rules.radius), dtype=np.int32)
                    self._to_target[smi] = float(
                        counted_tanimoto(v, self._target_fp[None, :])[0])
                except Exception:                                # noqa: BLE001
                    self._to_target[smi] = 0.0
        return self._to_target[smi]

    def set_target(self, smi: str) -> None:
        """Called once per route: `sim_to_target` is relative to the current search."""
        import numpy as np

        from morganbiopilot.core.chem import mol_ecfp

        self._to_target = {}
        try:
            self._target_fp = np.asarray(mol_ecfp(smi, self.rules.radius),
                                         dtype=np.float32)
        except Exception:                                        # noqa: BLE001
            self._target_fp = None

    def precedent(self, smi: str) -> int:
        """Source reactions behind the best-attested rule that applies here.

        Frequency, not resemblance: "this disconnection was extracted from forty
        MetaNetX reactions" is a different claim from "the substrate looks like this
        one", and the two can disagree.
        """
        if smi not in self._prec:
            self._profile(smi)
            idxs = self._idxs.get(smi, [])
            if self.rule_ec is None or not len(idxs):
                self._prec[smi] = 0
            else:
                self._prec[smi] = int(max(
                    (int(self.rule_ec.n_reactions[int(i)]) for i in idxs), default=0))
        return self._prec[smi]

    def is_known(self, smi: str) -> int:
        """1 when MetaNetX catalogues this compound, 0 when the template invented it.

        **Excluded from the benchmark, and this is the reason.** It scores 97% coverage,
        far above everything else, and the number is meaningless: the attested routes are
        mined from MetaNetX, so every on-route molecule is a MetaNetX compound by
        construction. Measured over 25 routes, 176 of 176 on-route frontier molecules are
        "known" against 26.6% of the off-route ones -- a predictor that cannot miss a
        positive because the ground truth defines it.

        Kept because the prior may well be sound at inference on a real target, where
        preferring observed metabolites over invented intermediates is good biology. It
        simply cannot be evaluated against a MetaNetX-derived ground truth, and any
        benchmark that includes it is measuring its own construction.
        """
        key = skeleton(smi)
        return int(bool(key) and key in self.known)

    def n_rules(self, smi: str) -> int:
        if smi not in self._nrules:
            self._profile(smi)
        return self._nrules[smi]

    def closeness(self, smi: str) -> float:
        if smi not in self._clo:
            try:
                self._clo[smi] = self._close.closeness(smi)
            except Exception:                                    # noqa: BLE001
                self._clo[smi] = 0.0
        return self._clo[smi]

    def heavy(self, smi: str) -> int:
        if smi not in self._heavy:
            from rdkit import Chem
            m = Chem.MolFromSmiles(smi)
            self._heavy[smi] = m.GetNumHeavyAtoms() if m else 0
        return self._heavy[smi]


def stratified_order(graph, frontier: List[int]) -> List[int]:
    """The production view: round-robin across depths, discovery order within one.

    Reproduced here rather than imported so that the baseline this benchmark reports is
    the ordering itself, independent of the top_k the live view happens to apply.
    """
    buckets: Dict[int, List[int]] = {}
    for node_id in sorted(frontier):
        buckets.setdefault(graph.molecules[node_id].depth, []).append(node_id)
    out, cursor = [], {d: 0 for d in buckets}
    depths = sorted(buckets)
    while len(out) < len(frontier):
        progressed = False
        for d in depths:
            i = cursor[d]
            if i < len(buckets[d]):
                out.append(buckets[d][i])
                cursor[d] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def orderings(graph, frontier: List[int], sc: Scorers) -> Dict[str, List[int]]:
    """Every candidate ordering of the frontier, best-first."""
    smi = {i: graph.molecules[i].smiles for i in frontier}
    dep = {i: graph.molecules[i].depth for i in frontier}
    sim = {i: sc.sim_max(smi[i]) for i in frontier}
    clo = {i: sc.closeness(smi[i]) for i in frontier}
    nrl = {i: sc.n_rules(smi[i]) for i in frontier}
    hvy = {i: sc.heavy(smi[i]) for i in frontier}
    prc = {i: sc.precedent(smi[i]) for i in frontier}
    sm3 = {i: sc.sim_mean3(smi[i]) for i in frontier}
    sbr = {i: sc.sim_breadth(smi[i]) for i in frontier}
    stt = {i: sc.sim_to_target(smi[i]) for i in frontier}
    # `is_known` is deliberately not an ordering here -- see Scorers.is_known.

    def by(key, reverse=True):
        # Ties break on node id -- discovery order -- so a scorer that separates
        # nothing scores exactly as `arbitrary` rather than as some new arbitrary thing.
        return [i for i in sorted(frontier, key=lambda n: (-key(n) if reverse
                                                           else key(n), n))]

    strat = stratified_order(graph, frontier)
    orders = {
        "arbitrary": sorted(frontier),
        # No longer the production view: this benchmark is what moved production to the
        # portfolio, so the label would now name the thing it replaced. Kept as the
        # baseline every other ordering is measured against.
        "stratified (former production)": strat,
        "depth_shallow": by(lambda n: dep[n], reverse=False),
        "depth_deep": by(lambda n: dep[n]),
        "similarity": by(lambda n: sim[n]),
        "closeness": by(lambda n: clo[n]),
        "few_rules": by(lambda n: nrl[n], reverse=False),
        "many_rules": by(lambda n: nrl[n]),
        "small": by(lambda n: hvy[n], reverse=False),
        "precedent": by(lambda n: prc[n]),
        # --- the similarity family: same Tanimoto, three ways of aggregating it over
        # the rules that apply, and one measured against a different reference.
        "sim_mean3": by(lambda n: sm3[n]),
        "sim_breadth": by(lambda n: sbr[n]),
        "sim_to_target": by(lambda n: stt[n]),
        "far_from_target": by(lambda n: stt[n], reverse=False),
        "sim x mean3": by(lambda n: sim[n] * sm3[n]),
        "sim - simtarget": by(lambda n: sim[n] - stt[n]),
        "sim x (1-simtarget)": by(lambda n: sim[n] * (1.0 - stt[n])),
        "sim x close": by(lambda n: sim[n] * clo[n]),
        "sim - 0.05*depth": by(lambda n: sim[n] - 0.05 * dep[n]),
        "sim / (1+depth)": by(lambda n: sim[n] / (1 + dep[n])),
        "sim x deep": by(lambda n: sim[n] * (1 + 0.2 * dep[n])),
        "sim, depth-strat": None,   # filled below: quality inside a diversity shell
    }
    # Portfolios: round-robin across whole orderings, so each metric contributes its few
    # confident picks instead of imposing its ranking. The benchmark rewards hedging --
    # the production view wins precisely by refusing to concentrate -- and every ordering
    # above is a single opinion. These are the first that are not.
    #
    # Members are chosen for independence of signal, not for individual score. Pairing
    # `similarity` with `sim/(1+depth)` would buy nothing: they rank the same molecules
    # for the same reason. Depth diversity, chemical resemblance, frequency of precedent
    # and molecular size are four different claims about the same frontier.
    orders["portfolio (strat/sim)"] = interleave(
        [strat, orders["similarity"]], frontier)
    orders["portfolio (strat/sim/prec)"] = interleave(
        [strat, orders["similarity"], orders["precedent"]], frontier)
    orders["portfolio (strat/sim/prec/small)"] = interleave(
        [strat, orders["similarity"], orders["precedent"], orders["small"]], frontier)
    # Two units of the diversity shell against one of each chemical signal: the shell is
    # the only member that wins on its own, so a flat round-robin under-weights it.
    orders["portfolio (2x strat/sim/prec)"] = interleave(
        [strat, strat[1:], orders["similarity"], orders["precedent"]], frontier)
    return orders


def interleave(orders: List[List[int]], frontier: List[int]) -> List[int]:
    """Round-robin across several orderings, first occurrence wins, then the rest."""
    out, seen = [], set()
    for rank in range(max(len(o) for o in orders)):
        for order in orders:
            if rank < len(order) and order[rank] not in seen:
                seen.add(order[rank])
                out.append(order[rank])
    out.extend(i for i in frontier if i not in seen)
    return out


def depth_stratified_by(graph, frontier, score) -> List[int]:
    """Round-robin across depths, but best-scoring first inside each depth.

    The two failure modes have opposite shapes: the production view is all diversity and
    no quality, a pure sort is all quality and no diversity -- and a pure sort makes the
    agent redundant, since taking candidate one reproduces the greedy heuristic exactly.
    """
    buckets: Dict[int, List[int]] = {}
    for node_id in frontier:
        buckets.setdefault(graph.molecules[node_id].depth, []).append(node_id)
    for d in buckets:
        buckets[d].sort(key=lambda n: (-score(n), n))
    out, cursor = [], {d: 0 for d in buckets}
    depths = sorted(buckets)
    while len(out) < len(frontier):
        progressed = False
        for d in depths:
            i = cursor[d]
            if i < len(buckets[d]):
                out.append(buckets[d][i])
                cursor[d] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def walk(target: str, tree: Dict[str, int], rules, prefilter, ranker, sc,
         max_depth: int, ranks: Dict[str, List[int]], sizes: List[int],
         top_n: int) -> str:
    """Replay one attested route, recording where each ordering puts the answer."""
    on_route = {}
    for molecule, cost in tree.items():
        key = skeleton(molecule)
        if key and not is_available(molecule):
            on_route[key] = cost

    # `sim_to_target` is relative to the molecule being made, so the reference has to be
    # reset per route -- and the per-molecule cache with it, since the same intermediate
    # can appear in two routes with different targets.
    sc.set_target(target)

    graph = SearchGraph(target)
    current = graph.root
    for _ in range(len(tree) + 2):
        node = graph.molecules[current]
        if node.depth >= max_depth:
            return "depth cap"
        try:
            report = expand(node.smiles, rules, prefilter, ranker=ranker, top_n=top_n)
        except Exception:                                        # noqa: BLE001
            return "expansion failed"
        if not report.neighbours:
            return "no candidate"
        for neighbour in report.neighbours:
            graph.add_neighbour(current, neighbour)
        node.expanded = True

        frontier = [i for i, m in graph.molecules.items()
                    if not m.solved and not m.expanded and not m.available]
        if not frontier:
            return "complete"

        good = {i for i in frontier if skeleton(graph.molecules[i].smiles) in on_route}
        if not good:
            # An empty on-route frontier has two opposite meanings and collapsing them
            # reported 75% "thread lost" on a corpus whose edges are rule applications
            # by construction -- a rate that cannot be true. Either every molecule of
            # the tree has been expanded, which is the trajectory finishing, or the
            # thread really was lost. `replay_routes` draws the same distinction.
            seen = {skeleton(m.smiles) for m in graph.molecules.values()
                    if m.expanded or m.available}
            return "complete" if on_route.keys() <= seen else "thread lost"

        orders = orderings(graph, frontier, sc)
        orders["sim, depth-strat"] = depth_stratified_by(
            graph, frontier, lambda n: sc.sim_max(graph.molecules[n].smiles))
        for name, order in orders.items():
            # Rank of the best-placed on-route molecule: any of them is a correct answer.
            ranks[name].append(min(order.index(i) + 1 for i in good))
        sizes.append(len(frontier))

        # Follow the attested route, hardest branch first -- the corpus convention.
        current = max(good, key=lambda i: on_route[skeleton(graph.molecules[i].smiles)])
    return "complete"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--routes", default="results/routes_ecoli_r1.jsonl")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--ranker", default="native_similarity")
    p.add_argument("--top-n", type=int, default=20, help="expansion cap")
    p.add_argument("--top-k", type=int, default=20, help="the view size to score against")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--out", default="frontier_ranking.tsv")
    p.add_argument("--with-known", action="store_true",
                   help="also load the MetaNetX membership prior. It is not an ordering "
                        "in this benchmark and cannot be: see `Scorers.is_known`")
    args = p.parse_args(argv)

    from morganbiopilot.core.ec import annotate_rules

    rules = load_rules(args.radius)
    prefilter = prefilter_from_rules(rules)
    ranker = make_ranker(args.ranker, rules)
    rule_ec = annotate_rules(rules)
    # Off by default: the prior is circular against a MetaNetX-derived ground truth
    # (see `Scorers.is_known`), and building its cache reads an 810 MB file.
    known = known_skeletons() if args.with_known else set()
    if known:
        print(f"  {len(known)} MetaNetX skeletons for the known-metabolite prior")
    sc = Scorers(rules, prefilter, ranker, SinkCloseness(args.radius),
                 rule_ec=rule_ec, known=known)
    print(f"r{args.radius}: {len(rules)} rules | expansion ranked by {args.ranker}, "
          f"capped at {args.top_n} | scoring a view of {args.top_k}")

    path = ROOT_DIR / args.routes
    routes = [json.loads(line) for line in open(path, encoding="utf-8")]
    routes = routes[:args.limit] if args.limit else routes
    print(f"{len(routes)} attested routes\n")

    ranks: Dict[str, List[int]] = defaultdict(list)
    sizes: List[int] = []
    outcomes: Dict[str, int] = defaultdict(int)
    for n, route in enumerate(routes, 1):
        outcomes[walk(route["target"], route["tree"], rules, prefilter, ranker,
                      sc, args.max_depth, ranks, sizes, args.top_n)] += 1
        if n % 25 == 0:
            print(f"  {n}/{len(routes)} routes -> {len(sizes)} decisions", flush=True)

    if not sizes:
        print("no decisions recorded", file=sys.stderr)
        return 1

    k = args.top_k
    SWEEP = (10, 20, 40, 80, 160)
    rows = []
    for name, rk in ranks.items():
        vis = [r for r, s in zip(rk, sizes) if s > k]     # only where truncation bites
        row = {
            "ordering": name,
            "decisions": len(rk),
            "MRR": statistics.mean(1 / r for r in rk),
            "median_rank": statistics.median(rk),
            "big_frontiers": len(vis),
        }
        # Coverage at several view sizes. Ordering turned out to matter far less than
        # k does, so the sweep is the useful output, not the ranking of orderings.
        for kk in SWEEP:
            row[f"cov@{kk}"] = 100 * sum(1 for r in rk if r <= kk) / len(rk)
            row[f"cov@{kk}_big"] = (100 * sum(1 for r in vis if r <= kk) / len(vis)
                                    if vis else float("nan"))
        rows.append(row)
    rows.sort(key=lambda r: -r[f"cov@{k}_big"])

    # Raw ranks, so any k can be recomputed without walking the routes again -- the
    # walk is the expensive part and the first version threw its output away.
    raw = RESULTS_DIR / (args.out.rsplit(".", 1)[0] + "_ranks.tsv")
    import csv as _csv
    with open(raw, "w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh, delimiter="\t")
        w.writerow(["ordering", "frontier_size", "rank_of_answer"])
        for name, rk in ranks.items():
            for r, s in zip(rk, sizes):
                w.writerow([name, s, r])
    print(f"\nraw ranks -> {raw}")

    print(f"\n{len(sizes)} decisions | median frontier {statistics.median(sizes):.0f}, "
          f"max {max(sizes)} | {sum(1 for s in sizes if s > k)} larger than {k}")
    print("outcomes: " + ", ".join(f"{v} {kk}" for kk, v in outcomes.items()))
    head = "".join(f"{f'@{kk}':>7}" for kk in SWEEP)
    print(f"\ncoverage on the {rows[0]['big_frontiers']} states whose frontier "
          f"exceeds {k}\n")
    print(f"{'ordering':<26}{'MRR':>7}{'median':>8}{head}")
    print("-" * (41 + 7 * len(SWEEP)))
    for r in rows:
        cells = "".join(f"{r[f'cov@{kk}_big']:>7.0f}" for kk in SWEEP)
        print(f"{r['ordering']:<26}{r['MRR']:>7.3f}{r['median_rank']:>8.0f}{cells}")

    out = RESULTS_DIR / args.out
    import csv
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print("\ncov@k when the frontier exceeds k is the number that matters: it is the")
    print("share of hard states where a top-k view keeps the answer reachable at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
