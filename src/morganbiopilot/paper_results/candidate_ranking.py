"""Which score should decide what an expansion keeps?

Template application here is exhaustive and unordered: every rule that matches is
applied, which is 29 candidates for vanillin at r2 and 96 for violacein at r1, and the
frontier reaches 15,000 nodes within 100 expansions. Two things break there. An LLM
policy sees `top_k = 20` of that frontier -- 0.13% -- and UCT never leaves depth 1
because its exploration term visits all 96 root reactions before revisiting any.

Every system that works does the opposite. RETRO-R1 and RETROAGENT put a learned
single-step model in front of the agent, which turns 381,302 templates into a top-50.
RetroPath RL, which is template-based like us and has no such model, builds a
biochemical score from cheminformatics and caps children at 10 per node -- and their
ablation reports that the chemical-similarity half is what guides the search.

So the question is not whether to cap, it is what to rank by. A cap that discards the
attested disconnection destroys exactly what it is meant to preserve: a first probe put
vanillin's reference step 24th of 29 under similarity to the rule's native substrate, so
a top-10 would have thrown it away.

The measurement
---------------
The 20 curated pathways give 70 attested steps, and `rule_coverage.steps_of` already
extracts them. For each step where the attested disconnection *is* among the candidates
-- recall is a separate question, answered by `rule_coverage` -- every scorer ranks the
candidates and we record where the attested one lands. Reported with the metrics the
field's own scoping review names (Gricourt et al. 2024): mean reciprocal rank, and
coverage at k, the share of steps whose attested disconnection survives a top-k cut.

`arbitrary` is the control and it is not decoration: it is what the engine does today,
rule-index order. A scorer that does not beat it is not ranking, it is reordering.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from morganbiopilot.core.building_blocks import is_cofactor, skeleton
from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.paths import RESULTS_DIR
from morganbiopilot.core.rules import load_rules
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import prefilter_from_rules
from morganbiopilot.paper_results.rule_coverage import load_reference_steps

KS = (1, 5, 10, 20)


def _skeletons(smiles: Sequence[str]) -> FrozenSet[str]:
    return frozenset(s for s in (skeleton(m) for m in smiles
                                 if m and not is_cofactor(m)) if s)


# --------------------------------------------------------------------------- scorers

def build_scorers(rules, rule_ec, radius: int, use_enzymatic: bool
                  ) -> Dict[str, Callable]:
    """Every scorer maps (query molecule, neighbour) -> float, higher is better.

    They are deliberately cheap. A ranker that costs more than the expansion it prunes
    saves nothing, and the whole point of capping is to spend less.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_cache: Dict[str, object] = {}

    def fp(smi: str):
        if smi not in fp_cache:
            mol = Chem.MolFromSmiles(smi)
            fp_cache[smi] = gen.GetFingerprint(mol) if mol is not None else None
        return fp_cache[smi]

    from morganbiopilot.one_step.ranking import NativeSimilarity

    all_sub = NativeSimilarity(rules, all_substrates=True)

    def native_similarity_all(query: str, n) -> float:
        """Best match against *every* substrate the rule's source reactions used.

        `smi_sub` keeps one representative; a rule can come from 388 reactions. This is
        the comparison RetroPath RL actually makes, and the only question is whether the
        extra substrates buy anything over the representative.
        """
        return all_sub.similarity(query, n.rule_idx)

    def native_similarity(query: str, n) -> float:
        """Tanimoto between the query and the substrate the rule was learned on.

        The first of the two terms Coley et al. use and RetroPath RL adopts. High means
        "this rule was recorded on a molecule like this one", which is a claim about
        specificity, not about the transformation being right.
        """
        a, b = fp(query), fp(str(rules.smi_sub[n.rule_idx]))
        return DataStructs.TanimotoSimilarity(a, b) if a is not None and b is not None else 0.0

    closeness = None

    def _closeness():
        nonlocal closeness
        if closeness is None:
            from morganbiopilot.multi_step.heuristics import SinkCloseness
            closeness = SinkCloseness(radius)
        return closeness

    def closeness_worst(query: str, n) -> float:
        """The AND-node value: a reaction is worth its weakest precursor.

        This is what `MCTS._evaluate` already computes, applied to a candidate rather
        than to a frontier node. Averaging instead would call a reaction half-solved
        when one of two precursors is unreachable.
        """
        c = _closeness()
        return min((c.closeness(m) for m in n.molecules), default=0.0)

    def closeness_best(query: str, n) -> float:
        c = _closeness()
        return max((c.closeness(m) for m in n.molecules), default=0.0)

    def has_ec(query: str, n) -> float:
        """Binary enzymatic grounding — already available as the `require_ec` filter.

        Read off the neighbour, not off `rule_ec[n.rule_idx]`: a neighbour now stands
        for every rule that reaches the same molecule set, and `ec_numbers` is their
        union. Going back to the representative rule alone would call a disconnection
        unannotated when a folded-in template carries the EC.
        """
        return 1.0 if n.ec_numbers else 0.0

    def fewer_precursors(query: str, n) -> float:
        """Prefer disconnections that need fewer things made. A crude prior, and
        exactly the kind a cap should be tested against before anything cleverer."""
        return -float(len(n.molecules))

    def combined(query: str, n) -> float:
        return native_similarity(query, n) * (0.5 + 0.5 * has_ec(query, n))

    def template_prior(query: str, n) -> float:
        """The rule set's own frequency prior, unused until now.

        `RuleSet.score` is documented as an NLL, lower meaning more frequent, and it
        behaves like one: it correlates at -0.51 with the log number of MetaNetX
        reactions a rule was derived from. That makes it the unconditional analogue of
        what a template classifier supplies -- P(template) rather than
        P(template | molecule) -- and it costs one array lookup.
        """
        return -float(rules.score[n.rule_idx])

    scorers: Dict[str, Callable] = {
        "arbitrary": lambda q, n: -float(n.rule_idx),   # today's behaviour, the control
        "template_prior": template_prior,
        "native_similarity": native_similarity,
        "native_similarity_all": native_similarity_all,
        "closeness_worst": closeness_worst,
        "closeness_best": closeness_best,
        "has_ec": has_ec,
        "fewer_precursors": fewer_precursors,
        "similarity_x_ec": combined,
    }

    if use_enzymatic:
        from morganbiopilot.data_processing.enzymatic_model import score_reactions

        def enzymatic(query: str, n) -> float:
            value = score_reactions([(query, list(n.molecules))])[0]
            return 0.0 if value != value else float(value)

        scorers["enzymatic"] = enzymatic
    return scorers


# --------------------------------------------------------------------------- ranking

def rank_of_attested(query: str, neighbours, attested: FrozenSet[str],
                     scorer: Callable) -> Optional[int]:
    """1-based rank of the attested candidate under `scorer`.

    Ties break on rule index, which is what an implementation would do and what the
    engine already does. The first version broke them pessimistically -- attested
    candidate behind everything sharing its score -- and that quietly decided the
    comparison: a two-valued scorer like `has_ec` puts dozens of candidates in one
    bucket and was charged for all of them, while a scorer with no ties at all, such as
    rule index, paid nothing. That is a property of the scores' granularity, not of
    their usefulness.
    """
    ordered = sorted(
        ((scorer(query, n), n.rule_idx, _skeletons(n.molecules) == attested)
         for n in neighbours),
        key=lambda t: (-t[0], t[1]))
    for position, (_score, _idx, is_attested) in enumerate(ordered, 1):
        if is_attested:
            return position
    return None


def evaluate(radius: int, use_enzymatic: bool, limit: int) -> Tuple[List[dict], Dict]:
    rules = load_rules(radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)
    scorers = build_scorers(rules, rule_ec, radius, use_enzymatic)
    print(f"r{radius}: {len(rules)} rules | scorers: {', '.join(scorers)}\n")

    rows: List[dict] = []
    skipped = defaultdict(int)
    for name, steps in load_reference_steps():
        for i, (substrate, precursors, _diameter) in enumerate(steps, 1):
            attested = _skeletons(precursors)
            if not attested:
                skipped["cofactor-only step"] += 1
                continue
            report = expand(substrate, rules, prefilter)
            if not report.neighbours:
                skipped["substrate not expandable"] += 1
                continue
            if not any(_skeletons(n.molecules) == attested for n in report.neighbours):
                skipped["attested step not reachable"] += 1
                continue

            row = {"pathway": name, "step": i, "n_candidates": len(report.neighbours)}
            for label, scorer in scorers.items():
                row[label] = rank_of_attested(substrate, report.neighbours,
                                              attested, scorer)
            rows.append(row)
            print(f"  {name:26s} step {i}  {len(report.neighbours):4d} candidates  "
                  + "  ".join(f"{k}:{row[k]}" for k in list(scorers)[:3]))
            if limit and len(rows) >= limit:
                return rows, dict(skipped)
    return rows, dict(skipped)


def report(rows: Sequence[dict], skipped: Dict, scorers: Sequence[str]) -> None:
    print(f"\n{len(rows)} attested steps are reachable and rankable")
    for reason, n in sorted(skipped.items()):
        print(f"  skipped, {reason}: {n}")
    if not rows:
        return
    sizes = sorted(r["n_candidates"] for r in rows)
    print(f"  candidates per step: median {sizes[len(sizes) // 2]}, "
          f"max {sizes[-1]}\n")

    head = "".join(f"{'cov@' + str(k):>9s}" for k in KS)
    print(f"{'scorer':22s}{'MRR':>8s}{head}{'median rank':>13s}")
    print("-" * (22 + 8 + 9 * len(KS) + 13))
    for label in scorers:
        ranks = [r[label] for r in rows if r.get(label)]
        if not ranks:
            continue
        mrr = sum(1.0 / r for r in ranks) / len(rows)
        line = f"{label:22s}{mrr:8.3f}"
        for k in KS:
            line += f"{sum(1 for r in ranks if r <= k) / len(rows) * 100:8.0f}%"
        ordered = sorted(ranks)
        line += f"{ordered[len(ordered) // 2]:13d}"
        print(line)

    print("\ncov@k is the share of attested disconnections that survive a top-k cut --"
          "\nthe quantity a capped expansion would preserve or destroy. `arbitrary` is"
          "\nthe engine's current order: a scorer below it is not ranking anything.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rank candidate disconnections and see which score keeps the "
                    "attested one inside a top-k cut.")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--no-enzymatic", action="store_true",
                   help="skip the fitted enzymatic score (needs the joblib artifact)")
    p.add_argument("--limit", type=int, default=0, help="0 = every attested step")
    p.add_argument("--out", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rows, skipped = evaluate(args.radius, not args.no_enzymatic, args.limit)
    scorers = [k for k in (rows[0] if rows else {})
               if k not in ("pathway", "step", "n_candidates")]
    report(rows, skipped, scorers)

    if rows:
        out = RESULTS_DIR / (args.out or f"candidate_ranking_r{args.radius}.tsv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            for row in rows:
                w.writerow({k: ("" if v is None else v) for k, v in row.items()})
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
