"""Can the rule set reproduce the attested route at all, search aside?

Every other number in this project mixes two things: whether the chemistry is in the
rule set, and whether the policy found it in its budget. This separates them. For each
step of each curated pathway it applies the whole rule set to the attested substrate and
asks whether the attested precursors come out. No search, no budget, no policy — the
ceiling that no `select` function can raise.

It answers the question the solve rates cannot: when we fail on a target, is it because
the disconnection does not exist in our templates, or because the search never looked?

Reading a step
--------------
`expand` emits precursor sets that are mono-component, sanitized and cofactor-free, so
the attested precursors are stripped of cofactors before comparison and matched on
InChIKey skeleton (the dataset stores neutral acids where the engine carries
phenolates). Four outcomes per step:

    exact       some neighbour's molecule set equals the attested one
    contains    some neighbour includes every attested precursor, plus extras
    partial     some neighbour shares a precursor, neither set contains the other
    missing     no neighbour touches the attested precursors at all

A route is *recoverable* only if every one of its steps is `exact`. That is strict on
purpose: a route is a conjunction, and one missing step breaks it however good the rest
looks.

`n_neighbours` is reported beside the outcome because coverage without selectivity is
not a solved problem. A step whose attested disconnection is one of 900 candidates is
covered in principle and hopeless in practice, and the two must not be confused — that
distinction is the whole reason a search policy exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from morganbiopilot.core.building_blocks import is_cofactor, skeleton, use_sink
from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.paths import GOLDEN_DATASET_DIR, RESULTS_DIR
from morganbiopilot.core.rules import load_rules
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import prefilter_from_rules

OUTCOMES = ("exact", "contains", "partial", "missing")


@dataclass
class StepResult:
    """One attested disconnection, tested against the whole rule set."""

    pathway: str
    radius: int
    step: int
    substrate: str
    n_attested: int
    n_neighbours: int
    outcome: str
    rule_idx: Optional[int]          # the rule that reproduced it, when one did
    reference_diameter: Optional[int]  # RetroRules diameter of the attested rule


def steps_of(path) -> Tuple[str, List[Tuple[str, List[str], Optional[int]]]]:
    """Attested steps as (substrate, precursors, diameter), read from the graph edges.

    The Cytoscape edges already run in the retro direction: `compound -> reaction` is
    the molecule being decomposed, `reaction -> compound` are its precursors. Using the
    edges rather than the reaction SMARTS avoids re-deriving what the dataset states.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    nodes = [n["data"] for n in raw["elements"]["nodes"]]
    edges = [e["data"] for e in raw["elements"]["edges"]]

    smiles = {n["id"]: n.get("SMILES", "") for n in nodes if n.get("type") == "compound"}
    diameters = {n["id"]: n.get("Diameter") for n in nodes if n.get("type") == "reaction"}

    substrate_of: Dict[str, str] = {}
    precursors_of: Dict[str, List[str]] = {}
    for edge in edges:
        src, dst = edge["source"], edge["target"]
        if src in smiles and dst in diameters:
            substrate_of[dst] = src
        elif src in diameters and dst in smiles:
            precursors_of.setdefault(src, []).append(dst)

    out = []
    for rxn_id in diameters:
        if rxn_id not in substrate_of or rxn_id not in precursors_of:
            continue
        diameter = diameters[rxn_id]
        out.append((
            smiles[substrate_of[rxn_id]],
            [smiles[p] for p in precursors_of[rxn_id]],
            int(diameter) if diameter is not None else None,
        ))
    return path.parent.name, out


def load_reference_steps() -> List[Tuple[str, List]]:
    files = sorted(GOLDEN_DATASET_DIR.glob("*/*_experimental.json"))
    if not files:
        raise SystemExit(f"no curated pathways under {GOLDEN_DATASET_DIR}")
    return [steps_of(p) for p in files]


def _skeletons(smiles_list: Sequence[str]) -> FrozenSet[str]:
    out = set()
    for smi in smiles_list:
        if not smi or is_cofactor(smi):
            continue
        sk = skeleton(smi)
        if sk:
            out.add(sk)
    return frozenset(out)


def classify_step(substrate: str, attested: Sequence[str], rules, prefilter,
                  rule_ec, ranker=None, top_n=None) -> Tuple[str, int, Optional[int]]:
    want = _skeletons(attested)
    report = expand(substrate, rules, prefilter, rule_ec=rule_ec,
                    ranker=ranker, top_n=top_n)
    neighbours = report.neighbours
    if not want:
        # Every attested precursor is a cofactor: the step makes the molecule out of
        # nothing the search would track. Not a coverage failure, so it is excluded
        # rather than scored.
        return "degenerate", len(neighbours), None

    best, best_rule = "missing", None
    for neighbour in neighbours:
        got = _skeletons(neighbour.molecules)
        if got == want:
            return "exact", len(neighbours), neighbour.rule_idx
        if want < got and best != "contains":
            best, best_rule = "contains", neighbour.rule_idx
        elif best == "missing" and (got & want):
            best, best_rule = "partial", neighbour.rule_idx
    return best, len(neighbours), best_rule


def run(radii: Sequence[int], require_ec: bool, ranker_name=None,
        top_n=None) -> List[StepResult]:
    references = load_reference_steps()
    results: List[StepResult] = []
    for radius in radii:
        rules = load_rules(radius)
        rule_ec = annotate_rules(rules)
        prefilter = prefilter_from_rules(rules)
        # The capped run is the one that decides whether a ranked expansion is
        # affordable: coverage is what a cap can only take away.
        from morganbiopilot.one_step.ranking import make_ranker
        ranker = make_ranker(ranker_name, rules)
        cap = f" | ranked by {ranker_name}, top {top_n}" if ranker else ""
        print(f"\nr{radius}: {len(rules)} rules | "
              f"EC coverage {100*rule_ec.coverage:.1f}%{cap}")
        for name, steps in references:
            marks = []
            for i, (substrate, precursors, diameter) in enumerate(steps, 1):
                outcome, n_neigh, rule_idx = classify_step(
                    substrate, precursors, rules, prefilter,
                    rule_ec if require_ec else None, ranker, top_n)
                results.append(StepResult(
                    pathway=name, radius=radius, step=i, substrate=substrate,
                    n_attested=len(_skeletons(precursors)), n_neighbours=n_neigh,
                    outcome=outcome, rule_idx=rule_idx, reference_diameter=diameter,
                ))
                marks.append({"exact": "=", "contains": "+", "partial": "~",
                              "missing": ".", "degenerate": "o"}[outcome])
            steps_scored = [m for m in marks if m != "o"]
            ok = steps_scored and all(m == "=" for m in steps_scored)
            print(f"  {name:28s} {''.join(marks):10s} "
                  f"{'RECOVERABLE' if ok else ''}")
    return results


def report(results: Sequence[StepResult], radii: Sequence[int]) -> None:
    print("\n  = exact   + contains   ~ partial   . missing   o cofactor-only step\n")
    print(f"{'radius':>8s}{'steps':>8s}{'exact':>8s}{'contains':>10s}"
          f"{'partial':>9s}{'missing':>9s}{'routes fully exact':>20s}"
          f"{'median cands':>14s}")
    print("-" * 86)
    for radius in radii:
        rows = [r for r in results if r.radius == radius and r.outcome != "degenerate"]
        if not rows:
            continue
        counts = Counter(r.outcome for r in rows)
        by_route: Dict[str, List[StepResult]] = {}
        for r in rows:
            by_route.setdefault(r.pathway, []).append(r)
        full = sum(1 for steps in by_route.values()
                   if all(s.outcome == "exact" for s in steps))
        cands = sorted(r.n_neighbours for r in rows if r.outcome == "exact")
        median = cands[len(cands) // 2] if cands else float("nan")
        print(f"{'r' + str(radius):>8s}{len(rows):8d}{counts['exact']:8d}"
              f"{counts['contains']:10d}{counts['partial']:9d}{counts['missing']:9d}"
              f"{full:>13d}/{len(by_route):<6d}{median:14.0f}")
    print("\n'median cands' is the number of neighbours the substrate produced on the"
          "\nsteps we did reproduce: coverage without selectivity is not a solved"
          "\nproblem, and it is what the search policy is asked to cut through.")

    # Where a route breaks. A conjunction fails at its weakest step, so the
    # distribution of failures across a route says more than the per-step average.
    for radius in radii:
        rows = [r for r in results if r.radius == radius and r.outcome != "degenerate"]
        by_route: Dict[str, List[StepResult]] = {}
        for r in rows:
            by_route.setdefault(r.pathway, []).append(r)
        broken = {n: [s for s in steps if s.outcome != "exact"]
                  for n, steps in by_route.items()}
        broken = {n: v for n, v in broken.items() if v}
        if not broken:
            continue
        print(f"\nr{radius}: routes broken by how many steps")
        hist = Counter(len(v) for v in broken.values())
        for k in sorted(hist):
            names = sorted(n for n, v in broken.items() if len(v) == k)
            print(f"  {k} step(s) missing: {hist[k]:2d}  {', '.join(names)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Does the rule set contain the attested route at all? "
                    "The ceiling no search policy can raise.")
    p.add_argument("--radii", default="2",
                   help="comma-separated ECFP radii to test, e.g. '1,2,3'")
    p.add_argument("--require-ec", action="store_true",
                   help="restrict to rules carrying a native EC")
    p.add_argument("--sink", default=None,
                   help="override the chassis sink; only affects cofactor stripping")
    p.add_argument("--ranker", default=None,
                   help="'native_similarity' to order rules before validation; "
                        "default is the engine's exhaustive, unordered expansion")
    p.add_argument("--top-n", type=int, default=None,
                   help="keep at most this many neighbours per expansion. Only "
                        "meaningful with --ranker; this is the measurement that says "
                        "whether a capped expansion still contains the attested routes")
    p.add_argument("--out", default="rule_coverage.tsv")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.sink:
        use_sink(args.sink)
    radii = [int(r) for r in args.radii.split(",") if r]

    results = run(radii, args.require_ec, args.ranker, args.top_n)
    report(results, radii)

    out = RESULTS_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0])), delimiter="\t")
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
