"""Can a score replace the ranking model retrobiosynthesis does not have?

Organic retrosynthesis agents filter candidates that a *trained neural policy* has
already ranked by learned probability -- top-10 for RETRO-R1, 50 narrowed to 20 for
RetroAgent. Template-based retrobiosynthesis has no such model: applying the rule set to
vanillin yields 28 distinct precursor sets with no probability, no ordering, and nothing
that separates attested enzymology from an extrapolation of the same rule. That is why
eleven of the twenty candidates shown to our agent were stilbene artefacts.

This asks whether a score can do the job, and it is the cheapest decisive experiment
available: the 20 curated pathways record the true disconnection at every step, so for
each step we can expand the product, collect every candidate precursor set our engine
produces, and ask where the *reference* set lands in the ranking.

Why the enzymatic score is better posed here than on the frontier
----------------------------------------------------------------
`data_processing.enzymatic_model` was trained on pairs sharing a substrate: same
molecule, two templates, matched element delta. Ranking frontier nodes asked it to
compare *different substrates at different depths*, which is not that comparison. Ranking
candidates asks it to compare *different templates on one substrate*, which is exactly
that comparison. The remaining extrapolation is milder and measurable: every candidate
here is a MetaNetX template, so the model must rank within the class it was trained to
separate from another. This script measures whether it can.

Baselines that decide what the numbers mean
-------------------------------------------
**Random** fixes the null: mean normalised rank 0.5 by construction.
**Sink closeness** is the quantity our greedy and MCTS baselines already use, so a score
that does not beat it adds nothing to the engine.
**Rule support** -- how many distinct rules produce the same precursor set -- is a
frequency prior needing no chemistry. If it wins, the signal is popularity.

    python -m morganbiopilot.paper_results.rank_disconnections
"""

import argparse
import random
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from morganbiopilot.core.building_blocks import inchikey_skeleton
from morganbiopilot.core.chem import sanitize
from morganbiopilot.core.golden_dataset import load_golden_dataset
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.heuristics import SinkCloseness
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import prefilter_from_rules


def _skeleton(smiles: str) -> Optional[str]:
    flat = sanitize(smiles)
    return inchikey_skeleton(flat) if flat else None


def reference_steps(pathways) -> List[Tuple[str, str, str]]:
    """(pathway, molecule to expand, reference precursor) for every reference step.

    Taken from the pathway topology, not from `GoldenReaction.reaction_smiles`: that
    field holds the RetroRules SMARTS template with atom maps, not a reaction between
    molecules, so matching molecules against it recovers almost nothing. The `edges`
    alternate compound -> reaction -> compound and already run in the retro direction,
    which is exactly the step we need.
    """
    steps = []
    for path in pathways:
        into: Dict[str, List[str]] = {}
        out: Dict[str, List[str]] = {}
        for a, b in path.edges:
            if a in path.compounds and b not in path.compounds:
                into.setdefault(b, []).append(a)       # compound -> reaction
            elif a not in path.compounds and b in path.compounds:
                out.setdefault(a, []).append(b)        # reaction -> compound
        for rxn_id, parents in into.items():
            for parent in parents:
                for child in out.get(rxn_id, ()):
                    product, precursor = path.compounds[parent], path.compounds[child]
                    if _skeleton(product) != _skeleton(precursor):
                        steps.append((path.name, product, precursor))
    return steps


def candidates(product: str, rules, prefilter, rule_ec):
    """Distinct precursor sets our engine produces, with how many rules gave each."""
    try:
        report = expand(product, rules, prefilter, rule_ec=rule_ec)
    except Exception:                                            # noqa: BLE001
        return []
    groups: Dict[tuple, List] = {}
    for neighbour in report.neighbours:
        groups.setdefault(tuple(sorted(neighbour.molecules)), []).append(neighbour)
    return [(list(mols), len(group)) for mols, group in groups.items()]


def rank_of(values: List[float], index: int, rng: random.Random) -> float:
    """1-based rank of `index` when sorting by `values` descending.

    Ties are broken at random rather than by position: several scorers here produce
    plateaus, and breaking ties by list order would let insertion order -- which is rule
    order -- leak in as a hidden scorer.
    """
    order = list(range(len(values)))
    rng.shuffle(order)
    order.sort(key=lambda i: -values[i])
    return order.index(index) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--variant", default="experimental")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from morganbiopilot.core.ec import annotate_rules
    from morganbiopilot.data_processing.enzymatic_model import score_reactions

    print(f"loading rules at r{args.radius} ...")
    rules = load_rules(radius=args.radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)
    closeness = SinkCloseness(args.radius)

    pathways = load_golden_dataset(variant=args.variant)
    steps = reference_steps(pathways)
    print(f"{len(pathways)} pathways -> {len(steps)} reference steps")

    rng = random.Random(args.seed)
    ranks: Dict[str, List[float]] = {k: [] for k in
                                     ("enzymatic", "closeness", "rule support", "random")}
    n_cands, recovered, missed = [], 0, Counter()

    for name, product, precursor in steps:
        cands = candidates(product, rules, prefilter, rule_ec)
        if not cands:
            missed["no candidate at all"] += 1
            continue
        wanted = _skeleton(precursor)
        hit = next((i for i, (mols, _) in enumerate(cands)
                    if any(_skeleton(m) == wanted for m in mols)), None)
        if hit is None:
            # The reference disconnection is not in our rule set's reach. Excluded, and
            # counted: a ranking metric conditioned on recall says nothing about recall.
            missed["reference not reachable"] += 1
            continue

        recovered += 1
        n_cands.append(len(cands))
        enz = score_reactions([(product, mols) for mols, _ in cands], default=0.5)
        near = [max((closeness.closeness(m) for m in mols), default=0.0)
                for mols, _ in cands]
        support = [float(n) for _, n in cands]

        ranks["enzymatic"].append(rank_of(list(enz), hit, rng) / len(cands))
        ranks["closeness"].append(rank_of(near, hit, rng) / len(cands))
        ranks["rule support"].append(rank_of(support, hit, rng) / len(cands))
        ranks["random"].append(rank_of([0.0] * len(cands), hit, rng) / len(cands))

    print(f"  reference recovered in {recovered}/{len(steps)} steps "
          f"({100 * recovered / max(len(steps), 1):.0f}%)")
    for reason, count in missed.most_common():
        print(f"    excluded, {reason}: {count}")
    if recovered < 20:
        print("too few usable steps to rank.", file=sys.stderr)
        return 1
    print(f"  candidates per step: median {int(np.median(n_cands))}, "
          f"max {max(n_cands)}")

    print()
    print("-" * 70)
    print(f"{'scorer':16s} {'norm.rank':>10s} {'top-1':>8s} {'top-5':>8s} {'MRR':>7s}")
    print("-" * 70)
    for label in ("enzymatic", "closeness", "rule support", "random"):
        # Normalised rank: rank / number of candidates. Lower is better, 0.5 is chance.
        norm = float(np.mean(ranks[label]))
        raw = [r * n for r, n in zip(ranks[label], n_cands)]
        top1 = float(np.mean([r <= 1 for r in raw]))
        top5 = float(np.mean([r <= 5 for r in raw]))
        mrr = float(np.mean([1.0 / r for r in raw]))
        print(f"{label:16s} {norm:10.3f} {100 * top1:7.0f}% {100 * top5:7.0f}% "
              f"{mrr:7.3f}")

    print()
    enz, near, rand = (float(np.mean(ranks[k]))
                       for k in ("enzymatic", "closeness", "random"))
    if enz > rand - 0.02:
        print("VERDICT: the enzymatic score does not rank the reference disconnection")
        print("above chance. It cannot stand in for the missing policy model, and the")
        print("candidate-filtering pivot needs a different signal -- or the LLM itself.")
    elif enz > near:
        print("VERDICT: the score beats chance but not sink closeness, which the engine")
        print("already computes. It adds nothing the baselines do not have.")
    else:
        print("VERDICT: the score ranks the reference disconnection better than chance")
        print("and better than sink closeness, on the comparison it was trained for")
        print("(one substrate, competing templates). It is a candidate for the ranking")
        print("model retrobiosynthesis lacks; the next question is whether an LLM given")
        print("the same candidates does better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
