"""An enzymatic score for mono-substrate reactions, learned from MetaNetX alone.

**This experiment returns a null, and the null is why it is kept.** It is the only
framing that asks the question the search actually has -- would *this* enzyme act on
*this* molecule -- and the answer is that our inputs do not contain it: AUC 0.534
against a rule-only control at exactly 0.500. Its sibling `enzymatic_model` does
separate biochemical from organic *transformations* (0.660), but that is a different
question, and this file is the record of why that score must not be read as substrate
plausibility on our own frontier.

The unit is the edge, not the node. A frontier molecule is reached by applying a
template to its parent, and what the search needs scored is that application:
`substrate >> products`. Two earlier framings missed it: one scored the pair (rule,
substrate) through fingerprint arithmetic without ever running the template, the other
scored the molecule alone, which cannot express that the same molecule is a fine
substrate for one enzyme and not for another. Both are deleted; their numbers survive
in the control notes below.

Construction
------------
Positives are attested mono-component reactions: each rule's template applied to
the substrate it was extracted from, so the reaction exists in MetaNetX by
construction. Unlabelled are the same templates applied to *other* metabolites
drawn from the same pool of original substrates.

**Templates are actually run**, and that is the point of this version rather than a
detail. The prefilter is a necessary condition only, so a pair it admits may fail
`RunReactants`. An earlier version used the prefilter alone, which left its unlabelled
pool contaminated with applications that are not reactions at all -- and a model can
separate "really applies" from "does not" without knowing any chemistry. Here every
pair on both sides is a validated application, so that difference cannot be what is
being learned.

Three controls, because every earlier version was inflated by one artefact
-------------------------------------------------------------------------
**Within-rule matching.** Each rule contributes its attested substrate against k
others it also transforms. Without this the label leaks through how promiscuous a
rule is: a rule admitting many molecules yields one positive and many unlabelled, a
restrictive one the reverse, and the reaction-centre vector encodes exactly that.
An unmatched version reached 0.948 AUC with no substrate information at all.

**Rule-disjoint split.** Every pair of a rule goes to the same side. Splitting by
*substrate* instead -- which looks stricter -- is incompatible with the matched
design: a rule's positive and its unlabelled partners then land on opposite sides,
the matching is destroyed in the test fold, and a model that learned "this rule
appeared with a positive" is wrong on that rule's unlabelled examples in a
systematic direction. That produced AUC 0.171, below chance, which is how the
contradiction was found. Splitting by rule keeps the matching intact in both folds;
substrates then recur across folds, which can only depress the score, never inflate
it, since a molecule is positive for one rule and unlabelled for another.

**Rule-only.** Trained on the reaction-centre vector with no substrate. If it scores
as well as the full model, the result is a per-rule prior and says nothing about
which substrates a template suits.

What the score can mean
-----------------------
The unlabelled pool holds real enzymatic reactions MetaNetX does not record --
enzyme promiscuity is ordinary -- so this is positive-unlabelled and the output
ranks *resemblance to attested reactions*. It is not P(enzymatic), and labelling
absence from the database as invalid would teach the search to reject exactly the
novelty template-based retrobiosynthesis exists to propose.

    python -m morganbiopilot.data_processing.reaction_pu --rules 6000
"""

import argparse
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from morganrxn.core.reaction_utils import apply_reaction

from morganbiopilot.core.chem import mol_ecfp, sanitize, split_components
from morganbiopilot.core.rules import load_rules
from morganbiopilot.one_step.prefilter import prefilter_from_rules


def _apply(template: str, substrate: str, radius: int
           ) -> Optional[Tuple[np.ndarray, int]]:
    """Run a template and return (summed product ECFP, product count), or None.

    None means the template did not apply at the graph level -- the false positives
    the fingerprint prefilter is expected to let through. Excluding them from both
    classes is what stops the model from learning applicability instead of
    plausibility.
    """
    try:
        products = apply_reaction(template, substrate)
    except Exception:                                          # noqa: BLE001
        return None
    if not products:
        return None
    molecules = split_components(products[0])
    if not molecules:
        return None
    total = None
    for smi in molecules:
        flat = sanitize(smi)
        if not flat:
            return None
        try:
            vec = np.asarray(mol_ecfp(flat, radius), dtype=np.int32)
        except Exception:                                      # noqa: BLE001
            return None
        total = vec if total is None else total + vec
    return total, len(molecules)


def build(rules, prefilter, n_rules: int, per_rule: int, seed: int):
    """Attested and unlabelled mono-component reactions, matched within rule."""
    rng = random.Random(seed)
    radius = rules.radius
    substrates = [str(s) for s in rules.smi_sub]

    order = list(range(len(substrates)))
    rng.shuffle(order)

    # Fingerprint the substrate pool once. It doubles as the source of unlabelled
    # candidates, so both classes are drawn from the same metabolites and the model
    # cannot separate them by recognising where the molecule came from.
    pool_smiles, pool_fp = [], []
    for idx in order[:max(n_rules * 2, 4000)]:
        flat = sanitize(substrates[idx])
        if not flat:
            continue
        try:
            pool_fp.append(np.asarray(mol_ecfp(flat, radius), dtype=np.int32))
        except Exception:                                      # noqa: BLE001
            continue
        pool_smiles.append(flat)
    pool_fp = np.asarray(pool_fp, dtype=np.int32)
    row_of = {smi: i for i, smi in enumerate(pool_smiles)}
    print(f"  substrate pool: {len(pool_smiles)} metabolites")

    admits: Dict[int, List[int]] = {}
    for row in range(len(pool_smiles)):
        for rule_idx in prefilter.applicable(pool_fp[row]):
            admits.setdefault(int(rule_idx), []).append(row)

    sub_fp, prod_fp, n_prod, rule_ids, labels, groups = [], [], [], [], [], []
    used, attempted = 0, 0
    for rule_idx in order:
        if used >= n_rules:
            break
        own = sanitize(substrates[rule_idx])
        own_row = row_of.get(own)
        if own_row is None:
            continue
        template = str(rules.template_reaction[rule_idx])

        attempted += 1
        positive = _apply(template, own, radius)
        if positive is None:
            continue      # the template does not even run on its own substrate

        others = [r for r in admits.get(rule_idx, ()) if r != own_row]
        rng.shuffle(others)
        drawn = []
        for row in others:
            if len(drawn) >= per_rule:
                break
            got = _apply(template, pool_smiles[row], radius)
            if got is not None:
                drawn.append((row, got))
        if len(drawn) < per_rule:
            continue      # too few *validated* alternatives to match against

        used += 1
        sub_fp.append(pool_fp[own_row]); prod_fp.append(positive[0])
        n_prod.append(positive[1]); rule_ids.append(rule_idx)
        labels.append(1); groups.append(rule_idx)
        for row, (vec, count) in drawn:
            sub_fp.append(pool_fp[row]); prod_fp.append(vec)
            n_prod.append(count); rule_ids.append(rule_idx)
            labels.append(0); groups.append(rule_idx)

    print(f"  {used} rules matched ({attempted} attempted), "
          f"{len(labels)} validated reactions")
    return (np.asarray(sub_fp), np.asarray(prod_fp), np.asarray(n_prod),
            np.asarray(rule_ids), np.asarray(labels), np.asarray(groups))


def features(sub, prod, n_prod, rule_ids, rules, with_substrate=True):
    centre = np.asarray(rules.ecfp_reaction_center, dtype=np.int32)[rule_ids]
    if not with_substrate:
        return centre.astype(np.float32)
    # The slack is the prefilter's own quantity, >= 0 by applicability: how far the
    # substrate exceeds what the template strictly demands. A substrate that only
    # just satisfies a precondition is not in the same position as one that
    # satisfies it amply, and that is the difference between attested use and
    # extrapolation.
    slack = sub + centre
    return np.hstack([sub, prod, slack,
                      n_prod.reshape(-1, 1)]).astype(np.float32)


def score(x, y, groups, label, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train, test = next(splitter.split(x, y, groups=groups))
    model = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
    model.fit(x[train], y[train])
    auc = roc_auc_score(y[test], model.predict_proba(x[test])[:, 1])
    print(f"{label:32s} AUC {auc:.3f}   ({x.shape[1]} features, "
          f"{len(set(groups[test]))} unseen rules)")
    return auc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rules", type=int, default=6000)
    parser.add_argument("--per-rule", type=int, default=3)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"loading rules at r{args.radius} ...")
    rules = load_rules(radius=args.radius)
    prefilter = prefilter_from_rules(rules)

    print("building validated mono-component reactions "
          "(templates are actually run) ...")
    t0 = time.perf_counter()
    sub, prod, n_prod, rule_ids, y, groups = build(
        rules, prefilter, args.rules, args.per_rule, args.seed)
    print(f"  {time.perf_counter() - t0:.0f}s: "
          f"{int(y.sum())} attested, {int((1 - y).sum())} unlabelled")
    if y.sum() < 200:
        print("too few reactions; raise --rules.", file=sys.stderr)
        return 1

    print()
    print("-" * 66)
    full = score(features(sub, prod, n_prod, rule_ids, rules, True),
                 y, groups, "substrate + products + slack", args.seed)
    rule_only = score(features(sub, prod, n_prod, rule_ids, rules, False),
                      y, groups, "rule only (no substrate)", args.seed)

    print()
    gap = full - rule_only
    if rule_only > 0.85 and gap < 0.05:
        print("VERDICT: the substrate adds nothing -- a per-rule prior, not a "
              "reaction score.")
    elif gap > 0.05:
        print(f"VERDICT: the substrate carries {gap:.3f} AUC the rule alone does "
              "not, on")
        print("substrates never seen in training, with every reaction on both sides "
              "validated")
        print("by RunReactants. This is the first framing where the separation "
              "cannot be")
        print("explained by applicability, rule frequency, or molecule memorisation.")
    else:
        print("VERDICT: no separation. The framing does not carry signal either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
