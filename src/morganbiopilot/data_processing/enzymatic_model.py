"""An enzymatic score for one mono-substrate reaction: MetaNetX against USPTO.

Input `substrate >> products`, output a score in [0, 1], high when the transformation
resembles metabolic chemistry rather than synthetic organic chemistry. Both rule sets
are already extracted in the same format at the same ECFP parameters -- 82k biochemical
templates under `data/reaction_rules/metanetx`, 85k organic ones under
`data/reaction_rules/uspto` -- which is what makes the framing possible.

**Fitted result: AUC 0.660, 95% CI [0.614, 0.707]** on 263 substrates never seen in
training, 1052 composition-matched pairs, with both controls pinned at exactly 0.500 by
construction rather than by argument (see below). At that AUC the classes overlap
heavily: a single edge's score decides nothing, and whatever value the score has must
come from aggregating along a route. Treat that as the hypothesis to test, not as a
property established here.

    python -m morganbiopilot.data_processing.enzymatic_model --molecules 20000
    python -m morganbiopilot.data_processing.enzymatic_model --report     # inspect
    python -m morganbiopilot.data_processing.enzymatic_model --ablation   # see below

    from morganbiopilot.data_processing.enzymatic_model import score_reactions
    score_reactions([("c1ccccc1O", ["Oc1ccccc1O"])])          # -> array([1.0])

The design, and the confound that killed four earlier framings
--------------------------------------------------------------
Contrasting the two corpora at the level of *reactions* reaches 0.995 AUC -- and 0.995
again for a model shown only the reactants, with the transformation hidden. The label
was "which database", because metabolites and patented intermediates are different
molecules written by different pipelines.

Here the substrate is held **identical**: each molecule is transformed once by a
MetaNetX template that applies to it and once by a USPTO template that applies to it.
Every substrate therefore carries both labels, so a model given the substrate alone sees
the same input twice with opposite labels and *must* score 0.500. That is arithmetic, not
a hoped-for control. Products come from RDKit and are canonicalised, so notation cannot
leak either, and every reaction is validated by `RunReactants` rather than merely
admitted by the fingerprint prefilter -- otherwise the model could separate "actually
applies" from "does not", which is applicability and not plausibility.

`--ablation` shows why that is not enough on its own, and it is the chemically
interesting part. With the pairs merely sharing a substrate, the full model reaches
0.986 -- but a model given only thirteen element counts and the fragment count reaches
0.956, with no single one of those fourteen features above 0.66 alone. The joint
elemental signature is what separates: biochemical templates emit oxygen and phosphorus
(+2.66 O, +0.23 P on average) and split into 1.23 fragments, organic ones emit carbon
(+6.0 C) in one piece. A mono-substrate template carries its original partner's atoms on
its product side, so the element delta reports *which co-substrate the reaction used* --
phosphate and water against aryl and alkyl. Real, chemically sensible, and needing no
model at all.

So the pairs used for fitting must also share their exact element delta. Both controls
are then vacuous by arithmetic: each substrate carries both labels, and so does each
element delta. Nothing is left except how the same atoms are arranged, and 0.660 is what
that is worth.

What the number is
------------------
`P(the transformation resembles biochemistry | 50/50 prior)`. The prior matters and is
not a detail: the training set holds exactly one biochemical and one organic
transformation per substrate, by construction, so the base rate is 0.5 by design and
has no relation to any deployment base rate. The calibrated output is therefore a
rescaled likelihood ratio, **not** P(enzymatic) in any population. Ranking with it is
legitimate; reading 0.7 as "70% chance an enzyme does this" is not.

Two further limits, both structural rather than fixable by more data:

*The negatives are unlabelled.* USPTO is not certified non-enzymatic -- hydrolysis,
reduction, oxidation and transamination happen in flasks and in cells alike, and patents
contain biocatalytic steps. High scores mean "looks metabolic"; low scores mean "looks
like flask chemistry", not "no enzyme does this".

*In our own engine this is an extrapolation.* The model was trained to separate
MetaNetX templates from USPTO templates. Every edge our search produces comes from a
MetaNetX template, so on our frontier the model is asked a question it was never
supervised on -- whether *this particular application* of a biochemical template is
plausible. The framing that does measure that (`reaction_pu`, substrate scope, matched
within rule and split by rule) returned a clean null: 0.534 against a rule-only control
of exactly 0.500. Use this score for triaging **mixed** rule sets, where the biochemical
/ organic distinction is a real decision; treat any use on our own frontier as an
untested hypothesis and run it as an ablation arm, not as a default column.

This is positive-unlabelled learning, of the case-control kind
-------------------------------------------------------------
P is MetaNetX: certainly enzymatic, attested in a biochemical database. U is USPTO: a
genuine mixture, mostly non-enzymatic but containing real enzymatic chemistry, since
enzymes and flasks share hydrolysis, reduction, oxidation and transamination. An
unlabelled pool that is a mixture with hidden positives is exactly the PU setting -- and
it is what an earlier, deleted version was *not*: that one kept a real substrate and
attached another reaction's products to it, size- then composition-matched, reaching
0.720. Fabricated negatives are self-supervised negative sampling, whatever they are
called, and no unlabelled pool means no PU.

The model here is the naive PU classifier -- U trained as if negative. Elkan and Noto
showed this recovers `g(x) = c * P(positive|x)`, a constant multiple of the true
posterior, so **the ranking is exactly correct and the AUC needs no correction**. The
probability scale is another matter. Ours is a *case-control* design: equal numbers were
drawn from P and from U by construction rather than sampled from one pool, so recovering
`P(enzymatic|x)` needs the contamination rate of U -- the fraction of USPTO template
applications an enzyme would in fact perform. Nothing in our data estimates that, and
the SCAR assumption that would let Elkan-Noto's `c` stand in fails anyway: MetaNetX is
not a random sample of enzymatic chemistry, it is the catalogued part of it, weighted
towards central metabolism.

So: rank with this score, do not read it as a probability. That is a limit of the
available labels, not of the fit, and more data will not lift it.

The raw score is the primary output, and the calibrated one is secondary
-----------------------------------------------------------------------
`score_reactions` returns the **uncalibrated** `g(x)` by default. That is deliberate.
Elkan-Noto guarantees the ranking of `g` and nothing about its scale, the scale is
unrecoverable here anyway (see above), and ranking is what every intended use needs --
a frontier column, a product along a route, a rule-set triage.

Calibration is available via `calibrated=True` but it costs resolution. The first
version of this module used isotonic regression, which with a few hundred calibration
points is a step function with about five levels: three chemically unrelated reactions
all came back as exactly 0.500, not because the model was undecided but because they
landed on the same plateau. Collapsing a ranking signal into five buckets defeats the
purpose. Platt scaling replaced it -- two parameters, smooth, strictly monotone when the
fitted slope is positive, which is checked at fit time and refuses to save if violated.

Both are monotone, so the test AUC is identical either way; only the Brier score and the
reliability curve move, and both are reported so the effect is visible rather than
assumed.

The split is three-way and grouped by substrate throughout: fit / calibrate / test. A
substrate's two transformations always travel together, since separating them would put
the same input in both folds with opposite labels -- the pathology that once drove an
earlier experiment to AUC 0.171, below chance.

The artifact records its own metrics
------------------------------------
Anything loading the model can read what it is worth (`meta["test_auc"]`, the CI, the
control values, the sample size) without going back to a log file. A score whose
provenance has been separated from its validation is how a 0.673 ranking signal turns
into a claimed probability three months later.
"""

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from morganrxn.core.reaction_utils import apply_reaction

from morganbiopilot.core.chem import mol_ecfp, sanitize, split_components
from morganbiopilot.core.paths import ENZYMATIC_SCORE_MODEL
from morganbiopilot.core.rules import load_rules
from morganbiopilot.one_step.prefilter import prefilter_from_rules

ELEMENTS = ("C", "H", "N", "O", "P", "S", "Cl", "F", "Br", "I", "Fe", "Mg", "Zn")

# How many prefilter-admitted templates to try before giving up on a substrate. The
# prefilter is a necessary condition only, so several candidates typically fail
# `RunReactants`. The budget has to be generous: composition matching needs a *choice*
# of transformations per substrate, not the first one that runs.
ATTEMPTS_PER_SUBSTRATE = 80

_SCORER: Optional[dict] = None


# ------------------------------------------------------------------- pair construction


def _counts(smiles: List[str]) -> Optional[Dict[str, int]]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    total: Counter = Counter()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            total[atom.GetSymbol()] += 1
            total["H"] += atom.GetTotalNumHs()
    return dict(total)


def _fingerprint(smiles: List[str], radius: int) -> Optional[np.ndarray]:
    """Counted ECFP summed over components, or None if any component fails."""
    total = None
    for smi in smiles:
        try:
            vec = np.asarray(mol_ecfp(smi, radius), dtype=np.int32)
        except Exception:                                        # noqa: BLE001
            return None
        total = vec if total is None else total + vec
    return total


def _delta_key(substrate: str, products: List[str]) -> Optional[tuple]:
    """Element counts gained and lost, as a hashable key."""
    before, after = _counts([substrate]), _counts(products)
    if before is None or after is None:
        return None
    return tuple(after.get(e, 0) - before.get(e, 0) for e in ELEMENTS)


def substrate_pool(rule_sets, limit: int, seed: int) -> List[str]:
    """Molecules drawn from both rule sets' original substrates.

    The union rather than MetaNetX alone, so the *substrate* space is not itself
    biochemical by construction. Which corpus a molecule came from is irrelevant to the
    labels -- both classes are built on every molecule that survives -- but a pool of
    metabolites only would leave the result describing metabolites only.
    """
    rng = random.Random(seed)
    seen, pool = set(), []
    for rules in rule_sets:
        smiles = [str(s) for s in rules.smi_sub]
        rng.shuffle(smiles)
        for smi in smiles[: limit * 2]:
            flat = sanitize(smi)
            if flat and flat not in seen:
                seen.add(flat)
                pool.append(flat)
    rng.shuffle(pool)
    return pool[: limit * 2]


def transform(pool: List[str], rules, prefilter, radius: int, cap: int, seed: int,
              per_substrate: int) -> Dict[str, Dict[tuple, Tuple[List[str], int]]]:
    """Validated transformations per substrate, indexed by element delta.

    `{substrate: {delta: (products, rule_idx)}}`. Several per substrate, because
    composition matching needs a choice: the pairing looks for a delta both rule sets
    reach on the same molecule.

    `cap` limits how often one template may be reused. Groups are substrates, so a
    popular template can appear in both folds; the cap keeps a handful of very
    promiscuous rules from letting the model memorise them instead of reading chemistry.
    """
    rng = random.Random(seed)
    used: Counter = Counter()
    out: Dict[str, Dict[tuple, Tuple[List[str], int]]] = {}

    for substrate in pool:
        vec = _fingerprint([substrate], radius)
        if vec is None:
            continue
        candidates = [int(i) for i in prefilter.applicable(vec)]
        if not candidates:
            continue
        rng.shuffle(candidates)

        found: Dict[tuple, Tuple[List[str], int]] = {}
        for rule_idx in candidates[:ATTEMPTS_PER_SUBSTRATE]:
            if len(found) >= per_substrate:
                break
            if used[rule_idx] >= cap:
                continue
            try:
                products = apply_reaction(str(rules.template_reaction[rule_idx]),
                                          substrate)
            except Exception:                                    # noqa: BLE001
                continue
            if not products:
                continue
            molecules = [sanitize(m) for m in split_components(products[0])]
            if not molecules or not all(molecules):
                continue
            key = _delta_key(substrate, molecules)
            if key is None or key in found:
                continue
            used[rule_idx] += 1
            found[key] = (molecules, rule_idx)
        if found:
            out[substrate] = found
    return out


def pair_up(pool, by_enzyme, by_flask, matched: bool, seed: int):
    """Matched triples (substrate, enzymatic products, organic products).

    With `matched`, the two transformations must also share their element delta, which
    makes the balance control uninformative by construction rather than by argument.
    """
    rng = random.Random(seed + 7)
    pairs, rules_used = [], set()
    for substrate in pool:
        left, right = by_enzyme.get(substrate), by_flask.get(substrate)
        if not left or not right:
            continue
        if matched:
            shared = sorted(set(left) & set(right))
            if not shared:
                continue
            key = rng.choice(shared)
            enzymatic, organic = left[key], right[key]
        else:
            enzymatic = left[rng.choice(sorted(left))]
            organic = right[rng.choice(sorted(right))]
        pairs.append((substrate, enzymatic[0], organic[0]))
        rules_used.add(("e", enzymatic[1]))
        rules_used.add(("o", organic[1]))
    return pairs, len(rules_used)


def build(pairs, radius: int):
    """Features for matched (substrate, enzymatic products, organic products) triples.

    Returns fingerprint rows, shallow-chemistry rows, substrate-only rows, labels and
    groups, with the two members of a pair adjacent and sharing a group.
    """
    fp_rows, shallow_rows, sub_rows, labels, groups = [], [], [], [], []

    for group, (substrate, enzymatic, organic) in enumerate(pairs):
        left = _fingerprint([substrate], radius)
        before = _counts([substrate])
        if left is None or before is None:
            continue

        sides = []
        for products in (enzymatic, organic):
            right = _fingerprint(products, radius)
            after = _counts(products)
            if right is None or after is None:
                break
            sides.append((right, after, len(products)))
        if len(sides) != 2:
            continue    # drop the whole pair, or the substrate stops being balanced

        for label, (right, after, n_prod) in zip((1, 0), sides):
            fp_rows.append(np.concatenate([left, right, right - left]))
            shallow_rows.append([after.get(e, 0) - before.get(e, 0) for e in ELEMENTS]
                                + [n_prod])
            sub_rows.append(left)
            labels.append(label)
            groups.append(group)

    return (np.asarray(fp_rows, dtype=np.float32),
            np.asarray(shallow_rows, dtype=np.float32),
            np.asarray(sub_rows, dtype=np.float32),
            np.asarray(labels), np.asarray(groups))


# --------------------------------------------------------------------------- features


def featurize(reactions: Sequence[Tuple[str, List[str]]], radius: int
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Feature rows for (substrate, products) pairs, and a mask of what survived.

    The layout must match `build` above exactly -- substrate ECFP, product ECFP, then
    their difference -- or the model is fed a permuted vector and fails silently rather
    than loudly. The artifact records the layout as a string for the same reason.
    """
    rows, ok = [], np.zeros(len(reactions), dtype=bool)
    for i, (substrate, products) in enumerate(reactions):
        left = _fingerprint([substrate], radius)
        right = _fingerprint(list(products), radius)
        if left is None or right is None:
            continue
        rows.append(np.concatenate([left, right, right - left]))
        ok[i] = True
    x = (np.asarray(rows, dtype=np.float32) if rows
         else np.empty((0, 0), dtype=np.float32))
    return x, ok


# --------------------------------------------------------------------------- inference


def load_scorer(path=None) -> dict:
    """Load and cache the artifact. Raises if it has not been trained yet."""
    global _SCORER
    if _SCORER is not None and path is None:
        return _SCORER
    import joblib

    target = path or ENZYMATIC_SCORE_MODEL
    if not target.exists():
        raise FileNotFoundError(
            f"No fitted enzymatic score at {target}. Train it first:\n"
            "  python -m morganbiopilot.data_processing.enzymatic_model")
    artifact = joblib.load(target)
    if path is None:
        _SCORER = artifact
    return artifact


def score_reactions(reactions: Sequence[Tuple[str, List[str]]], path=None,
                    calibrated: bool = False,
                    default: float = float("nan")) -> np.ndarray:
    """Scores in [0, 1], one per (substrate, products) pair. Higher = more metabolic.

    Raw by default: the ranking is the only thing the PU setting justifies, and Platt
    scaling -- while monotone -- buys nothing for it. Pass `calibrated=True` for values
    comparable against the reliability curve stored in the artifact.

    Batched because the engine produces edges by the thousand -- a per-edge call would
    pay the fingerprinting overhead one molecule at a time. Pairs whose fingerprints
    cannot be computed get `default` rather than a silent 0.5, so an unscorable edge is
    distinguishable from a genuinely uncertain one.
    """
    artifact = load_scorer(path)
    out = np.full(len(reactions), default, dtype=np.float64)
    if not reactions:
        return out
    x, ok = featurize(reactions, artifact["radius"])
    if len(x) == 0:
        return out
    raw = artifact["model"].predict_proba(x)[:, 1]
    out[ok] = (artifact["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
               if calibrated else raw)
    return out


def score_reaction(substrate: str, products: Sequence[str], path=None,
                   calibrated: bool = False) -> float:
    """Convenience wrapper for a single reaction. Prefer `score_reactions` in loops."""
    return float(score_reactions([(substrate, list(products))], path=path,
                                 calibrated=calibrated)[0])


# --------------------------------------------------------------------------- training


def _auc_ci(auc: float, n_pos: int, n_neg: int) -> Tuple[float, float, float]:
    """Hanley-McNeil standard error and 95% interval for an AUC.

    Reported because 0.673 on 86 test substrates and 0.673 on 2000 are different
    claims, and the sample size is the binding constraint on this experiment.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    return se, auc - 1.96 * se, auc + 1.96 * se


def _reliability(y, p, bins: int = 5) -> List[Tuple[float, float, int]]:
    """(mean predicted, observed frequency, count) per quantile bin."""
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        out.append((float(p[chunk].mean()), float(y[chunk].mean()), len(chunk)))
    return out


def collect(molecules: int, radius: int, cap: int, per_substrate: int, seed: int):
    """Run both rule sets over a shared substrate pool. The expensive step.

    Returns `(pool, by_enzyme, by_flask)` so the pairing can be redone in either arm
    without paying for template application twice.
    """
    print(f"loading both rule sets at r{radius} ...")
    enzymatic = load_rules(radius=radius, database_name="metanetx")
    organic = load_rules(radius=radius, database_name="uspto")
    print(f"  {len(enzymatic.smi_sub)} biochemical, {len(organic.smi_sub)} organic")

    pool = substrate_pool([enzymatic, organic], molecules, seed)
    print(f"  substrate pool: {len(pool)} molecules")

    print(f"running templates, up to {per_substrate} transformations per substrate "
          "per set ...")
    t0 = time.perf_counter()
    by_enzyme = transform(pool, enzymatic, prefilter_from_rules(enzymatic),
                          radius, cap, seed, per_substrate)
    by_flask = transform(pool, organic, prefilter_from_rules(organic),
                         radius, cap, seed + 1, per_substrate)
    print(f"  {time.perf_counter() - t0:.0f}s: {len(by_enzyme)} transformed "
          f"biochemically, {len(by_flask)} organically")
    return pool, by_enzyme, by_flask


def build_pairs(molecules: int, radius: int, cap: int, per_substrate: int, seed: int):
    """Composition-matched pairs: same substrate, same element delta, both worlds."""
    pool, by_enzyme, by_flask = collect(molecules, radius, cap, per_substrate, seed)
    pairs, n_rules = pair_up(pool, by_enzyme, by_flask, True, seed)
    print(f"  {len(pairs)} composition-matched pairs, {n_rules} distinct templates")
    return pairs


def ablation(molecules: int, radius: int, cap: int, per_substrate: int, seed: int
             ) -> int:
    """Loose against composition-matched, to show what the element delta was carrying.

    The loose arm is the one worth *not* believing: it scores 0.986, and a fourteen-
    feature atom counter scores 0.956 on the same data. Printing both side by side is
    the only way the matched arm's lower number reads as the stronger result.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    pool, by_enzyme, by_flask = collect(molecules, radius, cap, per_substrate, seed)

    for matched in (False, True):
        arm = "COMPOSITION-MATCHED" if matched else "LOOSE (substrate only)"
        pairs, n_rules = pair_up(pool, by_enzyme, by_flask, matched, seed)
        print()
        print("=" * 74)
        print(f"{arm}: {len(pairs)} pairs, {n_rules} distinct templates")
        if len(pairs) < 250:
            print("  too few pairs to read; raise --molecules.")
            continue

        x_fp, x_shallow, x_sub, y, groups = build(pairs, radius)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train, test = next(splitter.split(x_fp, y, groups=groups))
        print("-" * 74)
        for label, block in (("substrate only (design check)", x_sub),
                             ("element balance + fragment count", x_shallow),
                             ("substrate + products + difference", x_fp)):
            model = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
            model.fit(block[train], y[train])
            auc = roc_auc_score(y[test], model.predict_proba(block[test])[:, 1])
            print(f"{label:40s} AUC {auc:.3f}   ({block.shape[1]} features)")
    return 0


def fit(pairs, radius: int, seed: int) -> Tuple[dict, Dict[str, float]]:
    """Fit, calibrate and evaluate. Returns the artifact and its metrics."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    x_fp, x_shallow, x_sub, y, groups = build(pairs, radius)
    print(f"  {int(y.sum())} biochemical, {int((1 - y).sum())} organic, "
          f"{len(set(groups))} substrates, {x_fp.shape[1]} features")

    # Three-way, grouped by substrate at every level: a substrate's two transformations
    # never straddle a fold.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    rest, test = next(outer.split(x_fp, y, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    fit_idx, calib = (rest[i] for i in next(
        inner.split(x_fp[rest], y[rest], groups=groups[rest])))
    print(f"  split: {len(fit_idx)} fit / {len(calib)} calibrate / {len(test)} test")

    model = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
    model.fit(x_fp[fit_idx], y[fit_idx])

    # Platt scaling: logistic regression on the single raw-probability feature. Smooth
    # where isotonic quantises, and monotone provided the slope is positive -- checked
    # below, because a negative slope would mean the calibrator inverts the ranking the
    # AUC was measured on.
    calibrator = LogisticRegression()
    raw_calib = model.predict_proba(x_fp[calib])[:, 1]
    calibrator.fit(raw_calib.reshape(-1, 1), y[calib])
    slope = float(calibrator.coef_[0][0])

    raw_test = model.predict_proba(x_fp[test])[:, 1]
    cal_test = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    auc = float(roc_auc_score(y[test], raw_test))
    n_pos, n_neg = int(y[test].sum()), int((1 - y[test]).sum())
    se, lo, hi = _auc_ci(auc, n_pos, n_neg)

    # The controls are the result, not the AUC. They must land at chance -- substrate
    # because each one carries both labels, balance because each element delta does.
    controls = {}
    for name, block in (("substrate only", x_sub),
                        ("element balance + fragments", x_shallow)):
        ctrl = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
        ctrl.fit(block[fit_idx], y[fit_idx])
        controls[name] = float(roc_auc_score(
            y[test], ctrl.predict_proba(block[test])[:, 1]))

    metrics = {
        "test_auc": auc, "auc_se": se, "auc_ci95": (lo, hi),
        "brier_raw": float(brier_score_loss(y[test], raw_test)),
        "brier_calibrated": float(brier_score_loss(y[test], cal_test)),
        "n_pairs": len(pairs), "n_test_substrates": len(set(groups[test])),
        "controls": controls,
        "reliability": _reliability(y[test], cal_test),
        "platt_slope": slope,
        # Distinct values the score can take on the test fold. Isotonic collapsed this
        # to about five, which is how the quantisation problem was found; a ranking
        # signal with few levels cannot rank.
        "distinct_raw": int(len(np.unique(np.round(raw_test, 6)))),
        "distinct_calibrated": int(len(np.unique(np.round(cal_test, 6)))),
    }
    artifact = {
        "model": model, "calibrator": calibrator, "radius": radius,
        "feature_layout": "concat(ecfp(substrate), ecfp(products), difference)",
        "positives": "MetaNetX templates run on their own substrate",
        "negatives": "USPTO templates run on the same substrate, same element delta",
        "training_base_rate": 0.5,
        "means": "P(resembles biochemistry | 50/50 prior) -- a rescaled likelihood "
                 "ratio, not P(enzymatic) in any population",
        "metrics": metrics, "seed": seed,
    }
    return artifact, metrics


def report(metrics: Dict[str, float]) -> None:
    lo, hi = metrics["auc_ci95"]
    print()
    print("-" * 74)
    print(f"test AUC                     {metrics['test_auc']:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]   "
          f"({metrics['n_test_substrates']} unseen substrates)")
    for name, value in metrics["controls"].items():
        flag = "ok" if abs(value - 0.5) <= 0.05 else "LEAK"
        print(f"  control: {name:27s} {value:.3f}  <- {flag} "
              "(chance by construction)")
    print(f"Brier   raw {metrics['brier_raw']:.4f} -> calibrated "
          f"{metrics['brier_calibrated']:.4f}   "
          f"(Platt, slope {metrics['platt_slope']:+.2f}; monotone, AUC unchanged)")
    print(f"resolution: {metrics['distinct_raw']} distinct raw scores, "
          f"{metrics['distinct_calibrated']} calibrated "
          "(few levels cannot rank)")
    print()
    print("  reliability (equal-count bins)")
    print("     predicted   observed   n")
    for predicted, observed, count in metrics["reliability"]:
        print(f"       {predicted:.3f}      {observed:.3f}    {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--molecules", type=int, default=20000)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--cap", type=int, default=20,
                        help="maximum times one template may be reused")
    parser.add_argument("--per-substrate", type=int, default=20,
                        help="transformations per substrate per set; matched pairs "
                             "grow quadratically in this, runtime only linearly")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", action="store_true",
                        help="print the stored metrics of the fitted model and exit")
    parser.add_argument("--ablation", action="store_true",
                        help="loose vs composition-matched, without fitting anything")
    args = parser.parse_args()

    out = ENZYMATIC_SCORE_MODEL if args.out is None else Path(args.out)

    if args.report:
        artifact = load_scorer(out)
        print(f"{out}")
        print(f"  radius {artifact['radius']} | {artifact['feature_layout']}")
        print(f"  positives: {artifact['positives']}")
        print(f"  negatives: {artifact['negatives']}")
        print(f"  score means: {artifact['means']}")
        report(artifact["metrics"])
        return 0

    if args.ablation:
        return ablation(args.molecules, args.radius, args.cap,
                        args.per_substrate, args.seed)

    pairs = build_pairs(args.molecules, args.radius, args.cap,
                        args.per_substrate, args.seed)
    if len(pairs) < 250:
        print("too few matched pairs to fit; raise --molecules or --per-substrate.",
              file=sys.stderr)
        return 1

    artifact, metrics = fit(pairs, args.radius, args.seed)
    report(metrics)

    leaked = [n for n, v in metrics["controls"].items() if abs(v - 0.5) > 0.05]
    if leaked:
        print()
        print(f"NOT SAVED: {', '.join(leaked)} should be at chance by construction.")
        print("A control above chance means the matching is broken, and the model "
              "would be")
        print("scoring the artefact rather than the chemistry.")
        return 1
    if metrics["platt_slope"] <= 0:
        print()
        print(f"NOT SAVED: Platt slope {metrics['platt_slope']:+.2f} is not positive, "
              "so the")
        print("calibrator inverts the ranking the AUC was measured on.")
        return 1

    import joblib
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    print()
    print(f"wrote {out}")
    print("  score with: from morganbiopilot.data_processing.enzymatic_model "
          "import score_reactions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
