"""Load ECFP reaction rules from THIS project's data directory.

Why this module exists
----------------------
Two reasons, both about `morganrxn.core.reaction_rules.ReactionRules.load()`.

**1. It reads the wrong directory.** It takes no path and resolves through
``morganrxn.core.paths.REACTION_RULES_DIR``, i.e. ``<morganrxn repo root>/data``.
Called from MorganBioPilot it reads that package's copy, not ``code/data/``. Both
copies are currently identical, so the mistake is SILENT — until one is
regenerated. morganrxn is under review elsewhere and must not be modified, so we
reimplement path resolution here.

**2. It is too slow for a search loop.** It converts both fingerprint matrices to
lists of tuples::

    obj.ecfp_reaction = [tuple(x) for x in data["ecfp_reaction"]]

That is 82,434 x 1024 Python tuples per matrix: measured 28.2 s out of 39.1 s of
load time at r2 (72%), against 2.2 s to read the .npz. Those tuples exist only so
that `drop_duplicates` and `compute_score` can hash rows; the prefilter and the
expansion need plain numpy arrays. `RuleSet` keeps the arrays and skips the
conversion — an 18x faster load, which matters when the agent arbitrates between
six rule diameters.

`load_reaction_rules()` still returns a genuine `ReactionRules` for the cases that
need morganrxn's own filters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from morganbiopilot.core.paths import REACTION_RULES_DIR

# ECFP parameters of the rules shipped in data/ (see CLAUDE.md).
DEFAULT_FP_SIZE = 1024
DEFAULT_FOLDED = True
DEFAULT_CUSTOM = False


def ecfp_params_for_radius(
    radius: int,
    fp_size: int = DEFAULT_FP_SIZE,
    folded: bool = DEFAULT_FOLDED,
    custom: bool = DEFAULT_CUSTOM,
) -> dict:
    """Build the ecfp_params dict expected by morganrxn."""
    return {"radius": int(radius), "fpSize": int(fp_size), "folded": bool(folded), "custom": bool(custom)}


def _stem(ecfp_params: dict) -> str:
    """Rule directory name. Mirrors ``ReactionRules._ecfp_stem``."""
    suffix = "folded" if ecfp_params.get("folded", False) else "unfolded"
    suffix2 = "custom" if ecfp_params.get("custom", False) else "uncustom"
    return f"ecfp_r{int(ecfp_params['radius'])}_fp{int(ecfp_params['fpSize'])}_{suffix}_{suffix2}"


def rules_path(radius: int, database_name: str = "metanetx", **kwargs) -> Path:
    """Path to the rules .npz, inside MorganBioPilot's own data directory."""
    return REACTION_RULES_DIR / database_name / _stem(ecfp_params_for_radius(radius, **kwargs)) / "rules.npz"


@dataclass
class RuleSet:
    """Reaction rules as numpy arrays, at one ECFP radius.

    Directionality: rules are *not* retro or forward. A template is a directed
    graph rewrite, and the rule set contains both orientations of each reaction
    (``_L2R`` / ``_R2L`` suffixes in ``reaction_id``). Search direction is a
    property of how the engine walks the graph, never of a rule.
    """

    ecfp_reaction: np.ndarray            # (n, d) int32 — the translation
    ecfp_reaction_center: np.ndarray     # (n, d) int32, <= 0 — the precondition
    template_reaction: np.ndarray        # (n,) object — SMARTS
    smi_sub: np.ndarray                  # (n,) object
    nb_prod: np.ndarray                  # (n,) int32
    score: np.ndarray                    # (n,) float32 — NLL, lower = more frequent
    reaction_id: np.ndarray              # (n,) object
    reaction_monocomp_id: np.ndarray     # (n,) object
    ecfp_params: dict
    source_path: Optional[Path] = None

    def __len__(self) -> int:
        return int(self.ecfp_reaction.shape[0])

    @property
    def radius(self) -> int:
        return int(self.ecfp_params["radius"])

    def select(self, keep) -> "RuleSet":
        """Return a new RuleSet restricted to ``keep`` (indices or boolean mask)."""
        return RuleSet(
            ecfp_reaction=self.ecfp_reaction[keep],
            ecfp_reaction_center=self.ecfp_reaction_center[keep],
            template_reaction=self.template_reaction[keep],
            smi_sub=self.smi_sub[keep],
            nb_prod=self.nb_prod[keep],
            score=self.score[keep],
            reaction_id=self.reaction_id[keep],
            reaction_monocomp_id=self.reaction_monocomp_id[keep],
            ecfp_params=self.ecfp_params,
            source_path=self.source_path,
        )

    # Deliberately no `filter_by_smi_sub_atoms`.
    #
    # morganrxn has one, and it is right for that paper: dropping rules with tiny
    # substrate patterns raises reported prefilter precision (measured here: 19.6%
    # -> ~98% on L-tyrosine at r2), which is a property of the applicability
    # criterion worth reporting.
    #
    # In a pathway search it has no such justification, and it is actively harmful:
    # a rule with a small substrate pattern is exactly the rule that applies to a
    # small metabolite. With min_atoms=5, three golden-set targets — 2-amino-1,3-
    # propanediol, cis,cis-muconate, mesaconate — expanded to *zero* neighbours and
    # were unsolvable before any policy ran. Small targets are not noise here; they
    # are a third of the short pathways in the evaluation set.


def load_rules(radius: int, database_name: str = "metanetx", **kwargs) -> RuleSet:
    """Load rules for one radius from ``code/data/``, as numpy arrays.

    The radius is the promiscuity axis: r0 = generalized/speculative,
    r5 = conservative/close to a known reaction.
    """
    ecfp_params = ecfp_params_for_radius(radius, **kwargs)
    path = rules_path(radius, database_name, **kwargs)

    if not path.exists():
        raise FileNotFoundError(f"Rules not found: {path}")

    data = np.load(path, allow_pickle=True)
    n = len(data["nb_prod"])

    def opt(key):
        # Same backward compatibility as ReactionRules.load()
        return data[key] if key in data.files else np.array([None] * n, dtype=object)

    return RuleSet(
        ecfp_reaction=np.asarray(data["ecfp_reaction"], dtype=np.int32),
        ecfp_reaction_center=np.asarray(data["ecfp_reaction_center"], dtype=np.int32),
        template_reaction=data["template_reaction"],
        smi_sub=data["smi_sub"],
        nb_prod=np.asarray(data["nb_prod"], dtype=np.int32),
        score=np.asarray(data["score"], dtype=np.float32),
        reaction_id=opt("reaction_id"),
        reaction_monocomp_id=opt("reaction_monocomp_id"),
        ecfp_params=ecfp_params,
        # Record the path actually read: the bug we work around is silent, so we
        # want it visible in experiment logs.
        source_path=path,
    )


def load_reaction_rules(radius: int, database_name: str = "metanetx", **kwargs):
    """Load as a genuine `morganrxn` ReactionRules, for its own filter methods.

    Slow (tuple conversion, ~39 s at r2). Prefer `load_rules` unless you need
    `drop_duplicates`, `compute_score` or `merge`.
    """
    from morganrxn.core.reaction_rules import ReactionRules

    ecfp_params = ecfp_params_for_radius(radius, **kwargs)
    path = rules_path(radius, database_name, **kwargs)
    if not path.exists():
        raise FileNotFoundError(f"Rules not found: {path}")

    obj = ReactionRules(database_name, ecfp_params=ecfp_params)
    data = np.load(path, allow_pickle=True)

    obj.template_reaction = data["template_reaction"].tolist()
    obj.ecfp_reaction = [tuple(x) for x in data["ecfp_reaction"]]
    obj.ecfp_reaction_center = [tuple(x) for x in data["ecfp_reaction_center"]]
    obj.smi_sub = data["smi_sub"].tolist()
    obj.nb_prod = data["nb_prod"].tolist()
    obj.score = data["score"].tolist()

    n = len(obj.nb_prod)
    obj.reaction_monocomp_id = data["reaction_monocomp_id"].tolist() if "reaction_monocomp_id" in data.files else [None] * n
    obj.reaction_id = data["reaction_id"].tolist() if "reaction_id" in data.files else [None] * n

    # Guard rail: save() would write into the repo under review.
    def _forbid_save(*_a, **_k):
        raise RuntimeError(
            "ReactionRules.save() would write into the morganrxn repository, "
            "which is under review. Write to morganbiopilot.core.paths.REACTION_RULES_DIR."
        )

    obj.save = _forbid_save
    obj.save_chunk = _forbid_save
    obj.source_path = path
    return obj
