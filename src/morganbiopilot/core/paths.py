"""MorganBioPilot paths.

Deliberately distinct from ``morganrxn.core.paths``, which resolves to the root
of the morganrxn repository (``Path(__file__).parents[3]``) — and morganrxn is
under review elsewhere, so we do not touch it. All data access for THIS project
goes through the constants below.
"""

from pathlib import Path


def project_root() -> Path:
    # src/morganbiopilot/core/paths.py -> core -> morganbiopilot -> src -> code
    return Path(__file__).resolve().parents[3]


ROOT_DIR = project_root()
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

REACTION_RULES_DIR = DATA_DIR / "reaction_rules"
METANETX_DIR = DATA_DIR / "metanetx"

# Organism sink: the search stops at these. Name + InChI.
BUILDING_BLOCKS_DIR = DATA_DIR / "building_blocks"
BUILDING_BLOCKS_CSV = BUILDING_BLOCKS_DIR / "sink.csv"

# Cofactors, stripped from reaction sides: freely available, never expanded.
COFACTORS_DIR = DATA_DIR / "cofactors"
COFACTORS_TSV = COFACTORS_DIR / "cofactors_metanetx.tsv"

# 20 curated pathways (Koch et al. 2020) — the evaluation set.
GOLDEN_DATASET_DIR = DATA_DIR / "golden_dataset_pathways"

# Rule -> EC join table (see CLAUDE.md: the .npz carries no EC field).
METANETX_REACTIONS_TSV = METANETX_DIR / "processed" / "metanetx_reactions.tsv"

# Fitted models we produce ourselves, with the metrics that justify them. Beside
# ``data/`` rather than inside it: ``data/`` is ~1 GB of third-party corpora and is
# gitignored wholesale, while these are small deliverables we want under version
# control.
MODELS_DIR = ROOT_DIR / "models"
ENZYMATIC_SCORE_MODEL = MODELS_DIR / "enzymatic_score.joblib"
