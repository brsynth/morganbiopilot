"""Building blocks (the organism's sink) and cofactors.

Matching key
------------
Both sets are matched on the **first block of the InChIKey** (14 characters), not
on SMILES strings. That block is the connectivity skeleton hash: it ignores
stereochemistry, isotopes, charge and protonation state.

This is the right key here for two independent reasons:

- the whole framework works on flattened, stereo-free graphs (companion rule-set
  paper, under review),
  so a stereo-sensitive key would draw distinctions the representation cannot see;
- sinks and cofactor tables are full of protonation and charge variants of the
  same species (H+ as ``MNXM01``/``MNXM1``, ATP(3-) vs ATP(4-), ...). Matching on
  canonical SMILES would miss them and silently fail to reach the sink.

`data/building_blocks/sink.csv` gives Name + InChI; `data/cofactors/
cofactors_biochem.tsv` gives INCHI_PREFIX + INCHIKEY_PREFIX + NAME.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from rdkit import Chem, RDLogger

from morganbiopilot.core.chem import sanitize
from morganbiopilot.core.paths import (BUILDING_BLOCKS_CSV, BUILDING_BLOCKS_DIR,
                                       COFACTORS_TSV)

# Sink and cofactor files contain species RDKit grumbles about (bare protons,
# radical ions). Their parse failures are handled explicitly below.
RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=200_000)
def inchikey_skeleton(smi: str) -> Optional[str]:
    """First InChIKey block of a SMILES, or None if it cannot be computed."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key.split("-")[0] if key else None


def skeleton(smi: str) -> Optional[str]:
    """Sanitize, then take the connectivity hash. The key everything is matched on.

    Route SMILES and `chem_prop` SMILES disagree about protonation: the search carries
    phenolates (`O=Cc1ccc([O-])cc1`) where MetaNetX stores the neutral acid
    (`O=Cc1ccc(O)cc1`). Compared as strings they never match, which once broke every
    completion lookup silently. The first InChIKey block ignores charge and tautomer, so
    it matches what a chemist would call the same compound.

    Three modules had grown their own copy of this two-line composition; a shared one
    keeps the sink, the completion and the route replay agreeing on molecule identity.
    """
    flat = sanitize(smi)
    return inchikey_skeleton(flat) if flat else None


def _skeleton_from_inchi(inchi: str) -> Optional[str]:
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key.split("-")[0] if key else None


# The sink is swappable, because "available" is a property of the benchmark rather
# than of the engine. BioNavi-NP scores against 40 building blocks -- amino acids
# and a few common metabolites -- where our E. coli table has 753, and 31 of their
# 40 are already inside ours. Running their targets against our sink would answer
# an easier question than theirs and the numbers would not be comparable.
#
# Two formats are accepted, distinguished by whether an `InChI` column exists:
# the MetaNetX-derived CSV (Name, InChI) and the tab-separated name/SMILES lists
# that BioNavi-NP ships.
_SINK_PATH: Optional[Path] = None


def use_sink(path=None) -> None:
    """Point the chassis sink at another table, or back to the default.

    Every accessor below is cached, and several caches are keyed only by radius,
    so switching the file without clearing them would silently mix two sinks --
    the search would call a molecule available while the agent was shown the other
    table's nearest neighbour. Clear them together or not at all.
    """
    global _SINK_PATH
    _SINK_PATH = Path(path) if path else None
    for cached in (load_building_blocks, building_block_entries,
                   building_block_smiles, building_block_labels,
                   building_block_ecfps, _metanetx_names):
        cached.cache_clear()


def _sink_path() -> Path:
    return _SINK_PATH or BUILDING_BLOCKS_CSV


def _read_sink(path: Path):
    """(identifier, InChI or None, SMILES or None) rows, whatever the format."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "InChI" in df.columns:
            return [(str(n), str(i) if pd.notna(i) else None, None)
                    for n, i in zip(df["Name"], df["InChI"])]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip().split("\t")
        if len(parts) >= 2:
            rows.append((parts[0].strip(), None, parts[1].strip()))
        else:
            rows.append((parts[0].strip(), None, parts[0].strip()))
    return rows


@lru_cache(maxsize=1)
def load_building_blocks() -> Set[str]:
    """InChIKey skeletons of the organism's sink metabolites."""
    skeletons = set()
    for _, inchi, smi in _read_sink(_sink_path()):
        sk = _skeleton_from_inchi(inchi) if inchi else inchikey_skeleton(sanitize(smi or ""))
        if sk:
            skeletons.add(sk)
    return skeletons


@lru_cache(maxsize=1)
def load_cofactors() -> Set[str]:
    """InChIKey skeletons of the cofactors to strip from reaction sides."""
    df = pd.read_csv(COFACTORS_TSV, sep="\t")
    # The table already carries the prefix; fall back to the InChI when missing.
    skeletons = set()
    for _, row in df.iterrows():
        prefix = str(row.get("INCHIKEY_PREFIX", "") or "").strip()
        if prefix:
            skeletons.add(prefix)
            continue
        sk = _skeleton_from_inchi(str(row.get("INCHI_PREFIX", "")))
        if sk:
            skeletons.add(sk)
    return skeletons


def is_building_block(smi: str) -> bool:
    """True when the molecule is available in the chassis — i.e. a solved node."""
    sk = inchikey_skeleton(sanitize(smi))
    return sk is not None and sk in load_building_blocks()


def is_cofactor(smi: str) -> bool:
    """True when the molecule is a cofactor, hence not worth expanding."""
    sk = inchikey_skeleton(sanitize(smi))
    return sk is not None and sk in load_cofactors()


def strip_cofactors(smiles) -> list:
    """Drop cofactors from a list of (mono-component) SMILES."""
    return [smi for smi in smiles if not is_cofactor(smi)]


@lru_cache(maxsize=1)
def _metanetx_names(ids: frozenset) -> dict:
    """MetaNetX id -> chemical name, from chem_prop.tsv.

    The sink file names its rows with bare MNXM identifiers — 808 of 809 — so on
    its own it can only tell an agent that a target resembles "MNXM10". Joining
    against chem_prop recovers a real name for ~36% of them ("NADH", "butanal",
    "decanoate"), which is the difference between a number and a chemical fact.
    """
    from morganbiopilot.core.paths import METANETX_DIR

    names = {}
    path = METANETX_DIR / "chem_prop.tsv"
    if not path.exists():
        return names
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 2)
            if parts[0] in ids and len(parts) > 1 and parts[1]:
                names[parts[0]] = parts[1]
    return names


@lru_cache(maxsize=1)
def building_block_entries() -> tuple:
    """((sanitized SMILES, label), ...) for the sink, deduplicated.

    `label` is the chemical name when MetaNetX has one, else the raw identifier.
    Entries whose InChI is missing, or that reduce to nothing after sanitization
    (wildcard-only components), are dropped — see the sink-gap note in CLAUDE.md.
    """
    rows = _read_sink(_sink_path())
    ids = frozenset(name for name, inchi, _ in rows if inchi)
    names = _metanetx_names(ids) if ids else {}

    # Names recovered from PubChem, if the lookup script has been run.
    # See data_processing.name_building_blocks.
    #
    # Long PubChem titles are systematic IUPAC names, not names anyone uses:
    # a third of them run past 80 characters and one reaches 217. Such a string
    # costs ~50 prompt tokens per candidate and tells a model nothing the SMILES
    # on the same line does not already say. Filtering happens here rather than in
    # the lookup script so the cache stays faithful to what PubChem returned.
    MAX_LABEL_CHARS = 60

    extra = {}
    cache = BUILDING_BLOCKS_DIR / "names.tsv"
    if cache.exists():
        import csv

        with open(cache, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                label = row.get("label") or ""
                if row.get("source") == "pubchem" and 0 < len(label) <= MAX_LABEL_CHARS:
                    extra[row["smiles"]] = label

    seen = {}
    for raw_name, inchi, raw_smiles in rows:
        if inchi:
            mol = Chem.MolFromInchi(inchi)
            smi = sanitize(Chem.MolToSmiles(mol)) if mol is not None else ""
        else:
            # A name/SMILES table already carries a usable name, so the PubChem
            # cache and the MetaNetX join have nothing to add.
            smi = sanitize(raw_smiles or "")
        if not smi or smi in seen:
            continue
        key = str(raw_name)
        seen[smi] = extra.get(smi) or names.get(key, key)
    return tuple(sorted(seen.items()))


@lru_cache(maxsize=8)
def building_block_smiles() -> tuple:
    """Sanitized SMILES of the sink, for fingerprint-space similarity."""
    return tuple(smi for smi, _ in building_block_entries())


@lru_cache(maxsize=8)
def building_block_labels() -> tuple:
    """Names of the sink metabolites, aligned with `building_block_smiles()`."""
    return tuple(label for _, label in building_block_entries())


@lru_cache(maxsize=8)
def building_block_ecfps(radius: int):
    """(n, d) int32 matrix of sink ECFPs at one radius. Cached per radius."""
    import numpy as np

    from morganbiopilot.core.chem import mol_ecfp

    smiles = building_block_smiles()
    return np.asarray([mol_ecfp(s, radius) for s in smiles], dtype=np.int32)
