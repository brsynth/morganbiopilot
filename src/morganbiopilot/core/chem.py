"""Molecule sanitization and ECFP conventions for the whole project.

Sanitization follows the companion rule-set paper (under review), "Experimental
setup / Molecule sanitization": each structure is parsed into a 2D molecular graph and sanitized
into a canonical RDKit SMILES, wildcard-containing components are removed, and
each component is flattened to its 2D graph, **omitting stereochemistry**.

Dropping stereochemistry is not a shortcut. A template retains only a bounded
neighborhood around the reaction center, so it may truncate the context needed to
define a stereochemical descriptor at the template boundary; keeping stereo would
make the representation inconsistent across reactions (paper, "Reaction
templates"). Every SMILES entering or leaving the expansion engine must therefore
go through `sanitize()`.

The underlying implementation is `morganrxn.core.molecule_utils.sanitize_smiles`,
which takes a `remove_stereo` flag. This module hard-codes it to True so no code
path in the project can silently keep stereochemistry.
"""

from morganrxn.core.molecule_utils import get_mol_ecfp, sanitize_smiles

# Fingerprint dimension used throughout the paper and the shipped rules.
FP_SIZE = 1024

# Radii for which rules exist in data/. r0 = generalized/speculative,
# r5 = conservative.
AVAILABLE_RADII = (0, 1, 2, 3, 4, 5)


def sanitize(smi: str) -> str:
    """Canonical, stereo-free SMILES. The only sanitization entry point.

    Never call ``sanitize_smiles(..., remove_stereo=False)`` elsewhere: the ECFP
    representation of the whole project assumes flattened graphs.
    """
    return sanitize_smiles(smi, remove_stereo=True)


def sanitize_many(smiles) -> list:
    """Sanitize an iterable of SMILES, dropping empties. Order is not preserved."""
    out = {sanitize(smi) for smi in smiles}
    return [smi for smi in out if smi]


def split_components(smi: str) -> list:
    """Split a possibly multi-component SMILES into sanitized single components.

    The search works **mono-component**: every node of the graph is one molecule.
    `sanitize` already canonicalizes and sorts components, so splitting on '.' is
    exact.
    """
    return [c for c in sanitize(smi).split(".") if c]


def ecfp_params(radius: int, fp_size: int = FP_SIZE) -> dict:
    """Counted, folded, non-custom ECFP parameters — the shipped-rule convention.

    Note on the reaction-center radius: for a molecular ECFP of radius h, template
    compatibility requires a reaction-center radius of **2h** (paper, Proposition
    1). That radius is baked into the rules at generation time; `radius` here is h.
    """
    return {"radius": int(radius), "fpSize": int(fp_size), "folded": True, "custom": False}


def mol_ecfp(smi: str, radius: int, fp_size: int = FP_SIZE):
    """Counted ECFP of a SMILES. Sanitizes first — always."""
    return get_mol_ecfp(sanitize(smi), ecfp_params(radius, fp_size))
