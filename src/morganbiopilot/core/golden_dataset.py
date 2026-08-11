"""The golden dataset: 20 experimentally curated pathways (Koch et al. 2020).

Format: one directory per target, each holding ``<name>_core.json`` and
``<name>_experimental.json`` in Cytoscape shape::

    {"elements": {"nodes": [{"data": {...}}], "edges": [{"data": {...}}]}}

Compound nodes carry ``SMILES``, ``InChI``, ``inSink``, ``isSource`` and
``Names`` (InChIKey first). Reaction nodes carry ``Rule ID``, ``EC number``,
``Reaction SMILES`` and ``Diameter``.

Two cautions
------------
**`Diameter` is a RetroRules diameter, not our ECFP radius.** The reference
pathways use RetroRules diameters (2..16); our rule sets are indexed by the
morganrxn ECFP radius h (r0..r5), a different quantity — template compatibility
ties an ECFP radius h to a reaction-center radius 2h (paper, Proposition 1). The
two indexings must be reconciled explicitly before the golden `Diameter` can be
used as ground truth for the promiscuity axis. Do not assume r_k == diameter 2k.

**Sink membership here is the dataset's own**, computed against whatever chassis
Koch et al. used. `core.building_blocks` is the authority for this project; the
`inSink` flag is kept for comparison, not for the sink test.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from morganbiopilot.core.chem import sanitize
from morganbiopilot.core.paths import GOLDEN_DATASET_DIR


@dataclass(frozen=True)
class GoldenReaction:
    """One step of a reference pathway."""

    rule_id: Tuple[str, ...]
    ec_numbers: Tuple[str, ...]
    reaction_smiles: str
    diameter: Optional[int]          # RetroRules diameter — see module docstring
    node_id: str


@dataclass
class GoldenPathway:
    """One curated route, target included."""

    name: str
    variant: str                     # "experimental" | "core"
    target: str                      # sanitized SMILES
    compounds: Dict[str, str]        # node id -> sanitized SMILES
    sink_ids: Tuple[str, ...]        # nodes the dataset itself marks as sink
    reactions: Tuple[GoldenReaction, ...]
    edges: Tuple[Tuple[str, str], ...]

    def __len__(self) -> int:
        return len(self.reactions)

    @property
    def ec_numbers(self) -> Tuple[str, ...]:
        return tuple(ec for r in self.reactions for ec in r.ec_numbers)

    @property
    def diameters(self) -> Tuple[int, ...]:
        return tuple(r.diameter for r in self.reactions if r.diameter is not None)


def _as_tuple(value) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def load_pathway(path) -> GoldenPathway:
    """Parse one golden JSON file."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    nodes = [n["data"] for n in raw["elements"]["nodes"]]
    edges = tuple((e["data"]["source"], e["data"]["target"]) for e in raw["elements"]["edges"])

    compounds, sink_ids, target = {}, [], None
    reactions = []

    for node in nodes:
        if node.get("type") == "compound":
            smi = sanitize(str(node.get("SMILES", "")))
            compounds[node["id"]] = smi
            if node.get("inSink"):
                sink_ids.append(node["id"])
            if node.get("isSource"):
                target = smi
        elif node.get("type") == "reaction":
            diameter = node.get("Diameter")
            reactions.append(GoldenReaction(
                rule_id=_as_tuple(node.get("Rule ID")),
                ec_numbers=_as_tuple(node.get("EC number")),
                reaction_smiles=str(node.get("Reaction SMILES") or ""),
                diameter=int(diameter) if diameter is not None else None,
                node_id=node["id"],
            ))

    name = path.parent.name
    variant = "experimental" if "_experimental" in path.stem else "core"

    if target is None:
        raise ValueError(f"{path}: no compound flagged isSource")

    return GoldenPathway(
        name=name, variant=variant, target=target, compounds=compounds,
        sink_ids=tuple(sink_ids), reactions=tuple(reactions), edges=edges,
    )


@lru_cache(maxsize=4)
def load_golden_dataset(variant: str = "experimental") -> Tuple[GoldenPathway, ...]:
    """All 20 curated pathways, sorted by target name.

    `variant` is "experimental" (the published routes) or "core" (the reduced
    ones shipped alongside).
    """
    if variant not in ("experimental", "core"):
        raise ValueError(f"variant must be 'experimental' or 'core', got {variant!r}")

    paths = sorted(GOLDEN_DATASET_DIR.glob(f"*/*_{variant}.json"))
    if not paths:
        raise FileNotFoundError(f"No golden pathway found in {GOLDEN_DATASET_DIR}")
    return tuple(load_pathway(p) for p in paths)


def golden_targets(variant: str = "experimental") -> List[Tuple[str, str]]:
    """(name, sanitized target SMILES) for every pathway — the evaluation set."""
    return [(p.name, p.target) for p in load_golden_dataset(variant)]
