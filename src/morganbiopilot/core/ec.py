"""EC-number grounding: what enzyme class does a rule correspond to?

Why this is not free
--------------------
The project note (section 6, lever 1) calls the rule's native EC "almost free".
It is not: the rule ``.npz`` carries **no EC field** at all. EC must be joined
through ``reaction_id`` against ``data/metanetx/processed/metanetx_reactions.tsv``.

The join has three wrinkles, all handled here:

- a rule's ``reaction_id`` is a ``|``-separated list of every MetaNetX reaction
  that collapsed onto it, so a rule maps to a *set* of ECs, not one EC;
- those ids carry a ``__split<N>`` suffix (mono-component splitting) absent from
  the MetaNetX table, and must be stripped before lookup;
- ``ec_numbers`` uses the sentinel ``NOEC`` for "no EC known", which is *not* the
  same as a missing row and must not be counted as coverage.

Partial ECs such as ``1.14.-`` occur and are kept as-is: truncating them to a
level they do not reach would invent precision.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from morganbiopilot.core.paths import METANETX_REACTIONS_TSV

NOEC = "NOEC"
_SPLIT_SUFFIX = re.compile(r"__split\d+$")


@lru_cache(maxsize=1)
def reaction_ec_table() -> Dict[str, Tuple[str, ...]]:
    """MetaNetX reaction id (e.g. ``MNXR100024_L2R``) -> its EC numbers.

    Reactions annotated ``NOEC`` map to an empty tuple: present in the table, but
    carrying no enzyme annotation.
    """
    df = pd.read_csv(METANETX_REACTIONS_TSV, sep="\t", usecols=["id", "ec_numbers"])
    table: Dict[str, Tuple[str, ...]] = {}
    for rid, ecs in zip(df["id"], df["ec_numbers"]):
        if not isinstance(ecs, str) or ecs == NOEC:
            table[str(rid)] = ()
        else:
            table[str(rid)] = tuple(sorted({e.strip() for e in ecs.split(";") if e.strip()}))
    return table


def strip_split(reaction_id: str) -> str:
    """Drop the mono-component ``__split<N>`` suffix to match the MetaNetX table."""
    return _SPLIT_SUFFIX.sub("", reaction_id)


def ec_level(ec: str, level: int) -> str:
    """Truncate an EC number to `level` (1-4).

    Returns "" when the EC does not reach that level — e.g. ``1.14.-`` at level 4.
    Never pads with '-', which would claim precision the annotation lacks.
    """
    parts = ec.split(".")
    if level < 1 or level > 4 or len(parts) < level:
        return ""
    if any(p == "-" for p in parts[:level]):
        return ""
    return ".".join(parts[:level])


@dataclass
class RuleEC:
    """Native EC annotation of every rule in a rule set.

    ``ec[i]`` is the set of EC numbers carried by the MetaNetX reactions that
    produced rule ``i`` — possibly empty, possibly spanning several enzyme classes.
    """

    ec: Tuple[Tuple[str, ...], ...]
    n_reactions: np.ndarray          # MetaNetX reactions behind each rule
    n_unresolved: np.ndarray         # ids not found in the MetaNetX table

    def __len__(self) -> int:
        return len(self.ec)

    def at_level(self, rule_idx: int, level: int) -> Tuple[str, ...]:
        """EC numbers of one rule, truncated to `level` and deduplicated."""
        out = {ec_level(e, level) for e in self.ec[rule_idx]}
        out.discard("")
        return tuple(sorted(out))

    @property
    def coverage(self) -> float:
        """Share of rules carrying at least one EC. The honest grounding rate."""
        return float(sum(1 for e in self.ec if e) / len(self.ec)) if len(self.ec) else float("nan")

    def ambiguity(self, level: int = 1) -> float:
        """Share of EC-carrying rules spanning several classes at `level`."""
        carrying = [i for i, e in enumerate(self.ec) if e]
        if not carrying:
            return float("nan")
        multi = sum(1 for i in carrying if len(self.at_level(i, level)) > 1)
        return multi / len(carrying)


def annotate_rules(rules) -> RuleEC:
    """Join a `RuleSet` against MetaNetX to recover each rule's native EC."""
    table = reaction_ec_table()

    ecs = []
    n_reactions = np.zeros(len(rules), dtype=np.int32)
    n_unresolved = np.zeros(len(rules), dtype=np.int32)

    for i, field in enumerate(rules.reaction_id):
        ids = [p for p in str(field).split("|") if p]
        n_reactions[i] = len(ids)

        found = set()
        missing = 0
        for rid in ids:
            key = strip_split(rid)
            if key in table:
                found.update(table[key])
            else:
                missing += 1
        n_unresolved[i] = missing
        ecs.append(tuple(sorted(found)))

    return RuleEC(ec=tuple(ecs), n_reactions=n_reactions, n_unresolved=n_unresolved)
