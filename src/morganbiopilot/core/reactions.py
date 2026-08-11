"""Recovering complete reactions behind the mono-component rules.

The rule set was built by splitting multi-component MetaNetX reactions into
mono-component pieces — that is the ``__split<N>`` suffix on every
``reaction_id``. A rule therefore records one transformation skeleton and drops
everything else the reaction involved: co-substrates, co-products, stoichiometry.

A route found by the search is consequently a skeleton too. It says *which*
conversions occur, not what they consume or release, and that missing half is
exactly what makes a route judgeable — whether it needs O2, whether it releases
formaldehyde, whether the carbon balances. This module puts it back, by resolving
each rule's ``reaction_id`` against ``reac_prop.tsv`` and each participant against
``chem_prop.tsv``.

Direction: MetaNetX writes equations with ``=`` and its left/right ordering is not
a claim about which way the reaction runs. Orientation comes from the ``_L2R`` /
``_R2L`` suffix the rule carries, which is why `CompleteReaction` records it
explicitly rather than trusting the equation's layout.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from morganbiopilot.core.paths import METANETX_DIR

_SPLIT = re.compile(r"__split\d+$")
_DIRECTION = re.compile(r"_(L2R|R2L)$")
_TERM = re.compile(r"^\s*([0-9.]+)\s+(\S+?)(?:@\S+)?\s*$")


@dataclass(frozen=True)
class Participant:
    """One side of a reaction: how much of what."""

    mnx_id: str
    coefficient: float
    name: str = ""
    smiles: str = ""

    def render(self) -> str:
        amount = "" if self.coefficient == 1 else f"{self.coefficient:g} "
        return f"{amount}{self.name or self.mnx_id}"


@dataclass
class CompleteReaction:
    """A full MetaNetX reaction, oriented the way the rule uses it."""

    mnxr_id: str
    direction: str                      # "L2R" or "R2L"
    substrates: Tuple[Participant, ...]
    products: Tuple[Participant, ...]
    ec_numbers: Tuple[str, ...] = ()
    is_balanced: str = ""

    def render(self) -> str:
        left = " + ".join(p.render() for p in self.substrates)
        right = " + ".join(p.render() for p in self.products)
        return f"{left} --> {right}"

    def to_dict(self) -> dict:
        return {
            "mnxr_id": self.mnxr_id,
            "direction": self.direction,
            "is_balanced": self.is_balanced,
            "ec_numbers": list(self.ec_numbers),
            "equation": self.render(),
            "substrates": [{"id": p.mnx_id, "n": p.coefficient,
                            "name": p.name, "smiles": p.smiles} for p in self.substrates],
            "products": [{"id": p.mnx_id, "n": p.coefficient,
                          "name": p.name, "smiles": p.smiles} for p in self.products],
        }


def rule_reaction_ids(reaction_id_field: str) -> List[Tuple[str, str]]:
    """(bare MNXR id, direction) for each reaction a rule was extracted from."""
    out = []
    for token in str(reaction_id_field).split("|"):
        token = token.strip()
        if not token:
            continue
        token = _SPLIT.sub("", token)
        match = _DIRECTION.search(token)
        direction = match.group(1) if match else "L2R"
        out.append((_DIRECTION.sub("", token), direction))
    return out


def _parse_side(text: str) -> List[Tuple[str, float]]:
    terms = []
    for chunk in text.split(" + "):
        match = _TERM.match(chunk)
        if match:
            terms.append((match.group(2), float(match.group(1))))
    return terms


@lru_cache(maxsize=1)
def _equations(wanted: frozenset) -> Dict[str, tuple]:
    """MNXR id -> (left terms, right terms, classifs, is_balanced)."""
    out = {}
    path = METANETX_DIR / "reac_prop.tsv"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] not in wanted or len(parts) < 2 or "=" not in parts[1]:
                continue
            left, _, right = parts[1].partition("=")
            out[parts[0]] = (
                _parse_side(left), _parse_side(right),
                parts[3] if len(parts) > 3 else "",
                parts[4] if len(parts) > 4 else "",
            )
    return out


@lru_cache(maxsize=1)
def _compounds(wanted: frozenset) -> Dict[str, Tuple[str, str]]:
    """MNXM id -> (name, SMILES). One pass over a 772 MB file, so cache it."""
    out = {}
    path = METANETX_DIR / "chem_prop.tsv"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] in wanted:
                out[parts[0]] = (parts[1] if len(parts) > 1 else "",
                                 parts[8] if len(parts) > 8 else "")
    return out


def complete_reactions(reaction_id_fields: Iterable[str],
                       max_per_rule: int = 1) -> Dict[str, List[CompleteReaction]]:
    """Resolve rules' `reaction_id` fields into full, oriented reactions.

    `max_per_rule` caps how many source reactions to resolve per rule: a single
    rule can collapse hundreds of MetaNetX reactions (up to 2,584 at r2), and
    listing them all would drown the route rather than document it.
    """
    fields = list(reaction_id_fields)
    per_field = [rule_reaction_ids(f)[:max_per_rule] for f in fields]
    wanted_rxn = frozenset(rid for pairs in per_field for rid, _ in pairs)
    if not wanted_rxn:
        return {}

    equations = _equations(wanted_rxn)
    wanted_cpd = frozenset(
        cid
        for left, right, _, _ in equations.values()
        for cid, _ in left + right
    )
    compounds = _compounds(wanted_cpd)

    resolved: Dict[str, List[CompleteReaction]] = {}
    for field_text, pairs in zip(fields, per_field):
        entries = []
        for mnxr_id, direction in pairs:
            found = equations.get(mnxr_id)
            if not found:
                continue
            left, right, classifs, balanced = found

            def build(terms):
                return tuple(
                    Participant(cid, coeff, *compounds.get(cid, ("", "")))
                    for cid, coeff in terms
                )

            # The equation's layout carries no direction; the rule's suffix does.
            subs, prods = (build(left), build(right)) if direction == "L2R" \
                else (build(right), build(left))
            entries.append(CompleteReaction(
                mnxr_id=mnxr_id, direction=direction,
                substrates=subs, products=prods,
                ec_numbers=tuple(e for e in classifs.split(";") if e and e != "NOEC"),
                is_balanced=balanced,
            ))
        if entries:
            resolved[field_text] = entries
    return resolved
