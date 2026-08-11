"""Targets from a plain list, for benchmarks that ship no reference routes.

The golden set gives a curated pathway per target, so `compare_policies` can
stratify its table by reference route length -- the column where the interesting
differences live. External test sets rarely provide that. BioNavi-NP's ships 388
natural products as `name <TAB> SMILES` and nothing else: solve rate is the whole
contract.

This exposes such a list through the same shape `compare_policies` already reads,
with `len()` returning 0 to mean "no reference". The stratified table then puts
every run in one unnamed bucket, which is the honest rendering of "we do not know
how long these routes should be" -- better than inventing a length from the route
we happened to find, which would score a policy against itself.

Two things worth stating about running this set against our engine.

**Stereochemistry is discarded.** 69% of these targets carry it and the framework
this project builds on is defined on flattened graphs, so a stereospecific target
becomes its flat skeleton. That makes some targets easier than BioNavi-NP intends
and a few ill-posed. It is a property of the representation, not an oversight.

**The sink must be theirs.** They score against 40 building blocks; our E. coli
table has 753, and 31 of their 40 sit inside it. Running their targets against our
sink answers an easier question, so `core.building_blocks.use_sink` exists and
`compare_policies --sink` should point at their file whenever their targets are
used.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from morganbiopilot.core.chem import sanitize


@dataclass
class ListedTarget:
    """One target with no curated route behind it."""

    name: str
    target: str          # sanitized SMILES
    raw: str             # as shipped, stereochemistry included

    def __len__(self) -> int:
        """Reference route length — 0 stands for "unknown", not "zero steps"."""
        return 0

    @property
    def reactions(self) -> Tuple:
        return ()

    @property
    def ec_numbers(self) -> Tuple[str, ...]:
        return ()


def load_target_list(path) -> List[ListedTarget]:
    """Read `name <TAB> SMILES` (or bare SMILES) into targets.

    Entries that sanitize to nothing are dropped rather than carried as empty
    strings: a target the representation cannot express is not a target the
    engine failed on, and counting it as a miss would understate the solve rate
    for every policy equally but wrongly.
    """
    out, seen = [], set()
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        name, raw = (parts[0].strip(), parts[1].strip()) if len(parts) >= 2 \
            else (f"target{index + 1}", parts[0].strip())
        flat = sanitize(raw)
        if not flat or flat in seen:
            continue
        seen.add(flat)
        out.append(ListedTarget(name=name.replace(" ", "_"), target=flat, raw=raw))
    return out
