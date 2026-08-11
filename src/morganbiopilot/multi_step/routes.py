"""Turning a solved search into readable, storable pathways.

`SearchGraph.pathways()` returns lists of reaction-node ids, which mean nothing
once the run is over. This module resolves them into what a chemist actually wants
to see — target, each enzymatic step with its EC, and the chassis metabolites the
route terminates on — in a form that survives to disk.

Two things worth keeping in the record beyond the chemistry:

- **which leaves are building blocks and which are cofactors.** Both terminate a
  branch, but they are different claims: "the cell already makes this precursor"
  versus "this is ATP". A route that bottoms out entirely on cofactors is not a
  synthesis.
- **the cofactors each step consumed.** `expand` strips them from the search, so
  they are invisible in the graph, yet a route that silently requires four
  distinct cofactors is not equivalent to one that requires none. Section 8 of the
  project note asks for cofactor balance as part of enzymatic plausibility; this
  keeps the raw material for it.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from morganbiopilot.core.building_blocks import building_block_entries
from morganbiopilot.multi_step.graph import SearchGraph


@dataclass
class RouteStep:
    """One enzymatic step, written in the forward direction: products -> substrate."""

    depth: int
    substrate: str                  # the molecule this step makes
    precursors: Tuple[str, ...]     # what it is made from
    rule_idx: int
    reaction_id: str = ""
    ec_numbers: Tuple[str, ...] = ()
    cofactors: Tuple[str, ...] = ()
    template: str = ""


@dataclass
class Route:
    """A complete pathway from chassis metabolites up to the target."""

    target: str
    steps: Tuple[RouteStep, ...]
    leaves: Tuple[Tuple[str, str], ...] = ()      # (smiles, label) building blocks
    cofactor_leaves: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def ec_coverage(self) -> float:
        """Share of steps backed by a known enzyme."""
        return sum(1 for s in self.steps if s.ec_numbers) / len(self.steps) if self.steps else 0.0

    @property
    def distinct_cofactors(self) -> Tuple[str, ...]:
        return tuple(sorted({c for s in self.steps for c in s.cofactors}))

    def render(self) -> str:
        lines = [f"TARGET  {self.target}"]
        # Deepest first, so the route reads bottom-up: from the chassis to the target.
        for step in sorted(self.steps, key=lambda s: -s.depth):
            ec = f"  EC {','.join(step.ec_numbers)}" if step.ec_numbers else "  EC —"
            lines.append(f"  [{step.depth}] {' + '.join(step.precursors)}")
            lines.append(f"        --> {step.substrate}{ec}")
            if step.cofactors:
                lines.append(f"        cofactors: {', '.join(step.cofactors)}")
        if self.leaves:
            named = ", ".join(f"{label}" for _, label in self.leaves)
            lines.append(f"  FROM CHASSIS: {named}")
        if self.cofactor_leaves:
            lines.append(f"  (cofactor leaves: {', '.join(self.cofactor_leaves)})")
        lines.append(f"  {len(self)} steps | EC coverage {100*self.ec_coverage:.0f}%")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "n_steps": len(self),
            "ec_coverage": self.ec_coverage,
            "distinct_cofactors": list(self.distinct_cofactors),
            "leaves": [{"smiles": s, "label": l} for s, l in self.leaves],
            "cofactor_leaves": list(self.cofactor_leaves),
            "steps": [asdict(s) for s in sorted(self.steps, key=lambda s: -s.depth)],
        }


def _labels() -> Dict[str, str]:
    return {smi: label for smi, label in building_block_entries()}


def extract_routes(result, rule_ec=None, max_routes: int = 20) -> List[Route]:
    """Resolve a `SearchResult`'s solved pathways into `Route` objects."""
    graph: SearchGraph = result.graph
    labels = _labels()
    routes: List[Route] = []

    for rxn_ids in result.pathways()[:max_routes]:
        steps, leaves, cofactor_leaves = [], [], []
        molecules_in_route = set()

        for rxn_id in rxn_ids:
            rxn = graph.reactions[rxn_id]
            parent = graph.molecules[rxn.parent]
            precursors = tuple(graph.molecules[c].smiles for c in rxn.children)
            molecules_in_route.add(parent.smiles)
            molecules_in_route.update(precursors)

            ec = ()
            if rule_ec is not None:
                ec = tuple(rule_ec.ec[rxn.rule_idx])

            steps.append(RouteStep(
                depth=parent.depth,
                substrate=parent.smiles,
                precursors=precursors,
                rule_idx=rxn.rule_idx,
                reaction_id=str(rxn.neighbour.reaction_id),
                ec_numbers=ec,
                cofactors=tuple(rxn.neighbour.cofactors_removed),
                template=rxn.neighbour.template,
            ))

        # Leaves: molecules in the route that terminate a branch.
        for smiles in sorted(molecules_in_route):
            node_id = graph.molecule_id(smiles)
            if node_id is None:
                continue
            node = graph.molecules[node_id]
            if node.in_sink:
                leaves.append((smiles, labels.get(smiles, smiles)))
            elif node.is_cofactor:
                cofactor_leaves.append(smiles)

        routes.append(Route(
            target=result.target, steps=tuple(steps),
            leaves=tuple(leaves), cofactor_leaves=tuple(cofactor_leaves),
        ))

    return routes


def save_routes(routes: List[Route], path: Path, meta: Optional[dict] = None) -> None:
    """Write routes as JSON. One file per (policy, target, seed) run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "routes": [r.to_dict() for r in routes]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
