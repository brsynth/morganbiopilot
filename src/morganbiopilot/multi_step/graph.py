"""The AND-OR search graph: molecules, rule applications, and status propagation.

Structure
---------
- **OR-node** = one molecule. Solved when it is a building block, or when at least
  one of its rule applications is solved.
- **AND-node** = one validated rule application (`one_step.Neighbour`). Solved when
  *every* molecule it points to is solved.

Molecules are deduplicated by sanitized SMILES, so the structure is a graph, not a
tree: two branches reaching the same intermediate share it, and solving it once
solves it everywhere.

Against the RetroMorgan draft
-----------------------------
That version keyed nodes on ECFP vectors and needed a combinatorial test to decide
whether a node lay in building-block space (`N_combination.is_N_combination`,
subset-sum over the whole sink, with disk-backed bit indices). Mono-component
molecular nodes turn that into a dictionary lookup on the InChIKey skeleton — the
single largest simplification the graph-level formulation buys.

This module holds no search strategy. Which node to expand next is the policy's
decision, and the policy alone (see `policy.py`).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from morganbiopilot.core.building_blocks import is_building_block, is_cofactor
from morganbiopilot.core.chem import sanitize
from morganbiopilot.one_step.expand import Neighbour


@dataclass
class MoleculeNode:
    """An OR-node: one molecule to make."""

    smiles: str
    depth: int
    solved: bool = False
    expanded: bool = False
    in_sink: bool = False
    is_cofactor: bool = False
    parents: Set[int] = field(default_factory=set)      # AND-node ids
    children: List[int] = field(default_factory=list)   # AND-node ids

    @property
    def available(self) -> bool:
        """Obtainable without being made: a chassis metabolite, or a cofactor.

        Both terminate a branch, for the same reason — the cell already has them.
        Neither is ever expanded.
        """
        return self.in_sink or self.is_cofactor


@dataclass
class ReactionNode:
    """An AND-node: one rule application. All its molecules are required."""

    neighbour: Neighbour
    parent: int                                  # MoleculeNode id
    children: List[int] = field(default_factory=list)   # MoleculeNode ids
    solved: bool = False

    @property
    def rule_idx(self) -> int:
        return self.neighbour.rule_idx


class SearchGraph:
    """Explored AND-OR graph, with solved-status propagation."""

    def __init__(self, target: str):
        self.molecules: Dict[int, MoleculeNode] = {}
        self.reactions: Dict[int, ReactionNode] = {}
        self._by_smiles: Dict[str, int] = {}
        self._next_mol = 0
        self._next_rxn = 0

        self.root = self.add_molecule(sanitize(target), depth=0)

    # ---------------------------------------------------------------- molecules
    def add_molecule(self, smiles: str, depth: int) -> int:
        """Add a molecule, or return the existing node id (dedup by SMILES)."""
        smiles = sanitize(smiles)
        if smiles in self._by_smiles:
            node_id = self._by_smiles[smiles]
            # Keep the shallowest route to this molecule.
            node = self.molecules[node_id]
            node.depth = min(node.depth, depth)
            return node_id

        node_id = self._next_mol
        self._next_mol += 1

        in_sink = is_building_block(smiles)
        cofactor = is_cofactor(smiles)
        self.molecules[node_id] = MoleculeNode(
            smiles=smiles, depth=depth, in_sink=in_sink, is_cofactor=cofactor,
            solved=in_sink or cofactor,
        )
        self._by_smiles[smiles] = node_id
        return node_id

    def molecule_id(self, smiles: str) -> Optional[int]:
        return self._by_smiles.get(sanitize(smiles))

    # ---------------------------------------------------------------- reactions
    def add_neighbour(self, parent_id: int, neighbour: Neighbour) -> int:
        """Attach one validated rule application under a molecule node."""
        parent = self.molecules[parent_id]

        rxn_id = self._next_rxn
        self._next_rxn += 1
        rxn = ReactionNode(neighbour=neighbour, parent=parent_id)

        for smi in neighbour.molecules:
            child_id = self.add_molecule(smi, depth=parent.depth + 1)
            rxn.children.append(child_id)
            self.molecules[child_id].parents.add(rxn_id)

        self.reactions[rxn_id] = rxn
        parent.children.append(rxn_id)

        self._propagate_from(rxn_id)
        return rxn_id

    # -------------------------------------------------------------- propagation
    def _propagate_from(self, rxn_id: int) -> None:
        """Recompute solved status upward from one reaction node.

        A reaction is solved when all its molecules are; a molecule is solved when
        it is in the sink or any of its reactions is. Walks up until nothing
        changes, so a molecule solved deep in the graph unlocks every branch that
        shares it.
        """
        stack = [rxn_id]
        seen_rxn: Set[int] = set()

        while stack:
            rid = stack.pop()
            if rid in seen_rxn:
                continue
            seen_rxn.add(rid)

            rxn = self.reactions[rid]
            now = all(self.molecules[c].solved for c in rxn.children)
            if now == rxn.solved:
                continue
            rxn.solved = now

            parent = self.molecules[rxn.parent]
            parent_now = parent.available or any(self.reactions[r].solved for r in parent.children)
            if parent_now == parent.solved:
                continue
            parent.solved = parent_now

            stack.extend(parent.parents)

    # ------------------------------------------------------------------ queries
    @property
    def solved(self) -> bool:
        return self.molecules[self.root].solved

    def frontier(self) -> List[int]:
        """Molecules that can still be expanded: unsolved, unexpanded, not in sink.

        This is the set the policy chooses from — the only decision the LLM agent
        makes in this project.
        """
        return [
            i for i, m in self.molecules.items()
            if not m.solved and not m.expanded and not m.available
        ]

    # Enumeration steps allowed in one `pathways` call before it gives up. Reached only
    # on graphs that are both large and densely solved -- which budget 200 made routine
    # and budget 50 never did.
    MAX_PATHWAY_WORK = 2_000_000

    def shortest_route(self, node_id: Optional[int] = None) -> Optional[List[int]]:
        """The shortest solved route below a molecule, without enumerating any other.

        `pathways` answers this by building every route and taking the minimum, which
        is a cartesian product and the reason a solved search could hang. The shortest
        route needs no enumeration: on an AND-OR graph the cost of a molecule is 0 when
        it is available and otherwise `1 + sum(cost(children))` minimised over its
        solved reactions, which is a fixed point reached by relaxation in at most
        `len(molecules)` sweeps. Cycles need no special case: a cycle can only ever
        raise a cost, so it never enters the fixed point.

        Returns None when the molecule is unsolved, and [] when it is already available.
        """
        if node_id is None:
            node_id = self.root
        if not self.molecules[node_id].solved:
            return None

        INF = float("inf")
        cost = {i: (0 if m.available else INF) for i, m in self.molecules.items()}
        best: Dict[int, Optional[int]] = {i: None for i in self.molecules}

        changed = True
        while changed:
            changed = False
            for rid, rxn in self.reactions.items():
                if not rxn.solved:
                    continue
                total = 1
                for child in rxn.children:
                    total += cost[child]
                    if total == INF:
                        break
                if total < cost[rxn.parent]:
                    cost[rxn.parent] = total
                    best[rxn.parent] = rid
                    changed = True

        if cost[node_id] == INF:
            return None

        # Walk the argmin back down. `_seen` guards a malformed graph only: a cycle
        # cannot be on a finite-cost path, so this terminates on any graph the
        # relaxation above scored.
        route: List[int] = []
        stack, seen = [node_id], set()
        while stack:
            mid = stack.pop()
            if mid in seen or self.molecules[mid].available:
                continue
            seen.add(mid)
            rid = best[mid]
            if rid is None:
                continue
            route.append(rid)
            stack.extend(self.reactions[rid].children)
        return route

    def pathways(self, node_id: Optional[int] = None, _seen=None,
                 max_routes: int = 256, _work: Optional[List[int]] = None
                 ) -> List[List[int]]:
        """Solved routes below a molecule, as lists of reaction-node ids.

        Returns [[]] for a molecule already available (sink or cofactor), and []
        when the molecule is unsolved. Cycles are cut by `_seen`.

        `max_routes` caps what is RETURNED, and that is not enough on its own. It was
        assumed to bound the cost and does not: `_seen` differs along every path, so
        nothing can be memoised and the same subgraph is re-enumerated once per path
        that reaches it. On a 1382-molecule graph this call had not returned after nine
        minutes when asked for a single route, which is what stalled four runs of the
        campaign at budget 200 -- after they had already found their route, since an
        unsolved root returns [] immediately.

        `MAX_PATHWAY_WORK` therefore bounds the work as well, deterministically rather
        than by wall-clock: a research result should not depend on how loaded the node
        was. On exhaustion the routes found so far are returned, so the caller sees a
        truncated enumeration rather than a hang. **Use `shortest_route` when the
        shortest route is what you need** -- it is exact and needs no enumeration.
        """
        if node_id is None:
            node_id = self.root
        _seen = set() if _seen is None else _seen
        top_level = _work is None
        _work = [0] if _work is None else _work

        node = self.molecules[node_id]
        if node.available:
            return [[]]
        if not node.solved or node_id in _seen:
            return []

        _seen = _seen | {node_id}
        routes: List[List[int]] = []

        for rid in node.children:
            if len(routes) >= max_routes or _work[0] >= self.MAX_PATHWAY_WORK:
                break
            rxn = self.reactions[rid]
            if not rxn.solved:
                continue

            combos: List[List[int]] = [[rid]]
            for child in rxn.children:
                _work[0] += 1
                sub = self.pathways(child, _seen, max_routes=max_routes, _work=_work)
                if not sub:
                    combos = []
                    break
                combos = [c + s for c in combos for s in sub][:max_routes]
                _work[0] += len(combos)
            routes.extend(combos)

        # A solved molecule must never report zero routes. Budget exhaustion inside a
        # sub-enumeration surfaces as `sub == []`, which the AND-node rule turns into
        # "this reaction has no route", and that can empty the whole result — reporting
        # 0 routes for a graph that demonstrably has one. Fall back to the exact
        # shortest route, which costs nothing and is always right.
        if top_level and not routes and node.solved:
            fallback = self.shortest_route(node_id)
            if fallback:
                return [fallback]
        return routes[:max_routes]

    def __repr__(self) -> str:
        n_solved = sum(1 for m in self.molecules.values() if m.solved)
        return (f"SearchGraph(molecules={len(self.molecules)}, reactions={len(self.reactions)}, "
                f"solved_molecules={n_solved}, root_solved={self.solved})")
