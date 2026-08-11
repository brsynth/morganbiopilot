"""UCT over the AND-OR graph: the search-based baseline this literature expects.

Two deliberate departures from textbook MCTS, both forced by the setting.

**No learned value.** The baseline tier of agentic retrosynthesis -- Retro*,
EG-MCTS, PDVN, AiZynthFinder -- scores candidates with a learned single-step
model's probabilities. Our rules carry no usable score, because expansion applies
every applicable template rather than a model's top-k. This is therefore the
analogue of Retro*-0, Retro*'s explicitly non-learning variant: the search
structure with the learned part removed and a static sink-closeness evaluation in
its place. Rollouts are absent for accounting rather than taste -- the engine
charges one expansion per `select` call, so a policy simulating internally would
spend budget nobody counted, and budget is the comparison axis of every table.

**Selection descends from the root** instead of scoring the frontier. A first
version scored frontier molecules directly and was provably identical to
`GreedyECFP`: backpropagation only ever reaches the expanded node and its
ancestors, all of which are expanded by construction, so no frontier candidate
ever had `visits > 0`. The exploration term was then the same constant for every
candidate, cancelled, and left `Q = closeness` -- greedy under another name, and
it solved nothing on three targets where breadth-first solved two.

The cause is structural, and worth stating in the paper. UCT needs to *revisit*
nodes to accumulate statistics; this engine expands each molecule exactly once and
drops it from the frontier, so there is nothing to revisit. AiZynthFinder and
EG-MCTS escape this because their tree nodes are search *states* -- partial routes
-- and one molecule recurs along many paths. We search best-first over a DAG with a
global frontier, which is a different object. Descending from the root restores
what UCT needs: statistics accumulate along the path, and the path is retraversed
at every decision.

Descent alternates the two node kinds honestly: at an OR-node (molecule) it picks
the most promising reaction, at an AND-node (reaction) it moves to a precursor that
still needs making. It stops at the first molecule the engine would accept --
unexpanded, unsolved, inside the depth limit -- and returns it.
"""

import math
from typing import List, Optional, Set, Tuple

from morganbiopilot.multi_step.graph import SearchGraph
from morganbiopilot.multi_step.heuristics import SinkCloseness


class MCTS:
    """UCT by descent from the root, with a static evaluation instead of rollouts."""

    name = "mcts"

    def __init__(self, closeness: SinkCloseness, exploration: float = 1.0,
                 max_walk: int = 64):
        self.closeness = closeness
        self.c = float(exploration)
        self.max_walk = int(max_walk)
        self.mol_visits: dict = {}
        self.mol_value: dict = {}
        self.rxn_visits: dict = {}
        self.rxn_value: dict = {}
        self._path: List[Tuple[str, int]] = []
        self._pending: Optional[int] = None
        self._h_cache: dict = {}

    # ------------------------------------------------------------------ evaluation
    def _static(self, graph: SearchGraph, node_id: int) -> float:
        """Value of a molecule taken alone: how close it is to the chassis."""
        node = graph.molecules[node_id]
        if node.available or node.solved:
            return 1.0
        if node.smiles not in self._h_cache:
            self._h_cache[node.smiles] = self.closeness.closeness(node.smiles)
        return self._h_cache[node.smiles]

    def _evaluate(self, graph: SearchGraph, node_id: int) -> float:
        """Value of a molecule after expansion, one level down.

        A molecule is an OR-node, so its value is the *best* way to make it; a
        reaction is an AND-node, so its value is its *weakest* precursor. Averaging
        over an AND-node would call a reaction half-solved when one of two
        precursors is unreachable, which is not what "all children required" means.
        """
        node = graph.molecules[node_id]
        if node.available or node.solved:
            return 1.0
        best = self._static(graph, node_id)
        for rxn_id in node.children:
            precursors = graph.reactions[rxn_id].children
            if precursors:
                best = max(best, min(self._static(graph, c) for c in precursors))
        return best

    # -------------------------------------------------------------------- UCT parts
    def _score(self, visits: dict, value: dict, node_id: int,
               parent_visits: int, prior: float) -> float:
        n = visits.get(node_id, 0)
        q = value[node_id] / n if n else prior
        return q + self.c * math.sqrt(math.log(parent_visits + 1) / (1 + n))

    def _best_reaction(self, graph: SearchGraph, mol_id: int) -> Optional[int]:
        parent_visits = self.mol_visits.get(mol_id, 0)
        best, best_score = None, -math.inf
        for rxn_id in graph.molecules[mol_id].children:
            precursors = graph.reactions[rxn_id].children
            if not precursors:
                continue
            prior = min(self._static(graph, c) for c in precursors)
            score = self._score(self.rxn_visits, self.rxn_value, rxn_id,
                                parent_visits, prior)
            if score > best_score:
                best, best_score = rxn_id, score
        return best

    def _best_precursor(self, graph: SearchGraph, rxn_id: int,
                        seen: Set[int]) -> Optional[int]:
        """The precursor of an AND-node still worth working on.

        Solved and available precursors are skipped: descending into a branch that
        is already finished cannot buy anything.
        """
        parent_visits = self.rxn_visits.get(rxn_id, 0)
        best, best_score = None, -math.inf
        for mol_id in graph.reactions[rxn_id].children:
            node = graph.molecules[mol_id]
            if node.solved or node.available or mol_id in seen:
                continue
            score = self._score(self.mol_visits, self.mol_value, mol_id,
                                parent_visits, self._static(graph, mol_id))
            if score > best_score:
                best, best_score = mol_id, score
        return best

    def _descend(self, graph: SearchGraph, allowed: Set[int]) -> Optional[int]:
        """Walk root -> leaf by UCT, recording the path for backpropagation."""
        self._path = []
        node = graph.root
        seen = {node}
        for _ in range(self.max_walk):
            self._path.append(("m", node))
            if node in allowed:
                return node
            molecule = graph.molecules[node]
            if not molecule.expanded or not molecule.children:
                return None
            rxn_id = self._best_reaction(graph, node)
            if rxn_id is None:
                return None
            self._path.append(("r", rxn_id))
            nxt = self._best_precursor(graph, rxn_id, seen)
            if nxt is None:
                return None
            seen.add(nxt)
            node = nxt
        return None

    def _backpropagate(self, reward: float) -> None:
        """Credit the descent path only, not every ancestor.

        The graph is a DAG, so a node has many ancestors while the decision was
        taken along one path. Updating everything above the leaf would credit
        branches the descent never chose.
        """
        for kind, node_id in self._path:
            visits, value = ((self.mol_visits, self.mol_value) if kind == "m"
                             else (self.rxn_visits, self.rxn_value))
            visits[node_id] = visits.get(node_id, 0) + 1
            value[node_id] = value.get(node_id, 0.0) + reward

    # ---------------------------------------------------------------------- policy
    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        # Backpropagation is lazy: the engine expands the chosen node between calls,
        # so its children only exist by the time we are asked again.
        if self._pending is not None and self._pending in graph.molecules:
            self._backpropagate(self._evaluate(graph, self._pending))
            self._pending = None

        allowed = set(frontier)
        node = self._descend(graph, allowed)
        if node is None:
            # The descent hit a dead end or left the eligible set. Credit the path
            # nothing so UCT stops preferring it, and fall back to the frontier's
            # best static value for this turn.
            self._backpropagate(0.0)
            node = max(frontier, key=lambda i: (self._static(graph, i), -i))
            self._path = [("m", node)]

        self._pending = node
        return node
