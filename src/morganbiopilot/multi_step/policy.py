"""Search policies: the one decision that separates every system in the paper.

A policy answers a single question — *which frontier molecule should be expanded
next?* — and that is its whole interface. Everything else (expansion,
applicability, graph maintenance, status propagation, sink test) is deterministic
and shared.

This is deliberate. The project's claim is that all systems run on the same
engine and differ only in search strategy; making `select` the only extension
point turns that claim into a structural property of the code rather than a
promise in the paper. The LLM agent is one more implementation of this protocol.

The RetroMorgan draft already had this seam implicitly: its search loop was
identical across `astar` / `best_first` / `retro_star` / `dfs`, differing only in
the key used to pop from the frontier. Here that key becomes an object.
"""

import math
from typing import List, Optional, Protocol

from morganbiopilot.multi_step.graph import SearchGraph
from morganbiopilot.multi_step.heuristics import SinkCloseness


class Policy(Protocol):
    """Chooses the next molecule to expand."""

    name: str

    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        """Return a molecule-node id taken from `frontier` (never empty)."""
        ...


class BreadthFirst:
    """Shallowest first. Naive traversal reference of the project note."""

    name = "bfs"

    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        return min(frontier, key=lambda i: (graph.molecules[i].depth, i))


class DepthFirst:
    """Deepest first."""

    name = "dfs"

    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        return max(frontier, key=lambda i: (graph.molecules[i].depth, -i))


class GreedyECFP:
    """Closest to the sink first, by counted Tanimoto — no LLM.

    The "greedy best-first ECFP" baseline of the project note, and the closest
    thing to RetroPath2.0's enumeration logic guided by chemical similarity.
    """

    name = "greedy_ecfp"

    def __init__(self, closeness: SinkCloseness, depth_penalty: float = 0.0):
        self.closeness = closeness
        self.depth_penalty = float(depth_penalty)
        self._cache = {}

    def _h(self, graph: SearchGraph, node_id: int) -> float:
        node = graph.molecules[node_id]
        if node.smiles not in self._cache:
            self._cache[node.smiles] = self.closeness.h(node.smiles)
        return self._cache[node.smiles] + self.depth_penalty * node.depth

    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        return min(frontier, key=lambda i: (self._h(graph, i), i))
