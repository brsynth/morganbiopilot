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
import random
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


class RandomPolicy:
    """Uniform choice. The other naive reference, and the floor any policy must beat."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        return self.rng.choice(sorted(frontier))


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




class GreedySimilarity:
    """Expand the molecule with the best-precedented disconnection available.

    A second monotone heuristic, deliberately unlike the first. `GreedyECFP` scores a
    molecule by how close it is to the chassis, which is a proxy for depth; this scores
    it by the highest Tanimoto between the molecule and the native substrate of any rule
    that applies to it — "some enzyme was recorded acting on something like this". It is
    local and chemical, with no relation to distance from the sink.

    It exists to test a claim rather than to win. Following a monotone scalar greedily
    has produced the same flat budget curve three times with sink closeness — 25/25/25
    on the curated set at r2, 35/35/40 at r1, 52/52/52 on 141 LASER targets. If an
    unrelated heuristic plateaus the same way, that is a regularity; if it does not, the
    claim was about closeness in particular and the paper must say so.

    Cost: one prefilter pass per frontier node, which is the cheap half of an expansion
    (RunReactants is the expensive half and is not run here). Similarities are cached on
    the ranker, which already holds fingerprints for every rule substrate it has seen.
    """

    name = "greedy_similarity"

    def __init__(self, rules, prefilter, ranker):
        self.rules = rules
        self.prefilter = prefilter
        self.ranker = ranker
        self._cache = {}

    def _h(self, graph: SearchGraph, node_id: int) -> float:
        smiles = graph.molecules[node_id].smiles
        if smiles not in self._cache:
            import numpy as np

            from morganbiopilot.core.chem import mol_ecfp

            try:
                ecfp = np.asarray(mol_ecfp(smiles, self.rules.radius), dtype=np.int32)
                _vecs, rule_idxs = self.prefilter.one_step(ecfp)
            except Exception:                                    # noqa: BLE001
                self._cache[smiles] = 0.0
                return 0.0
            # `order` sorts most-similar first, so the head carries the maximum and we
            # never need the rest.
            best = self.ranker.order(smiles, rule_idxs)
            self._cache[smiles] = (
                self.ranker.similarity(smiles, int(best[0])) if len(best) else 0.0)
        return self._cache[smiles]

    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        # Higher is better here, unlike `GreedyECFP` where the heuristic is a distance.
        return max(frontier, key=lambda i: (self._h(graph, i), -i))
