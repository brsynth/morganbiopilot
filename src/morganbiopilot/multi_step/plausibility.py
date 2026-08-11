"""Route plausibility: the enzymatic score of every edge between the target and a node.

`data_processing.enzymatic_model` scores one reaction. This turns that into a property of
a *frontier node*: the product of the edge scores along the best path from the target
down to it, which is the plausibility of the partial route already built. In log space it
is a sum, which is what Retro* uses as its route cost.

Two quantities are reported per node, and the distinction matters more than it looks
------------------------------------------------------------------------------------
`cumulative` is the product itself. Every factor is at most 1, so it decays exponentially
with depth: a node at depth 2 and a node at depth 5 are not comparable on it, and shown
to a policy on its own it would mostly say "this one is shallower". That is a real
hazard here rather than a hypothetical -- the measured failure mode of the
`closest_chassis` column is that a monotone heuristic turns an LLM policy into a greedy
searcher which then plateaus, and a depth proxy would do the same thing while looking
like chemistry.

`per_step` is the geometric mean, `cumulative ** (1/edges)`: plausibility per enzymatic
step, depth-neutral. It is the one to rank on, and it is rendered first.

The path is chosen by max-product, not by the geometric mean. Optimising the product is
a shortest-path problem with weights `-log s >= 0`, so Dijkstra solves it exactly;
optimising the mean is not, and the extra fidelity is not worth a harder algorithm on a
score whose AUC is 0.660. So `per_step` is the mean *of the max-product path*, and the
edge count that normalises it comes from that same path.

Why Dijkstra rather than a topological sweep
--------------------------------------------
`SearchGraph.add_molecule` deduplicates by canonical SMILES, so a molecule reached by
several routes is one node and the graph is not a tree. It is not reliably a DAG either:
if the rule set contains A -> B and B -> A, expanding both closes a cycle through the
shared node. Depth is therefore not a safe traversal order. Dijkstra needs no ordering
and, since every weight is non-negative, returns the true optimum regardless of cycles.

What the number is not
----------------------
The scores multiplied here are uncalibrated `g(x)` from a positive-unlabelled model
whose validated question is "does this transformation look biochemical rather than
organic?" -- measured at AUC 0.660, which means heavily overlapping classes and no
verdict from any single edge. They are not independent either: consecutive edges of a
route share substructures. So the product is a **monotone cumulative score, not the
probability of a route**, and the framing that would license the probabilistic reading
(substrate scope, `data_processing.reaction_pu`) returned a clean null at 0.534.

Whether aggregation turns a weak per-edge signal into a useful per-node one is exactly
the open question. Ship this as an ablation arm, and give it to the baselines too: the
log-sum is a value function for MCTS, and if MCTS exploits it better than the LLM does,
that is a result rather than a failure.
"""

import heapq
import math
import threading
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from morganbiopilot.multi_step.graph import SearchGraph

# A score of exactly 0 would make the log infinite and erase every route through that
# edge, which is far stronger than anything a 0.660-AUC model has earned the right to
# say. Clamping keeps a bad edge very expensive but not disqualifying.
EPSILON = 1e-4


class NodeScore(NamedTuple):
    """Plausibility of the best path from the target down to one molecule."""

    cumulative: float               # product of the edge scores along that path
    per_step: Optional[float]       # its geometric mean; None at the target itself
    edges: int                      # path length in reactions


class RoutePlausibility:
    """Edge scores, cached, and the per-node aggregation built from them.

    One instance per search. Reaction nodes are immutable once added, so an edge is
    scored once and reused; only the aggregation is redone as the graph grows.
    """

    def __init__(self, model_path=None, fallback: float = 0.5):
        """`fallback` is used for edges the model cannot featurise.

        0.5 rather than 0: an unscorable edge is one whose fingerprint failed, which
        says nothing about its chemistry, and dropping it would silently prune routes.
        The count is exposed as `n_unscored` so a run can report how often it happened.
        """
        self._path = model_path
        self._fallback = float(fallback)
        self._edge: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self.n_unscored = 0
        # One instance is shared across concurrent searches (see `compare_policies`).
        # Dict reads and writes are individually safe under the GIL, so the lock is not
        # about corruption -- it is so two workers hitting the same new reaction do not
        # both pay for the model call, and so `n_unscored` stays an accurate count.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------- edges
    @staticmethod
    def _key(graph: SearchGraph, rxn_id: int) -> Tuple[str, Tuple[str, ...]]:
        """Cache key: what the reaction *is*, not where it sits.

        Keying on `rxn_id` would be the obvious choice and is a trap: ids are assigned
        per graph, so one instance shared across targets -- which is how the experiment
        harness builds tool surfaces -- would read target A's score for target B's
        reaction 5. Keying on content makes that impossible, and turns reuse across
        targets into a warm cache instead of a corruption.
        """
        rxn = graph.reactions[rxn_id]
        return (graph.molecules[rxn.parent].smiles, tuple(rxn.neighbour.molecules))

    def edge_scores(self, graph: SearchGraph) -> Dict[int, float]:
        """Scores for this graph's reactions, by node id, scoring what is new in a batch.

        Batched deliberately: per-edge calls would pay the fingerprinting overhead one
        molecule at a time, where a batch runs at ~2,200 edges/s.
        """
        from morganbiopilot.data_processing.enzymatic_model import score_reactions

        keys = {r: self._key(graph, r) for r in graph.reactions}
        with self._lock:
            missing = sorted({k for k in keys.values() if k not in self._edge})
            if missing:
                scored = score_reactions(
                    [(sub, list(prods)) for sub, prods in missing], path=self._path)
                for key, value in zip(missing, scored):
                    if value != value:                  # NaN: unfeaturisable
                        self.n_unscored += 1
                        value = self._fallback
                    self._edge[key] = float(value)
            return {r: self._edge[k] for r, k in keys.items()}

    # ------------------------------------------------------------------- nodes
    def node_scores(self, graph: SearchGraph,
                    only: Optional[Sequence[int]] = None) -> Dict[int, NodeScore]:
        """Best-path plausibility for every molecule reachable from the target.

        `only` restricts the returned dict, not the traversal -- the optimum for a
        frontier node can run through molecules that are not themselves on the frontier.
        """
        edge = self.edge_scores(graph)

        # Dijkstra on -log s. `dist` is the accumulated cost, minimised; `edges` counts
        # the reactions on the path that achieved it.
        dist: Dict[int, float] = {graph.root: 0.0}
        edges: Dict[int, int] = {graph.root: 0}
        heap: List[Tuple[float, int, int]] = [(0.0, 0, graph.root)]
        done = set()

        while heap:
            cost, n_edges, mol_id = heapq.heappop(heap)
            if mol_id in done:
                continue
            done.add(mol_id)

            for rxn_id in graph.molecules[mol_id].children:
                weight = -math.log(max(edge.get(rxn_id, self._fallback), EPSILON))
                for child_id in graph.reactions[rxn_id].children:
                    candidate = cost + weight
                    # Strict improvement only; ties keep the shorter path, which makes
                    # the geometric mean of an equally plausible route the larger one.
                    better = (candidate < dist.get(child_id, math.inf)
                              or (candidate == dist.get(child_id)
                                  and n_edges + 1 < edges.get(child_id, math.inf)))
                    if better:
                        dist[child_id] = candidate
                        edges[child_id] = n_edges + 1
                        heapq.heappush(heap, (candidate, n_edges + 1, child_id))

        wanted = dist.keys() if only is None else [i for i in only if i in dist]
        out: Dict[int, NodeScore] = {}
        for mol_id in wanted:
            cumulative = math.exp(-dist[mol_id])
            n = edges[mol_id]
            out[mol_id] = NodeScore(
                cumulative=cumulative,
                per_step=(cumulative ** (1.0 / n)) if n else None,
                edges=n,
            )
        return out
