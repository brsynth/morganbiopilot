"""Rendering a search state for a model — and the frontier truncation problem.

Why truncation is a methodological decision, not an implementation detail
------------------------------------------------------------------------
A BFS run at r2 reaches ~1,150 frontier molecules in 40 expansions; a greedy run
reaches ~5,200. Even with the enzymatic reality filter on, which roughly halves it
(~560), no prompt holds that. Something has to be cut.

**The criterion used to cut decides what the experiment measures.** Ranking the
frontier by ECFP distance to the sink would hand the agent the greedy best-first
baseline's own decision rule and then ask whether the agent beats it — the answer
would be an artefact. The order used here must therefore be uncorrelated with
every policy under test.

Three orders are provided:

- `stratified` (default) — round-robin across the depths present in the frontier,
  discovery order within each depth. Every depth is represented, and no depth is
  systematically at the top of the list.
- `discovery` — the order molecules entered the graph. **Kept only as a baseline;
  do not use it for experiments.** See below.
- `seeded_random` — a fixed-seed shuffle. Samples depth in proportion, so deep
  nodes appear only as often as they are frequent, and it discards an arbitrary
  slice of the frontier.

Why `discovery` is not neutral — a measured mistake
---------------------------------------------------
Discovery order was the first default here, chosen because it depends only on the
engine and favours no policy. That reasoning was wrong. Insertion order *is*
breadth-first order, so truncating it keeps the shallowest nodes: on vanillin at
budget 8, the agent was shown depths ``{0: 1, 1: 20}`` while ``{1: 17, 2: 6}``
stayed hidden — **it never once saw a node deeper than 1**, though depth-2 nodes
existed from step 3 onward.

The consequence was not subtle. The reference routes are 2-4 steps, so a policy
confined to depth 1 cannot finish one; plain depth-first search solved vanillin in
3 expansions while the agent failed in 8. That read as "DFS beats the agent" and
was an artefact of this function. Worse, the tooled/untooled ablation came out
null — unsurprisingly, since both conditions were choosing among the same
depth-1 candidates whatever the columns said.

The decision the project note delegates to the agent is *where to spend the next
expansion*, which includes choosing between going deeper and going wider. A
truncation that hides every deep node removes half of that decision.

`sink_distance` is still deliberately **not** offered: ordering by distance to the
sink would hand the agent the greedy baseline's own decision rule. Avoiding that
bias is necessary, and — as the above shows — not sufficient.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from morganbiopilot.multi_step.graph import SearchGraph

FRONTIER_ORDERS = ("portfolio", "stratified", "discovery", "seeded_random")
# "portfolio" needs a ranker and a prefilter; without them `build_frontier_view` falls
# back to "stratified", so the old behaviour is what a caller that passes nothing gets.
DEFAULT_FRONTIER_ORDER = "portfolio"

# How many frontier molecules the agent is shown per decision.
#
# No published system does this, and the reason is worth stating. RetroAgent's
# k = 20 is *not* comparable, despite an earlier comment here claiming it was: it
# truncates the single-step model's 50 top-scoring predictions for one already
# chosen molecule down to 20 candidate reactions, and exposes the frontier itself
# through an inspection tool rather than in the prompt. Retro-R1 likewise ranks
# reactions by model probability (k = 10) and lets the agent pick from a
# "molecular state" small enough not to need truncating. AOT does not use the LLM
# for node choice at all -- `SelectTarget` takes the least-explored molecule.
#
# Every one of those presupposes a learned single-step scorer to rank by. We have
# none: rules carry an NLL that is meaningless for this task and was dropped. And
# because we apply *every* applicable template instead of a model's top-k, our
# frontier reaches 1000-2800 molecules where theirs holds tens. The truncation
# problem is therefore a consequence of the rule-based setting, not a design whim,
# and the literature's answers do not transfer.
#
# Absent a ranking signal, selection is a quota (`_stratify`) rather than a top-k.
# 20 is a defensible size -- the same order as everyone's reaction lists, and AOT
# measured solve rate plateauing sharply against context size -- but it is not a
# measured optimum. Sweep --top-k over {10, 20, 40} to find that.
DEFAULT_TOP_K = 20


def _stratify(graph: SearchGraph, ids: Sequence[int], top_k: int) -> List[int]:
    """Round-robin across depths, discovery order within each depth.

    Guarantees every depth present in the frontier is represented as long as
    slots remain, and interleaves them so no depth sits systematically at the top
    of the list — position in a prompt is itself a bias.
    """
    buckets: Dict[int, List[int]] = {}
    for node_id in sorted(ids):
        buckets.setdefault(graph.molecules[node_id].depth, []).append(node_id)

    chosen: List[int] = []
    depths = sorted(buckets)
    cursor = {d: 0 for d in depths}
    while len(chosen) < top_k:
        progressed = False
        for depth in depths:
            bucket = buckets[depth]
            i = cursor[depth]
            if i < len(bucket):
                chosen.append(bucket[i])
                cursor[depth] = i + 1
                progressed = True
                if len(chosen) == top_k:
                    break
        if not progressed:
            break      # frontier exhausted before top_k
    return chosen


def _interleave(orders, ids):
    """Round-robin across several orderings, first occurrence wins, then the rest."""
    out, seen = [], set()
    for rank in range(max((len(o) for o in orders), default=0)):
        for order in orders:
            if rank < len(order) and order[rank] not in seen:
                seen.add(order[rank])
                out.append(order[rank])
    out.extend(i for i in ids if i not in seen)
    return out


# Per-molecule scores for the portfolio view, keyed by SMILES. A frontier molecule is
# re-examined at every subsequent step, so without this the view costs
# O(steps x frontier) prefilter passes instead of O(distinct molecules): a 30-route
# replay probe ran past ten minutes where the whole 12,251-route corpus took eighty.
# `GreedySimilarity` caches the same quantity for the same reason.
_PORTFOLIO_CACHE: Dict[str, tuple] = {}


def _portfolio_scores(smi, ranker, prefilter, rule_ec):
    """(similarity, precedent, heavy_atoms) for one molecule, computed once."""
    hit = _PORTFOLIO_CACHE.get(smi)
    if hit is not None:
        return hit

    import numpy as np
    from rdkit import Chem

    from morganbiopilot.core.chem import mol_ecfp

    sim, prec = 0.0, 0
    try:
        ecfp = np.asarray(mol_ecfp(smi, prefilter.radius), dtype=np.int32)
        _vecs, idxs = prefilter.one_step(ecfp)
        ordered = ranker.order(smi, idxs)
        if len(ordered):
            sim = ranker.similarity(smi, int(ordered[0]))
        if rule_ec is not None and len(idxs):
            prec = int(max((int(rule_ec.n_reactions[int(j)]) for j in idxs), default=0))
    except Exception:                                            # noqa: BLE001
        # One unparseable molecule must not sink the view; it scores last on these two
        # members and still reaches the agent through `_stratify`.
        pass

    m = Chem.MolFromSmiles(smi)
    out = (sim, prec, m.GetNumHeavyAtoms() if m else 10 ** 6)
    _PORTFOLIO_CACHE[smi] = out
    return out


def _portfolio(graph, ids, top_k, ranker, prefilter, rule_ec=None):
    """Depth diversity, chemical resemblance, precedent and size, round-robin.

    Each member contributes its few confident picks instead of imposing its ranking.
    Measured on 3,821 states whose frontier exceeds twenty, replaying attested routes:
    the on-route molecule survives a top-20 view 84% of the time here against 80% for
    `_stratify` alone -- 4.6 sigma, and the same ordering leads at every k from 10 to
    160. No single scorer beat `_stratify`; similarity alone reached 79%.

    Why a portfolio rather than the best scorer. `_stratify` wins among single
    orderings precisely because it refuses to concentrate: it guarantees a
    representative of every depth. A pure sort by similarity fills the twenty slots
    from whichever depth happens to score high and drops the answer when it sits
    elsewhere. Molecular size is the clearest case -- worst of the four on its own at
    66%, yet adding it to the other three buys two points, because the picks it is
    confident about are ones they miss.
    """
    if prefilter.radius is None:
        # Fingerprinting a query at the wrong radius yields a vector that matches
        # nothing, and every similarity would come back 0.0 -- a portfolio silently
        # degraded to two of its four members. Refuse instead.
        raise ValueError("the prefilter carries no radius; rebuild it with "
                         "prefilter_from_rules so the portfolio can fingerprint")

    sim, prec, heavy = {}, {}, {}
    for i in ids:
        smi = graph.molecules[i].smiles
        sim[i], prec[i], heavy[i] = _portfolio_scores(smi, ranker, prefilter, rule_ec)

    def by(key, reverse=True):
        return sorted(ids, key=lambda n: (-key(n) if reverse else key(n), n))

    members = [_stratify(graph, ids, len(ids)),
               by(lambda n: sim[n]),
               by(lambda n: prec[n]),
               by(lambda n: heavy[n], reverse=False)]
    return _interleave(members, ids)[:top_k]


@dataclass(frozen=True)
class Candidate:
    """One frontier molecule as the agent sees it."""

    node_id: int
    smiles: str
    depth: int
    ec_classes: tuple = ()          # level-1 EC classes reachable from this molecule
    closeness: Optional[float] = None   # best Tanimoto to the sink; None when untooled
    nearest_bb: str = ""            # name of that chassis metabolite
    nearest_bb_smiles: str = ""     # and its structure
    enzymatic_per_step: Optional[float] = None   # geometric mean along the best route
    enzymatic_route: Optional[float] = None      # the product itself
    enzymatic_edges: int = 0                     # reactions on that route

    def render(self, index: int) -> str:
        parts = [f"[{index}] {self.smiles}", f"depth={self.depth}"]
        if self.ec_classes:
            parts.append(f"EC={','.join(self.ec_classes)}")
        if self.enzymatic_per_step is not None:
            # Per-step first, product second. The product decays with depth whatever
            # the chemistry, so leading with it would make the column a depth proxy —
            # and a monotone depth proxy is the shape of signal that turned the
            # closest_chassis column into a greedy collapse.
            parts.append(
                f"enzymatic={self.enzymatic_per_step:.3f}/step "
                f"(route {self.enzymatic_route:.3f} over {self.enzymatic_edges})")
        if self.closeness is not None:
            # Name *and* structure, not just a number. The name grounds the
            # metabolite in biology; the SMILES lets the model compare skeletons
            # when the name is one it does not know — and a third of the sink is
            # still a bare MetaNetX identifier even after the PubChem lookup.
            near = ""
            if self.nearest_bb:
                near = f" to {self.nearest_bb}"
                if self.nearest_bb_smiles:
                    near += f" [{self.nearest_bb_smiles}]"
            parts.append(f"closest_chassis={self.closeness:.3f}{near}")
        return " | ".join(parts)


@dataclass
class FrontierView:
    """What the agent is shown, plus what was hidden from it.

    `n_total` vs `len(candidates)` is not bookkeeping — it is the number the paper
    needs to state how much of the search space each decision actually ranged over.
    """

    candidates: List[Candidate]
    n_total: int
    order: str
    seed: Optional[int] = None

    @property
    def truncated(self) -> bool:
        return len(self.candidates) < self.n_total

    def render(self) -> str:
        lines = [c.render(i) for i, c in enumerate(self.candidates)]
        if self.truncated:
            lines.append(
                f"({len(self.candidates)} of {self.n_total} frontier molecules shown, "
                f"ordered by {self.order})"
            )
        return "\n".join(lines)


def build_frontier_view(
    graph: SearchGraph,
    frontier: Sequence[int],
    top_k: int = DEFAULT_TOP_K,
    order: str = DEFAULT_FRONTIER_ORDER,
    seed: int = 0,
    closeness=None,
    rule_ec=None,
    plausibility=None,
    shuffle: bool = True,
    ranker=None,
    prefilter=None,
) -> FrontierView:
    """Select and render the frontier molecules the agent will choose among.

    `closeness` and `rule_ec` are the tool surface: pass them for the tooled
    condition, leave them None for the untooled one (section 8's grounding
    ablation). They enrich each candidate — they never reorder the frontier, which
    is what keeps the truncation policy-neutral.
    """
    if order not in FRONTIER_ORDERS:
        raise ValueError(f"order must be one of {FRONTIER_ORDERS}, got {order!r}")

    ids = list(frontier)
    if order == "portfolio" and ranker is not None and prefilter is not None:
        chosen = _portfolio(graph, ids, top_k, ranker, prefilter, rule_ec)
    elif order == "portfolio" or order == "stratified":
        # No ranker: the portfolio has nothing to interleave, so this is `_stratify`.
        # Silent rather than an error, because every caller that never ordered the
        # frontier keeps working -- and `_stratify` is the ordering the portfolio was
        # measured against, so the fallback is the documented baseline, not a surprise.
        chosen = _stratify(graph, ids, top_k)
    elif order == "discovery":
        ids.sort()          # node ids are assigned in insertion order
        chosen = ids[:top_k]
    else:
        random.Random(seed).shuffle(ids)
        chosen = ids[:top_k]

    if shuffle:
        # Selection and presentation are different decisions. Which candidates the
        # agent sees must be principled (stratified by depth); the order they are
        # listed in must not be, because position in a prompt is itself a signal.
        # This part *is* what the literature does. RetroAgent shuffles its
        # truncated reaction candidates before showing them, and Retro-R1 shuffles
        # the context on revisits and ablates the choice ("wos", without shuffle),
        # so position bias is a recognised concern rather than our invention.
        # Shuffling also makes the presentation seed a legitimate source of
        # run-to-run variance, which is how to get error bars from a model whose
        # temperature cannot be pinned.
        random.Random(seed + len(chosen)).shuffle(chosen)
    # One aggregation for the whole view: the traversal is global, so doing it per
    # candidate would repeat the same Dijkstra `top_k` times.
    routes = {} if plausibility is None else plausibility.node_scores(graph, chosen)

    candidates = []
    for node_id in chosen:
        node = graph.molecules[node_id]
        best_label, best_smiles, best_sim = "", "", None
        if closeness is not None:
            near = closeness.nearest(node.smiles, k=1)
            if near:
                best_label, best_smiles, best_sim = near[0]
        route = routes.get(node_id)
        candidates.append(Candidate(
            node_id=node_id,
            smiles=node.smiles,
            depth=node.depth,
            ec_classes=_reachable_ec_classes(graph, node_id, rule_ec),
            closeness=best_sim,
            nearest_bb=best_label,
            nearest_bb_smiles=best_smiles,
            enzymatic_per_step=None if route is None else route.per_step,
            enzymatic_route=None if route is None else route.cumulative,
            enzymatic_edges=0 if route is None else route.edges,
        ))

    return FrontierView(
        candidates=candidates, n_total=len(frontier), order=order,
        seed=seed if order == "seeded_random" else None,
    )


def _reachable_ec_classes(graph: SearchGraph, node_id: int, rule_ec) -> tuple:
    """Level-1 EC classes of the reactions that produced this molecule.

    Level 1 because aggregation over the reactions collapsed into a rule is only
    ~3% ambiguous there against ~9% at level 3 (see `core.ec`).
    """
    if rule_ec is None:
        return ()

    classes = set()
    for rxn_id in graph.molecules[node_id].parents:
        idx = graph.reactions[rxn_id].rule_idx
        for ec in rule_ec.ec[idx]:
            head = ec.split(".")[0]
            if head and head != "-":
                classes.add(head)
    return tuple(sorted(classes))
