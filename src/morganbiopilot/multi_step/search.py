"""The search loop. One loop, shared by every system in the paper.

The loop itself contains no strategy: it repeatedly asks the policy which frontier
molecule to expand, expands it with the deterministic `one_step` primitive, and
propagates solved status. Swapping `policy` is the *only* difference between the
BFS reference, the greedy ECFP baseline, MCTS, and the LLM agent.

Budget is counted in **expansions**, not wall-clock or tokens: it is the axis
along which the project note compares every system, and it is the only budget that
means the same thing for a 5 ms greedy step and a 3 s LLM call.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from morganbiopilot.core.chem import sanitize
from morganbiopilot.multi_step.graph import SearchGraph
from morganbiopilot.multi_step.policy import Policy
from morganbiopilot.one_step.expand import expand
from morganbiopilot.one_step.prefilter import ReactionCenterPrefilter


@dataclass
class SearchResult:
    """Outcome of one search. Everything the paper's tables need, per run."""

    target: str
    policy: str
    solved: bool
    n_expansions: int
    budget: int
    graph: SearchGraph
    elapsed_s: float
    stopped_because: str = ""            # "solved" | "budget" | "frontier_empty"
    decisions: List[dict] = field(default_factory=list)

    @property
    def n_molecules(self) -> int:
        return len(self.graph.molecules)

    @property
    def n_reactions(self) -> int:
        return len(self.graph.reactions)

    def pathways(self) -> List[List[int]]:
        return self.graph.pathways()

    @property
    def shortest_pathway_length(self) -> Optional[int]:
        routes = self.pathways()
        return min((len(r) for r in routes), default=None)


def search(
    target: str,
    rules,
    prefilter: ReactionCenterPrefilter,
    policy: Policy,
    budget: int = 100,
    max_depth: int = 0,
    max_rules_per_expansion: Optional[int] = None,
    stop_on_first_pathway: bool = True,
    rule_ec=None,
    require_ec: bool = False,
    max_seconds: Optional[float] = None,
    on_decision: Optional[Callable[[dict], None]] = None,
    ranker=None,
    top_n: Optional[int] = None,
) -> SearchResult:
    """Search for routes from `target` down to the chassis sink.

    `budget` is the number of node expansions — the comparison axis. `max_depth`
    of 0 means unlimited. Set `stop_on_first_pathway=False` to keep spending the
    budget after the first route, e.g. for top-k recovery.

    `max_seconds` bounds a single run in wall-clock and reports
    `stopped_because == "time"`. It is a defensive bound, not a tuned one: with
    `stop_on_first_pathway=False`, search cost across the 20 golden targets at budget
    50 measured 6-48 s (mean 19 s), yet one probe at budget 60 on 14-Butanediol had
    not returned after ten minutes and the cause was never established. A job that
    cannot be bounded cannot be scheduled, so the option exists — but leave it None,
    the default, for every measurement on the expansion axis, where a time cut-off
    would silently make the budget mean something different per target.

    `on_decision` is called with each decision row as it is produced. The rows already
    accumulate in `result.decisions`, but only a caller that waits for the return ever
    sees them; showing a search as it runs needs them one at a time. It runs on the
    search thread, so it must be cheap and must not raise — anything slower than the
    expansion itself would change what the timing columns measure.

    `ranker` and `top_n` cap each expansion (`one_step.ranking`). They are a property of
    the **environment**, not of the policy: set them once and every arm explores the
    same graph, which is the only way the comparison between arms means anything. Both
    or neither. Measured at r1 with `native_similarity` and 20: 8 of the 20 attested
    routes stay reproducible against 6 at r2 uncapped, and the median expansion drops
    from 96 candidates to 20 — the branching that otherwise leaves UCT stuck at depth 1
    and shows a language-model policy 0.13% of its frontier.
    """
    target = sanitize(target)
    graph = SearchGraph(target)
    result = SearchResult(
        target=target, policy=getattr(policy, "name", type(policy).__name__),
        solved=graph.solved, n_expansions=0, budget=budget, graph=graph, elapsed_s=0.0,
    )

    if graph.solved:
        result.stopped_because = "solved"
        return result

    t0 = time.perf_counter()

    while result.n_expansions < budget:
        # Checked before the expansion, not after: `expand` is the expensive call and
        # overshooting the limit by one of them is what the limit is there to prevent.
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            result.stopped_because = "time"
            break
        frontier = graph.frontier()
        if max_depth > 0:
            frontier = [i for i in frontier if graph.molecules[i].depth < max_depth]
        if not frontier:
            result.stopped_because = "frontier_empty"
            break

        node_id = policy.select(graph, frontier)
        node = graph.molecules[node_id]

        report = expand(
            node.smiles, rules, prefilter,
            max_rules=max_rules_per_expansion,
            rule_ec=rule_ec, require_ec=require_ec,
            ranker=ranker, top_n=top_n,
        )
        node.expanded = True
        result.n_expansions += 1

        for neighbour in report.neighbours:
            graph.add_neighbour(node_id, neighbour)

        # One row per decision: which node the policy picked, out of what, and what
        # it bought. This is what makes an LLM run reconstructible without re-running.
        result.decisions.append({
            "step": result.n_expansions,
            "node_id": node_id,
            "smiles": node.smiles,
            "depth": node.depth,
            "frontier_size": len(frontier),
            "n_prefiltered": report.n_prefiltered,
            "n_graph_valid": report.n_graph_valid,
            "n_neighbours": len(report.neighbours),
            "solved_after": graph.solved,
        })
        if on_decision is not None:
            on_decision(result.decisions[-1])

        if graph.solved and stop_on_first_pathway:
            result.stopped_because = "solved"
            break
    else:
        result.stopped_because = "budget"

    result.solved = graph.solved
    result.elapsed_s = time.perf_counter() - t0
    return result
