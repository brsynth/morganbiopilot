"""Is frontier truncation the bottleneck, and does the frontier cluster?

Two measurements that decide whether to rebuild how the agent sees the frontier.
Neither costs an LLM call: the search is driven by a classical policy and the
frontier view is computed alongside it.

    python -m morganbiopilot.paper_results.frontier_diagnostics
    python -m morganbiopilot.paper_results.frontier_diagnostics --targets naringenin --budget 50

**Coverage.** The agent chooses among `top_k` of a frontier that reaches
thousands of nodes, while every classical policy chooses among all of it. That
asymmetry is forced -- the prompt has a context limit -- but its size is
measurable: at each decision, would the move a classical policy makes even have
been available to the agent? If the answer is almost always yes, truncation is
not what limits the agent and there is nothing to fix. If it is usually no, then
the comparison is between observation windows rather than between policies, and
that has to be either fixed or stated plainly.

**Clusterability.** A proposed fix is to cluster the frontier and let the agent
navigate groups instead of a truncated list, which imposes a partition rather
than an order and so avoids handing the agent the greedy baseline's own decision
rule. That only works if the frontier has structure to partition. A frontier made
of near-identical fragments of one target would cluster arbitrarily, and the
hierarchy would be noise dressed as information. Silhouette against the
counted-Tanimoto distance answers it, as does the size of the largest cluster:
one group holding 95% of the nodes is a partition in name only.
"""

import argparse
import sys
from typing import Dict, List

import numpy as np

from morganbiopilot.agents.state import DEFAULT_TOP_K, build_frontier_view
from morganbiopilot.core.chem import mol_ecfp
from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.golden_dataset import load_golden_dataset
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.graph import SearchGraph
from morganbiopilot.multi_step.heuristics import SinkCloseness, counted_tanimoto
from morganbiopilot.multi_step.policy import BreadthFirst, DepthFirst, GreedyECFP
from morganbiopilot.multi_step.search import search
from morganbiopilot.one_step.prefilter import prefilter_from_rules


class Observing:
    """Wraps a classical policy and records what a truncated view would show.

    The wrapped policy drives the search, so the trajectory is its own; the view
    is computed at every step purely to ask whether its choice was visible.
    """

    def __init__(self, inner, top_k: int, closeness=None, rule_ec=None):
        self.inner = inner
        self.top_k = top_k
        self.closeness = closeness
        self.rule_ec = rule_ec
        self.rows: List[dict] = []

    @property
    def name(self) -> str:
        return f"observed({self.inner.name})"

    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        chosen = self.inner.select(graph, frontier)
        view = build_frontier_view(
            graph, frontier, top_k=self.top_k,
            closeness=self.closeness, rule_ec=self.rule_ec,
        )
        shown = {c.node_id for c in view.candidates}
        self.rows.append({
            "n_frontier": len(frontier),
            "n_shown": len(shown),
            "visible": chosen in shown,
            "depth": graph.molecules[chosen].depth,
        })
        return chosen


def coverage(targets, rules, prefilter, rule_ec, closeness, budget, top_k, max_depth):
    """How often a classical policy's move is inside the agent's window."""
    print(f"\n{'=' * 78}\nCOVERAGE -- was the classical policy's pick among the "
          f"{top_k} shown?\n{'=' * 78}")
    print(f"{'policy':12s} {'target':20s} {'decisions':>10s} {'frontier':>10s} "
          f"{'visible':>9s}")

    totals: Dict[str, List[int]] = {}
    for name, factory in (("bfs", BreadthFirst), ("dfs", DepthFirst),
                          ("greedy_ecfp", lambda: GreedyECFP(closeness))):
        for entry in targets:
            observer = Observing(factory(), top_k, closeness, rule_ec)
            search(entry.target, rules, prefilter, observer,
                   budget=budget, max_depth=max_depth, rule_ec=rule_ec)
            rows = observer.rows
            if not rows:
                continue
            seen = sum(r["visible"] for r in rows)
            mean_frontier = sum(r["n_frontier"] for r in rows) / len(rows)
            print(f"{name:12s} {entry.name:20s} {len(rows):10d} "
                  f"{mean_frontier:10.0f} {100 * seen / len(rows):8.0f}%")
            tally = totals.setdefault(name, [0, 0])
            tally[0] += seen
            tally[1] += len(rows)

    print()
    for name, (seen, total) in sorted(totals.items()):
        print(f"  {name:12s} visible in {seen}/{total} decisions "
              f"({100 * seen / total:.0f}%)")
    return totals


def clusterability(targets, rules, prefilter, rule_ec, radius, budget, max_depth, k):
    """Does the frontier have structure a partition could exploit?"""
    print(f"\n{'=' * 78}\nCLUSTERABILITY -- counted-Tanimoto structure of the "
          f"frontier\n{'=' * 78}")
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("  scikit-learn missing; skipping.", file=sys.stderr)
        return

    print(f"{'target':20s} {'nodes':>7s} {'sim mean':>9s} {'sim p95':>8s} "
          f"{'silhouette':>11s} {'largest':>9s}")
    for entry in targets:
        # Drive with dfs purely to grow a realistic frontier, then freeze it.
        graph_holder = {}

        class Capture(DepthFirst):
            def select(self, graph, frontier):
                graph_holder["graph"] = graph
                graph_holder["frontier"] = list(frontier)
                return super().select(graph, frontier)

        search(entry.target, rules, prefilter, Capture(),
               budget=budget, max_depth=max_depth, rule_ec=rule_ec)
        graph, frontier = graph_holder.get("graph"), graph_holder.get("frontier", [])
        if len(frontier) < 3 * k:
            print(f"{entry.name:20s} {len(frontier):7d}   frontier too small to "
                  f"partition into {k}")
            continue

        vectors = np.asarray(
            [mol_ecfp(graph.molecules[i].smiles, radius) for i in frontier],
            dtype=np.float32,
        )
        # Full similarity matrix, then 1 - T as the distance the clustering sees.
        sim = np.stack([counted_tanimoto(v, vectors) for v in vectors])
        off = sim[~np.eye(len(sim), dtype=bool)]
        distance = np.clip(1.0 - sim, 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)

        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average",
        ).fit_predict(distance)
        score = silhouette_score(distance, labels, metric="precomputed")
        largest = np.bincount(labels).max() / len(labels)

        print(f"{entry.name:20s} {len(frontier):7d} {off.mean():9.3f} "
              f"{np.percentile(off, 95):8.3f} {score:11.3f} {100 * largest:8.0f}%")

    print("\n  silhouette near 0 means no structure -- a hierarchy over it would be\n"
          "  arbitrary. 'largest' is the share of nodes in the biggest cluster: a\n"
          "  partition where one group holds almost everything buys nothing.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--targets", default="vanillin,naringenin,piceatannol")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--clusters", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--variant", default="experimental")
    args = parser.parse_args()

    rules = load_rules(radius=args.radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)
    closeness = SinkCloseness(args.radius)
    pathways = {p.name: p for p in load_golden_dataset(args.variant)}

    wanted = [n.strip() for n in args.targets.split(",") if n.strip()]
    missing = [n for n in wanted if n not in pathways]
    if missing:
        raise SystemExit(f"unknown target(s): {missing}")
    targets = [pathways[n] for n in wanted]

    print(f"{len(rules)} rules at r{args.radius} | budget {args.budget} | "
          f"top_k {args.top_k} | {len(targets)} target(s)")

    coverage(targets, rules, prefilter, rule_ec, closeness,
             args.budget, args.top_k, args.max_depth)
    clusterability(targets, rules, prefilter, rule_ec, args.radius,
                   args.budget, args.max_depth, args.clusters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
