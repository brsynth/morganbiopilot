"""Run one search from a browser, and watch it happen.

Everything in this project is a batch job that answers a question about sixty targets.
This answers a different need: point at one molecule, pick a policy and a radius, and
see what the search actually does, expansion by expansion, then read the routes it
found. It is the tool that would have saved most of the log-grepping this project has
needed.

Standard library only -- `http.server` plus server-sent events. No Flask, no Streamlit,
nothing to install into an environment that already took work to build, and it runs
anywhere the engine runs, including on a compute node through an SSH tunnel.

The design in one paragraph
---------------------------
A POST starts the search on its own thread and returns a run id. The search pushes one
event per expansion into a queue; a GET on that id drains the queue as an SSE stream, so
the page fills in while the search is still running. Rule sets and the sink-closeness
matrix are cached per radius, because loading 82,000 rules takes seconds and nobody
wants to pay that on every click.

What it is not
--------------
Not a benchmark. One search, one target, no seeds, no repeats -- the numbers it shows
are a single draw and must not be quoted. `paper_results.compare_policies` is the
measuring instrument; this is the microscope.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
POLICIES = ("bfs", "dfs", "random", "greedy", "mcts", "llm")


# --------------------------------------------------------------------------- caching

class Engine:
    """Rule sets, EC annotations, prefilters and closeness, cached per radius.

    Loading a radius costs seconds and hundreds of megabytes, and a user comparing r1
    against r2 will switch back and forth. Cached under a lock because two browser tabs
    can start searches at the same moment and the first call is the expensive one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rules: Dict[int, tuple] = {}
        self._closeness: Dict[int, object] = {}
        self._rankers: Dict[int, object] = {}

    def rules(self, radius: int):
        with self._lock:
            if radius not in self._rules:
                from morganbiopilot.core.ec import annotate_rules
                from morganbiopilot.core.rules import load_rules
                from morganbiopilot.one_step.prefilter import prefilter_from_rules

                rules = load_rules(radius)
                self._rules[radius] = (rules, annotate_rules(rules),
                                       prefilter_from_rules(rules))
            return self._rules[radius]

    def closeness(self, radius: int):
        with self._lock:
            if radius not in self._closeness:
                from morganbiopilot.multi_step.heuristics import SinkCloseness
                self._closeness[radius] = SinkCloseness(radius)
            return self._closeness[radius]

    def ranker(self, radius: int):
        """Cached per radius: it holds a fingerprint for every rule substrate it sees,
        and rebuilding it on each search would throw that cache away every click.

        The rule set is fetched *before* taking the lock, deliberately. `self._lock` is
        a plain `Lock`, so calling `self.rules()` from inside it deadlocks -- and it did:
        the first ranked search hung while holding the lock and every later request then
        blocked forever on "loading the rule set".
        """
        rules, _ec, _pf = self.rules(radius)
        with self._lock:
            if radius not in self._rankers:
                from morganbiopilot.one_step.ranking import make_ranker
                self._rankers[radius] = make_ranker("native_similarity", rules)
            return self._rankers[radius]


ENGINE = Engine()


# --------------------------------------------------------------------------- runs

@dataclass
class Run:
    """One search in flight, plus the events it has produced so far."""

    run_id: str
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    done: bool = False
    # Kept so a page reloaded mid-run still gets the whole story rather than the tail.
    history: List[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: dict) -> None:
        with self.lock:
            self.history.append(event)
        self.events.put(event)


RUNS: Dict[str, Run] = {}
RUNS_LOCK = threading.Lock()


def build_policy(name: str, radius: int, seed: int, model: str):
    """One policy instance. Mirrors `paper_results.compare_policies.build_policies`."""
    from morganbiopilot.multi_step.mcts import MCTS
    from morganbiopilot.multi_step.policy import (BreadthFirst, DepthFirst, GreedyECFP,
                                                  GreedySimilarity, RandomPolicy)

    if name == "greedy_similarity":
        rules, _ec, prefilter = ENGINE.rules(radius)
        return GreedySimilarity(rules, prefilter, ENGINE.ranker(radius))
    if name == "bfs":
        return BreadthFirst()
    if name == "dfs":
        return DepthFirst()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "greedy":
        return GreedyECFP(ENGINE.closeness(radius))
    if name == "mcts":
        return MCTS(ENGINE.closeness(radius))
    if name == "llm":
        # Imported here only: the agent package pulls in an HTTP client and a backend
        # that the classical policies have no reason to load.
        from morganbiopilot.agents.backends import make_backend
        from morganbiopilot.agents.policy import LLMPolicy
        from morganbiopilot.agents.tools import untooled

        return LLMPolicy(tools=untooled(), backend=make_backend(model), seed=seed)
    raise ValueError(f"unknown policy {name!r}")


def run_search(run: Run, params: dict) -> None:
    """The search thread. Every exit path must emit a terminal event."""
    from morganbiopilot.core.chem import sanitize
    from morganbiopilot.multi_step.routes import extract_routes
    from morganbiopilot.multi_step.search import search
    from morganbiopilot.one_step.expand import expand

    try:
        radius = int(params.get("radius", 2))
        policy_name = params.get("policy", "bfs")
        budget = int(params.get("budget", 50))
        max_depth = int(params.get("max_depth", 7))
        require_ec = bool(params.get("require_ec"))
        exhaustive = bool(params.get("exhaustive"))
        seed = int(params.get("seed", 0))
        # Ranked, capped expansion. Off by default so the page can show what the
        # unranked engine does, which is the comparison the paper rests on: at r1 an
        # uncapped expansion of violacein yields 96 precursor sets and the frontier
        # passes 15,000 nodes within 100 expansions.
        top_n = int(params.get("top_n") or 0) or None
        use_rank = bool(params.get("ranked")) and top_n
        target = sanitize(str(params.get("smiles", "")).strip())
        if not target:
            run.emit({"type": "error", "text": "that SMILES could not be parsed"})
            return

        run.emit({"type": "status", "text": f"loading the r{radius} rule set..."})
        rules, rule_ec, prefilter = ENGINE.rules(radius)
        ranker = ENGINE.ranker(radius) if use_rank else None
        if ranker is not None:
            run.emit({"type": "status",
                      "text": f"expansions ranked by native similarity, top {top_n}"})
        run.emit({"type": "status",
                  "text": f"{len(rules)} rules at r{radius}, "
                          f"EC coverage {100 * rule_ec.coverage:.1f}%"})

        # A target no rule touches is unsolvable before any policy runs, and saying so
        # up front is kinder than a progress bar that never moves.
        probe = expand(target, rules, prefilter, rule_ec=rule_ec, require_ec=require_ec,
                       ranker=ranker, top_n=top_n if ranker else None)
        if not probe.neighbours:
            run.emit({"type": "error",
                      "text": "no reaction rule applies to this molecule at "
                              f"r{radius} — it cannot be expanded at all. "
                              "A lower radius admits more general templates."})
            return
        run.emit({"type": "status",
                  "text": f"target expandable: {probe.n_prefiltered} rules passed the "
                          f"prefilter, {len(probe.neighbours)} gave valid precursor sets"})

        if policy_name in ("greedy", "mcts"):
            run.emit({"type": "status", "text": "building sink closeness..."})
        policy = build_policy(policy_name, radius, seed, params.get("model", ""))
        run.emit({"type": "status", "text": f"policy ready: {policy_name}"})

        result = search(
            target, rules, prefilter, policy,
            budget=budget, max_depth=max_depth,
            rule_ec=rule_ec, require_ec=require_ec,
            stop_on_first_pathway=not exhaustive,
            on_decision=lambda row: run.emit({"type": "decision", **row}),
            ranker=ranker, top_n=top_n if ranker else None,
        )

        # Enumerate widely, show a slice: routes differing in one step are common, and
        # the interesting ones -- those that branch, and branch again -- are rarely the
        # first the cartesian product produces. Naringenin needs about forty before an
        # AND node appears under another AND node.
        routes = (extract_routes(result, rule_ec, max_routes=16, max_pathways=256)
                  if result.solved else [])
        routes.sort(key=lambda r: (-sum(1 for s in r.steps if len(s.precursors) > 1),
                                   len(r)))
        run.emit({
            "type": "done",
            "solved": result.solved,
            "stopped_because": result.stopped_because,
            "n_expansions": result.n_expansions,
            "n_molecules": result.n_molecules,
            "n_reactions": result.n_reactions,
            "elapsed_s": round(result.elapsed_s, 2),
            "target": result.target,
            "routes": [r.to_dict() for r in routes],
        })
    except Exception:                                    # noqa: BLE001
        run.emit({"type": "error", "text": traceback.format_exc(limit=4)})
    finally:
        run.done = True
        run.events.put({"type": "_eof"})


# --------------------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    server_version = "MorganBioPilot"

    def log_message(self, fmt, *args):                   # noqa: A003
        # One line per expansion would drown the console the search is printing to.
        pass

    # -- helpers ------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    # -- routes -------------------------------------------------------------
    def do_GET(self):                                    # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (HERE / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/presets":
            self._json({"presets": presets(), "policies": list(POLICIES)})
        elif path == "/api/events":
            self.stream_events(parse_qs(urlparse(self.path).query).get("run", [""])[0])
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):                                   # noqa: N802
        if urlparse(self.path).path != "/api/search":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return

        run = Run(run_id=uuid.uuid4().hex[:12])
        with RUNS_LOCK:
            RUNS[run.run_id] = run
            # Unbounded growth otherwise: this runs for days on a workstation.
            for old in [k for k, v in RUNS.items() if v.done][:-20]:
                RUNS.pop(old, None)
        threading.Thread(target=run_search, args=(run, params), daemon=True).start()
        self._json({"run": run.run_id})

    def stream_events(self, run_id: str) -> None:
        run = RUNS.get(run_id)
        if run is None:
            self._json({"error": "unknown run"}, 404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Replay what already happened, so a page opened late still gets the run from
        # its first expansion rather than from wherever the stream happened to attach.
        with run.lock:
            backlog = list(run.history)
        try:
            for event in backlog:
                self._event(event)
            seen = len(backlog)
            while True:
                event = run.events.get()
                if event.get("type") == "_eof":
                    self._event({"type": "eof"})
                    return
                # The backlog already carried everything emitted before we attached.
                if seen > 0:
                    seen -= 1
                    continue
                self._event(event)
        except (BrokenPipeError, ConnectionResetError):
            return                                       # the tab was closed

    def _event(self, event: dict) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.flush()


def presets() -> List[dict]:
    """A few real targets, so the first click does not require typing a SMILES."""
    try:
        from morganbiopilot.core.golden_dataset import load_golden_dataset
        return [{"name": g.name, "smiles": g.target}
                for g in load_golden_dataset("experimental")]
    except Exception:                                    # noqa: BLE001
        return [{"name": "vanillin", "smiles": "O=Cc1ccc(O)c(OC)c1"},
                {"name": "styrene", "smiles": "C=Cc1ccccc1"}]


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 keeps it off the network; use 0.0.0.0 behind a tunnel")
    p.add_argument("--port", type=int, default=8500)
    args = p.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MorganBioPilot — open http://{args.host}:{args.port}")
    print("rule sets load on first use, so the first search of each radius is slow")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
