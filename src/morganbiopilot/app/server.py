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

The page is *told* what it can offer -- policies, toolings, rankers and radii all come
from `/api/presets`. They were once written twice, once here and once in the HTML, and
the two drifted: the page went on offering `greedy_similarity` and `random` long after
those policies were deleted, and picking either put a traceback in the live log. One
list, served, is the only arrangement that cannot drift.

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

# Served to the page, which builds its <select> from them. Every id here must be one
# `build_policy` handles, and vice versa.
POLICIES = (
    {"id": "bfs", "label": "bfs — breadth first"},
    {"id": "dfs", "label": "dfs — depth first"},
    {"id": "greedy", "label": "greedy — closest to the chassis"},
    {"id": "mcts", "label": "mcts — UCT on sink closeness"},
    {"id": "llm", "label": "llm — language-model policy"},
)
POLICY_IDS = tuple(p["id"] for p in POLICIES)

# The agent arms of `paper_results.compare_policies.build_policies`, same names and same
# meanings, so a cell of that grid can be inspected here one target at a time.
TOOLINGS = (
    {"id": "tooled", "label": "tooled — sink closeness + native EC"},
    {"id": "tooled_plain", "label": "tooled_plain — same columns, unexplained"},
    {"id": "untooled", "label": "untooled — SMILES and depth only"},
    {"id": "ec_only", "label": "ec_only — native EC alone"},
    {"id": "closeness_only", "label": "closeness_only — sink closeness alone"},
    {"id": "tooled_enz", "label": "tooled_enz — tooled + route plausibility"},
    {"id": "enz_only", "label": "enz_only — route plausibility alone"},
)
TOOLING_IDS = tuple(t["id"] for t in TOOLINGS)

RANKERS = (
    {"id": "native_similarity",
     "label": "native_similarity — one substrate per rule"},
    {"id": "native_similarity_all",
     "label": "native_similarity_all — every native substrate"},
)
RANKER_IDS = tuple(r["id"] for r in RANKERS)


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
        self._rankers: Dict[tuple, object] = {}
        self._plausibility: Optional[object] = None

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

    def ranker(self, radius: int, name: str = "native_similarity"):
        """Cached per (radius, name): it holds a fingerprint for every rule substrate it
        sees, and rebuilding it on each search would throw that cache away every click.
        Keyed on the name too, because `native_similarity_all` loads the whole MetaNetX
        reaction table and must not be served from the cheap one's slot.

        The rule set is fetched *before* taking the lock, deliberately. `self._lock` is
        a plain `Lock`, so calling `self.rules()` from inside it deadlocks -- and it did:
        the first ranked search hung while holding the lock and every later request then
        blocked forever on "loading the rule set".
        """
        rules, _ec, _pf = self.rules(radius)
        key = (radius, name)
        with self._lock:
            if key not in self._rankers:
                from morganbiopilot.one_step.ranking import make_ranker
                self._rankers[key] = make_ranker(name, rules)
            return self._rankers[key]

    def plausibility(self):
        """Shared for the same reason as the ranker: the scorer loads a fitted model and
        caches edges on reaction content, so one instance across searches is both safe
        and warmer than a fresh one per click.
        """
        with self._lock:
            if self._plausibility is None:
                from morganbiopilot.multi_step.plausibility import RoutePlausibility
                self._plausibility = RoutePlausibility()
            return self._plausibility


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


def build_surface(tooling: str, radius: int, rule_ec):
    """The grounding signals one agent arm receives.

    A transcription of `compare_policies.build_policies`, deliberately: the point of
    running an arm here is to see what that arm does, so the two must build the same
    `ToolSurface` from the same name.
    """
    from morganbiopilot.agents.tools import ToolSurface, tooled, untooled

    if tooling == "untooled":
        return untooled()
    if tooling == "ec_only":
        return ToolSurface(rule_ec=rule_ec)
    if tooling == "closeness_only":
        return ToolSurface(closeness=ENGINE.closeness(radius))
    if tooling == "enz_only":
        return ToolSurface(plausibility=ENGINE.plausibility())
    if tooling == "tooled_enz":
        return tooled(radius, rule_ec, plausibility=ENGINE.plausibility())
    # "tooled" and "tooled_plain" show the same columns; only the prompt differs.
    return tooled(radius, rule_ec)


def build_policy(name: str, radius: int, seed: int, params: dict, *,
                 rule_ec=None, prefilter=None, ranker=None, top_k: int = 20):
    """One policy instance. Mirrors `paper_results.compare_policies.build_policies`."""
    from morganbiopilot.multi_step.mcts import MCTS
    from morganbiopilot.multi_step.policy import (BreadthFirst, DepthFirst,
                                                  GreedyECFP)

    if name == "bfs":
        return BreadthFirst()
    if name == "dfs":
        return DepthFirst()
    if name == "greedy":
        return GreedyECFP(ENGINE.closeness(radius))
    if name == "mcts":
        return MCTS(ENGINE.closeness(radius))
    if name == "llm":
        # Imported here only: the agent package pulls in an HTTP client and a backend
        # that the classical policies have no reason to load.
        from morganbiopilot.agents.backends import make_backend
        from morganbiopilot.agents.policy import (DEFAULT_EFFORT, DEFAULT_MODEL,
                                                  LLMPolicy)

        tooling = params.get("tooling") or "untooled"
        # Falling back to the package default rather than to the empty string, which
        # `make_backend` read as an Anthropic model named "" and only failed on the
        # first decision, several rule-loading seconds later.
        model = str(params.get("model") or "").strip() or DEFAULT_MODEL
        effort = str(params.get("effort") or "").strip() or DEFAULT_EFFORT
        return LLMPolicy(
            tools=build_surface(tooling, radius, rule_ec),
            backend=make_backend(model, effort=effort),
            top_k=top_k, seed=seed,
            # `tooled_plain` shows the same engine-computed columns as `tooled` but
            # leaves the system prompt silent about what they mean.
            explain=tooling != "tooled_plain",
            # The frontier order is part of the environment, not of the policy. Without
            # these three the view silently degrades to `_stratify`, and the page would
            # then be showing an arm the benchmark never ran.
            ranker=ranker, prefilter=prefilter, rule_ec=rule_ec,
        )
    raise ValueError(f"unknown policy {name!r}")


def run_search(run: Run, params: dict) -> None:
    """The search thread. Every exit path must emit a terminal event."""
    from morganbiopilot.agents.state import DEFAULT_TOP_K
    from morganbiopilot.core.chem import AVAILABLE_RADII, sanitize, split_components
    from morganbiopilot.multi_step.routes import extract_routes
    from morganbiopilot.multi_step.search import search
    from morganbiopilot.one_step.expand import expand

    try:
        # Every choice is validated before anything expensive loads, and reported as a
        # sentence rather than as the traceback an unguarded ValueError used to leave in
        # the live log. A stale option in a cached page must read as a message.
        radius = int(params.get("radius", 1))
        if radius not in AVAILABLE_RADII:
            have = ", ".join(f"r{r}" for r in AVAILABLE_RADII)
            run.emit({"type": "error",
                      "text": f"no rule set at radius {radius}; have {have}"})
            return
        policy_name = str(params.get("policy") or "bfs")
        if policy_name not in POLICY_IDS:
            run.emit({"type": "error",
                      "text": f"unknown policy {policy_name!r}; choose from "
                              f"{', '.join(POLICY_IDS)}"})
            return
        tooling = str(params.get("tooling") or "untooled")
        if policy_name == "llm" and tooling not in TOOLING_IDS:
            run.emit({"type": "error",
                      "text": f"unknown tooling {tooling!r}; choose from "
                              f"{', '.join(TOOLING_IDS)}"})
            return
        ranker_name = str(params.get("ranker") or "native_similarity")
        if ranker_name not in RANKER_IDS:
            run.emit({"type": "error",
                      "text": f"unknown ranker {ranker_name!r}; choose from "
                              f"{', '.join(RANKER_IDS)}"})
            return

        budget = int(params.get("budget", 50))
        max_depth = int(params.get("max_depth", 7))
        require_ec = bool(params.get("require_ec"))
        exhaustive = bool(params.get("exhaustive"))
        seed = int(params.get("seed", 0))
        # A wall-clock bound on this one search. Off by default, as in the benchmark --
        # a bound that fires turns a solve into a censored measurement -- but a browser
        # tab is where an unbounded run is least tolerable: one probe at budget 60 on
        # 14-Butanediol had not returned after ten minutes.
        max_seconds = float(params.get("max_seconds") or 0) or None
        # The matched control: restrict every policy to the portfolio's top-k frontier
        # view, which is what the agent arms already see. It doubles as the agent's own
        # `top_k`, so the two cannot be given different numbers by accident.
        view_top_k = int(params.get("view_top_k") or 0) or None
        # Ranked, capped expansion. Off by default so the page can show what the
        # unranked engine does, which is the comparison the paper rests on: at r1 an
        # uncapped expansion of violacein yields 96 precursor sets and the frontier
        # passes 15,000 nodes within 100 expansions.
        top_n = int(params.get("top_n") or 0) or None
        use_rank = bool(params.get("ranked")) and top_n
        # `sanitize` cannot be asked whether it worked: `morganrxn.sanitize_smiles`
        # swallows the exception and returns its input unchanged, so the falsy test
        # this guard used to make never fired and a typed SMILES reached RDKit as
        # `None` -- eight lines of Boost.Python signature mismatch in the live log.
        # Parsing here is the only way to tell the user what is actually wrong.
        from rdkit import Chem

        raw = str(params.get("smiles", "")).strip()
        target = sanitize(raw) if raw else ""
        if not target or Chem.MolFromSmiles(target) is None:
            run.emit({"type": "error",
                      "text": f"that SMILES could not be parsed: {raw[:120]!r}"})
            return
        # The search works mono-component: every node of the graph is one molecule, and
        # `expand` raises rather than guess which half was meant.
        if len(split_components(target)) > 1:
            run.emit({"type": "error",
                      "text": "this is a mixture; search one component at a time "
                              f"({' , '.join(split_components(target))})"})
            return

        run.emit({"type": "status", "text": f"loading the r{radius} rule set..."})
        rules, rule_ec, prefilter = ENGINE.rules(radius)
        ranker = ENGINE.ranker(radius, ranker_name) if use_rank else None
        if ranker is not None:
            run.emit({"type": "status",
                      "text": f"expansions ranked by {ranker_name}, top {top_n}"})
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

        # The agent arms that carry a closeness column pay for the same matrix, so the
        # wait is announced for them too.
        needs_closeness = policy_name in ("greedy", "mcts") or (
            policy_name == "llm"
            and tooling in ("tooled", "tooled_plain", "closeness_only", "tooled_enz"))
        if needs_closeness:
            run.emit({"type": "status", "text": "building sink closeness..."})
        policy = build_policy(policy_name, radius, seed, params, rule_ec=rule_ec,
                              prefilter=prefilter, ranker=ranker,
                              top_k=view_top_k or DEFAULT_TOP_K)
        label = policy_name if policy_name != "llm" else f"llm / {tooling}"
        run.emit({"type": "status", "text": f"policy ready: {label}"})
        if policy_name == "llm" and ranker is None:
            # Not fatal, but it changes what is being shown: the portfolio's ranked
            # member is what orders the candidates the agent chooses among.
            run.emit({"type": "status",
                      "text": "note: no ranker, so the frontier view falls back to a "
                              "stratified sample rather than the portfolio"})

        result = search(
            target, rules, prefilter, policy,
            budget=budget, max_depth=max_depth,
            rule_ec=rule_ec, require_ec=require_ec,
            stop_on_first_pathway=not exhaustive,
            on_decision=lambda row: run.emit({"type": "decision", **row}),
            ranker=ranker, top_n=top_n if ranker else None,
            max_seconds=max_seconds, view_top_k=view_top_k,
        )

        # Enumerate widely, show a slice: routes differing in one step are common, and
        # the interesting ones -- those that branch, and branch again -- are rarely the
        # first the cartesian product produces. Naringenin needs about forty before an
        # AND node appears under another AND node.
        routes = (extract_routes(result, rule_ec, max_routes=16, max_pathways=256)
                  if result.solved else [])
        # Shortest first, branching only as a tiebreak. It used to be the other way
        # round -- most-branching first, to surface convergent routes -- and on a
        # decorable scaffold that inverted the intent: the AND steps it ranks on are
        # the transferases, so violacein led with a 45-step route that prenylates,
        # glycosylates and sialylates the indole and then strips it all off again, over
        # a curated route of 6. A detour that reaches the sink solves the search, not
        # the biology, and it must not be what the page shows first.
        routes.sort(key=lambda r: (len(r),
                                   -sum(1 for s in r.steps if len(s.precursors) > 1)))
        run.emit({
            "type": "done",
            "solved": result.solved,
            "stopped_because": result.stopped_because,
            "n_expansions": result.n_expansions,
            "n_molecules": result.n_molecules,
            "n_reactions": result.n_reactions,
            # Exact and free -- `graph.shortest_route()` walks the graph rather than
            # enumerating it. It is the honest answer to "how long is the pathway":
            # a route's step count includes every branch of its AND-tree, so a
            # convergent route of 12 steps can still be 4 reactions deep.
            "shortest_pathway": result.shortest_pathway_length,
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
            self._json(options())
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


def options() -> dict:
    """Everything the page needs to build its form -- the single source of truth.

    Radii come from `core.chem.AVAILABLE_RADII` and `top_k` from `agents.state`, so
    adding a radius, or moving the default the benchmark uses, does not need the HTML
    touched. Restating either here is how the two copies drifted the first time.
    """
    from morganbiopilot.agents.state import DEFAULT_TOP_K
    from morganbiopilot.core.chem import AVAILABLE_RADII

    return {
        "presets": presets(),
        "policies": list(POLICIES),
        "toolings": list(TOOLINGS),
        "rankers": list(RANKERS),
        "radii": list(AVAILABLE_RADII),
        # r1 is the reported configuration: 80,116 ECFP-compatible templates, priced
        # against the branching factor it costs by `paper_results.radius_ladder`.
        "defaults": {"radius": 1, "policy": "mcts", "tooling": "untooled",
                     "ranker": "native_similarity", "top_k": DEFAULT_TOP_K},
    }


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
