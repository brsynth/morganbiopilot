"""First-run diagnostic: one real call, everything printed.

Run this before any experiment, and again whenever the model, the account, or the
request shape changes. It is deliberately not a paper result — its job is to make
failures legible, so it prints the request it sent, the raw answer, and the usage,
rather than an aggregate.

    python -m morganbiopilot.agents.smoke
    python -m morganbiopilot.agents.smoke --target styrene --budget 3 --effort low

Cost is a few cents: one small target, a handful of calls, `low` effort by default.

Three things it checks that nothing else can, because they need a live server:

- that `output_config` accepts `effort` and `format` in the same request;
- that the server-side fallback beta is enabled on this account (if not, it retries
  without it and tells you, rather than failing the run);
- that the model returns JSON matching the schema.
"""

import argparse
import os
import sys
from typing import Optional

from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.golden_dataset import golden_targets
from morganbiopilot.core.paths import ROOT_DIR
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.search import search
from morganbiopilot.one_step.prefilter import prefilter_from_rules

# Rough Claude Opus 5 rates, $ per million tokens. Only used for the estimate line.
RATES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (3.0, 15.0),
         "claude-haiku-4-5": (1.0, 5.0)}


LOCAL_ENDPOINTS = (
    ("ollama", "http://localhost:11434/v1"),
    ("vLLM", "http://localhost:8000/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
)


def detect_local_server() -> Optional[str]:
    """Return the base URL of a local OpenAI-compatible server, if one answers."""
    import urllib.error
    import urllib.request

    for _, url in LOCAL_ENDPOINTS:
        try:
            urllib.request.urlopen(f"{url}/models", timeout=2)
            return url
        except (urllib.error.URLError, OSError):
            continue
    return None


def check_credentials(spec: str) -> bool:
    """Provider-aware. A local model needs no key; Anthropic does."""
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")

    if spec.startswith("openai:"):
        base = os.environ.get("OPENAI_BASE_URL")
        if base:
            print(f"endpoint: OPENAI_BASE_URL={base}")
            return True
        found = detect_local_server()
        if found:
            os.environ["OPENAI_BASE_URL"] = found
            print(f"endpoint: no OPENAI_BASE_URL set, but a server answers at {found} "
                  "— using it")
            return True
        listed = "\n".join(f"      {n}: {u}" for n, u in LOCAL_ENDPOINTS)
        print(
            "endpoint: NOT FOUND.\n"
            "  An OpenAI-compatible model needs a base URL. Either start a local\n"
            "  server (none is answering on the usual ports):\n"
            f"{listed}\n"
            f"  or set OPENAI_BASE_URL (and OPENAI_API_KEY if the host needs one)\n"
            f"  in {ROOT_DIR / '.env'}.",
            file=sys.stderr,
        )
        return False

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("credentials: ANTHROPIC_API_KEY is set")
        return True
    print(
        "credentials: NOT FOUND.\n"
        f"  Put a key in {ROOT_DIR / '.env'} as:\n"
        "      ANTHROPIC_API_KEY=sk-ant-...\n"
        "  That file is already excluded from git. Alternatively export the\n"
        "  variable in your shell, or log in with the `ant` CLI.\n"
        "  For a free run with no account, use a local model instead:\n"
        "      --model openai:<model-served-locally>",
        file=sys.stderr,
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", default="styrene", help="golden-set target name")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--budget", type=int, default=3, help="expansions, i.e. decisions")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--untooled", action="store_true", help="withhold ECFP and EC")
    parser.add_argument("--no-fallback", action="store_true",
                        help="skip the server-side fallback beta")
    args = parser.parse_args()

    if not check_credentials(args.model):
        return 1

    from morganbiopilot.agents.policy import LLMPolicy
    from morganbiopilot.agents.tools import tooled, untooled

    targets = dict(golden_targets())
    if args.target not in targets:
        print(f"unknown target {args.target!r}; choose from {sorted(targets)}", file=sys.stderr)
        return 1

    print(f"loading rules r{args.radius} ...")
    rules = load_rules(args.radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)
    surface = untooled() if args.untooled else tooled(args.radius, rule_ec)
    print(f"  {len(rules)} rules | tools={surface.name} | model={args.model} "
          f"| effort={args.effort} | budget={args.budget}\n")

    policy = LLMPolicy(
        tools=surface, model=args.model, effort=args.effort, top_k=args.top_k,
        server_side_fallback=not args.no_fallback,
    )

    result = search(
        targets[args.target], rules, prefilter, policy,
        budget=args.budget, max_depth=4, rule_ec=rule_ec, require_ec=True,
    )

    print("=== decisions ===")
    for call in policy.calls:
        status = call.fallback or "ok"
        print(f"  step {call.step}: {call.n_candidates:3d} candidates "
              f"(of {call.n_frontier:4d}) -> [{call.choice}] {status} "
              f"| {call.elapsed_s:5.1f}s | in={call.input_tokens} out={call.output_tokens}")
        if call.reason:
            print(f"      reason: {call.reason[:150]}")
        if call.smiles:
            print(f"      chose : {call.smiles[:80]}")

    failures = [c for c in policy.calls if c.is_failure]
    print("\n=== verdict ===")
    print(f"  API calls issued : {policy.n_calls} (decisions: {policy.n_decisions})")
    print(f"  usable answers   : {policy.n_calls - len(failures)}/{policy.n_calls}")
    if failures:
        print(f"  FAILURES         : {sorted({c.fallback for c in failures})}")

    rate_in, rate_out = RATES.get(args.model, (0.0, 0.0))
    cost = (policy.total_input_tokens * rate_in
            + policy.total_output_tokens * rate_out) / 1_000_000
    print(f"  tokens           : in={policy.total_input_tokens} "
          f"out={policy.total_output_tokens}")
    if rate_in:
        print(f"  approx. cost     : ${cost:.4f}")
    print(f"  search           : solved={result.solved} "
          f"expansions={result.n_expansions} molecules={result.n_molecules}")

    if policy.n_calls == 0:
        print("\n  No API call was issued — the frontier never offered a choice.\n"
              "  Raise --budget or --top-k.", file=sys.stderr)
        return 1
    if len(failures) == policy.n_calls:
        print("\n  Every call failed. The request shape or the account is the problem,\n"
              "  not the prompt. If the failures are api_error, try --no-fallback:\n"
              "  the server-side fallback beta may not be enabled here.", file=sys.stderr)
        return 1

    print("\n  OK — the request shape is accepted and answers parse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
