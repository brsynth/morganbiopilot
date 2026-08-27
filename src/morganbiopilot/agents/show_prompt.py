"""Print the exact prompt the agent is sent, without spending an LLM call.

    python -m morganbiopilot.agents.show_prompt vanillin 4 tooled
    python -m morganbiopilot.agents.show_prompt vanillin 4 untooled
    python -m morganbiopilot.agents.show_prompt vanillin 4 tooled_enz

Every arm `compare_policies` can run is available here, so the prompt printed is the
prompt a real run sends: untooled, ec_only, closeness_only, enz_only, tooled, tooled_enz.
`enz_only` and `tooled_enz` need models/enzymatic_score.joblib to exist.


A fake backend records what `LLMPolicy` hands it and answers with a fixed choice,
so the search advances and we can look at a decision taken a few steps in, when
the frontier is no longer trivial.
"""
import json
import sys

from morganbiopilot.agents.backends import Completion
from morganbiopilot.agents.policy import LLMPolicy
from morganbiopilot.agents.tools import ToolSurface, tooled, untooled
from morganbiopilot.multi_step.heuristics import SinkCloseness
from morganbiopilot.multi_step.plausibility import RoutePlausibility
from morganbiopilot.core.chem import sanitize
from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.golden_dataset import load_golden_dataset
from morganbiopilot.core.rules import load_rules
from morganbiopilot.one_step.prefilter import prefilter_from_rules
from morganbiopilot.multi_step.search import search

TARGET_NAME = sys.argv[1] if len(sys.argv) > 1 else "vanillin"
# Early decisions have a trivial frontier and say nothing about the format;
# the default asks for a decision taken once the graph has branched.
WANTED_STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TOOLING = sys.argv[3] if len(sys.argv) > 3 else "tooled"
RADIUS = 2


class RecordingBackend:
    name = "fake(recording)"
    supports_effort = False

    def __init__(self):
        self.prompts = []

    def complete(self, system, prompt, schema):
        self.prompts.append(prompt)
        # Always take the first candidate: any fixed rule advances the search.
        return Completion(text=json.dumps({"choice": 0}),
                          input_tokens=0, output_tokens=0)


rules = load_rules(radius=RADIUS)
rule_ec = annotate_rules(rules)
prefilter = prefilter_from_rules(rules)
entries = {e.name: e for e in load_golden_dataset(variant="experimental")}
entry = entries[TARGET_NAME]

backend = RecordingBackend()


def make_surface(name: str):
    """The same arms `compare_policies` offers, so this shows what a run really sends."""
    if name == "untooled":
        return untooled()
    if name == "ec_only":
        return ToolSurface(rule_ec=rule_ec)
    if name == "closeness_only":
        return ToolSurface(closeness=SinkCloseness(RADIUS))
    if name == "enz_only":
        return ToolSurface(plausibility=RoutePlausibility())
    if name == "tooled_enz":
        return tooled(RADIUS, rule_ec, plausibility=RoutePlausibility())
    if name == "tooled":
        return tooled(RADIUS, rule_ec)
    raise SystemExit(f"unknown tooling {name!r}; choose from untooled, ec_only, "
                     "closeness_only, enz_only, tooled, tooled_enz")


surface = make_surface(TOOLING)
policy = LLMPolicy(tools=surface, backend=backend)

search(entry.target, rules, prefilter, policy,
       budget=12, max_depth=5, rule_ec=rule_ec)

print(f"target      : {TARGET_NAME}  ({entry.target})")
print(f"radius      : {RADIUS}   top_k: {policy.top_k}   tooling: {surface.name}")
print(f"decisions   : {policy.n_decisions}   API calls: {policy.n_calls}")
print(f"prompts captured: {len(backend.prompts)}")

index = min(WANTED_STEP, len(backend.prompts)) - 1
if index < 0:
    print("\n(no prompt captured -- frontier never had more than one candidate)")
    raise SystemExit(0)

print("\n" + "=" * 78)
print("SYSTEM PROMPT")
print("=" * 78)
print(policy.system_prompt)
print("\n" + "=" * 78)
print(f"USER PROMPT  (decision #{index + 1})")
print("=" * 78)
print(backend.prompts[index])

chars = len(policy.system_prompt) + len(backend.prompts[index])
print("\n" + "=" * 78)
print(f"size: {chars} characters, ~{chars // 4} tokens (rough 4 chars/token)")
