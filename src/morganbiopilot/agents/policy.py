"""The LLM search policy: an agent that answers one question per expansion.

Mono-agent, as in section 5 of the project note: its only responsibility is to
choose which frontier molecule to expand next. Expansion, applicability, graph
maintenance, status propagation and the sink test stay deterministic, so this
class implements exactly the same `Policy` protocol as BFS or greedy ECFP.

Every call is logged (`LLMPolicy.calls`) with the candidates offered, the choice,
the token counts and the latency. Runs are not reproducible — sampling parameters
were removed from this model generation, so `temperature=0` is not available and
returns a 400. Repeat runs and report variance; do not present a single run as a
point estimate.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from morganbiopilot.agents.backends import parse_json_choice
from morganbiopilot.agents.state import (
    DEFAULT_FRONTIER_ORDER, DEFAULT_TOP_K, build_frontier_view,
)
from morganbiopilot.agents.tools import ToolSurface
from morganbiopilot.multi_step.graph import SearchGraph

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
# No module-level output budget: each backend knows its own. Anthropic needs
# thousands of tokens because adaptive thinking is charged against the output
# budget; a locally served 8k-context model needs about a hundred and rejects
# anything that does not fit beside the prompt. One number cannot serve both, and
# the one that used to be here -- 16000 -- made every vLLM request a 400.
# Imported rather than restated: a second copy of this default silently overrode
# the one in `state.build_frontier_view` and kept the agent on breadth-first
# truncation after that default had been fixed.
DEFAULT_ORDER = DEFAULT_FRONTIER_ORDER

SYSTEM_PROMPT = """\
You are the search policy of a retrobiosynthesis planner.

The planner decomposes a target molecule into molecules the host organism already \
makes. It works backwards: each expansion applies enzymatic reaction rules to one \
molecule and adds the molecules on the other side of those reactions to the search \
graph. A branch ends when it reaches a chassis metabolite.

Your only job, each turn, is to pick which molecule from the frontier to expand \
next. You do not propose reactions, choose rules, or judge chemistry — the engine \
does that deterministically. You decide where to spend the next expansion.

The chassis is E. coli. Its sink is roughly 750 metabolites: central carbon \
metabolism intermediates, amino acids, nucleotides, fatty acids, cofactors such as \
NADH and acetyl-CoA, and common precursors of secondary metabolism. A branch is \
solved the moment it reaches any of them — you do not need to reach glucose.

The expansion budget is limited and shared across the whole search. Spending it on \
molecules that are large, exotic, or far from primary metabolism wastes it; so does \
exhaustively deepening one branch while other promising ones sit untouched.

Reason as a metabolic engineer: which of these molecules most plausibly sits one or \
a few enzymatic steps from something E. coli already makes?

Answer with the index of your chosen candidate and one sentence of justification."""

# Each grounding column gets its own paragraph, added only when that column is
# actually rendered. Before this, the tooled variant showed two engine-computed
# fields that the prompt never mentioned: the model had to infer what `EC=2`
# meant and, worse, guess whether `closest_chassis` was a distance or a
# similarity. Getting that polarity backwards inverts the signal entirely.
# Keeping the paragraphs independent also allows an EC-only condition, which
# separates "grounding does not help" from "this particular grounding is noisy".
EC_EXPLANATION = """\
Some candidate lines carry an EC field, e.g. `EC=1,2`. These are the level-1 \
classes of the enzymatic reactions that produced that molecule in the graph: \
1 oxidoreductase, 2 transferase, 3 hydrolase, 4 lyase, 5 isomerase, 6 ligase, \
7 translocase. It describes the chemistry that led to the molecule, not what is \
available from it. The field is absent when none of the rules involved carries an \
annotation, which is the case for about half of them — absent means "not \
annotated", never "not enzymatic"."""

CLOSENESS_EXPLANATION = """\
Some candidate lines carry a closest_chassis field, e.g. \
`closest_chassis=0.591 to 4-aminobenzoate [Nc1ccc(C(=O)[O-])cc1]`. The number is \
a counted-Tanimoto similarity in [0, 1] between the candidate and the most \
similar metabolite of the chassis sink, and the name and structure that follow \
are that metabolite. Higher means more similar: around 0.7 a close structural \
analogue is already available, around 0.35 nothing in the sink resembles this \
molecule. Judge the remaining gap from the two structures — the number measures \
shared substructures, not how many enzymatic steps separate them. Some sink \
metabolites have no common name and appear as a bare identifier such as MNXM208; \
use the SMILES in that case."""

# Stated with its own uncertainty, deliberately. The column comes from a model at AUC
# 0.660 -- heavily overlapping classes -- and a prompt that presented it as a verdict
# would invite exactly the over-reliance the closest_chassis column already produced.
PLAUSIBILITY_EXPLANATION = """\
Some candidate lines carry an enzymatic field, e.g. \
`enzymatic=0.612/step (route 0.229 over 3)`. Every reaction between the target and \
that candidate has been scored for how much it resembles known enzymatic chemistry \
rather than synthetic organic chemistry. The first number is the average per step, \
the second the product over the whole partial route, and the last is how many \
reactions that route contains. Higher is more biochemical; around 0.5 means the \
scorer cannot tell. Compare candidates on the per-step figure: the product falls \
with every extra step whatever the chemistry, so a low product often just means a \
longer route. The scorer is weak — it is right about which of two transformations is \
more biochemical roughly two times in three — so treat it as one weak indication \
among the structures you can read yourself, never as a verdict on a single \
candidate."""


def build_system_prompt(tools: ToolSurface, explain: bool = True) -> str:
    """The system prompt for one tool surface — only columns actually shown.

    `explain=False` shows the same columns without saying what they mean, which is
    how this project ran until the columns were documented. It exists as an
    ablation, not as an option anyone should pick: the accidental comparison it
    formalises is the most interesting agentic signal we have. A 3B model given
    unexplained columns did *worse* with them than without (33% vs 67%); a 7B model
    given explained ones did better (75% vs 50%). Those runs differ in both model
    and prompt, so they confound the effect. Holding the model fixed and toggling
    only the explanation isolates it, and the question generalises past
    retrosynthesis: engine-computed features may need to be explained to a model or
    they degrade its decisions.
    """
    sections = [SYSTEM_PROMPT]
    if not explain:
        return sections[0]
    if tools.rule_ec is not None:
        sections.append(EC_EXPLANATION)
    if tools.closeness is not None:
        sections.append(CLOSENESS_EXPLANATION)
    if tools.plausibility is not None:
        sections.append(PLAUSIBILITY_EXPLANATION)
    return "\n\n".join(sections)

# How many consecutive API failures before a run gives up. Low, because the
# failures that matter here are account- or request-level and will not clear on
# their own: a bad key, an exhausted balance, a rejected parameter.
MAX_CONSECUTIVE_ERRORS = 5

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "description": "Index of the chosen candidate."},
        "reason": {"type": "string", "description": "One sentence of justification."},
    },
    "required": ["choice", "reason"],
    "additionalProperties": False,
}


@dataclass
class LLMCall:
    """One decision, recorded so a run can be reconstructed without re-running it."""

    step: int
    n_candidates: int
    n_frontier: int
    choice: Optional[int]
    smiles: str
    reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    elapsed_s: float = 0.0
    fallback: str = ""      # non-empty when the model's answer could not be used
    called_api: bool = True  # False when the decision was made without asking
    # The state exactly as the model saw it, kept only when `record_prompts` is on.
    # Rejection-sampling fine-tuning turns successful episodes into training pairs, and
    # a pair is only usable if its prompt is byte-for-byte what inference will produce
    # -- re-rendering it afterwards is impossible, because the frontier view depends on
    # the graph at that instant and the graph has moved on.
    prompt: str = ""

    @property
    def is_failure(self) -> bool:
        """A fallback that reflects the model failing, not a skipped decision.

        `skipped_single` means the frontier had one candidate and there was
        nothing to decide — counting it as a failure would make a perfectly
        reliable agent look unreliable in the paper's table.
        """
        return bool(self.fallback) and self.fallback != "skipped_single"


class LLMPolicy:
    """Search policy backed by a language model."""

    def __init__(
        self,
        tools: ToolSurface,
        backend=None,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        top_k: int = DEFAULT_TOP_K,
        order: str = DEFAULT_ORDER,
        seed: int = 0,
        max_tokens: Optional[int] = None,
        explain: bool = True,
        client=None,
        server_side_fallback: bool = True,
        record_prompts: bool = False,
        ranker=None,
        prefilter=None,
    ):
        # `ranker` and `prefilter` are what the "portfolio" frontier order needs. They
        # are not tools: the agent never sees them, they decide which twenty molecules
        # it is shown. That is a property of the environment, and every arm -- prompted,
        # fine-tuned, and the classical baselines through their own frontier -- must
        # face the same one or the comparison measures the view, not the policy.
        self.ranker = ranker
        self.prefilter = prefilter
        self.tools = tools
        # Off by default: a 150-expansion run would otherwise hold 150 prompts of a
        # thousand tokens each, times however many runs a worker pool has in flight.
        # Only the episode generator for rejection sampling needs them.
        self.record_prompts = bool(record_prompts)
        # Built once: it is identical at every decision, and a stable prefix is
        # what makes prompt caching possible on backends that offer it.
        self.explain = bool(explain)
        self.system_prompt = build_system_prompt(tools, explain=self.explain)
        self.top_k = top_k
        self.order = order
        self.seed = seed

        if backend is None:
            from morganbiopilot.agents.backends import make_backend

            # `max_tokens` is left to the backend unless the caller insists. The
            # policy used to force 16000 on everything, a figure that only makes
            # sense for Anthropic, where adaptive thinking is charged against the
            # output budget. Against a locally served 8k-context model the same
            # number is simply invalid -- vLLM rejects the request with a 400,
            # since 16000 > 8192 - prompt -- and every decision fell back.
            extra = {} if max_tokens is None else {"max_tokens": max_tokens}
            backend = make_backend(
                model, effort=effort, client=client,
                server_side_fallback=server_side_fallback, **extra,
            )
        self.backend = backend
        self.calls: List[LLMCall] = []
        self._step = 0
        self._consecutive_errors = 0

    @property
    def name(self) -> str:
        # The explanation state belongs in the name: two rows differing only in
        # whether the columns were documented must not collide in the table.
        # Untooled has no columns to explain, so the marker would be noise there.
        suffix = "" if self.explain or self.tools.name == "none" else "|unexplained"
        return f"llm[{self.backend.name}|tools={self.tools.name}{suffix}]"

    # ------------------------------------------------------------------ metrics
    @property
    def n_decisions(self) -> int:
        """Expansions this policy chose — one per search step."""
        return len(self.calls)

    @property
    def n_calls(self) -> int:
        """API calls actually issued. Fewer than decisions: single-candidate
        frontiers are resolved without asking the model, and that difference is
        what the cost column must report."""
        return sum(1 for c in self.calls if c.called_api)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def n_fallbacks(self) -> int:
        """Decisions where the model's answer was unusable. Report this.

        Excludes decisions taken without an API call — see `LLMCall.is_failure`.
        """
        return sum(1 for c in self.calls if c.is_failure)

    # ------------------------------------------------------------------- policy
    def select(self, graph: SearchGraph, frontier: List[int]) -> int:
        self._step += 1
        view = build_frontier_view(
            graph, frontier,
            top_k=self.top_k, order=self.order, seed=self.seed,
            closeness=self.tools.closeness, rule_ec=self.tools.rule_ec,
            plausibility=self.tools.plausibility,
            ranker=self.ranker, prefilter=self.prefilter,
        )

        if len(view.candidates) == 1:
            # No decision to make; don't spend a call on it.
            node_id = view.candidates[0].node_id
            self.calls.append(LLMCall(
                step=self._step, n_candidates=1, n_frontier=view.n_total,
                choice=0, smiles=graph.molecules[node_id].smiles,
                reason="single candidate", fallback="skipped_single",
                called_api=False,
            ))
            return node_id

        prompt = self._render_prompt(graph, view)
        t0 = time.perf_counter()
        record = LLMCall(
            step=self._step, n_candidates=len(view.candidates),
            n_frontier=view.n_total, choice=None, smiles="",
            prompt=prompt if self.record_prompts else "",
        )

        try:
            completion = self.backend.complete(self.system_prompt, prompt, CHOICE_SCHEMA)
        except Exception as exc:
            record.fallback = f"api_error:{type(exc).__name__}"
            self._consecutive_errors += 1
            # A run whose API keeps refusing is not producing a measurement, and every
            # further call costs wall-clock while degrading silently towards the
            # fallback rule. This is not hypothetical: an exhausted credit balance
            # once let 1668 consecutive calls fail over two hours, and the resulting
            # solve rate was reported as an arm of an ablation. Fail loudly instead.
            if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"{self._consecutive_errors} consecutive API failures "
                    f"({record.fallback}); the last error was: {exc}"
                ) from exc
            return self._fallback(view, record, t0)
        self._consecutive_errors = 0

        record.input_tokens = completion.input_tokens
        record.output_tokens = completion.output_tokens
        record.cache_read_tokens = completion.cache_read_tokens

        if completion.refused:
            record.fallback = "refusal"
            return self._fallback(view, record, t0)

        index, reason = parse_json_choice(completion.text)
        record.reason = reason
        if index is None:
            record.fallback = "unparseable"
            return self._fallback(view, record, t0)

        if not 0 <= index < len(view.candidates):
            # The schema cannot bound an integer, so the range is checked here.
            record.fallback = f"out_of_range:{index}"
            return self._fallback(view, record, t0)

        node_id = view.candidates[index].node_id
        record.choice = index
        record.smiles = graph.molecules[node_id].smiles
        record.elapsed_s = time.perf_counter() - t0
        self.calls.append(record)
        return node_id

    # ------------------------------------------------------------------ helpers
    def _fallback(self, view, record: LLMCall, t0: float) -> int:
        """Take the first candidate and record why. Never silently substitute.

        Falling back to the first candidate in `discovery` order is the
        policy-neutral choice — it is not the greedy pick, so a run with many
        fallbacks degrades toward BFS rather than toward a competing baseline.
        """
        record.choice = 0
        record.smiles = view.candidates[0].smiles
        record.elapsed_s = time.perf_counter() - t0
        self.calls.append(record)
        return view.candidates[0].node_id

    def _render_prompt(self, graph: SearchGraph, view) -> str:
        target = graph.molecules[graph.root].smiles
        solved = sum(1 for m in graph.molecules.values() if m.solved)
        lines = [
            f"Target molecule: {target}",
            f"Search graph so far: {len(graph.molecules)} molecules, "
            f"{len(graph.reactions)} reactions, {solved} already reachable from the chassis.",
            "",
            "Frontier candidates:",
            view.render(),
            "",
            "Which candidate should be expanded next?",
        ]
        return "\n".join(lines)
