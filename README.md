# MorganBioPilot

Agentic retrobiosynthesis: a language model acting as the **search policy** on top of a
deterministic expansion engine in ECFP fingerprint space.

Node expansion is symbolic and reproducible — applicability prefiltering via the
reaction-center ECFP, graph-level validation, precursor generation. The agent decides
strategy only: which frontier molecule to expand next. Everything else is shared, byte
for byte, with the classical baselines.

---

## The design in one paragraph

A retrosynthetic search is two things glued together: an *engine* that answers "what
reactions produce this molecule?", and a *policy* that answers "which molecule should I
work on next?". Most systems entangle them, which makes it impossible to say whether a
result comes from better chemistry or better search. Here the seam is explicit: a policy
is any object with a `select(graph, frontier) -> molecule_id` method, and it is the
**only** extension point. Breadth-first, depth-first, greedy best-first, UCT and the LLM
agent are five implementations of that one protocol, handed the same rule set, the same
prefilter, the same expansion cap and the same budget. The reproducibility claim of the
paper is therefore a structural property of the code rather than a promise in the text —
and the corollary matters just as much: **do not let search logic leak into the engine.**

Note what this does *not* mean. The policy changes which parts of the graph are ever
built, so two policies explore different graphs; what is identical is the machinery that
builds them.

---

## Core concepts

**AND-OR search graph.** Molecules are OR-nodes: one is solved if it is in the sink, or
if *any* reaction producing it is solved. Reactions are AND-nodes: one is solved only if
*all* of its precursors are. Status propagates upward on every expansion. It is a graph,
not a tree — two branches reaching the same intermediate share it, and solving it once
solves it everywhere.

**The sink.** The chassis: E. coli metabolites plus a cofactor list that is stripped from
reaction sides and never expanded. A route is solved when every leaf is one of them.

**Budget.** The comparison axis is the number of **expansions**, one per policy decision.
Wall-clock is reported but is not the axis: it conflates a local baseline with a hosted
model behind an HTTP call.

**Two caps, often confused:**

| | `top_n` | `top_k` |
|---|---|---|
| what it limits | distinct **disconnections** kept per expansion | frontier **molecules** shown to the model |
| ordered by | substrate similarity of the rule | portfolio interleave, then shuffled |
| applies to | every policy, LLM or not | the LLM arms only |
| effect | destructive: the rest is never added to the graph | non-destructive: a molecule not shown this step is still selectable later |

`top_n = 20` and `top_k = 20` in the reported runs, and the coincidence is the whole
reason the two get mixed up.

**The frontier view.** The frontier routinely exceeds ten thousand molecules, so the
agent sees a bounded observation of twenty. They are chosen by round-robin interleaving
of four criteria — depth stratification, substrate similarity, reaction-precedent count,
molecular size — and then **shuffled** before display, so position carries no signal.

---

## Repository layout

```
src/morganbiopilot/
├── core/              shared primitives; no search, no agents
│   ├── paths.py           every data path in the project, in one place
│   ├── chem.py            sanitization and ECFP conventions
│   ├── rules.py           rule-set loading (fast path, project-local data dir)
│   ├── ec.py              rule → EC-number join
│   ├── building_blocks.py the sink and the cofactor table
│   ├── golden_dataset.py  the 20 curated pathways
│   ├── target_list.py     targets for benchmarks with no reference routes
│   └── reactions.py       full reactions behind the mono-component rules
│
├── one_step/          expand ONE molecule — the deterministic primitive
│   ├── prefilter.py       sparse O(d) reaction-center applicability test
│   ├── expand.py          prefilter → RDKit validation → precursor sets
│   └── ranking.py         which rules an expansion tries, so `top_n` is principled
│
├── multi_step/        the search itself
│   ├── graph.py           AND-OR graph, status propagation, route extraction
│   ├── search.py          THE loop — every system in the paper runs this one
│   ├── policy.py          BreadthFirst, DepthFirst, GreedyECFP + the Policy protocol
│   ├── mcts.py            UCT over the AND-OR graph
│   ├── heuristics.py      sink closeness in fingerprint space
│   ├── routes.py          solved search → readable, storable pathways
│   └── plausibility.py    enzymatic score along a route
│
├── agents/            the only non-deterministic part
│   ├── policy.py          LLMPolicy: one question per expansion, one JSON answer
│   ├── state.py           rendering a search state; frontier selection and truncation
│   ├── tools.py           the tool surface, and the grounding ablation
│   ├── backends.py        provider backends (hosted API or local OpenAI-compatible)
│   ├── train_policy.py    LoRA fine-tuning of the policy
│   ├── show_prompt.py     print the exact prompt, spending no call
│   └── smoke.py           first-run diagnostic: one real call, everything printed
│
├── data_processing/   one-off, rerunnable preparation; writes into data/ or results/
│   ├── route_corpus.py    mine attested routes from MetaNetX
│   ├── replay_routes.py   routes → the decisions the agent actually faces
│   ├── rewrite_corpus_prompt.py  re-stamp a corpus with the current prompt format
│   ├── self_play.py       the policy's own solved searches as training data
│   ├── enzymatic_model.py / reaction_pu.py   learned enzymatic scores
│   └── name_building_blocks.py   fill missing sink names from PubChem
│
├── paper_results/     every table and figure of the submission
│   ├── compare_policies.py   the main table: solve rate vs budget, per policy
│   ├── route_recovery.py     is the attested route among the ones we return?
│   ├── frontier_ranking.py   which ordering keeps the on-route molecule visible?
│   ├── candidate_ranking.py / rank_disconnections.py   which score ranks rules?
│   ├── rule_coverage.py / radius_ladder.py   what the templates reproduce, and at what
│   ├── frontier_diagnostics.py, cost_normalised.py, complete_pathways.py
│   ├── summarise_partial.py  rebuild tables from a killed job's partial TSV
│   ├── tables.py             the result row and its printing — stdlib only
│   └── visualize_routes.py   solved pathways as a standalone HTML page
│
└── app/               a local web UI: one search, watched expansion by expansion
    └── server.py          http.server + server-sent events; standard library only
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate morganbiopilot
pip install -e .
pip install -e <path to the companion rule-set package>/morganrxn
```

The `morganrxn` package supplies the ECFP reaction-rule machinery — rule loading,
`apply_reaction`, the fingerprint conventions — and is the subject of a companion paper
currently under review. It is a dependency, not a vendored copy: this project never
modifies it, and reimplements path resolution and rule loading locally where the
upstream behaviour would read the wrong directory or be too slow for a search loop.

**RDKit must be the pinned version.** Counted ECFP fingerprints have to be bit-for-bit
identical across both packages, or the templates and the applicability criterion stop
being comparable.

---

## Data

`data/` is roughly 1 GB of third-party corpora and is not version-controlled. Expected
layout:

```
data/
├── reaction_rules/metanetx/ecfp_r{0..5}_fp1024_folded_uncustom/   rule sets by radius
├── metanetx/                 chem_prop.tsv, reac_prop.tsv + processed/
├── building_blocks/          sink.csv (the chassis), names.tsv
├── cofactors/                cofactors_biochem.tsv
├── golden_dataset_pathways/  20 curated pathways, SBML
├── laser_dataset/            141 metabolic-engineering targets
└── bionavinp/                60 natural-product targets + their building blocks
```

The reported configuration is **radius 1** — 80,116 ECFP-compatible templates. The radius
is a genuine trade-off and is measured, not assumed: `paper_results/radius_ladder.py`
prices reproduction of attested single steps against the branching factor it costs.

---

## Running

**One search, interactively.** The tool that saves the log-grepping:

```bash
python -m morganbiopilot.app.server --port 8500
```

Point it at a molecule, pick a policy and a radius, watch the expansions stream in. Rule
sets load on first use, so the first search of each radius is slow. It is a microscope,
not a measuring instrument — one draw, no seeds; do not quote its numbers.

**The main benchmark table:**

```bash
python -m morganbiopilot.paper_results.compare_policies --radius 1 --budgets 10,25,50,100,200 --policies bfs,dfs,greedy,mcts --seeds 0
```

Add `--llm --models <spec>` to include the agent arms; they cost money or a GPU, so they
are opt-in. Every run writes a TSV whose rows are recomputed into tables by
`paper_results/tables.py` — no number in the paper is copied from a log summary.

**The training pipeline**, in order:

```bash
python -m morganbiopilot.data_processing.route_corpus      # mine attested routes
python -m morganbiopilot.data_processing.replay_routes     # routes → state/choice pairs
python -m morganbiopilot.agents.train_policy --pairs <corpus.jsonl> --out models/policy_sft
```

The split is **by target, never by example**: a single route contributes many decisions
over one graph, so splitting at random would put a target's depth-1 decision in training
and its depth-2 decision in validation. Evaluation targets and any state whose
observation displays one are dropped from the corpus outright.

---

## Reading the results

- **Solve rate is a curve, not a point.** With `stop_on_first_pathway`, a run halts the
  moment the target is solved, so `n_expansions` *is* the expansions-to-solution and the
  whole curve over budgets falls out of one run at the largest budget.
- **Everything to the right of the budget columns is measured at the largest budget
  only.** Route length, expansions and seconds-per-success describe the N=200 run, never
  the N=10 column beside them.
- **Partial files overstate.** Runs finish in completion order and solved runs finish
  fastest: on a completed benchmark the fastest 20% of runs are 96% solved against 68%
  over all of them. Rows below 95% coverage are not reported.
- **Ceiling.** A target with no applicable rule is unsolvable before any policy runs, and
  caps the achievable rate. Every table states that cap.
- **Fallbacks.** When a model's answer is unusable — refusal, unparseable, out of range —
  the policy takes the first candidate and records it. A run with many fallbacks is
  partly breadth-first, and the tables say so.

---

## Conventions

Seeds drive the frontier presentation shuffle only. Breadth-first, depth-first, greedy
and UCT are deterministic functions of the graph, so one seed is their complete answer
and they carry no error bars; the agent arms are repeated because the shuffle changes
what they see.

API keys are read from the environment (`OPENAI_API_KEY`, `OPENAI_BASE_URL` for a
local OpenAI-compatible server) or a gitignored `.env`. None is needed for the classical
policies.

`data/`, `results/`, `logs/`, `models/` and cluster job scripts are gitignored — the
first two by size, the last because paths, accounts and partitions are specific to one
machine.

---

## Anonymity

This source tree accompanies a double-blind submission. It carries no author names,
institutions, repository URLs or local filesystem paths, and the companion rule-set work
is referred to descriptively rather than by citation. Keep it that way when editing: the
easiest way to break it is a commit message, a git remote, or a path in a docstring.
