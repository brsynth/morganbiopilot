"""LLM agents: the only non-deterministic part of this project.

The package boundary is the point. Everything in `core`, `one_step`, and
`multi_step` is deterministic and reproducible; everything that calls a model
lives here. That makes the paper's central claim ("all systems share the same
engine, only the policy varies") a structural property of the tree rather than a
promise in the text — you can point at the directory listing and say which half
is which.

Plural, because the project note keeps a second agent in reserve: an enzymatic
feasibility judge (section 6, lever 3; section 11). Nothing here should assume a
single agent.

Modules:

- `state`  — renders a search state into something a model can read, including
             the frontier truncation. Methodologically load-bearing: see below.
- `tools`  — the tool surface offered to the agent. Toggling it is the
             tooled/untooled ablation of section 8.

An agent is still just a `multi_step.policy.Policy`: one method, `select`. If a
class here grows a second responsibility, that logic belongs in `multi_step`.
"""
