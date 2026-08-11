"""Multi-step search: AND-OR graph, status propagation, sink test, policies.

The only place a search policy lives — including the LLM agent. The expansion
primitive it calls must stay in `one_step` so that every baseline and every model
shares the exact same engine.
"""
