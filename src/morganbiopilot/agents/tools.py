"""The tool surface offered to an agent — and the grounding ablation.

Section 8 of the project note compares "the same agent deprived of ECFP distance
and EC hints" against the tooled agent. That comparison is this object: a
`ToolSurface` either carries the two grounding signals or it does not, and nothing
else about the agent changes.

Information injection, not tool calling
--------------------------------------
The signals are computed by the environment and rendered into the state the model
reads, rather than exposed as callable tools it must invoke. Three reasons:

- **Cost.** A callable tool turns one API call per decision into several. Over
  20 targets x a fixed expansion budget x a model grid, that multiplies the bill
  by the number of tool round-trips for information the environment already has.
- **A cleaner ablation.** Both conditions issue exactly one call per decision with
  the same prompt structure; only the presence of two fields differs. With
  callable tools, the untooled condition would also differ in call count and
  conversation shape, confounding the comparison.
- **No information asymmetry.** Every candidate carries its annotation, so the
  agent never has to spend a call to learn about a molecule it might ignore.

True tool calling is a defensible variant — it lets the agent query molecules
outside the shown frontier — but it is a different experiment and must be reported
as one.
"""

from dataclasses import dataclass
from typing import Optional

from morganbiopilot.core.ec import RuleEC
from morganbiopilot.multi_step.heuristics import SinkCloseness
from morganbiopilot.multi_step.plausibility import RoutePlausibility


@dataclass(frozen=True)
class ToolSurface:
    """Which grounding signals the agent receives. `None` means withheld."""

    closeness: Optional[SinkCloseness] = None
    rule_ec: Optional[RuleEC] = None
    plausibility: Optional[RoutePlausibility] = None

    @property
    def name(self) -> str:
        """Short label for tables and log filenames."""
        bits = []
        if self.closeness is not None:
            bits.append("ecfp")
        if self.rule_ec is not None:
            bits.append("ec")
        if self.plausibility is not None:
            bits.append("enz")
        return "+".join(bits) if bits else "none"

    @property
    def is_tooled(self) -> bool:
        return any(s is not None for s in
                   (self.closeness, self.rule_ec, self.plausibility))


def tooled(radius: int, rule_ec: RuleEC,
           plausibility: Optional[RoutePlausibility] = None) -> ToolSurface:
    """Full grounding: ECFP distance to the sink, plus native EC classes.

    Route plausibility is opt-in rather than part of `tooled`, so the arm that has
    carried every agentic result so far keeps meaning exactly what it meant. Adding a
    column silently to the established condition would make the new runs incomparable
    with the ones already in the tables.
    """
    return ToolSurface(closeness=SinkCloseness(radius), rule_ec=rule_ec,
                       plausibility=plausibility)


def untooled() -> ToolSurface:
    """No grounding. The agent sees SMILES and depth only."""
    return ToolSurface()
