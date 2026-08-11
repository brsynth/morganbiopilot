"""Expansion of a single node: the deterministic primitive of the whole project.

Three stages, following Meyer et al. 2026 and the project note (section 4):

1. **Prefilter** the applicable rules with the reaction-center ECFP — O(d),
   necessary condition only (`prefilter.py`).
2. **Validate at graph level** with RDKit `RunReactants`, through
   `morganrxn.core.reaction_utils.apply_reaction`. This is what rules out the
   candidates the fingerprint relaxation lets through.
3. **Emit precursor sets**: sanitized (canonical, stereo-free), split into
   mono-component molecules, cofactors stripped.

Two conventions that shape this module
--------------------------------------
**Mono-component.** Every node is a single molecule. A rule application yields a
set of molecules, each of which becomes its own node — an AND-node in the search
graph, since all of them are needed.

**No retro/forward direction.** A template is a directed graph rewrite, and the
rule set contains both orientations of every reaction (``_L2R`` / ``_R2L`` in
``reaction_id``). Nothing here is "retro": we apply templates to a molecule and
obtain the molecules on the other side. Direction is a property of how the search
walks the graph, never of a rule — so this module says *neighbours*, not
*precursors of a retro-step*.

No search logic here, and no LLM. Every baseline and every model calls this same
function, which is what makes the paper's comparison meaningful.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from morganrxn.core.reaction_utils import apply_reaction

from morganbiopilot.core.building_blocks import strip_cofactors
from morganbiopilot.core.chem import mol_ecfp, sanitize, split_components
from morganbiopilot.one_step.prefilter import ReactionCenterPrefilter


@dataclass(frozen=True)
class Neighbour:
    """One validated rule application: the molecule set on the other side.

    `molecules` is mono-component and cofactor-free. All of them are required,
    so this is an AND-node.
    """

    rule_idx: int
    molecules: tuple                     # sanitized, mono-component SMILES
    template: str = ""
    reaction_id: str = ""
    cofactors_removed: tuple = ()        # kept for provenance, not for search
    ec_numbers: tuple = ()               # native EC of the rule; empty ~half the time

    @property
    def ec_classes(self) -> tuple:
        """Level-1 EC classes, deduplicated. Coarse but far less ambiguous.

        Level-1 ambiguity is ~3% of EC-carrying rules against ~9% at level 3, so
        this is the granularity that survives aggregation over the reactions
        collapsed into a rule.
        """
        out = {e.split(".")[0] for e in self.ec_numbers if e and e.split(".")[0] != "-"}
        return tuple(sorted(out))


@dataclass
class ExpansionReport:
    """Bookkeeping for one node expansion. Cheap to log, and worth logging."""

    target: str
    n_prefiltered: int = 0      # rules surviving the O(d) criterion
    n_graph_valid: int = 0      # of those, applicable at graph level
    n_apply_errors: int = 0
    n_empty_after_cofactors: int = 0     # applications yielding only cofactors
    n_dropped_no_ec: int = 0             # graph-valid but no enzyme behind the rule
    neighbours: List[Neighbour] = field(default_factory=list)

    @property
    def prefilter_precision(self) -> float:
        """Share of prefiltered rules that really apply. Quantifies the relaxation."""
        return self.n_graph_valid / self.n_prefiltered if self.n_prefiltered else float("nan")

    @property
    def molecules(self) -> set:
        """All distinct molecules reachable from this node in one step."""
        return {m for n in self.neighbours for m in n.molecules}


def expand(
    target: str,
    rules,
    prefilter: ReactionCenterPrefilter,
    max_rules: Optional[int] = None,
    keep_cofactors: bool = False,
    rule_ec=None,
    require_ec: bool = False,
) -> ExpansionReport:
    """Expand one mono-component molecule into its validated neighbour sets.

    `rules` is a `RuleSet` (see `core.rules`); `prefilter` must be built from the
    same rule set, otherwise indices refer to different rules. The ECFP radius is
    read from the rule set, so the two cannot disagree.

    `rule_ec` is an optional `core.ec.RuleEC` over the *same* rule set. When given,
    every neighbour carries its rule's native EC — the grounding signal of the
    project note (section 6). It is absent for ~51% of rules, so an empty
    `ec_numbers` means "not annotated", never "not enzymatic".

    `require_ec=True` drops rules with no EC annotation: the **enzymatic reality
    filter**. This is a property of the environment, not of any policy — every
    baseline and every model then explores the same graph, which is what makes the
    paper's comparison meaningful. Note the consequence for the tooled/untooled
    ablation: with this on, the "agent without EC" still searches an EC-filtered
    space, so the ablation measures *EC shown to the agent*, not *EC used at all*.
    Say which one the paper reports.

    `max_rules` caps graph-level validation to the first candidates by rule index.
    The cap is arbitrary by construction — there is no rule ranking in this project
    — so anything but None makes the expansion incomplete in a way that must be
    reported. Leave it None for exact behaviour.
    """
    target = sanitize(target)
    report = ExpansionReport(target=target)

    if len(split_components(target)) > 1:
        raise ValueError(f"expand() works mono-component; got {target!r}")

    ecfp = np.asarray(mol_ecfp(target, rules.radius), dtype=np.int32)
    _child_vecs, rule_idxs = prefilter.one_step(ecfp)
    report.n_prefiltered = int(rule_idxs.size)

    if max_rules is not None and rule_idxs.size > max_rules:
        # No ranking: rule indices are already sorted, so this is a deterministic
        # but arbitrary truncation. Report it whenever it is used.
        rule_idxs = rule_idxs[:max_rules]

    if require_ec and rule_ec is None:
        raise ValueError("require_ec=True needs rule_ec (see core.ec.annotate_rules)")

    for rule_idx in rule_idxs:
        rule_idx = int(rule_idx)

        if require_ec and not rule_ec.ec[rule_idx]:
            # Enzymatic reality filter: a rule with no EC behind it has no known
            # enzyme catalysing it, so it is not a biosynthetic step. Dropped here,
            # in the engine, so every policy explores the same graph.
            report.n_dropped_no_ec += 1
            continue

        template = str(rules.template_reaction[rule_idx])

        try:
            products = apply_reaction(template, target)
        except Exception:
            report.n_apply_errors += 1
            continue

        if not products:
            # Prefiltered but not graph-applicable: exactly the false positives the
            # necessary-condition relaxation is expected to produce.
            continue

        report.n_graph_valid += 1

        for product in products:
            molecules = split_components(product)
            if not molecules:
                continue

            if keep_cofactors:
                kept, removed = molecules, []
            else:
                kept = strip_cofactors(molecules)
                removed = [m for m in molecules if m not in kept]

            if not kept:
                # Everything on this side is a cofactor: nothing left to search.
                report.n_empty_after_cofactors += 1
                continue

            report.neighbours.append(
                Neighbour(
                    rule_idx=rule_idx,
                    molecules=tuple(sorted(kept)),
                    template=template,
                    reaction_id=str(rules.reaction_id[rule_idx]),
                    cofactors_removed=tuple(sorted(removed)),
                    ec_numbers=rule_ec.ec[rule_idx] if rule_ec is not None else (),
                )
            )

    return report
