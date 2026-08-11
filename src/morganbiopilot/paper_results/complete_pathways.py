"""Rebuild complete pathways from the mono-component routes.

The search works mono-component: a rule records one transformation skeleton and
drops the co-substrates, co-products and stoichiometry of the MetaNetX reaction it
came from. A solved route is therefore a skeleton, and cannot be judged as
chemistry. This script resolves every step back to its full reaction and writes
the completed pathways next to the originals.

    python -m morganbiopilot.paper_results.complete_pathways
    python -m morganbiopilot.paper_results.complete_pathways --target vanillin --print

Reads  results/routes/*.json          (mono-component, from compare_policies)
Writes results/complete_routes/*.json (full equations, per step)

Completion is post-processing, by design
----------------------------------------
The search stays mono-substrate over the AND-OR graph and never reads these files.
Pushing co-substrates into `expand()` would make the solve rate directly
interpretable, but it would also change what a node is, add a `reac_prop` lookup
to every expansion, and invalidate every run already recorded. The mono-component
search plus post-hoc completion is what RetroPath2.0 and RetroRules do.

The consequence has to be reported rather than hidden: a route the search calls
solved may consume reagents nothing supplies. `fully_satisfied` below is that
check. It leaves the comparison between policies valid — every policy is
optimistic in the same way — while making the absolute numbers honest.

On direction — and a metric that was removed
--------------------------------------------
MetaNetX writes equations with ``=``: the side a compound appears on is a
typographic convention, not a claim that the reaction runs that way. Both
directions occur in an organism, which is why the rule set is symmetrised into
66,725 ``_L2R`` plus 66,725 ``_R2L`` entries — an enumeration device to obtain
both directed templates, not a directionality assertion.

An earlier version of this script counted steps "running against the recorded
direction" and reported 58%. That number measured which side of the ``=`` a
compound had been typed on. It is deleted rather than kept with a caveat, because
a plausible-looking metric that measures nothing is worse than none.

Equations are still *oriented* here, to the direction the route uses them in, so
the printed arrow agrees with the arrow drawn above it. That is presentation, not
an audit.

What a genuine reversibility check would need is thermodynamic data — standard
Gibbs free energies per reaction, as eQuilibrator provides — which is not in
`data/`. Until then the honest position is that feasibility is not modelled, and
the paper should say so, the way RetroAgent states that yield, selectivity and
conditions are outside its validity notion.
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

from morganbiopilot.core.building_blocks import (
    building_block_entries, inchikey_skeleton, is_building_block,
)
from morganbiopilot.core.chem import sanitize
from morganbiopilot.core.paths import RESULTS_DIR
from morganbiopilot.core.reactions import CompleteReaction, complete_reactions

# Ubiquitous carriers. Their presence in an equation is expected and says nothing;
# anything else on the substrate side is a co-substrate the route must supply.
CARRIERS = {
    "H(+)", "H2O", "O2", "CO2", "NAD(+)", "NADH", "NADP(+)", "NADPH",
    "ATP", "ADP", "AMP", "phosphate", "diphosphate", "CoA", "FAD", "FADH2",
    # MetaNetX also writes the redox pair without committing to phosphorylation.
    # Six of 36 "unmet co-substrates" in the first full audit were this spelling
    # alone -- a naming gap reported as missing chemistry.
    "NAD(P)+", "NAD(P)H",
    # Flavins are prosthetic groups of the enzyme, not reagents a pathway sources:
    # 127 of 680 unmet co-substrates in the full grid were one of these two.
    "a reduced flavin", "an oxidized flavin", "FMN", "FMNH2",
    "a reduced ferredoxin [iron-sulfur] cluster",
    "an oxidized ferredoxin [iron-sulfur] cluster",
}

# Names MetaNetX uses for an unspecified participant rather than a compound: a
# generic acceptor, a placeholder in a template reaction. They cannot be supplied
# because they do not denote anything, so counting them as missing reagents
# inflates the audit. "A" appeared 7 times in the first full run.
GENERIC_NAMES = {
    "A", "B", "R", "X", "Y", "an alcohol", "an aldehyde", "a ketone",
    # "Acceptor" and "Donor" are how MetaNetX writes a half-reaction whose partner
    # is left unspecified. 190 of 680 unmet co-substrates in the full grid were
    # "Acceptor" alone -- by far the largest single entry, and not a compound.
    "Acceptor", "Donor", "acceptor", "donor",
    "an electron acceptor", "an electron donor",
    "Oxidized electron acceptor", "Reduced electron acceptor",
}

# How many source reactions to consider per rule before choosing one. A rule can
# collapse thousands of MetaNetX reactions; taking the first (the old behaviour)
# attached whichever happened to come first in the file, which for a
# sulfotransferase rule meant an L-tyrosine methyl ester reaction on a route about
# 4-vinylphenol. The route's own molecules decide which candidate is right, so
# enough of them have to be resolved for the right one to be among them.
CANDIDATES_PER_RULE = 50

# Template matching against thousands of participants logs a valence warning for
# every odd MetaNetX structure; the parse failures that matter are counted instead.
RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=1)
def _labels() -> Dict[str, str]:
    """Chassis metabolite names, so a substituted role reads as a compound."""
    return {smiles: label for smiles, label in building_block_entries()}


@lru_cache(maxsize=100_000)
def _skeleton(smi: str) -> str:
    """Connectivity hash of a SMILES — the same key the sink is matched on.

    Route SMILES and `chem_prop` SMILES disagree about protonation: the search
    carries phenolates (`O=Cc1ccc([O-])cc1`) where MetaNetX stores the neutral
    acid (`O=Cc1ccc(O)cc1`). Compared as strings they never match, which broke
    both the reaction choice and the link between consecutive steps. The first
    InChIKey block ignores charge and tautomer, so it matches what a chemist
    would call the same compound.
    """
    return inchikey_skeleton(sanitize(smi)) or "" if smi else ""


@lru_cache(maxsize=100_000)
def _mol(smi: str):
    return Chem.MolFromSmiles(smi) if smi else None


def _formula(smi: str) -> Optional[Dict[str, int]]:
    """Element tally of one molecule, hydrogens included."""
    molecule = _mol(smi) if smi else None
    if molecule is None:
        return None
    counts: Dict[str, int] = {}
    for atom in molecule.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        if atom.GetTotalNumHs():
            counts["H"] = counts.get("H", 0) + atom.GetTotalNumHs()
    return counts


def _delta(before: str, after: str) -> Optional[Dict[str, int]]:
    """What the transformation adds and removes, element by element."""
    start, end = _formula(before), _formula(after)
    if start is None or end is None:
        return None
    return {e: end.get(e, 0) - start.get(e, 0)
            for e in set(start) | set(end) if end.get(e, 0) != start.get(e, 0)}


def _pick_reaction(step: dict, candidates: List[CompleteReaction]):
    """Choose the source reaction behind this step, its orientation, and its roles.

    Most steps are *predicted*, not catalogued: a template extracted from a
    sulfotransferase acting on L-tyrosine methyl ester gets applied to
    4-vinylphenol, and no MetaNetX reaction describes that. So requiring the
    route's own molecules to appear in the source reaction rejected 10 of 43
    steps. What transfers is the rest of the equation — the cosubstrates and
    coproducts the template implies — with the source reaction's transformed
    compound standing in for the route's.

    Returns `(reaction, agrees, consumed_role, released_role, catalogued)`, the
    roles being indices into the oriented sides, or `(None, ...)` when the
    template cannot be read at all.
    """
    want_made = _skeleton(step["substrate"])
    want_used = _skeleton(step["precursors"][0]) if step["precursors"] else ""
    route_delta = _delta(step["precursors"][0], step["substrate"]) \
        if step["precursors"] else None

    best, best_score = None, 0
    for reaction in candidates:
        for agrees in (True, False):
            consumed = reaction.substrates if agrees else reaction.products
            released = reaction.products if agrees else reaction.substrates

            for i, taken in enumerate(consumed):
                if taken.name in CARRIERS or not taken.smiles:
                    continue
                for j, given in enumerate(released):
                    if given.name in CARRIERS or not given.smiles:
                        continue
                    exact = (_skeleton(taken.smiles) == want_used
                             and _skeleton(given.smiles) == want_made)
                    # The decisive test. The pair standing in for the route's own
                    # pair must undergo the *same* transformation, element for
                    # element. Matching each side independently against the
                    # template let a methyltransferase's donor be mistaken for its
                    # acceptor -- protocatechualdehyde was aligned with
                    # S-adenosyl-L-methionine, so the completed equation acquired a
                    # methyl group from nowhere and dragged caffeoyl-CoA in as a
                    # co-substrate the route was then reported unable to supply.
                    # Requiring equal deltas also makes balance hold by
                    # construction: substituting an equimolar pair with an
                    # identical formula change cannot unbalance an equation that
                    # was balanced.
                    same_change = (route_delta is not None
                                   and _delta(taken.smiles, given.smiles) == route_delta)
                    if not (exact or same_change):
                        continue
                    score = (1000 * exact + 100 * same_change) * 100 \
                        - len(consumed) - len(released)
                    if score > best_score:
                        best = (reaction, agrees, i, j, exact)
                        best_score = score
    return best if best else (None, True, -1, -1, False)


def _atom_counts(participants: List[dict]) -> Tuple[Dict[str, float], bool]:
    """Element tally of one side of an equation, with stoichiometry.

    Returns `(counts, complete)`; `complete` is False when any participant has no
    structure in `chem_prop` (an oxidised azurin, a generic
    "a 5-methyltetrahydrofolate"), in which case the tally cannot be compared.
    """
    counts: Dict[str, float] = {}
    for participant in participants:
        molecule = _mol(participant["smiles"]) if participant["smiles"] else None
        if molecule is None:
            return counts, False
        n = participant["n"]
        for atom in molecule.GetAtoms():
            counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + n
            if atom.GetTotalNumHs():
                counts["H"] = counts.get("H", 0) + atom.GetTotalNumHs() * n
    return counts, True


def _check_balance(step: dict) -> str:
    """Does *our* completion conserve atoms? "yes", "no", or "unknown".

    This catches a bad completion. When a step is novel, co-substrates and
    co-products are transferred from a source reaction about a *different*
    substrate, which is sound only when they do not depend on it. On a
    sulfotransferase the donor/acceptor pair transfers cleanly; elsewhere it does
    not, and the missing atoms are the sibling split that the mono-component rule
    set dropped.

    Two things must be excluded or the check accuses us of other people's
    problems. First, 47% of MetaNetX reactions are not certified balanced, and an
    equation that never balanced cannot be evidence that completion broke it --
    those are "unknown", not "no". Second, hydrogens: protonation is a convention
    (MetaNetX carries explicit H(+) participants) and our own substitution swaps a
    phenolate for a neutral phenol, so counting H would report our display choice
    as a chemical error. Heavy atoms only.
    """
    if step["complete"].get("is_balanced") != "B":
        return "unknown"
    left, left_ok = _atom_counts(step.get("full_substrates", []))
    right, right_ok = _atom_counts(step.get("full_products", []))
    if not (left_ok and right_ok):
        return "unknown"
    heavy = {element for element in set(left) | set(right) if element != "H"}
    return "yes" if all(left.get(e, 0) == right.get(e, 0) for e in heavy) else "no"


def _check_satisfaction(steps: List[dict]) -> List[dict]:
    """Mark every co-substrate as supplied or not, and flag the unmet ones.

    A co-substrate counts as supplied when it is a carrier, a chassis metabolite,
    or released by another step of the route. The last clause is deliberately not
    ordered by depth: a route is a DAG converging on the target, so anything it
    produces is produced before the target is. Ordering would only matter for a
    route that consumed a metabolite before making it, which the AND-OR graph
    cannot construct.

    Matching is by connectivity skeleton, falling back to the MetaNetX name for
    participants `chem_prop` gives no structure for ("Oxidized azurin"). Those can
    never be matched against a route SMILES, so they are always reported unmet —
    correctly: the chassis is not known to supply them either.
    """
    made = {_skeleton(p["smiles"]) for st in steps
            for p in st.get("full_products", []) if p["smiles"]} - {""}
    made_names = {p["name"] for st in steps for p in st.get("full_products", [])}
    # A step's own product is supplied by that step; without this a route that
    # makes its intermediate would still be told the intermediate is missing.
    made |= {_skeleton(st["substrate"]) for st in steps} - {""}

    for step in steps:
        for participant in step.get("co_substrates", []):
            supplied = (
                participant["in_sink"]
                or (participant["smiles"] and _skeleton(participant["smiles"]) in made)
                or (not participant["smiles"] and participant["name"] in made_names)
            )
            participant["supplied"] = bool(supplied)
    return [dict(p, depth=st["depth"]) for st in steps
            for p in st.get("co_substrates", []) if not p["supplied"]]


def complete_route(route: dict, resolved: Dict[str, List[CompleteReaction]]) -> dict:
    """Attach the full reaction of every step, oriented as the route uses it.

    `resolved` is passed in rather than looked up here: resolving reactions means
    a pass over a 772 MB file, and doing it per route re-read it once per route.
    """
    steps, unresolved, cosubstrates, predicted = [], 0, set(), 0
    for step in route["steps"]:
        entries: List[CompleteReaction] = resolved.get(step["reaction_id"], [])
        # Pick the candidate whose transformation matches this step, and where in
        # it the transformed compound sits, rather than taking whichever reaction
        # the file listed first.
        reaction, agrees, consumed_role, released_role, catalogued = \
            _pick_reaction(step, entries) if entries else (None, True, -1, -1, False)
        if reaction is None:
            unresolved += 1
            steps.append({**step, "complete": None})
            continue
        predicted += not catalogued

        # Orient the equation the way the route uses it, so the printed arrow
        # agrees with the arrow drawn above it. Both orientations are the same
        # reaction; this is layout, not chemistry.
        consumed = reaction.substrates if agrees else reaction.products
        released = reaction.products if agrees else reaction.substrates

        def describe(participants):
            out = []
            for p in participants:
                canonical = sanitize(p.smiles) if p.smiles else ""
                out.append({
                    "name": p.name or p.mnx_id,
                    "smiles": canonical,
                    "n": p.coefficient,
                    "carrier": p.name in CARRIERS,
                    "in_sink": bool(canonical) and is_building_block(canonical),
                })
            return out

        full_substrates = describe(consumed)
        full_products = describe(released)

        # Put the route's own molecules in the transformed roles. Without this the
        # completed equation described the reaction the template came from, so a
        # route about 4-vinylphenol was reported as consuming L-tyrosine methyl
        # ester 4-sulfate — counted as a co-substrate nothing supplies, when in
        # truth it is the analogue of the route's own precursor and is not
        # required at all.
        def substitute(participants: List[dict], index: int, smiles: str) -> None:
            if index < 0 or not smiles:
                return
            canonical = sanitize(smiles)
            stood_in_for = participants[index]["name"]
            # When the source participant is the same compound, its MetaNetX name
            # is the right name for it; only the SMILES needs to become the
            # route's, since the two differ in protonation. Substituting the name
            # too would print a bare SMILES where a name was available.
            same = _skeleton(participants[index]["smiles"]) == _skeleton(canonical)
            participants[index] = {
                "name": stood_in_for if same else _labels().get(canonical, canonical),
                "smiles": canonical,
                "n": participants[index]["n"],
                "carrier": False,
                "in_sink": is_building_block(canonical),
                "from_route": True,
                "template_analogue": "" if same else stood_in_for,
            }

        substitute(full_substrates, consumed_role,
                   step["precursors"][0] if step["precursors"] else "")
        substitute(full_products, released_role, step["substrate"])

        for participant in full_substrates:
            if not participant["carrier"] and participant["name"] not in GENERIC_NAMES:
                cosubstrates.add(participant["name"])

        def equation(left, right):
            def term(p):
                amount = "" if p["n"] == 1 else f"{p['n']:g} "
                return f"{amount}{p['name']}"
            return (" + ".join(term(p) for p in left) + " --> "
                    + " + ".join(term(p) for p in right))

        # `CompleteReaction.render()` always writes the equation left-to-right as
        # MetaNetX stored it, which contradicts `full_substrates` whenever the
        # route runs the reaction the other way — the reagent list said "consumes
        # Oxidized azurin" while the equation above it showed azurin as a product.
        # Both describe the same reaction; only one can sit next to the route's
        # arrow. Overwrite the string so the record is self-consistent.
        complete = reaction.to_dict()
        complete["equation_as_stored"] = complete["equation"]
        complete["equation"] = equation(full_substrates, full_products)

        steps.append({
            **step,
            "complete": complete,
            "oriented_as_written": agrees,
            # The completed step: every substrate the reaction consumes and every
            # product it releases, in the route's own direction. `precursors` keeps
            # the mono-component skeleton the search actually reasoned over.
            "full_substrates": full_substrates,
            "full_products": full_products,
            "co_substrates": [p for p in full_substrates
                              if not p["carrier"] and p["name"] not in GENERIC_NAMES],
        })

    unmet = _check_satisfaction(steps)
    for step in steps:
        if step.get("complete"):
            step["balanced"] = _check_balance(step)
    unbalanced = [s["depth"] for s in steps if s.get("balanced") == "no"]

    return {
        **route,
        "steps": steps,
        "n_unresolved": unresolved,
        "n_predicted": predicted,
        "co_substrates": sorted(cosubstrates),
        # The honest verdict on the route. The search called it solved on the
        # mono-component skeleton; this says whether the full reactions close --
        # every substrate accounted for AND every equation conserving atoms.
        "unmet_co_substrates": unmet,
        "unbalanced_depths": unbalanced,
        "fully_satisfied": not unmet and unresolved == 0 and not unbalanced,
    }


def render(route: dict) -> str:
    lines = [f"TARGET {route['target']}  ({route['n_steps']} steps)"]
    for step in route["steps"]:
        complete = step.get("complete")
        if not complete:
            lines.append(f"  [{step['depth']}] rule {step['rule_idx']}: "
                         "no source reaction resolved")
            continue
        lines.append(f"  [{step['depth']}] {complete['mnxr_id']} "
                     f"balanced={complete['is_balanced']}")
        lines.append(f"        {complete['equation']}")
    if route["co_substrates"]:
        lines.append(f"  co-substrates beyond carriers: {', '.join(route['co_substrates'])}")
    if route["fully_satisfied"]:
        lines.append("  FULLY SATISFIED: every substrate is a carrier, a chassis "
                     "metabolite, or made by the route")
    else:
        for participant in route["unmet_co_substrates"]:
            lines.append(f"  UNMET at depth {participant['depth']}: "
                         f"{participant['name']}"
                         f"{'  (no structure in chem_prop)' if not participant['smiles'] else ''}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--routes-dir", default=str(RESULTS_DIR / "routes"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "complete_routes"))
    parser.add_argument("--target", default="", help="only this target")
    parser.add_argument("--print", dest="do_print", action="store_true")
    args = parser.parse_args()

    routes_dir = Path(args.routes_dir)
    files = sorted(routes_dir.glob("*.json"))
    if args.target:
        files = [f for f in files if f.name.startswith(f"{args.target}__")]
    if not files:
        print(f"No route files in {routes_dir}.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One resolution pass for every step of every route in every file. Both
    # MetaNetX tables are read once; resolving per route re-read `chem_prop.tsv`
    # once per route.
    payloads = {path: json.loads(path.read_text(encoding="utf-8")) for path in files}
    all_fields = [step["reaction_id"]
                  for payload in payloads.values()
                  for route in payload.get("routes", [])
                  for step in route["steps"]]
    print(f"Resolving {len(set(all_fields))} distinct rules "
          f"(up to {CANDIDATES_PER_RULE} source reactions each)...")
    resolved = complete_reactions(all_fields, max_per_rule=CANDIDATES_PER_RULE)

    totals = {"routes": 0, "steps": 0, "unresolved": 0, "predicted": 0,
              "unmet": 0, "satisfied": 0, "unbalanced": 0}
    unmet_by_name: Dict[str, int] = {}
    per_policy: Dict[str, List[int]] = {}
    for path in files:
        payload = payloads[path]
        completed = [complete_route(r, resolved) for r in payload.get("routes", [])]
        policy = payload.get("meta", {}).get("policy", path.stem)
        for route in completed:
            totals["routes"] += 1
            totals["steps"] += route["n_steps"]
            totals["unresolved"] += route["n_unresolved"]
            totals["predicted"] += route["n_predicted"]
            totals["unbalanced"] += len(route["unbalanced_depths"])
            totals["unmet"] += len(route["unmet_co_substrates"])
            totals["satisfied"] += bool(route["fully_satisfied"])
            for participant in route["unmet_co_substrates"]:
                unmet_by_name[participant["name"]] = \
                    unmet_by_name.get(participant["name"], 0) + 1
            tally = per_policy.setdefault(policy, [0, 0])
            tally[0] += bool(route["fully_satisfied"])
            tally[1] += 1
        (out_dir / path.name).write_text(
            json.dumps({"meta": payload.get("meta", {}), "routes": completed}, indent=2),
            encoding="utf-8",
        )
        if args.do_print and completed:
            print(f"\n===== {path.stem} =====")
            print(render(completed[0]))

    print(f"\n{len(files)} file(s) -> {out_dir}")
    print(f"  routes completed          : {totals['routes']}")
    print(f"  steps                     : {totals['steps']}")
    print(f"  steps not completed       : {totals['unresolved']}")
    print(f"  predicted, not catalogued : {totals['predicted']}"
          f" of {totals['steps'] - totals['unresolved']} completed steps")
    # The metric that survives: whether the full reactions of a route close. It
    # does not depend on direction, which is why it means something.
    share = 100 * totals["satisfied"] / totals["routes"] if totals["routes"] else 0
    print(f"  fully satisfied routes    : {totals['satisfied']}/{totals['routes']}"
          f" ({share:.0f}%)")
    print(f"  unmet co-substrates       : {totals['unmet']}")
    print(f"  steps with unbalanced eqn : {totals['unbalanced']}")

    if unmet_by_name:
        print("\n  what the routes cannot supply:")
        for name, count in sorted(unmet_by_name.items(), key=lambda kv: -kv[1]):
            print(f"    {count:3d}x  {name}")

    if len(per_policy) > 1:
        print("\n  fully satisfied, by policy:")
        for policy, (ok, total) in sorted(per_policy.items()):
            print(f"    {policy:24s} {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
