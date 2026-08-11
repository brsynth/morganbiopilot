"""Render solved pathways as a standalone HTML page.

Reads the JSON written by `multi_step.routes.save_routes` and produces one
self-contained file: molecules drawn with RDKit and inlined as SVG, no server, no
CDN, no JavaScript library. It opens from `file://` and survives being emailed.

    python -m morganbiopilot.paper_results.visualize_routes
    python -m morganbiopilot.paper_results.visualize_routes --target vanillin
    python -m morganbiopilot.paper_results.visualize_routes --open

Layout follows the convention BioNavi-NP uses: the target on the left, precursors
to the right, so the eye travels the retrosynthetic direction while the arrows
point the way the chemistry actually runs.

Enzymatic and non-enzymatic steps are coloured differently, as in BioNavi-NP — but
here the distinction is not decorative. A step is "enzymatic" when its rule carries
a native EC number, and roughly a third of applicable rules do not (see
`core.ec`). Colour is therefore the honest way to show which parts of a route are
backed by a known enzyme and which are chemistry the search proposed without one.
"""

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from morganbiopilot.core.paths import RESULTS_DIR

RDLogger.DisableLog("rdApp.*")

TARGET_COLOUR = "#c2185b"
SINK_COLOUR = "#2e7d32"
INTERMEDIATE_COLOUR = "#1976d2"


def molecule_svg(smiles: str, size: int = 150) -> str:
    """Inline SVG for a molecule, or a placeholder when RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f'<div class="unparsed">{html.escape(smiles[:40])}</div>'
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _node(smiles: str, kind: str, label: str = "") -> str:
    colour = {"target": TARGET_COLOUR, "sink": SINK_COLOUR}.get(kind, INTERMEDIATE_COLOUR)
    caption = html.escape(label or smiles)
    return (
        f'<div class="mol {kind}" style="border-color:{colour}" title="{html.escape(smiles)}">'
        f"{molecule_svg(smiles)}"
        f'<div class="cap">{caption[:38]}</div></div>'
    )


def _amount(participant: dict) -> str:
    """'2 H(+)' — the stoichiometric coefficient, omitted when it is 1."""
    coefficient = participant.get("n", 1)
    prefix = "" if coefficient == 1 else f"{coefficient:g} "
    return prefix + participant.get("name", "")


def _reagent(participant: dict) -> str:
    """A substrate the route does not have to make: carrier, or chassis metabolite.

    Rendered small and terminal. The colour is the load-bearing part — a
    substrate that is neither a carrier nor in the sink is something the route
    silently assumes it can obtain, and that is where formally valid routes turn
    out to be biologically impossible.
    """
    kind = "carrier" if participant["carrier"] else (
        "have" if participant["in_sink"] or participant.get("supplied") else "missing")
    amount = "" if participant["n"] == 1 else f'{participant["n"]:g} '
    smiles = participant.get("smiles") or ""
    drawing = molecule_svg(smiles, size=72) if smiles and kind != "carrier" else ""
    return (
        f'<div class="branch"><div class="reagent {kind}" '
        f'title="{html.escape(smiles)}">{drawing}'
        f'<div class="rname">{html.escape(amount + participant["name"][:26])}</div>'
        "</div></div>"
    )


def _subtree(smiles: str, by_product: dict, leaves: dict, target: str,
             seen: frozenset) -> str:
    """Render one molecule and, recursively, everything it is made from.

    The layout is a tree growing rightward from the target, which is what a route
    is: read left to right you are walking backwards through the synthesis, and
    every leaf on the right is something the chassis already provides. The earlier
    step-per-row layout showed each reaction in isolation, so every intermediate
    appeared twice and the reader had to reassemble the chain.
    """
    kind = "target" if smiles == target else ("sink" if smiles in leaves else "intermediate")
    node = _node(smiles, kind, leaves.get(smiles, ""))

    step = by_product.get(smiles)
    if step is None or smiles in seen:
        return f'<div class="branch">{node}</div>'

    seen = seen | {smiles}
    enzymatic = bool(step.get("ec_numbers"))
    ec_text = ", ".join(step["ec_numbers"]) if enzymatic else "no EC"

    extras = ""
    complete = step.get("complete")
    if complete:
        released = " + ".join(_amount(p) for p in step.get("full_products", []))
        extras = (
            f'<div class="mnxr">{html.escape(complete["mnxr_id"])}</div>'
            + (f'<div class="eqn">releases {html.escape(released[:60])}</div>'
               if released else "")
        )

    rxn = (
        f'<div class="{"rxn enzymatic" if enzymatic else "rxn nonenzymatic"}" '
        f'title="rule {step["rule_idx"]}">'
        f'<div class="arr">&larr;</div>'
        f'<div class="ec">{html.escape(ec_text)}</div>'
        + extras
        + "</div>"
    )

    # Branch on every substrate the complete reaction consumes, not just the
    # mono-component precursor the search reasoned over. A step that needs
    # 4-nitrophenyl sulfate alongside its main precursor is a different claim from
    # one that needs only the precursor, and the tree has to show it.
    #
    # Recursing requires a step that actually makes the substrate. Without the
    # `in by_product` test, a co-substrate nothing supplies was drawn as an
    # ordinary blue intermediate with no children — visually identical to a
    # solved branch, so a route missing a reagent looked complete. Anything the
    # route neither makes nor gets from the chassis now terminates in red.
    substrates = step.get("full_substrates")
    if substrates:
        children = "".join(
            _subtree(s["smiles"], by_product, leaves, target, seen)
            if (s["smiles"] and not s["carrier"] and not s["in_sink"]
                and s["smiles"] in by_product)
            else _reagent(s)
            for s in substrates
        )
    else:
        children = "".join(
            _subtree(p, by_product, leaves, target, seen) for p in step["precursors"]
        )
    return (f'<div class="branch">{node}{rxn}'
            f'<div class="children">{children}</div></div>')


def render_route(route: dict, index: int, total: int) -> str:
    """One route, as a tree: target on the left, chassis metabolites on the right."""
    leaves = {leaf["smiles"]: leaf["label"] for leaf in route.get("leaves", [])}
    target = route["target"]
    by_product = {step["substrate"]: step for step in route["steps"]}

    head = (
        f'<div class="route-head">Route {index + 1} of {total} &middot; '
        f'{route["n_steps"]} steps &middot; EC coverage '
        f'{100 * route.get("ec_coverage", 0):.0f}%'
        + (f' &middot; cofactors: {", ".join(route["distinct_cofactors"])}'
           if route.get("distinct_cofactors") else "")
        + "</div>"
    )
    tree = _subtree(target, by_product, leaves, target, frozenset())
    return f'<div class="route" id="route-{index}">{head}<div class="tree">{tree}</div></div>'


STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px;
     background:#fafafa;color:#222}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#666;font-size:13px;margin-bottom:18px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin-bottom:22px;
        padding:10px 14px;background:#fff;border:1px solid #e0e0e0;border-radius:6px}
.legend span{display:flex;align-items:center;gap:6px}
.swatch{width:13px;height:13px;border-radius:3px;border:2px solid}
.run{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px;
     margin-bottom:22px;overflow-x:auto}
.run-head{font-size:14px;font-weight:600;margin-bottom:2px}
.run-meta{font-size:12px;color:#777;margin-bottom:12px;font-family:ui-monospace,monospace}
.route{border-top:1px solid #eee;padding-top:12px;margin-top:12px}
.route-head{font-size:12px;color:#555;margin-bottom:10px}
/* Tree growing rightward: each branch is [molecule][reaction][its precursors],
   and precursors stack vertically so an AND-step reads as one bracket. */
.tree{display:inline-block}
.branch{display:flex;align-items:center;gap:8px}
.children{display:flex;flex-direction:column;gap:8px;
          border-left:2px solid #d8d8d8;padding-left:8px;margin-left:2px}
.children > .branch:only-child{border-left:none}
.mol{border:2px solid;border-radius:6px;padding:4px;background:#fff;text-align:center}
.mol svg{display:block}
.cap{font-size:10px;color:#555;max-width:150px;overflow:hidden;
     text-overflow:ellipsis;white-space:nowrap}
.unparsed{font-family:ui-monospace,monospace;font-size:10px;padding:20px;color:#b71c1c}
.rxn{min-width:104px;text-align:center;border-radius:5px;padding:6px 8px;font-size:11px}
.enzymatic{background:#b2dfdb;border:1px solid #4db6ac}
.nonenzymatic{background:#fff3c4;border:1px solid #ffd54f}
.arr{font-size:19px;line-height:1}
.ec{font-family:ui-monospace,monospace;font-size:10px}
.cof{font-size:10px;color:#666;margin-top:2px}
.rxn{max-width:250px}
.mnxr{font-family:ui-monospace,monospace;font-size:9px;color:#555;margin-top:4px}
.eqn{font-size:9px;color:#666;margin-top:3px;line-height:1.3;
     overflow:hidden;text-overflow:ellipsis}
.chips{display:flex;flex-wrap:wrap;gap:3px;justify-content:center;margin-top:4px}
.chip{font-size:9px;padding:1px 5px;border-radius:8px;border:1px solid}
.chip.have{background:#e8f5e9;border-color:#66bb6a;color:#1b5e20}
.chip.missing{background:#ffebee;border-color:#ef5350;color:#b71c1c}
.reagent{border:2px dashed;border-radius:6px;padding:3px 6px;text-align:center;
         font-size:10px;background:#fff}
.reagent.carrier{border-color:#bdbdbd;color:#757575}
.reagent.have{border-color:#66bb6a;color:#1b5e20}
.reagent.missing{border-color:#ef5350;color:#b71c1c;font-weight:600}
.rname{max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.warn{font-size:9px;font-weight:600;color:#b71c1c;margin-top:3px}
@media (prefers-color-scheme:dark){
  body{background:#181818;color:#e8e8e8}
  .run,.legend,.mol{background:#242424;border-color:#3a3a3a}
  .children{border-left-color:#444}
  .cap,.run-meta,.route-head{color:#aaa}
  .mol svg{filter:invert(0.92) hue-rotate(180deg)}
  .enzymatic{background:#26514c;border-color:#3d8b82}
  .nonenzymatic{background:#4a4326;border-color:#8b7a3d}
}
"""


def build_page(files: List[Path]) -> str:
    sections = []
    for path in sorted(files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta, routes = payload.get("meta", {}), payload.get("routes", [])
        if not routes:
            continue
        meta_line = " &middot; ".join(
            f"{k}={html.escape(str(v))}" for k, v in meta.items() if k != "target"
        )
        body = "\n".join(render_route(r, i, len(routes)) for i, r in enumerate(routes))
        sections.append(
            f'<div class="run"><div class="run-head">{html.escape(meta.get("target", path.stem))}'
            f'</div><div class="run-meta">{meta_line}</div>{body}</div>'
        )

    legend = (
        f'<span><i class="swatch" style="border-color:{TARGET_COLOUR}"></i>target</span>'
        f'<span><i class="swatch" style="border-color:{SINK_COLOUR}"></i>chassis metabolite</span>'
        f'<span><i class="swatch" style="border-color:{INTERMEDIATE_COLOUR}"></i>intermediate</span>'
        '<span><i class="swatch enzymatic" style="border-color:#4db6ac"></i>step with a known EC</span>'
        '<span><i class="swatch nonenzymatic" style="border-color:#ffd54f"></i>step with no EC</span>'
        '<span><i class="chip have">co-substrate in chassis</i></span>'
        '<span><i class="chip missing">co-substrate NOT in chassis</i></span>'
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MorganBioPilot — pathways</title>"
        f"<style>{STYLE}</style></head><body>"
        "<h1>Retrobiosynthetic pathways</h1>"
        "<div class='sub'>Read each row right to left for the retrosynthetic "
        "direction; arrows point the way the chemistry runs.</div>"
        f"<div class='legend'>{legend}</div>"
        + "\n".join(sections)
        + "</body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--complete", action="store_true",
                        help="read results/complete_routes (full reactions, "
                             "co-substrates, direction audit)")
    parser.add_argument("--routes-dir", default="")
    parser.add_argument("--target", default="", help="only this target")
    parser.add_argument("--policy", default="", help="substring filter on policy")
    parser.add_argument("--out", default="", help="output HTML path")
    parser.add_argument("--open", action="store_true", help="open in a browser")
    args = parser.parse_args()

    routes_dir = Path(args.routes_dir) if args.routes_dir else (
        RESULTS_DIR / ("complete_routes" if args.complete else "routes"))
    files = sorted(routes_dir.glob("*.json"))
    if args.target:
        files = [f for f in files if f.name.startswith(f"{args.target}__")]
    if args.policy:
        files = [f for f in files if args.policy in f.name]

    if not files:
        print(f"No route files in {routes_dir}. Run compare_policies first "
              "(routes are written for every solved run).", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else routes_dir / "pathways.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(files), encoding="utf-8")
    print(f"{len(files)} run(s) rendered -> {out}")

    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
