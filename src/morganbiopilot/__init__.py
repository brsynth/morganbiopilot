"""MorganBioPilot: agentic retrobiosynthesis on a deterministic ECFP expansion engine."""

import sys
import types

__version__ = "0.1"


def _stub_ipython_display() -> None:
    """Keep IPython out of the process, because importing it can crash RDKit.

    On the IFB cluster, `from rdkit.Chem import Draw` followed by `import IPython`
    segfaults inside expat while prompt_toolkit parses an HTML string at import
    time. Each import is fine alone; only the pair kills the interpreter. It is a
    collision between libraries the two bundle, not a bug in either project, and
    it took a four-step bisection to find because the traceback points at XML
    parsing, several layers away from either culprit.

    We pull in RDKit's Draw only by accident: `morganrxn.core.reaction_utils`
    provides `apply_reaction`, which the expansion engine needs, and that module
    imports `core.visualization` for a `plot_reaction` helper called once, in a
    debugging branch we never take. `visualization` in turn imports
    `IPython.display`. The whole crash therefore sits behind a Jupyter
    convenience that is useless in a batch job.

    morganrxn is under review elsewhere and must not be modified, so the fix goes
    here: a minimal `IPython.display` is placed in `sys.modules` before morganrxn
    can import the real one. The stubbed functions raise if anyone actually calls
    them, which is better than silently drawing nothing.

    Guarded on IPython being absent from `sys.modules`, so a notebook -- where the
    kernel has already imported it -- is untouched and plotting still works there.
    """
    if "IPython" in sys.modules:
        return

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "IPython.display is stubbed out by morganbiopilot: importing IPython "
            "alongside rdkit.Chem.Draw segfaults on some platforms. Plotting "
            "helpers are unavailable in this process; call them from a notebook, "
            "where the real IPython is already loaded and this stub is skipped."
        )

    display = types.ModuleType("IPython.display")
    display.display = _unavailable
    display.SVG = _unavailable
    display.HTML = _unavailable

    ipython = types.ModuleType("IPython")
    ipython.display = display

    sys.modules["IPython"] = ipython
    sys.modules["IPython.display"] = display


_stub_ipython_display()
