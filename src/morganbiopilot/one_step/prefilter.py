"""Sparse O(d) reaction-center prefilter.

Criterion (Meyer et al. 2026, Eq. 8): a reaction is ECFP-applicable to a molecular
vector ``v`` when

    ECFP_rc,h(r) + v >= 0    coordinate-wise.

This is a **necessary condition only** — a relaxation of graph-level
applicability, since ECFPs do not preserve full molecular connectivity. Every
surviving candidate must still be validated at graph level (see `expand.py`).

Why not use ``morganrxn.core.reaction_utils.one_step`` directly
--------------------------------------------------------------
It evaluates the criterion densely over all d = 1024 coordinates:

    mask = np.all((v[None, :] + centers) >= 0, axis=1)

But reaction-center vectors are all <= 0 and extremely sparse — median 13 non-zero
coordinates out of 1024 at r2 (1.57% density), 28 at r5 (3.38%). Only those
coordinates can ever violate the criterion, so the dense form does 30-60x more
arithmetic than the test requires. Measured cost of the dense form: ~240 ms per
expansion at r2, ~470 ms at r5, which does not survive an expansion-budget
experiment repeated over 20 targets and a dozen configurations.

This module evaluates the same criterion over the non-zero coordinates only, and
keeps both matrices in CSR form to cut resident memory. `ReactionCenterPrefilter`
is verified to return bit-identical results to `one_step` (see
`paper_results`/tests).
"""

import numpy as np
from scipy.sparse import csr_matrix


class ReactionCenterPrefilter:
    """Applicability prefilter over a fixed rule set, at one ECFP radius.

    Semantics are identical to ``morganrxn.core.reaction_utils.one_step``,
    including deduplication of identical child fingerprints and the choice of a
    single representative rule index per distinct child vector.
    """

    def __init__(self, ecfp_reaction_centers, ecfp_reactions, radius=None):
        # The radius the vectors were built at. Carried so a caller that holds only the
        # prefilter can fingerprint a query consistently with it, instead of guessing --
        # `mol_ecfp(smi, wrong_radius)` produces a vector that silently matches nothing.
        self.radius = radius
        centers = np.asarray(ecfp_reaction_centers, dtype=np.int32)
        reactions = np.asarray(ecfp_reactions, dtype=np.int32)

        if centers.shape != reactions.shape:
            raise ValueError(f"shape mismatch: centers {centers.shape} vs reactions {reactions.shape}")
        if (centers > 0).any():
            # The criterion below assumes centers are non-positive (they encode
            # *required* environments as negative counts). If this ever fires, the
            # rule file was built with a different convention.
            raise ValueError("reaction-center vectors must be <= 0")

        self.n_rules, self.dim = centers.shape

        # Non-zero entries of the centers: rule index, coordinate, required count.
        rows, cols = np.nonzero(centers)
        self._rows = rows.astype(np.int32)
        self._cols = cols.astype(np.int32)
        self._required = (-centers[rows, cols]).astype(np.int32)  # > 0

        # Reaction vectors stay sparse; only the few applicable rows are densified.
        self._reactions = csr_matrix(reactions)

    @property
    def nnz(self) -> int:
        return self._rows.size

    def applicable(self, v) -> np.ndarray:
        """Indices of rules whose reaction-center criterion holds for ``v``."""
        v = np.asarray(v, dtype=np.int32)

        # A rule fails as soon as one required coordinate is not covered.
        violated = v[self._cols] < self._required

        ok = np.ones(self.n_rules, dtype=bool)
        ok[self._rows[violated]] = False
        return np.where(ok)[0]

    def one_step(self, v):
        """Prefilter + translation, deduplicated.

        Returns ``(child_vecs_unique, rxn_idxs_unique)``: the distinct predicted
        child fingerprints ``v + ECFP(r)``, and one rule index per distinct child.
        Matches ``morganrxn.core.reaction_utils.one_step`` exactly.
        """
        v = np.asarray(v, dtype=np.int32)
        applicable_idxs = self.applicable(v)

        if applicable_idxs.size == 0:
            return (np.empty((0, self.dim), dtype=np.int32),
                    np.empty((0,), dtype=applicable_idxs.dtype))

        child_vecs = v + self._reactions[applicable_idxs].toarray().astype(np.int32)
        child_vecs_unique, idx_unique = np.unique(child_vecs, axis=0, return_index=True)
        return child_vecs_unique, applicable_idxs[idx_unique]


def prefilter_from_rules(rules) -> ReactionCenterPrefilter:
    """Build a prefilter from a `ReactionRules` object (see `core.rules`)."""
    return ReactionCenterPrefilter(rules.ecfp_reaction_center, rules.ecfp_reaction,
                                   radius=getattr(rules, "radius", None))
