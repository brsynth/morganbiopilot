"""Distance to the sink in fingerprint space.

Serves two roles at once, and the distinction matters for the paper:

- **heuristic** for the non-LLM baselines (greedy best-first, A*, MCTS rollouts),
  playing the role the chemical-similarity guidance plays in RetroPath RL
  (Koch et al. 2020);
- **tool** exposed to the LLM agent. The tooled/untooled ablation of the project
  note is exactly "does the agent get to call this or not".

Simplification against the RetroMorgan draft
--------------------------------------------
There, a node was a *sum* of ECFP vectors, so closeness required greedily
selecting `nfrag` building blocks whose sum best matched the node
(`score_tanimoto_sum_nfrag`). Here a node is one molecule, so closeness is the
best single-building-block similarity: one matrix product.
"""

import numpy as np

from morganbiopilot.core.building_blocks import (building_block_ecfps,
                                                 building_block_labels,
                                                 building_block_smiles)
from morganbiopilot.core.chem import mol_ecfp


def counted_tanimoto(v: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Generalized Tanimoto of a count vector against every row of `matrix`.

        T(u, v) = <u,v> / (<u,u> + <v,v> - <u,v>)

    Same definition as `retromorgan.scoring_function.tani_counts`, batched.
    """
    v = np.asarray(v, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)

    dot = m @ v
    denom = (m * m).sum(axis=1) + float(v @ v) - dot
    out = np.ones_like(dot)
    np.divide(dot, denom, out=out, where=denom > 0)
    return out


class SinkCloseness:
    """Similarity of a molecule to the closest building block, at one radius."""

    def __init__(self, radius: int):
        self.radius = int(radius)
        self.matrix = building_block_ecfps(self.radius)
        self.smiles = building_block_smiles()
        self.labels = building_block_labels()

    def closeness(self, smi: str) -> float:
        """Best counted-Tanimoto to any building block, in [0, 1]. Higher is closer."""
        sims = counted_tanimoto(np.asarray(mol_ecfp(smi, self.radius), dtype=np.int32), self.matrix)
        return float(sims.max()) if sims.size else 0.0

    def nearest(self, smi: str, k: int = 5):
        """The k closest building blocks, as (label, smiles, similarity), best first.

        This is the agent-facing form. A bare number tells a model nothing it can
        reason about; "0.59 to NADH" is a chemical fact it can act on. Costs the
        same matrix product as `closeness`, so prefer it wherever the result is
        shown to a model.
        """
        sims = counted_tanimoto(np.asarray(mol_ecfp(smi, self.radius), dtype=np.int32), self.matrix)
        if sims.size == 0:
            return []
        top = np.argsort(sims)[::-1][:k]
        return [(self.labels[int(i)], self.smiles[int(i)], float(sims[int(i)])) for i in top]

    def h(self, smi: str) -> float:
        """Heuristic cost-to-go in [0, 1]: 0 when the molecule is in the sink."""
        return 1.0 - self.closeness(smi)
