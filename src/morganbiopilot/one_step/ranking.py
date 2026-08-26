"""Order the rules an expansion will try, so a cap can be principled.

Template application here is exhaustive: every rule the prefilter admits is validated
and every valid one becomes a neighbour. That is 29 candidates for vanillin at r2 and 96
for violacein at r1, and a search reaches a frontier of 15,000 within 100 expansions.
Two things break there — a language-model policy sees `top_k = 20` of that frontier, and
UCT never leaves depth 1 because its exploration term visits all 96 root reactions
before revisiting any.

Every system that works caps this. RETRO-R1 and RETROAGENT put a learned single-step
model in front of the agent, turning 381,302 templates into a top-50. RetroPath RL, which
is template-based like us and has no such model, builds a biochemical score from
cheminformatics and allows at most 10 children per node.

`expand` already had the cap — `max_rules` — but it truncated by rule index, which its
own docstring calls arbitrary by construction. This supplies the order that makes it mean
something.

Why similarity to the native substrate
--------------------------------------
Measured on the attested disconnections of the curated set that our rules reproduce at
all, against the metrics the field's scoping review names (Gricourt et al. 2024). The
campaign radius is r1, which reproduces 55 of the 70 attested steps; the r2 column is
kept because it is a different and easier problem, not a contradiction -- r1 admits more
rules, so there is more to rank. 102 candidates per step on average at r1, median 92,
max 474.

    scorer                     r1 MRR   cov@10   cov@20   median | r2 MRR  cov@10
    native_similarity_all       0.410      76%      89%        4 |      -       -
    native_similarity           0.398      78%      89%        5 |  0.587     88%
    similarity_x_ec             0.398      78%      89%        5 |      -       -
    closeness_worst             0.371      60%      72%        5 |  0.486     86%
    closeness_best              0.265      58%      70%        6 |      -       -
    template_prior              0.225      52%      65%       10 |  0.276     65%
    arbitrary (rule index)      0.184      49%      69%       11 |  0.333     65%
    enzymatic score             0.074      21%      41%       26 |  0.238     70%

Three results worth keeping:

- cov@20 is the quantity the cap consumes, and 89% of it predicts losing ~6 of the 55
  steps. Capping at 20 was measured independently to cost 5 (55 exact -> 50). The two
  benchmarks agree without having been fitted to each other.
- `closeness_worst` leads at rank 1 (27% against 23%) and loses badly by rank 10. Same
  lesson as the frontier orderings: a scorer that is often first but sometimes far is
  worse under a cap than one that is steadily near. Rank the window, not the winner.
- `similarity_x_ec` is identical to `native_similarity` on all four thresholds. The EC
  annotation adds nothing to the ordering, and `enzymatic` alone ranks below arbitrary.

It wins, but the decisive argument is *when* it can be computed. It needs only the query
molecule and `RuleSet.smi_sub`, so it ranks **before** RDKit validation: we validate only
the top candidates and never pay for the rest. `closeness_worst` needs the products, so
it can only reorder work already done.

The fingerprint radius here is fixed at 2 and is deliberately **not** the template
radius. Molecular similarity has to keep meaning the same thing when the rule set's
promiscuity changes, or r1 and r2 stop being comparable.

Known limitation, and what it actually costs
--------------------------------------------
`smi_sub` carries one representative substrate per rule, while a rule may be derived from
hundreds of MetaNetX reactions — rule 0 lists 385. RetroPath RL compares against all
native substrates and keeps the best match.

`native_similarity_all` does exactly that and is in the table above: it buys 0.012 of MRR
and one place of median rank, and **the same cov@20 of 89%**. Since cov@20 is the only
quantity the cap consumes, the approximation is free at the operating point. It is the
cheaper scorer that ships, and the limitation is measured rather than merely declared.

The earlier note here predicted that fixing the `reaction_id` join "would likely raise the
MRR". It does, negligibly. Recorded because a prediction that was checked and came back
small is worth more than one left open.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class NativeSimilarity:
    """Rank rules by Tanimoto between the query and the substrate they were learned on.

    Fingerprints are cached on both sides: the query changes once per expansion, and the
    rule substrates recur across every expansion of a search.
    """

    name = "native_similarity"

    def __init__(self, rules, fp_radius: int = 2, fp_bits: int = 2048,
                 all_substrates: bool = False) -> None:
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.*")
        self._rules = rules
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_bits)
        self._rule_fp: Dict[int, object] = {}
        self._query_fp: Dict[str, object] = {}
        self._all = all_substrates
        self._sources = self._load_sources() if all_substrates else None

    @staticmethod
    def _load_sources():
        """rule reaction id -> the left side of the MetaNetX reaction it came from.

        `RuleSet.smi_sub` holds one representative substrate per rule, but rules are
        deduplicated across their source reactions and one can come from 388 of them.
        RetroPath RL compares the query against every native substrate and keeps the
        best match; this is what makes that possible.

        The join needs one repair: `reaction_id` is a `|`-separated list whose entries
        carry a `__split0` suffix the reaction table does not use. Stripped, it matches
        every sampled rule.
        """
        import csv
        import re

        from morganbiopilot.core.paths import METANETX_REACTIONS_TSV

        strip = re.compile(r"__split\d+$")
        out: Dict[str, str] = {}
        with open(METANETX_REACTIONS_TSV, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                left = row["reaction"].split(">>")[0] if ">>" in row["reaction"] else ""
                if left:
                    out[row["id"]] = left
        return (out, strip)

    def _fp(self, smiles: str):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return self._gen.GetFingerprint(mol) if mol is not None else None

    def _for_query(self, smiles: str):
        if smiles not in self._query_fp:
            # One entry per molecule ever expanded; a search builds thousands, not
            # millions, and the dict is dropped with the ranker.
            self._query_fp[smiles] = self._fp(smiles)
        return self._query_fp[smiles]

    def _for_rule(self, rule_idx: int):
        """Fingerprints of the substrates this rule was learned on.

        A tuple, because with `all_substrates` a rule carries as many as it has source
        reactions and the score is the best match among them.
        """
        if rule_idx not in self._rule_fp:
            smiles = [str(self._rules.smi_sub[rule_idx])]
            if self._all:
                table, strip = self._sources
                for raw in str(self._rules.reaction_id[rule_idx]).split("|"):
                    left = table.get(strip.sub("", raw))
                    if left:
                        smiles.extend(left.split("."))
            seen, fps = set(), []
            for smi in smiles:
                if smi and smi not in seen:
                    seen.add(smi)
                    fp = self._fp(smi)
                    if fp is not None:
                        fps.append(fp)
            self._rule_fp[rule_idx] = tuple(fps)
        return self._rule_fp[rule_idx]

    def similarity(self, target: str, rule_idx: int) -> float:
        """Tanimoto between the query and one rule's native substrate.

        Exposed because a *policy* can want the same quantity the expansion cap uses:
        `GreedySimilarity` scores a frontier molecule by the best-precedented
        disconnection available to it, which is the maximum of this over the rules that
        apply.
        """
        from rdkit import DataStructs

        query, fps = self._for_query(target), self._for_rule(int(rule_idx))
        if query is None or not fps:
            return 0.0
        return max(DataStructs.TanimotoSimilarity(query, f) for f in fps)

    def order(self, target: str, rule_idxs: Sequence[int]) -> List[int]:
        """Rule indices, most similar native substrate first.

        Ties keep rule-index order, which is what the engine did before this existed —
        so a rule set where every substrate is equally similar behaves exactly as it
        used to rather than in some new arbitrary way.
        """
        from rdkit import DataStructs

        query = self._for_query(target)
        if query is None:
            return list(rule_idxs)

        scored = []
        for idx in rule_idxs:
            idx = int(idx)
            fps = self._for_rule(idx)
            sim = max((DataStructs.TanimotoSimilarity(query, f) for f in fps),
                      default=0.0)
            scored.append((-sim, idx))
        scored.sort()
        return [idx for _, idx in scored]


def make_ranker(name: Optional[str], rules):
    """`None` or "none" keeps exhaustive, unordered expansion — the current behaviour."""
    if not name or name == "none":
        return None
    if name in ("native_similarity", "similarity"):
        return NativeSimilarity(rules)
    if name in ("native_similarity_all", "similarity_all"):
        return NativeSimilarity(rules, all_substrates=True)
    raise ValueError(f"unknown ranker {name!r}; have 'native_similarity' or 'none'")
