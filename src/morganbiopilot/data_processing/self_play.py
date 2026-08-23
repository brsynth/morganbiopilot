"""Turn the policy's own successful searches into training data.

The corpus this project trains on has two defects, and they are the same defect twice.
Every state in it sits **on a route that works**, because it was built by replaying
mined reference routes. So the model never sees a dead end, never learns when to abandon
a branch, and at inference it spends most of its time in states no training example
resembles. That is textbook behaviour cloning: fine until the policy deviates, and the
error compounds from there.

Rejection sampling fixes both at once, for a fraction of what RL costs. Run the current
policy, keep the episodes that **reached the chassis**, and train on the decisions it
actually made in them. The supervision is then outcome-level -- nothing is kept unless
the whole search succeeded -- and the states are drawn from the policy's own
distribution, which is the one it will meet again.

Why not RL
----------
RETRO-R1 (Table 2) is the only paper that separates the two stages on the same model:
supervised fine-tuning reaches 63.16% at N=50 and PPO 64.63%. **1.5 points**, growing to
about six only by N=500. RETROAGENT's much larger ablation gap (4.4% -> 53.3%) compares
against *zero-shot*, not against supervised training, so it measures something we already
have. Against one to four points in our budget range, multi-turn PPO with environment
masking and an actor-critic pair on a 48 GB card is not the right use of ten days.

What is kept, and what is not
-----------------------------
Only decisions where the model was actually asked. A frontier with one candidate is
recorded by the policy as `skipped_single`; there was no choice to learn from, and
training on it would teach the format rather than the task. Fallbacks are dropped for
the same reason: the label would be the fallback rule's answer, not the model's.

Episodes are also **not** all equal. `--max-solved-at` keeps only searches that finished
within a given number of expansions, which is the cheap analogue of the budget penalty
RETROAGENT puts in its reward. Set it to 0 to keep every success.

Leakage is handled at generation time rather than after: an evaluation target that never
runs cannot contaminate anything downstream.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from morganbiopilot.core.building_blocks import skeleton, use_sink
from morganbiopilot.core.ec import annotate_rules
from morganbiopilot.core.paths import ROOT_DIR
from morganbiopilot.core.rules import load_rules
from morganbiopilot.multi_step.search import search
from morganbiopilot.one_step.prefilter import prefilter_from_rules
from morganbiopilot.one_step.ranking import make_ranker


def banned_skeletons(targets_files: Sequence[str], variant: str) -> Set[str]:
    """Evaluation targets, excluded before a single episode is generated."""
    from morganbiopilot.agents.train_policy import benchmark_skeletons

    return benchmark_skeletons(targets_files, variant)


def load_targets(path: str, banned: Set[str], limit: int, seed: int) -> List[str]:
    """Distinct target SMILES from the mined corpus, minus the benchmarks.

    Shuffled before truncation: the corpus is written in depth order, so a prefix would
    be all short routes -- the same mistake that cost this project two corpora.
    """
    seen: Dict[str, None] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            smiles = json.loads(line)["target"]
            sk = skeleton(smiles)
            if sk and sk not in banned and smiles not in seen:
                seen[smiles] = None
    targets = list(seen)
    random.Random(seed).shuffle(targets)
    return targets[:limit] if limit else targets


def episode_pairs(policy, target: str) -> List[dict]:
    """One training pair per decision the model was actually asked to make."""
    out = []
    for call in policy.calls:
        if not call.called_api or call.fallback or call.choice is None:
            continue
        if not call.prompt:
            continue
        out.append({
            "system": policy.system_prompt,
            "user": call.prompt,
            "assistant": json.dumps({"choice": int(call.choice), "reason": ""}),
            "depth": None,          # filled by the caller, which has the graph
            # The molecule the policy chose, which is what `depth` must describe. Kept
            # in the record so a wrong depth can be recomputed from the corpus instead
            # of by regenerating it -- the first version stored neither, and the whole
            # 8,015-pair corpus had to be trained on with its depths unusable.
            "smiles": call.smiles,
            "n_frontier": call.n_frontier,
            "n_shown": call.n_candidates,
            "target": target,
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--routes", default="results/routes_ecoli_r1.jsonl",
                   help="mined corpus, used only as a source of targets")
    p.add_argument("--out", default="results/sft_pairs_r1_selfplay.jsonl")
    p.add_argument("--model", default="openai:policy",
                   help="the served adapter; 'openai:policy' with run_serve_policy")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--top-k", type=int, default=20, help="frontier candidates shown")
    p.add_argument("--ranker", default="native_similarity")
    p.add_argument("--top-n", type=int, default=20, help="expansion cap")
    p.add_argument("--limit", type=int, default=1500, help="0 = every corpus target")
    p.add_argument("--max-solved-at", type=int, default=0,
                   help="keep only episodes solved within this many expansions; "
                        "0 keeps every success")
    # The measured failure mode is impatience, not incompetence: on 141 LASER targets
    # the trained policy goes 63 -> 65 -> 67% from N=10 to N=50 while breadth-first
    # goes 55 -> 69. It finds the easy targets fastest of any policy (exp2sol 3.5) and
    # stops converting budget into solutions after that. Keeping every success would
    # feed it mostly one-expansion episodes and reinforce exactly that.
    p.add_argument("--min-solved-at", type=int, default=3,
                   help="drop episodes solved in fewer expansions than this: their "
                        "decisions were not decisions. 0 keeps every success")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sink", default=None)
    p.add_argument("--eval-targets",
                   default="data/bionavinp/testset.txt,"
                           "data/laser_dataset/laser_targets.tsv")
    args = p.parse_args(argv)

    if args.sink:
        use_sink(args.sink)

    files = [f.strip() for f in (args.eval_targets or "").split(",") if f.strip()]
    banned = banned_skeletons(files, "experimental")
    targets = load_targets(args.routes, banned, args.limit, args.seed)
    print(f"{len(targets)} targets (evaluation targets excluded: "
          f"{len(banned)} skeletons)")

    rules = load_rules(args.radius)
    rule_ec = annotate_rules(rules)
    prefilter = prefilter_from_rules(rules)
    ranker = make_ranker(args.ranker, rules)
    print(f"r{args.radius}: {len(rules)} rules | ranked by {args.ranker}, "
          f"capped at {args.top_n} | budget {args.budget} | top_k {args.top_k}")

    from morganbiopilot.agents.backends import make_backend
    from morganbiopilot.agents.policy import LLMPolicy
    from morganbiopilot.agents.tools import untooled

    # Relative to the repository root, exactly like `--routes` a few lines above and
    # like the `cat` that the slurm runs on this same path. Resolving it against
    # RESULTS_DIR instead wrote `results/results/...`, which cost one 5-hour
    # generation: the pairs were on disk, and the job died concatenating them.
    out_path = Path(args.out) if args.out.startswith("/") else ROOT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outcomes: Counter = Counter()
    depths: Counter = Counter()
    emitted = 0
    t0 = time.perf_counter()
    lock = __import__("threading").Lock()

    def one(index_target):
        i, target = index_target
        policy = LLMPolicy(tools=untooled(), backend=make_backend(args.model),
                           top_k=args.top_k, seed=args.seed + i, explain=False,
                           record_prompts=True)
        try:
            result = search(target, rules, prefilter, policy,
                            budget=args.budget, max_depth=args.max_depth,
                            rule_ec=rule_ec, ranker=ranker, top_n=args.top_n)
        except Exception as exc:                                # noqa: BLE001
            return "search failed", [], str(exc)[:80]
        if not result.solved:
            return "unsolved", [], ""
        if args.max_solved_at and result.n_expansions > args.max_solved_at:
            return "solved but too long", [], ""
        if result.n_expansions < args.min_solved_at:
            return "solved too easily", [], ""
        pairs = episode_pairs(policy, target)
        for pair in pairs:
            # The chosen molecule, not the first prompt line -- that line reads
            # "Target molecule: <smiles>", so it never resolved, and the `0 if None`
            # fallback below turned every failure into a plausible depth. Unresolved
            # is now -1, which is visibly not a depth.
            node = result.graph.molecule_id(pair["smiles"])
            pair["depth"] = -1 if node is None else result.graph.molecules[node].depth
        return "kept", pairs, ""

    jobs = list(enumerate(targets))
    # LF everywhere: see route_corpus for why this is not cosmetic.
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(one, j) for j in jobs]
                for k, fut in enumerate(as_completed(futures), 1):
                    outcome, pairs, note = fut.result()
                    outcomes[outcome] += 1
                    for pair in pairs:
                        fh.write(json.dumps(pair) + "\n")
                        depths[pair["depth"]] += 1
                        emitted += 1
                    fh.flush()
                    if k % 50 == 0:
                        print(f"  {k}/{len(jobs)} episodes -> {emitted} pairs "
                              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        else:
            for k, j in enumerate(jobs, 1):
                outcome, pairs, note = one(j)
                outcomes[outcome] += 1
                for pair in pairs:
                    fh.write(json.dumps(pair) + "\n")
                    depths[pair["depth"]] += 1
                    emitted += 1
                fh.flush()

    print(f"\n{emitted} pairs from {len(jobs)} episodes")
    for k, v in outcomes.most_common():
        print(f"  {k:22s} {v:5d}  ({100 * v / len(jobs):.0f}%)")
    kept = outcomes.get("kept", 0)
    if kept:
        print(f"  pairs per kept episode: {emitted / kept:.2f}")

    print("  depth: " + ", ".join(f"{k}:{depths[k]}" for k in sorted(depths)))
    # A decision is taken on a frontier molecule, so depth 0 -- the target itself --
    # is not a value this can legitimately take, and -1 means the lookup failed.
    # Both silently made every pair look shallow to `subsample` once already.
    bad = depths.get(-1, 0) + depths.get(0, 0)
    if bad:
        print(f"\nWARNING: {bad} of {emitted} pairs ({100 * bad / emitted:.0f}%) carry "
              f"depth 0 or -1.\n  Those are not decision depths; `train_policy "
              f"--max-pairs` would treat them all as\n  shallow. Train with "
              f"--max-pairs 0, or fix the lookup, before relying on depth.",
              file=sys.stderr)
    print(f"\nwrote {out_path}")
    print("\nTrain on the union of this and the replayed corpus -- self-play alone "
          "would drop\nevery state the policy never reaches, which is most of the "
          "reference routes:")
    print(f"  cat results/sft_pairs_r1_ranked.jsonl {args.out} > "
          f"results/sft_pairs_r1_union.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
