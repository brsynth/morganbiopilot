"""LoRA fine-tuning of the search policy on attested biosynthetic routes.

Everything we measured says the prompted agent is not failing for want of information but
because of it: sink closeness accounts for the entire early advantage (30% against 23% at
N=10) and the entire plateau (+5 points to N=50 against +18 without), EC contributes
nothing, and the enzymatic column actively harms. Meanwhile the systems that work in
organic retrosynthesis do not prompt at all -- RetroAgent trains a 4B model, RETRO-R1 a
7B, the same size as ours -- and RETRO-R1's supervised stage alone reaches 63.2% against
64.6% for the full RL pipeline at N=50, which is the budget regime we measure in.

So this trains rather than prompts, on the state rendered **without columns**. The model
has to learn value from attested chemistry instead of being handed a heuristic that makes
it greedy.

Following RETRO-R1, the reasoning field is left empty: reference routes become
multi-turn exchanges with nothing between the thinking tags, so no justification has to be
annotated or distilled. Only the choice is supervised.

The split is by target, not by example
--------------------------------------
A route yields several decisions and they share a target molecule, its early frontier and
much of its graph. Splitting at random would put a target's depth-1 decision in training
and its depth-3 decision in validation, and the reported accuracy would be memorisation.
Grouping by target is the same discipline that turned an earlier experiment in this
project from 0.948 to a trustworthy null.

What the validation number is, and is not
-----------------------------------------
Next-node accuracy on held-out targets is a training monitor, not the result. The result
is solve rate against MCTS at 52% and the untooled prompted agent at 41±3% on the same 60
BioNavi-NP targets, and it can only be measured by running the search
(`paper_results.compare_policies`). A model can improve next-node accuracy and still
search worse, because one wrong turn early costs more than several right ones later.

    python -m morganbiopilot.agents.train_policy --pairs results/sft_pairs_andor.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_pairs(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def subsample(pairs: List[dict], keep_deep: int, limit: int, seed: int) -> List[dict]:
    """Cut the corpus down, taking the cut out of the shallow decisions.

    Deep decisions are the scarce resource: the attested routes that produce them are a
    small minority, and they are what the measured plateau is made of. Shallow ones are
    plentiful. So everything at depth >= `keep_deep` is kept and the reduction falls
    entirely on the rest, in the proportions they already have.

    This is a deliberate shift of the training distribution away from the inference
    distribution, where most decisions are shallow -- a search takes many depth-1 steps
    for each deep one. It is a bet, and it is falsifiable: if the trained policy improves
    late and *loses* the early ground the untooled agent already lacks (23% against 30%
    at N=10), this re-weighting is the first thing to suspect.
    """
    if not limit or limit >= len(pairs):
        return pairs
    rng = random.Random(seed)
    deep = [p for p in pairs if p["depth"] >= keep_deep]
    shallow = [p for p in pairs if p["depth"] < keep_deep]
    room = max(0, limit - len(deep))
    rng.shuffle(shallow)
    return deep + shallow[:room]


def split_by_target(pairs: List[dict], fraction: float, seed: int):
    """Hold out whole targets, so no target appears on both sides."""
    targets = sorted({pair["user"].split("\n", 1)[0] for pair in pairs})
    rng = random.Random(seed)
    rng.shuffle(targets)
    held = set(targets[:max(1, int(len(targets) * fraction))])
    train = [p for p in pairs if p["user"].split("\n", 1)[0] not in held]
    valid = [p for p in pairs if p["user"].split("\n", 1)[0] in held]
    return train, valid, len(held)


def as_conversation(pair: dict) -> Dict[str, list]:
    """Prompt-completion form, which is what makes the loss completion-only.

    TRL supervises only the `completion` side when a dataset carries both columns, so the
    thousands of tokens of frontier in the prompt contribute no gradient -- we are teaching
    a choice, not teaching the model to reproduce SMILES.
    """
    return {
        "prompt": [{"role": "system", "content": pair["system"]},
                   {"role": "user", "content": pair["user"]}],
        "completion": [{"role": "assistant", "content": pair["assistant"]}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", default="results/sft_pairs_andor.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="models/policy_sft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Batch 2, not 4: padding is dynamic (4096 measured no slower than
    # 2048), so a batch of four long sequences reaches 16k tokens of
    # activations and OOMs on the L40S at step 53 of 58. The effective
    # batch is unchanged at 16.
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=3072,
                        help="measured with the Qwen tokenizer: median 979, p95 2143, "
                             "max 4229. 3072 keeps 99%% and caps the worst-case batch")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="0 = the whole corpus; the reduction is taken out of the "
                             "shallow decisions, see `subsample`")
    parser.add_argument("--keep-deep", type=int, default=3,
                        help="depth at or above which every decision is kept")
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    full = len(pairs)
    pairs = subsample(pairs, args.keep_deep, args.max_pairs, args.seed)
    if len(pairs) < full:
        kept = Counter(p["depth"] for p in pairs)
        print(f"subsampled {full} -> {len(pairs)} pairs, keeping every decision at "
              f"depth >= {args.keep_deep}")
        print("  kept by depth: " + ", ".join(f"{k}:{kept[k]}" for k in sorted(kept)))
    if len(pairs) < 2000:
        print(f"only {len(pairs)} pairs; below the floor we set for a 7B.",
              file=sys.stderr)
    train, valid, n_held = split_by_target(pairs, args.holdout, args.seed)
    depths = Counter(p["depth"] for p in pairs)
    print(f"{len(pairs)} pairs | train {len(train)} | validation {len(valid)} "
          f"({n_held} held-out targets)")
    print("  depth: " + ", ".join(f"{k}:{depths[k]}" for k in sorted(depths)))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    sets = {name: Dataset.from_list([as_conversation(p) for p in rows])
            for name, rows in (("train", train), ("validation", valid))}

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
    )
    # `all-linear` rather than a hand-picked module list: the attention/MLP naming differs
    # between model families and a stale list silently trains fewer adapters than intended.
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=sets["train"],
        eval_dataset=sets["validation"],
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.out)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "corpus.json").write_text(json.dumps({
        "pairs": args.pairs, "n_pairs": len(pairs), "corpus_size": full,
        "max_pairs": args.max_pairs, "keep_deep": args.keep_deep, "n_train": len(train),
        "n_validation": len(valid), "held_out_targets": n_held,
        "depths": dict(sorted(depths.items())), "base_model": args.model,
        "lora_r": args.lora_r, "epochs": args.epochs, "lr": args.lr,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote adapter to {args.out}")
    print("Next-node accuracy is a monitor. The result is a search:")
    print("  python -m morganbiopilot.paper_results.compare_policies --llm ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
