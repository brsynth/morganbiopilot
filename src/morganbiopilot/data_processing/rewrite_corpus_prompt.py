"""Re-stamp an existing SFT corpus with the current prompt and target format.

The corpus stores the system prompt and the supervised answer on every pair, so
changing `SYSTEM_PROMPT` or the answer schema makes an existing corpus disagree with
inference — and a training state that is not byte-for-byte an inference state is the
one thing the replay pipeline exists to guarantee.

Regenerating the corpus would replay 12,251 routes for nothing: the prompt text has no
influence on which decisions were recorded, only on how they are rendered. This
rewrites the two affected fields in place, in seconds, and leaves the decisions
untouched.

It is deliberately narrow. It never invents a `choice`, edits `user` only under
`--strip-ec` and then only by deleting a column whose exact shape is known, and refuses
to run if a pair does not carry the fields it expects — a corpus in an unexpected shape
should stop the run, not be silently half-converted.

`--strip-ec` exists because the corpus was built with the EC column while the arms the
paper reports run `--tooling untooled`, which renders none: the model was trained to
read a column it never sees. Stripping it is exact rather than approximate, and the
selection of candidates is unaffected — measured identical over 183 frontier states.

    python -m morganbiopilot.data_processing.rewrite_corpus_prompt \
        results/sft_pairs_r1_ranked.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from morganbiopilot.agents.policy import SYSTEM_PROMPT

# The EC column as `render` emits it: appended last, so it always ends the line.
EC_FIELD = re.compile(r" \| EC=[0-9,]+")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--keep-reason", action="store_true",
                    help="leave the answer as {choice, reason}; only re-stamp the prompt")
    ap.add_argument("--strip-ec", action="store_true",
                    help="remove the `| EC=...` column from the rendered candidates, "
                         "making the corpus match `--tooling untooled` inference")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"no such corpus: {args.corpus}", file=sys.stderr)
        return 1

    tmp = args.corpus.with_suffix(".jsonl.rewriting")
    n = 0
    prompts_before = set()
    answers_changed = 0
    ec_removed = [0]

    # newline="\n" so the file stays byte-identical wherever it is produced; a corpus
    # differing only by line endings once broke a reproducibility check.
    with open(args.corpus, encoding="utf-8") as src, \
            open(tmp, "w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            pair = json.loads(line)
            for field in ("system", "user", "assistant"):
                if field not in pair:
                    print(f"line {n + 1}: missing {field!r}", file=sys.stderr)
                    return 1
            prompts_before.add(pair["system"])
            pair["system"] = SYSTEM_PROMPT
            if args.strip_ec:
                # Exactly what `render` produces with `rule_ec=None`: the EC part is
                # appended last and only when present, so deleting it reconstructs the
                # untooled line character for character. Verified on the corpus -- of
                # 20,570 occurrences in a 2,000-pair sample, every one matched this
                # pattern and ended the line.
                stripped, n_ec = EC_FIELD.subn("", pair["user"])
                pair["user"] = stripped
                ec_removed[0] += n_ec
            if not args.keep_reason:
                answer = json.loads(pair["assistant"])
                if "reason" in answer:
                    answer.pop("reason")
                    answers_changed += 1
                pair["assistant"] = json.dumps(answer)
            dst.write(json.dumps(pair) + "\n")
            n += 1

    print(f"{n} pairs | {len(prompts_before)} distinct prompt(s) before "
          f"| {answers_changed} answers stripped of `reason` "
          f"| {ec_removed[0]} EC fields removed")
    if len(prompts_before) > 1:
        print("WARNING: the corpus held more than one system prompt; it was built from "
              "more than one version of the code.", file=sys.stderr)

    if args.dry_run:
        os.remove(tmp)
        print("dry run: nothing written")
        return 0

    os.replace(tmp, args.corpus)
    print(f"rewrote {args.corpus}")
    print("The corpus now matches `agents.policy.SYSTEM_PROMPT`. Re-train before "
          "evaluating: an adapter trained on the old prompt is a different model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
