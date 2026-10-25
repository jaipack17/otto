"""Stage 1: pass the generated prompt sets through CuratorKit.

Our tasks are generated programmatically and verified by construction (screen.py), so
CuratorKit's LLM generation tasks do not apply: they are built for document-grounded
text, and a small model rewriting our instructions would break the match between an
instruction and its verifier. What CuratorKit does give us, and what this stage uses:

  * a schema gate and a text cleaner over every row,
  * a secrets gate (detect-secrets) that REJECTS credential-shaped rows,
  * a provenance manifest, checksums and a dataset card for the writeup.

Left off, deliberately: PII pseudonymisation (needs spaCy en_core_web_lg) and the
toxicity gate (needs a Detoxify model). Both are large downloads, and our rows are
fixed templated shell instructions with no personal text.

Three settings are not defaults, each for a measured reason:

  dedup="none"    CuratorKit defaults to exact dedup. Every seed of a template shares
                  one instruction string, so 990 rows are only 33 distinct texts and
                  dedup would collapse the dataset to 33. (test_curate pins this.)
  format          set explicitly; auto-detection of a prompt-only file is low confidence.
  verify()        TextCleaner is on, so we assert it did not rewrite any prompt. The
                  rollout finds each task by EXACT instruction match, so one altered
                  character would raise a KeyError in the middle of training.

Run:  ./.venv/bin/python -m src.tasks.curate
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Optional

from curatorkit import Curator, CuratorConfig

from .template import load_templates

SPLITS = {"train": "data/dataset/train.jsonl", "eval": "data/dataset/eval.jsonl"}
OUT_ROOT = "data/curated"


def curate(path: str, out_dir: str, name: str):
    """Run one prompt file through CuratorKit and return its CuratorResult."""
    config = CuratorConfig(
        dataset=path,
        format="prompt_only",
        field_mapping={"prompt": "instruction"},
        dedup="none",                      # see the module docstring
        schema_gate=True,
        clean=True,
        secrets_gate=True,
        output_dir=out_dir,
        export_formats=["ppo"],            # prompt-only rows export as {"prompt": ...}
        name=name,
    )
    return Curator(config).run()


def _prompts(path: str) -> list[str]:
    with open(path) as fh:
        return [json.loads(line)["prompt"] for line in fh if line.strip()]


def verify(src_path: str, out_dir: str, known_instructions: set[str]) -> list[str]:
    """Return a list of problems; empty means the curated set is safe to train on."""
    problems: list[str] = []
    before = _prompts(src_path)
    after = _prompts(os.path.join(out_dir, "ppo.jsonl"))

    if len(after) != len(before):
        problems.append(f"{len(before)} rows in, {len(after)} out")
    if Counter(before) != Counter(after):
        changed = list((Counter(before) - Counter(after)))[:2]
        problems.append(f"the cleaner rewrote or dropped prompt text, e.g. {changed!r}")
    unknown = set(after) - known_instructions
    if unknown:
        problems.append(f"{len(unknown)} curated prompt(s) match no template, e.g. {sorted(unknown)[0]!r}")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", default="data/templates/templates.jsonl")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args(argv)

    known = {t.instruction for t in load_templates(a.templates)}
    failed = False
    for split, path in SPLITS.items():
        out_dir = os.path.join(a.out, split)
        result = curate(path, out_dir, f"otto-{split}")
        problems = verify(path, out_dir, known)
        passed, rejected = len(result.passed), len(result.rejected)   # both are sample lists
        status = "OK " if not problems and rejected == 0 else "BAD"
        print(f"[{status}] {split:5s} passed {passed}  rejected {rejected}  -> {out_dir}")
        for p in problems:
            print(f"        problem: {p}")
        failed |= bool(problems) or rejected != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
