"""Build the train and held-out prompt sets from the accepted templates.

A row is a pointer, not a task: (template_id, seed, prompt). The fixture and the
verifier are regenerated from template_id + seed by TaskTemplate.instantiate, which
is deterministic (tests/test_templates.py::test_determinism). So the dataset stays
small, holds no answers, and the reward function needs only the two pointer values.

The instruction is identical for every seed of a template -- all variation is in the
fixture -- so prompt text has ONE distinct value per template, and any dedup keyed on
it would collapse N rows to len(templates).

Train and eval seeds come from disjoint ranges, asserted: a held-out set that shares
instances with training measures memorisation.

Every row is instantiated at build time. Screening only used 5 seeds per template,
and scaling to dozens can hit a seed whose random counts make a template invalid
(e.g. every expected tag drawing 0). Those rows are dropped and reported.

Run:
  ./.venv/bin/python -m src.tasks.build_dataset --templates data/templates/templates.jsonl \\
      --out data/dataset --train 30 --eval 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from typing import Optional

from ..verifier.task import TaskError
from .template import load_templates

# Far apart, so no realistic --train value reaches the eval range.
TRAIN_START, EVAL_START = 1_000, 900_000


def build(templates_path: str, out_dir: str, n_train: int, n_eval: int) -> dict:
    templates = load_templates(templates_path)
    splits = {"train": range(TRAIN_START, TRAIN_START + n_train),
              "eval": range(EVAL_START, EVAL_START + n_eval)}
    assert not set(splits["train"]) & set(splits["eval"]), "train/eval seeds overlap"

    kept: dict[str, list[dict]] = {"train": [], "eval": []}
    dropped: list[list] = []
    for split, seeds in splits.items():
        for t in templates:
            for seed in seeds:
                d = tempfile.mkdtemp(prefix="ds-")
                try:
                    t.instantiate(seed, d)            # must build and validate
                except (TaskError, KeyError, ValueError) as e:
                    dropped.append([t.id, seed, str(e)[:90]])
                    continue
                finally:
                    shutil.rmtree(d, ignore_errors=True)
                kept[split].append({"id": f"{t.id}#{seed}", "prompt": t.instruction,
                                    "template_id": t.id, "seed": seed, "split": split})

    os.makedirs(out_dir, exist_ok=True)
    for split, rows in kept.items():
        with open(f"{out_dir}/{split}.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)

    with open(templates_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    manifest = {                                      # provenance for the writeup
        "templates_sha256": digest,
        "seed_ranges": {k: [v.start, v.stop - 1] for k, v in splits.items()},
        "rows": {k: len(v) for k, v in kept.items()},
        "distinct_prompts": len({r["prompt"] for r in kept["train"]}),
        "per_template_train": dict(Counter(r["template_id"] for r in kept["train"])),
        "dropped": dropped,
    }
    with open(f"{out_dir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", default="data/templates/templates.jsonl")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--train", type=int, default=30, help="seeds per template, train")
    ap.add_argument("--eval", type=int, default=5, help="seeds per template, held out")
    a = ap.parse_args(argv)
    m = build(a.templates, a.out, a.train, a.eval)
    print(f"train rows {m['rows']['train']}, eval rows {m['rows']['eval']}, "
          f"distinct train prompts {m['distinct_prompts']}, dropped {len(m['dropped'])}")
    for t, s, why in m["dropped"][:10]:
        print(f"  dropped {t}#{s}: {why}")
    print(f"wrote {a.out}/train.jsonl, eval.jsonl, manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
