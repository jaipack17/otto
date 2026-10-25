"""Score a policy (base model, or base + a LoRA adapter) on fixed tasks through the SAME
rollout used for training: same prompt, thinking budget, harness and verifier.

  # base model
  ./.venv-train/bin/python -u -m src.training.evaluate --tag base
  # trained adapter (a run directory or a checkpoint-N inside it)
  ./.venv-train/bin/python -u -m src.training.evaluate --tag trained --adapter runs/run3

Two groups of templates are reported separately, because they answer different questions:
  trained-on   the templates the run trained on, on fixtures it never saw
  held-out     the eight templates that no code path trains on (see src/tasks/splits.py)
Results go to eval/<tag>.jsonl (every episode) and eval/<tag>.md (the table).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..harness.episode import SYSTEM_PROMPT
from ..tasks.splits import HOLDOUT
from ..tasks.template import load_templates

EVAL_STEP_BASE = 100_000        # training uses steps < 1000, so fixtures never coincide by construction


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a pass rate. With ~4 episodes per template, a bare percentage
    would overstate what we know, so the report always carries this."""
    if n == 0:
        return (0.0, 0.0)
    p = passed / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows: list[dict], trained_on: set[str]) -> dict:
    by_t: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        by_t[r["template"]][0] += r["reward"] > 0
        by_t[r["template"]][1] += 1
    out = {"templates": {t: tuple(v) for t, v in sorted(by_t.items())}}
    for name, keep in (("trained-on", lambda t: t in trained_on), ("held-out", lambda t: t in HOLDOUT)):
        p = sum(v[0] for t, v in by_t.items() if keep(t))
        n = sum(v[1] for t, v in by_t.items() if keep(t))
        out[name] = {"passed": p, "n": n, "rate": p / n if n else None, "ci": wilson(p, n)}
    return out


def render(tag: str, s: dict, trained_on: set[str]) -> str:
    lines = [f"# Evaluation: {tag}", "", "| group | passed | episodes | pass rate | 95% interval |",
             "|---|---|---|---|---|"]
    for name in ("trained-on", "held-out"):
        g = s[name]
        rate = "-" if g["rate"] is None else f"{g['rate']:.1%}"
        lo, hi = g["ci"]
        lines.append(f"| {name} | {g['passed']} | {g['n']} | {rate} | {lo:.1%} to {hi:.1%} |")
    lines += ["", "| template | group | passed |", "|---|---|---|"]
    for t, (p, n) in s["templates"].items():
        lines.append(f"| {t} | {'trained-on' if t in trained_on else 'held-out'} | {p}/{n} |")
    return "\n".join(lines) + "\n"


def load_policy(model_name: str, adapter: Optional[str], device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter if adapter else model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.to(device).eval()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="output name, e.g. base or trained")
    ap.add_argument("--adapter", help="LoRA adapter dir; omit to evaluate the base model")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--templates", default="data/templates/templates.jsonl")
    ap.add_argument("--trained-on", default="", metavar="ID,ID,...",
                    help="templates the run trained on (default: read from the adapter's RUN_INFO)")
    ap.add_argument("--G", type=int, default=4, help="episodes per template (one fixture each)")
    ap.add_argument("--think-budget", type=int, default=256)
    ap.add_argument("--answer-cap", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval")
    a = ap.parse_args(argv)

    import torch
    from ..harness.rollout import make_rollout, release_memory

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    templates = load_templates(a.templates)

    trained_on = {x for x in a.trained_on.split(",") if x}
    if not trained_on and a.adapter:
        info = Path(a.adapter) / "RUN_INFO.json"
        if not info.exists():
            info = Path(a.adapter).parent / "RUN_INFO.json"
        if info.exists():
            trained_on = set(json.loads(info.read_text())["train_templates"])
    chosen = [t for t in templates if t.id in HOLDOUT or t.id in trained_on]
    if not trained_on:
        print("note: no --trained-on / RUN_INFO, so only the held-out templates are scored")

    tok, model = load_policy(a.model, a.adapter, dev)
    traces: list = []
    rollout = make_rollout(templates, think_budget=a.think_budget, answer_cap=a.answer_cap,
                           max_tokens=a.max_tokens, trace=traces, batch=a.batch)
    stub = types.SimpleNamespace(
        processing_class=tok, model=model, num_generations=a.G,
        accelerator=types.SimpleNamespace(device=dev, unwrap_model=lambda m: m),
        state=types.SimpleNamespace(global_step=0))

    Path(a.out).mkdir(exist_ok=True)
    jsonl = Path(a.out) / f"{a.tag}.jsonl"
    jsonl.write_text("")
    per_call = max(1, a.batch // a.G)                     # templates scored per rollout call
    rows: list[dict] = []
    t0 = time.time()
    for i in range(0, len(chosen), per_call):
        group = chosen[i:i + per_call]
        stub.state.global_step = EVAL_STEP_BASE + i
        prompts = []
        for t in group:
            prompts += [[{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": t.instruction}]] * a.G
        rollout(prompts, stub)
        release_memory()
        new = [{"template": tr["template"], "seed": tr["seed"], "reward": tr["reward"],
                "stop_reason": tr["stop_reason"], "commands": tr["commands"]} for tr in traces]
        traces.clear()
        rows += new
        with open(jsonl, "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in new)
        done = sum(1 for r in rows if r["reward"] > 0)
        print(f"[{min(i + per_call, len(chosen))}/{len(chosen)} templates] "
              f"{done}/{len(rows)} passed so far, {time.time() - t0:.0f}s", flush=True)

    s = summarize(rows, trained_on)
    md = render(a.tag, s, trained_on)
    (Path(a.out) / f"{a.tag}.md").write_text(md)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
