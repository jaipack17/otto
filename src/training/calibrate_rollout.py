"""Calibrate every template through the REAL training rollout.

GRPO only learns from a group whose rollouts score DIFFERENTLY: all-fail and all-pass
groups have zero advantage and cost a full step for nothing. So before spending
minutes per step, find which templates actually produce mixed groups for this model.

This calls make_rollout exactly as TRL would -- same in-process generation, same
thinking budget, same stitching, same verifier -- so the numbers describe the run we
are about to do, not a stand-in. (The earlier llama.cpp calibration ran a different
harness and was invalidated by it.)

One group per template (G rollouts of one fixture instance). A different seed per
template, reproducible. Results are appended as they finish.

Run (training env):
  ./.venv-train/bin/python -u -m src.training.calibrate_rollout --G 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
import types
from dataclasses import asdict
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..harness.episode import SYSTEM_PROMPT
from ..harness.rollout import group_seed, make_rollout
from ..tasks.calibrate import GroupOutcome, report
from ..tasks.template import load_templates


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--templates", default="data/templates/templates.jsonl")
    ap.add_argument("--G", type=int, default=4, help="rollouts per template")
    ap.add_argument("--think-budget", type=int, default=256)
    ap.add_argument("--answer-cap", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--only", nargs="+", help="limit to these template ids")
    ap.add_argument("--system-prompt-file", help="use this system prompt instead of "
                    "SYSTEM_PROMPT, for comparing prompts under identical conditions")
    ap.add_argument("--step-offset", type=int, default=0, help="shifts every seed")
    ap.add_argument("--seed", type=int, default=0, help="torch sampling seed")
    ap.add_argument("--out", default="data/rollouts/calibration_rollout.jsonl")
    ap.add_argument("--mixed-out", default="data/templates/mixed.txt")
    ap.add_argument("--traces-out", default="data/rollouts/traces.jsonl",
                    help="every episode transcript, with why it scored what it did")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    templates = load_templates(a.templates)
    chosen = [t for t in templates if not a.only or t.id in set(a.only)]

    system = open(a.system_prompt_file).read().strip() if a.system_prompt_file else SYSTEM_PROMPT
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    traces: list = []
    rollout = make_rollout(templates, think_budget=a.think_budget,
                           answer_cap=a.answer_cap, max_tokens=a.max_tokens, trace=traces)
    # What TRL would pass: just the attributes the rollout reads.
    stub = types.SimpleNamespace(
        processing_class=tok, model=model, num_generations=a.G,
        accelerator=types.SimpleNamespace(device=dev, unwrap_model=lambda m: m),
        state=types.SimpleNamespace(global_step=0))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").close()
    open(a.traces_out, "w").close()
    outcomes: list[GroupOutcome] = []
    print(f"calibrating {len(chosen)} templates x {a.G} rollouts on {dev}, "
          f"think budget {a.think_budget}\n", flush=True)
    for i, t in enumerate(chosen):
        stub.state.global_step = a.step_offset + i
        started = time.monotonic()
        prompt = [{"role": "system", "content": system},
                  {"role": "user", "content": t.instruction}]
        o = GroupOutcome(template_id=t.id, seed=group_seed(stub.state.global_step, 0))
        try:
            out = rollout([prompt] * a.G, stub)
            o.rewards, o.stop_reasons = out["episode_reward"], out["stop_reason"]
            o.commands_per_episode = out["n_commands"]
        except Exception as e:                           # noqa: BLE001 - record, continue
            o.error = f"{type(e).__name__}: {e}"
        o.seconds = time.monotonic() - started
        outcomes.append(o)
        print(f"  [{i + 1:2d}/{len(chosen)}] {t.id:22s} {o.passed}/{a.G} {o.verdict:9s} "
              f"{o.seconds:5.0f}s  cmds={o.commands_per_episode} stops={sorted(set(o.stop_reasons))}"
              + (f"  {o.error}" if o.error else ""), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(asdict(o) | {"verdict": o.verdict}) + "\n")
        with open(a.traces_out, "a") as fh:
            fh.writelines(json.dumps(tr) + "\n" for tr in traces)
        traces.clear()

    print("\n" + report(outcomes, g=a.G))
    mixed = [o.template_id for o in outcomes if o.useful]
    with open(a.mixed_out, "w") as fh:
        fh.write("\n".join(mixed) + "\n")
    print(f"\n{len(mixed)} mixed templates written to {a.mixed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
