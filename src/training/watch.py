"""Live view of a run. Reads the files the run writes; never touches the process.

  ./.venv-train/bin/python -m src.training.watch runs/<name>            # once
  ./.venv-train/bin/python -m src.training.watch runs/<name> --follow   # refresh every 20 s
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .analysis import load_jsonl

COLS = [("step", "step", 5), ("reward", "reward", 7), ("zero_std", "frac_reward_zero_std", 8),
        ("loss", "loss", 9), ("gnorm", "grad_norm", 7), ("entropy", "entropy", 8),
        ("len", "completions/mean_length", 7), ("lora", "lora_norm", 7), ("secs", "step_secs", 6)]


def cell(v, w):
    if v is None:
        return "-".rjust(w)
    return (f"{v:.4g}" if isinstance(v, float) else str(v)).rjust(w)


def render(run: Path, last: int = 15) -> str:
    metrics = [r for r in load_jsonl(str(run / "metrics.jsonl")) if "train_runtime" not in r]
    stats = {r["step"]: r for r in load_jsonl(str(run / "rollout_stats.jsonl"))}
    lines = [f"run: {run}   {'STOP requested' if (run / 'STOP').exists() else ''}".rstrip(),
             " ".join(h.rjust(w) for h, _, w in COLS) + "   pass  mixed  stops"]
    for r in metrics[-last:]:
        s = stats.get(r["step"], {})
        extra = ""
        if s:
            extra = f"  {s['passed']:>2}/{s['n']:<2} {s['mixed_groups']:>2}/{len(s['groups'])}   {s['stop_reasons']}"
        lines.append(" ".join(cell(r.get(k), w) for _, k, w in COLS) + extra)
    if not metrics:
        lines.append("(no steps logged yet)")
    prog = load_jsonl(str(run / "progress.jsonl"))
    if prog:
        e = prog[-1]
        what = (f"{e['event']}" + (f" chunk {e['chunk']}/{e['chunks']} turn {e['turn']}: "
                f"{e['running']} running, {e['finished']} finished" if e["event"] == "turn" else
                f" micro-batch {e['k']}/{e['of']}" if e["event"] == "micro" else ""))
        lines.append(f"now: step {e['step']} {what}  ({int(time.time() - e['t'])}s ago)")
    ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    lines.append(f"checkpoints: {[c.name for c in ckpts]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args(argv)
    while True:
        print(render(Path(a.run_dir)), flush=True)
        if not a.follow:
            return 0
        print("-" * 60, flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
