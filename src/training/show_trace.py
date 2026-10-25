"""Print a saved episode as a readable conversation. Stdlib only.

  python -m src.training.show_trace runs/main                      # list the latest step
  python -m src.training.show_trace runs/main --step 12            # list a given step
  python -m src.training.show_trace runs/main --step 12 --index 3  # show episode 3
  python -m src.training.show_trace runs/main --best | --worst     # of the chosen step
  python -m src.training.show_trace runs/main --template stale-logs # first match
  add --full to stop truncating long thinking and observations
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .analysis import load_jsonl

_THINK = re.compile(r"<think>\n?(.*?)\n?</think>", re.S)
_CMD = re.compile(r"<command>(.*?)</command>", re.S)
_DONE = "<done>"
_ENV_WRAP = re.compile(r"^\s*<\|im_start\|>user\n|<\|im_end\|>\s*<\|im_start\|>assistant\s*$")


def clip(text: str, full: bool, n: int) -> str:
    text = text.strip()
    return text if full or len(text) <= n else text[:n].rstrip() + f" ...[+{len(text) - n} chars]"


def step_files(run: Path) -> list[Path]:
    return sorted((run / "traces").glob("step-*.jsonl"))


def list_step(traces: list[dict], name: str) -> str:
    lines = [f"{name}: {len(traces)} episodes",
             f"{'#':>3} {'reward':>6}  {'stop':<14} {'cmds':>4} {'tok':>5}  template (seed)"]
    for i, t in enumerate(traces):
        lines.append(f"{i:>3} {t['reward']:>6.1f}  {t['stop_reason']:<14} {len(t['commands']):>4} "
                     f"{t['model_tokens']:>5}  {t['template']} ({t['seed']})")
    lines.append("show one with --index N (also --best, --worst, --template NAME, --full)")
    return "\n".join(lines)


def render(t: dict, full: bool = False) -> str:
    out = [f"{t['template']}  seed {t['seed']}   reward {t['reward']}   stop={t['stop_reason']}   "
           f"commands={len(t['commands'])}   model tokens={t['model_tokens']}   "
           f"thinking cuts={t.get('think_cuts', 0)}"]
    if t.get("failures"):
        out.append("failed checks:")
        out += [f"  - {f}" for f in t["failures"]]
    if t.get("gate_failed"):
        out.append("(failed the hard gate: wrong set of changed paths)")
    out.append(f"filesystem diff: {clip(str(t.get('diff') or '(no changes)'), full, 400)}")
    turn = 0
    for seg in t.get("segments", []):
        text = seg["text"]
        if seg["role"] == "model":
            turn += 1
            m = _THINK.search(text)
            think = m.group(1) if m else text.split("<command>")[0].replace("<think>", "")
            cmd = _CMD.search(text)
            out += ["", f"--- turn {turn}   [model, {seg.get('tokens', '?')} tokens, trained]"]
            if think.strip():
                out.append("think: " + clip(think, full, 600).replace("\n", "\n       "))
            if cmd:
                out.append(f"  $ {cmd.group(1).strip()}")
            elif _DONE in text:
                out.append("  <done>")
            elif "</think>" not in text:
                out.append("  (still thinking when the budget ran out; see the forced cut below)")
            else:
                out.append(f"  (no command) {clip(text.split('</think>')[-1], full, 200)!r}")
        else:
            body = _ENV_WRAP.sub("", text)
            if "I should stop deliberating" in body:
                out.append(f"--- [thinking budget hit: forced to answer]   (not trained)")
            else:
                out.append("--- observation (not trained): "
                           + clip(body.replace("\n", " ⏎ "), full, 300))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--index", type=int)
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--worst", action="store_true")
    ap.add_argument("--template")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)

    run = Path(a.run_dir)
    files = step_files(run)
    if not files:
        print(f"no traces in {run}/traces yet")
        return 1
    if a.step is None:
        path = files[-1]
    else:
        path = run / "traces" / f"step-{a.step:04d}.jsonl"
        if not path.exists():
            print(f"{path.name} does not exist; have steps "
                  f"{files[0].stem[5:]}..{files[-1].stem[5:]}")
            return 1
    traces = load_jsonl(str(path))
    if not traces:
        print(f"{path.name} is empty")
        return 1

    pick = None
    if a.index is not None:
        if not 0 <= a.index < len(traces):
            print(f"--index must be 0..{len(traces) - 1}")
            return 1
        pick = traces[a.index]
    elif a.best:
        pick = max(traces, key=lambda t: t["reward"])
    elif a.worst:
        pick = min(traces, key=lambda t: t["reward"])
    elif a.template:
        pick = next((t for t in traces if t["template"] == a.template), None)
        if pick is None:
            print(f"no episode of template {a.template!r} in {path.name}")
            return 1
    if pick is None:
        print(list_step(traces, path.name))
    else:
        print(f"[{path.name}]\n" + render(pick, a.full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
