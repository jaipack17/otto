"""Pure analysis of rollout traces. Stdlib only, so it runs anywhere (the torch-free env,
the report generator, the live watcher) and is cheap to test.

A trace is one episode as the rollout records it: template, seed, reward, stop_reason,
commands, failures, gate_failed, model_tokens, think_cuts, and the transcript segments.

The behaviour features follow River (arXiv 2608.22631, Table 9): rule-based, no model in
the loop. We record them to OBSERVE what the policy does. They are never put in the reward.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable, Optional

# River: a command is "inspection" when its head is a read-only tool and it does not write.
INSPECT = {"ls", "cat", "head", "tail", "find", "grep", "stat", "wc", "tree", "file", "du",
           "pwd", "diff", "cmp"}
_WRITES = re.compile(r">|\btee\b|-delete\b|-exec\s+(rm|mv|chmod|cp)\b|\bsed\s+(-\w*i\w*)\b")
_HEAD = re.compile(r"\s*(?:\w+=\S*\s+)*([^\s|;&(){}]+)")
_EXIT = re.compile(r"\[exit (-?\d+)\]")


def load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass                      # a half-written last line while the run is live
    except FileNotFoundError:
        pass
    return out


def head(cmd: str) -> str:
    m = _HEAD.match(cmd)
    return m.group(1).rsplit("/", 1)[-1] if m else ""


def is_inspection(cmd: str) -> bool:
    return head(cmd) in INSPECT and not _WRITES.search(cmd)


def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def exit_codes(trace: dict) -> list[int]:
    """Exit status of each command, read from the observation segments in order."""
    text = "\n".join(s["text"] for s in trace.get("segments", []) if s["role"] == "env")
    return [int(c) for c in _EXIT.findall(text)]


def behaviour(trace: dict) -> dict:
    """Rule-based features of one episode. None means 'not defined for this episode'."""
    cmds = trace.get("commands", [])
    done = trace.get("stop_reason") == "done"
    codes = exit_codes(trace)
    repeated = sum(1 for i, c in enumerate(cmds) if any(jaccard(c, p) >= 0.8 for p in cmds[:i]))
    return {
        "inspect_first": (is_inspection(cmds[0]) if cmds else None),
        "verify_before_done": (any(is_inspection(c) for c in cmds[-3:]) if done and cmds else None),
        "premature_done": ((len(cmds) <= 1 or (bool(codes) and codes[-1] != 0)) if done else None),
        "repetition": (repeated / len(cmds) if cmds else None),
        "error_rate": (sum(1 for c in codes if c != 0) / len(codes) if codes else None),
        "python_use": (any(head(c) in ("python", "python3") for c in cmds) if cmds else None),
        "cmd_diversity": (len({head(c) for c in cmds}) / len(cmds) if cmds else None),
    }


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def group_verdict(passed: int, n: int) -> str:
    return "MIXED" if 0 < passed < n else ("ALL-PASS" if passed == n else "ALL-FAIL")


def summarize_rollout(traces: list[dict], G: int) -> dict:
    """Aggregate one step's episodes. Traces must be ordered group by group, which is how
    the rollout produces them, so consecutive blocks of G share one fixture instance."""
    n = len(traces)
    passed = sum(1 for t in traces if t["reward"] > 0)
    groups = []
    for i in range(0, n, G):
        block = traces[i:i + G]
        p = sum(1 for t in block if t["reward"] > 0)
        groups.append({"template": block[0]["template"], "seed": block[0]["seed"],
                       "passed": p, "n": len(block), "verdict": group_verdict(p, len(block))})
    by_template: dict[str, list[int]] = {}
    for t in traces:
        row = by_template.setdefault(t["template"], [0, 0])
        row[0] += t["reward"] > 0
        row[1] += 1
    feats = [behaviour(t) for t in traces]
    return {
        "n": n, "passed": passed, "pass_rate": passed / n if n else 0.0,
        "groups": groups,
        "mixed_groups": sum(1 for g in groups if g["verdict"] == "MIXED"),
        "zero_std_groups": sum(1 for g in groups if g["verdict"] != "MIXED"),
        "stop_reasons": dict(Counter(t["stop_reason"] for t in traces)),
        "think_cut_rate": _mean(1.0 if t.get("think_cuts", 0) > 0 else 0.0 for t in traces),
        "mean_think_cuts_per_episode": _mean(t.get("think_cuts", 0) for t in traces),
        "mean_commands": _mean(len(t["commands"]) for t in traces),
        "mean_model_tokens": _mean(t["model_tokens"] for t in traces),
        "max_model_tokens": max((t["model_tokens"] for t in traces), default=0),
        "gate_failed_rate": _mean(1.0 if t.get("gate_failed") else 0.0 for t in traces),
        "collateral_rate": _mean(1.0 if any("unexpected changes" in f for f in t.get("failures", []))
                                 else 0.0 for t in traces),
        "by_template": by_template,
        "behaviour": {k: _mean(f[k] for f in feats) for k in feats[0]} if feats else {},
    }
