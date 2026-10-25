"""Measure whether a template is at the right difficulty for our policy.

GRPO's advantage is (r - mean)/std over a group of G rollouts of the SAME
prompt. If every rollout scores the same, every advantage is zero and the step
teaches nothing:

    0 0 0 0 0 0 0 0  -> no gradient, task too hard
    1 1 1 1 1 1 1 1  -> no gradient, task too easy
    0 1 0 0 1 0 0 1  -> signal

So the number we optimise is the fraction of templates producing a MIXED group,
not the aggregate pass rate: a set averaging 50% built from half-trivial and
half-impossible tasks produces no gradient at all.

A group shares ONE fixture instance. Varying the seed makes a different group,
never a bigger one -- rollouts facing different fixtures face different
difficulties, so their mean stops being a baseline.

Results are appended as they finish, because a full pass takes hours and a crash
three quarters through should not cost the whole run.

Run:
  ./.venv/bin/python -m src.tasks.calibrate data/templates/templates.jsonl \\
      --base-url http://127.0.0.1:8080/v1 --g 8 --concurrency 2
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..harness.episode import Episode, run_episode
from ..harness.policy import APIPolicy
from .template import TaskTemplate, load_templates


@dataclass
class GroupOutcome:
    """One group: G rollouts against a single instance of one template."""

    template_id: str
    seed: int
    rewards: list[float] = field(default_factory=list)
    stop_reasons: list[str] = field(default_factory=list)
    commands_per_episode: list[int] = field(default_factory=list)
    seconds: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rewards if r > 0)

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        if not self.rewards:
            return "EMPTY"
        if self.passed == 0:
            return "ALL-FAIL"
        if self.passed == len(self.rewards):
            return "ALL-PASS"
        return "MIXED"

    @property
    def useful(self) -> bool:
        """Only a mixed group produces a gradient."""
        return self.verdict == "MIXED"

    @property
    def advice(self) -> str:
        return {"MIXED": "keep",
                "ALL-PASS": "make harder",
                "ALL-FAIL": "make easier, or drop",
                "ERROR": "investigate",
                "EMPTY": "investigate"}[self.verdict]


def calibrate_group(
    template: TaskTemplate,
    seed: int,
    policy: APIPolicy,
    *,
    g: int = 8,
    concurrency: int = 2,
    save_to: Optional[str] = None,
) -> GroupOutcome:
    """Run G rollouts against ONE instance of this template."""
    out = GroupOutcome(template_id=template.id, seed=seed)
    started = time.monotonic()
    fixture = tempfile.mkdtemp(prefix="calib-")
    try:
        spec = template.instantiate(seed, fixture)

        def one(_: int) -> Episode:
            # run_episode copies the fixture, so rollouts cannot see each other.
            return run_episode(spec, fixture, policy)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            episodes = list(pool.map(one, range(g)))

        out.rewards = [e.reward for e in episodes]
        out.stop_reasons = [e.stop_reason for e in episodes]
        out.commands_per_episode = [len(e.commands) for e in episodes]

        if save_to:
            # Keep the trajectories: they are the input to the monitor, and
            # River's 12 behaviour features are extracted from exactly this.
            os.makedirs(save_to, exist_ok=True)
            path = os.path.join(save_to, f"{template.id}-seed{seed}.jsonl")
            with open(path, "w") as fh:
                for e in episodes:
                    fh.write(json.dumps({
                        "task_id": e.task_id, "reward": e.reward,
                        "stop_reason": e.stop_reason, "messages": e.messages,
                        "commands": e.commands, "diff": e.diff,
                        "failures": (e.result.failures if e.result else []),
                        "gate_failed": (e.result.gate_failed if e.result else None),
                    }) + "\n")
    except Exception as e:                       # noqa: BLE001 - record and continue
        out.error = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(fixture, ignore_errors=True)
        out.seconds = time.monotonic() - started
    return out


def calibrate(
    templates: list[TaskTemplate],
    policy: APIPolicy,
    *,
    seeds: tuple[int, ...] = (1,),
    g: int = 8,
    concurrency: int = 2,
    save_to: Optional[str] = None,
    append_to: Optional[str] = None,
    verbose: bool = True,
) -> list[GroupOutcome]:
    out: list[GroupOutcome] = []
    total = len(templates) * len(seeds)
    for i, template in enumerate(templates):
        for seed in seeds:
            res = calibrate_group(template, seed, policy, g=g,
                                  concurrency=concurrency, save_to=save_to)
            out.append(res)
            if verbose:
                n = len(out)
                print(f"  [{n:3d}/{total}] {res.template_id:22s} seed={seed} "
                      f"{res.passed}/{g:<2d} {res.verdict:9s} {res.seconds:6.1f}s"
                      + (f"  {res.error}" if res.error else ""), flush=True)
            if append_to:
                with open(append_to, "a") as fh:
                    fh.write(json.dumps(asdict(res) | {"verdict": res.verdict}) + "\n")
    return out


def report(outcomes: list[GroupOutcome], g: int = 8) -> str:
    lines = [f"{'template':24s}{'seed':>5s}{'passed':>8s}  {'verdict':10s}{'advice':22s}{'mean s':>8s}"]
    lines.append("-" * 80)
    for o in outcomes:
        lines.append(f"{o.template_id:24s}{o.seed:5d}{o.passed:5d}/{len(o.rewards) or g:<2d}  "
                     f"{o.verdict:10s}{o.advice:22s}{o.seconds / max(len(o.rewards), 1):8.1f}")
    mixed = [o for o in outcomes if o.useful]
    lines.append("-" * 80)
    lines.append(f"{len(mixed)}/{len(outcomes)} groups are MIXED "
                 f"({100 * len(mixed) / max(len(outcomes), 1):.0f}%) -- these are the ones that train")
    all_r = [r for o in outcomes for r in o.rewards]
    if all_r:
        lines.append(f"aggregate pass rate {100 * sum(1 for r in all_r if r > 0) / len(all_r):.1f}% "
                     f"over {len(all_r)} episodes "
                     f"(NOT the metric -- a 50% average of trivial and impossible tasks trains nothing)")
    stops: dict[str, int] = {}
    for o in outcomes:
        for s in o.stop_reasons:
            stops[s] = stops.get(s, 0) + 1
    if stops:
        lines.append("stop reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(stops.items())))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="JSONL file of task templates")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--g", type=int, default=8, help="rollouts per group")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-thinking", action="store_true",
                    help="Qwen3 non-thinking mode (note: it stops following the format)")
    ap.add_argument("--only", nargs="+", help="limit to these template ids")
    ap.add_argument("--save-to", default="data/rollouts",
                    help="directory for full trajectories ('' to skip)")
    ap.add_argument("--append-to", default="data/rollouts/calibration.jsonl")
    args = ap.parse_args(argv)

    templates = load_templates(args.path)
    if args.only:
        templates = [t for t in templates if t.id in set(args.only)]
    policy = APIPolicy(base_url=args.base_url, model=args.model,
                       temperature=args.temperature, max_tokens=args.max_tokens,
                       thinking=not args.no_thinking)

    if args.append_to:
        os.makedirs(os.path.dirname(args.append_to) or ".", exist_ok=True)
    print(f"calibrating {len(templates)} templates x {len(args.seeds)} seed(s) "
          f"x {args.g} rollouts = {len(templates) * len(args.seeds) * args.g} episodes "
          f"| thinking={not args.no_thinking}\n")
    outcomes = calibrate(templates, policy, seeds=tuple(args.seeds), g=args.g,
                         concurrency=args.concurrency,
                         save_to=args.save_to or None, append_to=args.append_to)
    print("\n" + report(outcomes, g=args.g))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
