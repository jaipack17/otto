"""Acceptance tests for candidate task templates.

No template enters the training set without passing every gate here. The gates are
River's eight defect categories, turned into things we can actually execute --
their audit found that >60% of TMax's environments carry at least one quality
issue, 40.4% of them VERIFIER-TOO-WEAK, and that is the predictable cost of their
decision to skip validation. See docs/TASK_DESIGN.md Sec 3.4.

Gates, cheapest first, so a bad candidate dies before it costs anything:

  1. SCHEMA      TaskTemplate.validate() + every instantiated TaskSpec validates.
  2. BUDGET      the reference must fit inside max_turns, with room to inspect.
  3. LEAK        the instruction must not reveal a value instantiation computes.
  4. VARIETY     the answer must differ across seeds. A template whose expected
                 paths AND resolved check values are identical for every seed has
                 one fixed answer, so a policy can memorise it and never look.
  5. REFERENCE   the task's own solution must score exactly 1.0, on every seed.
                 (INSTR-ENV-MISMATCH, buggy verifiers.)
  6. CHEATS      every cheat must score 0.0, on every seed. (VERIFIER-TOO-WEAK.)

Gates 1-4 need no container. Gates 5-6 do, and are skipped with a warning when
no daemon is available -- useful on Kaggle, where there may be no Docker at all,
but a result produced that way is explicitly marked incomplete.

Run:  ./.venv/bin/python -m src.tasks.screen data/templates/templates.jsonl
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from ..harness.episode import _force_rmtree, copy_fixture, run_reference
from ..harness.sandbox import IMAGE, Sandbox
from ..verifier.cheats import from_dir, known_cheats, random_cheats, score_action
from ..verifier.state import snapshot
from ..verifier.task import TaskError
from .template import TaskTemplate, load_templates

DEFAULT_SEEDS = (1, 2, 3, 7, 11)


def docker_available(image: str = IMAGE) -> bool:
    """Retried, because Docker Desktop's resource saver pauses its VM when idle
    and the first call after a pause fails while the engine wakes."""
    for attempt in range(3):
        try:
            if not subprocess.run(["docker", "info"], capture_output=True,
                                  timeout=30).returncode and \
               not subprocess.run(["docker", "image", "inspect", image],
                                  capture_output=True, timeout=30).returncode:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
        if attempt < 2:
            subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    return False


def content_state(root: str) -> dict:
    """What a filesystem holds, ignoring permission bits. Docker Desktop forces the
    owner-write bit on bind mounts, so modes differ between a container run and a host
    run of the very same commands; kind and content do not."""
    return {p: (e.kind, e.sha256, e.target) for p, e in snapshot(root).items()}


def reference_state(spec, fixture_src: str, image: str = IMAGE) -> dict:
    """Final filesystem after the task's own reference solution, run in the sandbox."""
    root = tempfile.mkdtemp(prefix="refstate-")
    try:
        copy_fixture(fixture_src, root)
        with Sandbox(root, image=image, timeout=60.0) as sb:
            for command in spec.reference:
                sb.exec(command)
        return content_state(root)
    finally:
        _force_rmtree(root)


def recording(action):
    """Wrap a cheat so the filesystem it leaves behind is captured during the same run
    that gets scored. Random cheats hold a stateful RNG, so running one twice applies a
    different mutation the second time; recording and scoring must be one application."""
    holder: dict = {}

    def wrapped(root: str) -> None:
        try:
            action(root)
        finally:
            holder["state"] = content_state(root)

    return wrapped, holder


@dataclass
class SeedOutcome:
    """What one seed of one template produced."""

    seed: int
    answer_key: str = ""                 # fingerprint of the expected answer
    reference_reward: Optional[float] = None
    reference_failures: list[str] = field(default_factory=list)
    cheats_run: int = 0
    escaped: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ScreenResult:
    template_id: str
    accepted: bool = False
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seeds: list[SeedOutcome] = field(default_factory=list)
    executed: bool = False               # were gates 4-5 actually run?

    def render(self) -> str:
        head = "ACCEPT" if self.accepted else "REJECT"
        if self.accepted and not self.executed:
            head = "ACCEPT?"              # gates 4-5 skipped
        lines = [f"{head:8s} {self.template_id}"]
        for r in self.rejections:
            lines.append(f"           reject: {r}")
        for w in self.warnings:
            lines.append(f"           warn:   {w}")
        if self.executed and self.seeds:
            keys = len({s.answer_key for s in self.seeds})
            lines.append(f"           {len(self.seeds)} seeds, {keys} distinct answers, "
                         f"{sum(s.cheats_run for s in self.seeds)} cheat runs, "
                         f"{sum(len(s.escaped) for s in self.seeds)} escaped")
        return "\n".join(lines)


def _answer_key(spec) -> str:
    """Fingerprint of everything a policy would have to get right.

    Both the expected paths and the resolved check values, because a template can
    legitimately keep fixed paths while varying the content behind them -- "fix
    the broken config" always writes config.ini. What must not be constant is the
    *answer*.
    """
    return json.dumps(
        {"paths": sorted(spec.expected_paths), "checks": spec.checks},
        sort_keys=True, default=str,
    )


def screen(
    template: TaskTemplate,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_random_cheats: int = 25,
    image: str = IMAGE,
    execute: Optional[bool] = None,
) -> ScreenResult:
    """Run every acceptance test against one template."""
    res = ScreenResult(template_id=template.id)
    if execute is None:
        execute = docker_available(image)
    res.executed = execute
    if not execute:
        res.warnings.append("no docker: reference and cheat gates were NOT run")

    # --- gate 1: schema -----------------------------------------------------
    try:
        template.validate()
    except TaskError as e:
        res.rejections.append(f"schema: {e}")
        return res

    # --- gate 2: budget -----------------------------------------------------
    # A reference needing more commands than the turn limit allows describes a
    # task no policy can finish, however capable. The reference is also the
    # *shortest* known solution, so equality leaves no turn to look first.
    if len(template.reference) > template.max_turns:
        res.rejections.append(
            f"budget: reference needs {len(template.reference)} commands but "
            f"max_turns is {template.max_turns}, so the task is unsolvable")
    elif len(template.reference) == template.max_turns:
        res.warnings.append(
            f"budget: reference uses all {template.max_turns} turns, leaving none "
            "for the agent to inspect the fixture first")

    # --- gate 3: leak -------------------------------------------------------
    for w in template.check_no_leak():
        res.rejections.append(f"leak: {w}")

    # --- gates 4-6, per seed ------------------------------------------------
    for seed in seeds:
        out = SeedOutcome(seed=seed)
        root = tempfile.mkdtemp(prefix="screen-")
        try:
            try:
                spec = template.instantiate(seed, root)
            except (TaskError, KeyError, ValueError) as e:
                res.rejections.append(f"seed {seed}: instantiation failed: {e}")
                continue
            out.answer_key = _answer_key(spec)

            if execute:
                ref = run_reference(spec, root, image=image)
                out.reference_reward = ref.reward
                out.reference_failures = list(ref.failures)
                if ref.reward != 1.0:
                    res.rejections.append(
                        f"seed {seed}: reference scored {ref.reward:.2f}, "
                        f"not 1.0 ({'; '.join(ref.failures[:2])})")

                build = from_dir(root)
                cheats = known_cheats(spec) + random_cheats(
                    spec, n=n_random_cheats, seed=seed)
                out.cheats_run = len(cheats)
                ref_state = None
                for c in cheats:
                    action, held = recording(c.apply)
                    r = score_action(spec, build, action).reward
                    if r <= 0.0:
                        continue
                    # A random mutation can land on the real solution (for instance by
                    # picking `rm` of the one directory the task asks to delete). If its
                    # final filesystem is identical to the reference's, it is a correct
                    # answer reached by luck, not an exploit, so it is not a defect.
                    if ref_state is None:
                        ref_state = reference_state(spec, root, image)
                    if held.get("state") == ref_state:
                        res.warnings.append(
                            f"seed {seed}: random cheat {c.label!r} is an accidental "
                            "solution (same final state as the reference)")
                    else:
                        out.escaped.append((c.label, r))
                for label, r in out.escaped:
                    res.rejections.append(
                        f"seed {seed}: cheat {label!r} scored {r:.2f}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
        res.seeds.append(out)

    # --- gate 4: variety ----------------------------------------------------
    keys = {s.answer_key for s in res.seeds if s.answer_key}
    if len(keys) == 1 and len(res.seeds) > 1:
        res.rejections.append(
            f"variety: all {len(res.seeds)} seeds produce the same answer, so the "
            "task is memorisable -- randomise filenames or counts")

    res.accepted = not res.rejections
    return res


def screen_all(
    templates: list[TaskTemplate],
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_random_cheats: int = 25,
    image: str = IMAGE,
) -> list[ScreenResult]:
    """Screen a whole candidate set. Probes docker once, not per template."""
    execute = docker_available(image)
    return [screen(t, seeds=seeds, n_random_cheats=n_random_cheats,
                   image=image, execute=execute) for t in templates]


def report(results: list[ScreenResult]) -> str:
    accepted = [r for r in results if r.accepted]
    lines = [r.render() for r in results]
    lines.append("")
    lines.append(f"{len(accepted)}/{len(results)} accepted")
    if results and not results[0].executed:
        lines.append("NOTE: reference and cheat gates were skipped (no docker). "
                     "This run does not qualify a template for training.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="JSONL file of task templates")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--cheats", type=int, default=25, help="random cheats per seed")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--accepted-out", help="write accepted templates to this JSONL")
    args = ap.parse_args(argv)

    templates = load_templates(args.path)
    results = screen_all(templates, seeds=tuple(args.seeds),
                         n_random_cheats=args.cheats, image=args.image)
    print(report(results))

    if args.accepted_out:
        ok = {r.template_id for r in results if r.accepted}
        with open(args.accepted_out, "w") as fh:
            for t in templates:
                if t.id in ok:
                    fh.write(json.dumps(t.to_dict()) + "\n")
        print(f"\nwrote {len(ok)} accepted template(s) -> {args.accepted_out}")

    return 0 if all(r.accepted for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
