"""Task definitions and verification -- turning a filesystem diff into a reward.

Verification is layered, and the layering is the anti-hacking property:

  Layer 1  the SET of changed paths must equal what the task expected, and each
           path must have changed in the WAY the task expected.
           A hard gate: wrong set -> reward 0, content checks never run.
           This is what makes `rm -rf /work/*` score zero without any task
           author having anticipated it.
  Layer 2  content checks on the paths that were legitimately touched.
           Graded: the fraction that passed.
  Layer 3  reward. Binary (Layer 1 AND all of Layer 2) by default.

Layer 1 cannot be graded. If a rollout earned partial credit for deleting the
right files *among* the wrong ones, the reward would be paying for collateral
damage -- so an unexpected change is fatal, not a deduction.

Checks are declarative JSON rather than Python callables because tasks have to
round-trip through JSONL and parquet: they come out of CuratorKit, live in a HF
dataset, and reach AlignTune as dataset columns. A callable cannot be stored in
a parquet file. New check kinds are one decorated function in the registry below.

Malformed tasks raise TaskError at LOAD time, never at verify time. A task with a
typo'd check kind would otherwise be silently unsatisfiable -- it would train the
model that correct behaviour scores zero, which is worse than any crash. We would
rather fail before a single GPU-second is spent.

Stdlib only -- see state.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .state import UNREADABLE, Diff, Entry, diff


class TaskError(Exception):
    """A task definition is malformed. Always a bug in the task, never the agent."""


# A check returns None when it passes, or a human-readable reason when it fails.
# "None means OK" keeps the call site a plain truthiness test while still carrying
# a diagnosis string, which the monitor and our debugging both need.
CheckFn = Callable[[str, dict, dict], Optional[str]]

_CHECKS: dict[str, CheckFn] = {}
_CHECK_REQUIRES: dict[str, tuple[str, ...]] = {}


def check(kind: str, requires: tuple[str, ...] = ("path",)):
    """Register a check kind so `{"kind": "<kind>", ...}` dispatches to it.

    `requires` names the keys the spec must carry. TaskSpec.validate() enforces
    them at load time so a check can index its spec directly without defensive
    .get() calls -- a missing key is impossible by the time verify() runs.
    """
    def deco(fn: CheckFn) -> CheckFn:
        _CHECKS[kind] = fn
        _CHECK_REQUIRES[kind] = requires
        return fn
    return deco


def _read(root: str, rel: str) -> Optional[bytes]:
    """Read a fixture file, or None if absent/unreadable.

    Never raises: this runs inside the reward function, and the agent controls
    these files. A crash here kills a training step.
    """
    try:
        with open(f"{root}/{rel}", "rb") as fh:
            return fh.read()
    except OSError:
        return None


# --- check kinds -------------------------------------------------------------

@check("sha256", requires=("path", "value"))
def _sha256(root: str, spec: dict, after: dict) -> Optional[str]:
    """Exact content match by hash. The strictest check; needs a known-good file."""
    entry = after.get(spec["path"])
    if entry is None:
        return f"{spec['path']}: absent"
    if entry.sha256 == UNREADABLE:
        return f"{spec['path']}: unreadable"
    if entry.sha256 != spec["value"]:
        return f"{spec['path']}: sha256 mismatch"
    return None


@check("equals", requires=("path", "text"))
def _equals(root: str, spec: dict, after: dict) -> Optional[str]:
    """Exact text match. Note trailing newlines: `echo` adds one, `printf` does not."""
    data = _read(root, spec["path"])
    if data is None:
        return f"{spec['path']}: absent or unreadable"
    want = str(spec["text"]).encode()   # a selector may resolve to an int
    if data != want:
        return f"{spec['path']}: expected {want!r}, got {data[:80]!r}"
    return None


@check("contains", requires=("path", "text"))
def _contains(root: str, spec: dict, after: dict) -> Optional[str]:
    """Substring present. Lenient -- use when output format is not fully pinned."""
    data = _read(root, spec["path"])
    if data is None:
        return f"{spec['path']}: absent or unreadable"
    if str(spec["text"]).encode() not in data:
        return f"{spec['path']}: missing {spec['text']!r}"
    return None


@check("regex", requires=("path", "pattern"))
def _regex(root: str, spec: dict, after: dict) -> Optional[str]:
    """Pattern match over the decoded text. `multiline` makes ^/$ per-line."""
    data = _read(root, spec["path"])
    if data is None:
        return f"{spec['path']}: absent or unreadable"
    flags = re.MULTILINE if spec.get("multiline") else 0
    if not re.search(spec["pattern"], data.decode("utf-8", "replace"), flags):
        return f"{spec['path']}: no match for /{spec['pattern']}/"
    return None


@check("line_count", requires=("path", "value"))
def _line_count(root: str, spec: dict, after: dict) -> Optional[str]:
    """Number of non-empty lines. A cheap proxy for 'found the right N things'."""
    data = _read(root, spec["path"])
    if data is None:
        return f"{spec['path']}: absent or unreadable"
    n = len([ln for ln in data.decode("utf-8", "replace").splitlines() if ln.strip()])
    if n != spec["value"]:
        return f"{spec['path']}: expected {spec['value']} lines, got {n}"
    return None


@check("contains_each", requires=("path", "texts"))
def _contains_each(root: str, spec: dict, after: dict) -> Optional[str]:
    """Every string in `texts` must appear. Templates fill this from a tag, so a
    randomised fixture's generated filenames become the expected receipt lines."""
    data = _read(root, spec["path"])
    if data is None:
        return f"{spec['path']}: absent or unreadable"
    missing = [t for t in spec["texts"] if str(t).encode() not in data]
    if missing:
        return f"{spec['path']}: missing {missing}"
    return None


@check("mode", requires=("path", "value"))
def _mode(root: str, spec: dict, after: dict) -> Optional[str]:
    """Permission bits, e.g. {"kind":"mode","path":"run.sh","value":"0755"}."""
    entry = after.get(spec["path"])
    if entry is None:
        return f"{spec['path']}: absent"
    want = int(spec["value"], 8)
    if entry.mode != want:
        return f"{spec['path']}: mode {entry.mode:04o}, expected {want:04o}"
    return None


# --- task spec ---------------------------------------------------------------

@dataclass
class TaskSpec:
    """One verifiable terminal task.

    `reference` is a list of shell commands known to solve the task, run in order
    in one persistent shell. A list rather than a single line because a one-liner
    over a randomised fixture gets fragile fast (pipelines that break when a glob
    matches nothing), and a failing step is far easier to locate. It is never shown to
    the model -- it exists so we can validate the task itself (does the task's own
    solution pass its own verifier?). TMax deliberately skips this validation and
    lets RL soft-filter zero-pass tasks instead; at 0.6B we cannot, because we
    could not distinguish "task is broken" from "model can't do it".
    """

    id: str
    instruction: str                                   # the prompt the model sees
    fixture: str                                       # fixture directory name
    expect_deleted: set[str] = field(default_factory=set)
    expect_created: set[str] = field(default_factory=set)
    expect_modified: set[str] = field(default_factory=set)
    checks: list[dict] = field(default_factory=list)   # Layer 2, declarative
    reference: Optional[list[str]] = None              # task-validation only
    max_turns: int = 5                                 # see docs/README.md §7

    @property
    def expected_paths(self) -> set[str]:
        return self.expect_deleted | self.expect_created | self.expect_modified

    def validate(self) -> "TaskSpec":
        """Raise TaskError on anything that would make this task unsatisfiable.

        Returns self so it can be chained. Called automatically by from_dict().
        """
        if not self.id or not self.instruction or not self.fixture:
            raise TaskError(f"{self.id!r}: id, instruction and fixture are required")

        if self.max_turns < 1:
            raise TaskError(f"{self.id!r}: max_turns must be >= 1")

        if self.reference is not None:
            if not isinstance(self.reference, list) or not self.reference:
                raise TaskError(f"{self.id!r}: reference must be a non-empty list of commands")
            if not all(isinstance(c, str) and c.strip() for c in self.reference):
                raise TaskError(f"{self.id!r}: reference commands must be non-empty strings")

        if not self.expected_paths:
            raise TaskError(
                f"{self.id!r}: expects no changes at all; a task whose correct "
                "solution is to do nothing teaches nothing"
            )

        # A path cannot be in two expectation sets: deleted-and-created is not a
        # state the diff can ever report (it reports one or the other).
        for a, b in (("deleted", "created"), ("deleted", "modified"), ("created", "modified")):
            overlap = getattr(self, f"expect_{a}") & getattr(self, f"expect_{b}")
            if overlap:
                raise TaskError(f"{self.id!r}: {sorted(overlap)} in both expect_{a} and expect_{b}")

        checked_paths = set()
        for i, spec in enumerate(self.checks):
            if not isinstance(spec, dict) or "kind" not in spec:
                raise TaskError(f"{self.id!r}: check {i} has no 'kind'")
            kind = spec["kind"]
            if kind not in _CHECKS:
                raise TaskError(
                    f"{self.id!r}: check {i} has unknown kind {kind!r}; "
                    f"known kinds are {sorted(_CHECKS)}"
                )
            missing_keys = [k for k in _CHECK_REQUIRES[kind] if k not in spec]
            if missing_keys:
                raise TaskError(f"{self.id!r}: check {i} ({kind}) missing {missing_keys}")
            path = spec["path"]
            # A content check on a deleted path can never pass -- the file is gone.
            if path in self.expect_deleted:
                raise TaskError(
                    f"{self.id!r}: check {i} targets {path!r}, which expect_deleted "
                    "says will not exist"
                )
            checked_paths.add(path)

        # "Modified somehow" is a weak assertion that a stray byte satisfies, so
        # every modified path must carry at least one content check.
        unchecked = self.expect_modified - checked_paths
        if unchecked:
            raise TaskError(
                f"{self.id!r}: expect_modified paths {sorted(unchecked)} have no "
                "content check; 'modified' alone is too weak to verify"
            )

        return self

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(
            id=str(d["id"]),
            instruction=d["instruction"],
            fixture=d["fixture"],
            expect_deleted=set(d.get("expect_deleted", [])),
            expect_created=set(d.get("expect_created", [])),
            expect_modified=set(d.get("expect_modified", [])),
            checks=list(d.get("checks", [])),
            # Tolerate the single-command form so older task files still load.
            reference=([ref] if isinstance(ref := d.get("reference"), str) else ref),
            max_turns=int(d.get("max_turns", 5)),
        ).validate()

    def to_dict(self) -> dict:
        """Round-trip to JSONL. Sets become sorted lists so output is stable."""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "fixture": self.fixture,
            "expect_deleted": sorted(self.expect_deleted),
            "expect_created": sorted(self.expect_created),
            "expect_modified": sorted(self.expect_modified),
            "checks": self.checks,
            "reference": self.reference,
            "max_turns": self.max_turns,
        }


def load_tasks(path: str) -> list[TaskSpec]:
    """Load a JSONL task file. Raises TaskError naming the line that is broken."""
    tasks, seen = [], set()
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                task = TaskSpec.from_dict(json.loads(line))
            except (TaskError, KeyError, ValueError) as e:
                raise TaskError(f"{path}:{lineno}: {e}") from e
            if task.id in seen:
                raise TaskError(f"{path}:{lineno}: duplicate task id {task.id!r}")
            seen.add(task.id)
            tasks.append(task)
    return tasks


# --- verification ------------------------------------------------------------

@dataclass
class VerifyResult:
    """Why a rollout scored what it scored.

    `reward` is what GRPO consumes. Everything else exists for diagnosis: the
    monitor needs to tell collateral damage apart from an honest wrong answer,
    because they are the same number and completely different problems.
    """

    passed: bool
    reward: float
    score: float                    # graded: fraction of Layer 2 checks passed
    gate_failed: bool               # True if Layer 1 rejected it
    failures: list[str] = field(default_factory=list)
    unexpected: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)

    def render(self) -> str:
        head = "PASS" if self.passed else ("GATE FAIL" if self.gate_failed else "FAIL")
        lines = [f"{head}  reward={self.reward:.2f} score={self.score:.2f}"]
        lines += [f"    {f}" for f in self.failures]
        return "\n".join(lines)


def verify(
    task: TaskSpec,
    root: str,
    before: dict[str, Entry],
    after: dict[str, Entry],
    graded: bool = False,
) -> VerifyResult:
    """Score one finished episode.

    `root` is the mutated fixture copy on the host -- never a path inside the
    container. `graded=True` returns partial Layer 2 credit as the reward instead
    of a strict boolean; use it only if difficulty calibration shows all-zero
    groups (see docs/README.md §3a).
    """
    d = diff(before, after)

    # --- Layer 1: hard gate on the set of changed paths ---------------------
    unexpected = d.changed_paths - task.expected_paths
    missing = task.expected_paths - d.changed_paths

    # Also require each path to have changed in the way the task expected: a file
    # that was supposed to be deleted but was merely emptied is not a pass.
    wrong_kind = []
    for p in task.expect_deleted & d.changed_paths:
        if p not in d.deleted:
            wrong_kind.append(f"{p}: expected deletion, was modified")
    for p in task.expect_created & d.changed_paths:
        if p not in d.created:
            wrong_kind.append(f"{p}: expected creation, was modified")

    if unexpected or missing or wrong_kind:
        failures = []
        if unexpected:
            failures.append(f"unexpected changes: {sorted(unexpected)}")
        if missing:
            failures.append(f"expected but unchanged: {sorted(missing)}")
        failures += wrong_kind
        return VerifyResult(
            passed=False, reward=0.0, score=0.0, gate_failed=True,
            failures=failures, unexpected=unexpected, missing=missing,
        )

    # --- Layer 2: content checks on legitimately-touched paths --------------
    failures = []
    for spec in task.checks:
        fn = _CHECKS.get(spec["kind"])
        if fn is None:
            # Unreachable for any task built through from_dict/validate. If we get
            # here the task bypassed validation, and scoring 0 would quietly teach
            # the model that correct behaviour is wrong -- so fail loudly instead.
            raise TaskError(
                f"{task.id!r}: unknown check kind {spec['kind']!r} reached verify(); "
                "the task was not validated"
            )
        reason = fn(root, spec, after)
        if reason:
            failures.append(reason)

    n = len(task.checks)
    score = 1.0 if n == 0 else (n - len(failures)) / n
    passed = not failures

    return VerifyResult(
        passed=passed,
        reward=(score if graded else (1.0 if passed else 0.0)),
        score=score,
        gate_failed=False,
        failures=failures,
    )
