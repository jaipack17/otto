"""Adversarial action generation -- racing the policy to our own verifier's bugs.

GRPO is already searching for these: it samples thousands of episodes and keeps
whatever scores highest, with no interest in whether the reward came from solving
the task or from a flaw in the check. If `touch result.txt` scores 1.0, the policy
will find it, converge on it, and the reward curve will look excellent. The point
of this module is to run that search ourselves first. (The general technique is
called *fuzzing*; `random_cheats` below is the part that literally is one.)

The property being tested is two-sided, and the positive control is NOT optional:

    reference solution  -> reward == 1.0
    every cheat         -> reward == 0.0

A verifier that returns 0.0 for everything satisfies the second clause perfectly.
Without the first, the suite proves nothing.

This finds bugs; it never proves their absence. Treat it as a ratchet: every cheat
a real rollout discovers gets added here as a permanent regression test.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from .state import snapshot
from .task import TaskSpec, VerifyResult, verify

# An action mutates a fixture copy in place, exactly as an episode would.
Action = Callable[[str], None]


@dataclass
class Cheat:
    """A named adversarial action that must score zero."""

    label: str
    apply: Action


def from_dir(src: str) -> Action:
    """Make a fixture builder that copies a real fixture directory."""
    def build(root: str) -> None:
        # dirs_exist_ok: tempfile already created root.
        shutil.copytree(src, root, symlinks=True, dirs_exist_ok=True)
    return build


def score_action(
    task: TaskSpec,
    build: Action,
    action: Action,
    graded: bool = False,
) -> VerifyResult:
    """Build a throwaway fixture, apply `action`, and score it.

    This is the episode cycle with the model removed: copy fixture -> mutate ->
    snapshot before/after -> verify. The phase-1 harness replaces `action` with
    "let the policy drive the container" and keeps everything else.

    The fixture copy is always a fresh temp dir and is always destroyed, so a
    destructive action cannot leak into the next case or touch the real fixture.
    """
    root = tempfile.mkdtemp(prefix="cheat-")
    try:
        build(root)
        before = snapshot(root)
        try:
            action(root)
        except OSError:
            # A cheat that fails partway through is still a legitimate episode:
            # the agent tried something and the filesystem refused. Score the
            # state it actually left behind.
            pass
        return verify(task, root, before, snapshot(root), graded=graded)
    finally:
        # chmod back up first: a cheat may have removed our own write permission
        # on a directory, which would make rmtree fail and leak temp dirs.
        for dirpath, dirnames, _ in os.walk(root):
            for d in dirnames:
                try:
                    os.chmod(os.path.join(dirpath, d), 0o700)
                except OSError:
                    pass
        shutil.rmtree(root, ignore_errors=True)


# --- the cheats we can name --------------------------------------------------

def _existing(root: str) -> list[str]:
    """Relative paths present in the fixture, deepest first (safe delete order)."""
    return sorted(snapshot(root), key=lambda p: p.count("/"), reverse=True)


def _rm(root: str, rel: str) -> None:
    full = f"{root}/{rel}"
    try:
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        else:
            os.remove(full)
    except OSError:
        pass


def known_cheats(task: TaskSpec, correct: Optional[Action] = None) -> list[Cheat]:
    """The attacks we can name, derived from the task spec so they fit any task.

    If `correct` is supplied (a Python stand-in for the reference solution), the
    strongest cases become available: a *correct* solution plus collateral damage.
    Those are the ones that isolate Layer 1, because every content check passes.
    """
    cheats: list[Cheat] = []

    def add(label: str, fn: Action) -> None:
        cheats.append(Cheat(label, fn))

    # The laziest possible policy.
    add("do nothing", lambda r: None)

    # Scorched earth. Satisfies every "X is absent" predicate ever written.
    def nuke(r: str) -> None:
        for rel in _existing(r):
            _rm(r, rel)
    add("rm -rf everything", nuke)

    if task.expect_deleted and (task.expect_created or task.expect_modified):
        # Satisfy the deletions, skip the rest.
        #
        # Guarded: for a task whose entire specification IS the deletion, this
        # cheat performs exactly the reference solution and would score 1.0 --
        # a false positive that rejects a sound task rather than finding a weak
        # verifier. Only a cheat when there is something else left undone.
        def delete_only(r: str) -> None:
            for rel in task.expect_deleted:
                _rm(r, rel)
        add("delete expected, create nothing", delete_only)

        # Truncate rather than delete: the path still exists, so a careless
        # verifier that only checked "file is empty" would pass this.
        def truncate(r: str) -> None:
            for rel in task.expect_deleted:
                try:
                    open(f"{r}/{rel}", "w").close()
                except OSError:
                    pass
        add("truncate instead of delete", truncate)

        # Hide from the verifier instead of acting.
        def hide(r: str) -> None:
            for rel in task.expect_deleted:
                try:
                    os.chmod(f"{r}/{rel}", 0o000)
                except OSError:
                    pass
        add("chmod 000 instead of delete", hide)

    if task.expect_created:
        # Fake the outputs: right paths, empty content.
        def touch_outputs(r: str) -> None:
            for rel in task.expect_created:
                try:
                    parent = os.path.dirname(f"{r}/{rel}")
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    open(f"{r}/{rel}", "w").close()
                except OSError:
                    pass
        add("touch expected outputs (empty)", touch_outputs)

        # Point an expected output at something else on disk.
        def symlink_output(r: str) -> None:
            for rel in task.expect_created:
                try:
                    os.symlink("/etc/hostname", f"{r}/{rel}")
                except OSError:
                    pass
        add("symlink expected output elsewhere", symlink_output)

    # Unexpected creation, with nothing else done.
    add("create junk file", lambda r: open(f"{r}/.cheat-junk", "w").write("x"))

    # Open everything up -- a mode change on every path.
    def chmod_all(r: str) -> None:
        for rel in _existing(r):
            try:
                os.chmod(f"{r}/{rel}", 0o777)
            except OSError:
                pass
    add("chmod 777 everything", chmod_all)

    # --- the sharp ones: a correct solution, plus damage -------------------
    if correct is not None:
        def correct_plus_junk(r: str) -> None:
            correct(r)
            open(f"{r}/.cheat-extra", "w").write("x")
        add("correct + one junk file", correct_plus_junk)

        def correct_plus_deletion(r: str) -> None:
            correct(r)
            survivors = [p for p in _existing(r) if p not in task.expected_paths]
            files = [p for p in survivors if not os.path.isdir(f"{r}/{p}")]
            if files:
                _rm(r, files[0])
        add("correct + delete a bystander", correct_plus_deletion)

        def correct_plus_chmod(r: str) -> None:
            correct(r)
            survivors = [p for p in _existing(r) if p not in task.expected_paths]
            if survivors:
                try:
                    os.chmod(f"{r}/{survivors[0]}", 0o777)
                except OSError:
                    pass
        add("correct + chmod a bystander", correct_plus_chmod)

    return cheats


# --- the cheats we did not think of ------------------------------------------

def random_cheats(task: TaskSpec, n: int = 20, seed: int = 0) -> list[Cheat]:
    """Mutations we did not think of. Seeded, so a failure is reproducible.

    We enumerate attacks badly; the policy explores exhaustively. These cover the
    space between the named cases.
    """
    cheats: list[Cheat] = []
    for i in range(n):
        rng = random.Random(seed + i)

        def mutate(r: str, rng: random.Random = rng) -> None:
            paths = _existing(r)
            for _ in range(rng.randint(1, 4)):
                op = rng.choice(["rm", "truncate", "chmod", "create", "append"])
                if op == "create":
                    name = f".cheat-{rng.randrange(10**6)}"
                    open(f"{r}/{name}", "w").write("x" * rng.randint(0, 32))
                    continue
                if not paths:
                    continue
                rel = rng.choice(paths)
                full = f"{r}/{rel}"
                try:
                    if op == "rm":
                        _rm(r, rel)
                    elif op == "truncate" and os.path.isfile(full):
                        open(full, "w").close()
                    elif op == "chmod":
                        os.chmod(full, rng.choice([0o000, 0o400, 0o644, 0o755, 0o777]))
                    elif op == "append" and os.path.isfile(full):
                        with open(full, "a") as fh:
                            fh.write("cheat\n")
                except OSError:
                    pass

        cheats.append(Cheat(f"random#{i} (seed={seed + i})", mutate))
    return cheats
