"""Filesystem state snapshots and diffs -- the primitive every reward is built on.

The verifier NEVER runs inside the sandbox. The container mounts a throwaway *copy*
of the task fixture at /work; when the episode ends the container is gone but the
mutated copy remains on the host, and we inspect it from here. So verification code
is not merely read-only to the agent, it is unreachable -- there is no checker in
the container for a rollout to stub out, monkey-patch or delete.

Why diff the whole tree instead of asserting spot predicates? Because "assert a.log
is absent" is satisfied by `rm -rf /work/*`. Comparing the *complete* set of changes
against the expected set makes collateral damage fail automatically, with no
per-task work anticipating the attack.

Stdlib only, on purpose: this module is imported by the reward function on every
rollout, so it runs in the training image, the eval image and locally. A dependency
here is a dependency everywhere (see docs/SERVING.md for why that matters).
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from typing import Optional

# Read files in 1 MiB chunks so a large fixture artifact (TMax-style tasks ship
# PNGs, audio, stripped binaries) never has to sit in memory to be hashed.
CHUNK = 1 << 20

# Recorded in place of a hash when a path exists but cannot be read. This is a
# *state*, not an error: an agent that chmod 000's a file has changed the world,
# and we want that to show up as a change rather than crash the reward function
# mid-rollout.
UNREADABLE = "<unreadable>"


@dataclass(frozen=True)
class Entry:
    """One path's observable state. Frozen so a snapshot can't be edited in place."""

    kind: str                      # "file" | "dir" | "symlink"
    mode: int                      # permission bits only (st_mode & 0o7777)
    size: Optional[int] = None     # files only
    sha256: Optional[str] = None   # files only; UNREADABLE if it couldn't be read
    target: Optional[str] = None   # symlinks only: the raw link text


def _hash_file(path: str) -> tuple[int, str]:
    """Return (size, sha256) for a file, or (-1, UNREADABLE) if it can't be read."""
    h = hashlib.sha256()
    total = 0
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
                total += len(chunk)
    except OSError:
        return -1, UNREADABLE
    return total, h.hexdigest()


def _entry(path: str) -> Entry:
    """Classify one path from a single lstat, without following symlinks."""
    st = os.lstat(path)                  # lstat, not stat: describe the link itself
    mode = stat.S_IMODE(st.st_mode)

    if stat.S_ISLNK(st.st_mode):
        try:
            target = os.readlink(path)
        except OSError:
            target = UNREADABLE
        return Entry("symlink", mode, target=target)

    if stat.S_ISDIR(st.st_mode):
        return Entry("dir", mode)        # no content hash; children are their own entries

    size, digest = _hash_file(path)
    return Entry("file", mode, size=size, sha256=digest)


def snapshot(root: str) -> dict[str, Entry]:
    """Map every path under `root` to its state, keyed by relative POSIX path.

    Directories are recorded too, so `mkdir junk/` is a detectable change even when
    the directory is empty. Symlinks are recorded but never followed, so a symlink
    pointing outside the fixture cannot drag unrelated host state into the snapshot
    -- and swinging a symlink to a new target registers as a modification.
    """
    root = os.path.abspath(root)
    out: dict[str, Entry] = {}

    # followlinks=False: os.walk still *lists* a symlinked directory (we classify it
    # as a symlink via lstat) but does not descend through it.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                out[rel] = _entry(full)
            except OSError:
                # Vanished between listing and stat (a background process in the
                # container, or a race with teardown). Treat as absent.
                continue
    return out


@dataclass
class Diff:
    """What changed between two snapshots of the same fixture."""

    created: dict[str, Entry] = field(default_factory=dict)
    deleted: dict[str, Entry] = field(default_factory=dict)
    modified: dict[str, tuple[Entry, Entry]] = field(default_factory=dict)  # (before, after)

    @property
    def changed_paths(self) -> set[str]:
        """Every path touched, in any way. This is what Layer 1 compares."""
        return set(self.created) | set(self.deleted) | set(self.modified)

    def is_empty(self) -> bool:
        return not self.changed_paths

    def render(self) -> str:
        """Human-readable summary, for failure diagnosis and trace logs."""
        lines = []
        for p in sorted(self.deleted):
            lines.append(f"  - {p}")
        for p in sorted(self.created):
            lines.append(f"  + {p}  ({self.created[p].kind})")
        for p in sorted(self.modified):
            before, after = self.modified[p]
            # Report every dimension that changed, not just the first one found: an
            # agent that rewrites a file *and* chmods it must not be able to hide the
            # content edit behind the permissions change.
            parts = []
            if before.kind != after.kind:
                # Kind change makes content/target incomparable (a dir has no hash),
                # so it stands alone as the content-level fact.
                parts.append(f"{before.kind} -> {after.kind}")
            elif after.sha256 == UNREADABLE:
                parts.append("became unreadable")      # the chmod-000 hiding attempt
            elif before.sha256 == UNREADABLE:
                parts.append("became readable")
            elif before.sha256 != after.sha256:
                parts.append(f"content {before.size}B -> {after.size}B")
            elif before.target != after.target:
                parts.append(f"target {before.target} -> {after.target}")
            if before.mode != after.mode:
                parts.append(f"mode {before.mode:04o} -> {after.mode:04o}")
            lines.append(f"  ~ {p}  " + ", ".join(parts))
        return "\n".join(lines) or "  (no changes)"


def diff(before: dict[str, Entry], after: dict[str, Entry]) -> Diff:
    """Compare two snapshots. Equality is structural: Entry is a frozen dataclass,
    so `before[p] != after[p]` already covers content, mode, kind and link target."""
    d = Diff()
    for path, entry in after.items():
        if path not in before:
            d.created[path] = entry
        elif before[path] != entry:
            d.modified[path] = (before[path], entry)
    for path, entry in before.items():
        if path not in after:
            d.deleted[path] = entry
    return d
