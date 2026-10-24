"""Task templates: fixture recipes that emit a fixture AND its expectations.

A static fixture is memorisable. Training draws ~4,000 prompts over 500 steps, so
with ~100 tasks each is seen ~40 times; a 0.6B policy can learn a literal command
sequence per instruction and never inspect anything. So the fixture is randomised
per episode.

That forces a change of shape. If filenames and counts vary by seed, the expected
diff cannot be a static list -- it has to be *derived from the fixture that was
just generated*. The fixture builder and the verifier therefore stop being
independent artifacts and become two outputs of one generator, which is the only
way they can be guaranteed to agree.

    TaskTemplate --instantiate(seed, dir)--> (fixture on disk, TaskSpec)

Expectations are written as selectors over *tags* rather than literal paths:

    {"tag": "stale"}            -> the paths that rule tagged "stale"
    {"count_of_tag": "stale"}   -> how many there were
    {"names_of_tag": "stale"}   -> their paths, for a contains_each check
    {"basenames_of_tag": "x"}   -> their basenames

Recipes are declarative data, not Python, for the same reason checks are: a
template is what CuratorKit generates and what round-trips through JSONL and
parquet, and a closure cannot live in a parquet cell.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..verifier.task import TaskError, TaskSpec
from .generators import WordPool, render


@dataclass
class FileRule:
    """One generation rule: make between count[0] and count[1] files like this."""

    name: str                      # pattern, may contain "/" and {word}/{int}/{hex}
    content: str = ""              # pattern too
    count: tuple[int, int] = (1, 1)
    tag: str = ""
    mode: Optional[str] = None     # octal string, e.g. "0755"

    @classmethod
    def from_dict(cls, d: dict) -> "FileRule":
        c = d.get("count", 1)
        lo, hi = (c, c) if isinstance(c, int) else (int(c[0]), int(c[1]))
        return cls(name=d["name"], content=d.get("content", ""),
                   count=(lo, hi), tag=d.get("tag", ""), mode=d.get("mode"))

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "content": self.content,
                             "count": list(self.count), "tag": self.tag}
        if self.mode:
            d["mode"] = self.mode
        return d


@dataclass
class TaskTemplate:
    """A task generator. Validated once; instantiated many times."""

    id: str
    instruction: str
    files: list[FileRule]
    expect_deleted: Any = field(default_factory=list)   # selector or literal list
    expect_created: Any = field(default_factory=list)
    expect_modified: Any = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    reference: list[str] = field(default_factory=list)  # run in order, one shell
    max_turns: int = 5

    # --- instantiation ---------------------------------------------------

    def instantiate(self, seed: int, dest: str) -> TaskSpec:
        """Write a fixture into `dest` and return the TaskSpec that verifies it.

        The returned spec is a plain TaskSpec, fully validated, so everything in
        phase 0 applies unchanged -- the Layer 1 gate, the cheat suite, verify().
        """
        rng = random.Random(seed)
        pool = WordPool(rng)
        by_tag: dict[str, list[str]] = {}
        lines_by_tag: dict[str, int] = {}
        ints_by_tag: dict[str, list[int]] = {}

        os.makedirs(dest, exist_ok=True)
        for rule in self.files:
            n = rng.randint(rule.count[0], rule.count[1])
            for _ in range(n):
                rel = render(rule.name, pool, rng)
                content_ints: list[int] = []
                body = render(rule.content, pool, rng, ints=content_ints)
                full = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(full) or dest, exist_ok=True)
                with open(full, "w") as fh:
                    fh.write(body)
                if rule.mode:
                    os.chmod(full, int(rule.mode, 8))
                by_tag.setdefault(rule.tag, []).append(rel)
                lines_by_tag[rule.tag] = lines_by_tag.get(rule.tag, 0) + len(
                    [ln for ln in body.splitlines() if ln.strip()])
                ints_by_tag.setdefault(rule.tag, []).extend(content_ints)

        for paths in by_tag.values():
            paths.sort()
        facts = {"paths": by_tag, "lines": lines_by_tag, "ints": ints_by_tag}

        # Expand set-wide checks. `{"kind": "mode", "for_each_tag": "script", ...}`
        # becomes one check per path that rule produced, which is the only way to
        # assert a per-file property when the filenames are generated.
        expanded: list[dict] = []
        for spec in self.checks:
            tag = spec.get("for_each_tag")
            if tag is None:
                expanded.append(spec)
                continue
            base = {k: v for k, v in spec.items() if k != "for_each_tag"}
            for rel in by_tag.get(tag, []):
                expanded.append({**base, "path": rel})
        self_checks = expanded

        return TaskSpec.from_dict({
            "id": f"{self.id}#{seed}",
            "instruction": self.instruction,
            "fixture": dest,
            "expect_deleted": _resolve(self.expect_deleted, facts),
            "expect_created": _resolve(self.expect_created, facts),
            "expect_modified": _resolve(self.expect_modified, facts),
            "checks": _resolve(self_checks, facts),
            "reference": self.reference,
            "max_turns": self.max_turns,
        })

    # --- validation ------------------------------------------------------

    def validate(self) -> "TaskTemplate":
        """Raise TaskError on anything that makes this template unusable."""
        if not self.id or not self.instruction:
            raise TaskError(f"{self.id!r}: id and instruction are required")
        if not self.files:
            raise TaskError(f"{self.id!r}: no file rules")
        if not self.reference:
            raise TaskError(f"{self.id!r}: reference must be a non-empty list of commands")
        if not all(isinstance(c, str) and c.strip() for c in self.reference):
            raise TaskError(f"{self.id!r}: reference commands must be non-empty strings")

        for i, r in enumerate(self.files):
            lo, hi = r.count
            if lo < 0 or hi < lo:
                raise TaskError(f"{self.id!r}: file rule {i} has invalid count {r.count}")
            if not r.tag:
                raise TaskError(f"{self.id!r}: file rule {i} has no tag")
            if r.name.startswith("/") or ".." in r.name:
                raise TaskError(f"{self.id!r}: file rule {i} name must be relative: {r.name!r}")

        for i, spec in enumerate(self.checks):
            if not isinstance(spec, dict) or "kind" not in spec:
                raise TaskError(f"{self.id!r}: check {i} has no 'kind'")
            if "for_each_tag" not in spec and "path" not in spec:
                raise TaskError(
                    f"{self.id!r}: check {i} needs either 'path' or 'for_each_tag'")

        tags = {r.tag for r in self.files}
        for t in _referenced_tags(self):
            if t not in tags:
                raise TaskError(
                    f"{self.id!r}: expectations reference tag {t!r}, "
                    f"which no file rule produces (have {sorted(tags)})"
                )

        # A fixture with nothing to preserve makes `rm -rf *` indistinguishable
        # from a correct solution: every path is expected to change, so the
        # Layer 1 gate has nothing to catch. Require at least one bystander that
        # is always present and never expected to change.
        touched = _referenced_tags(self)
        bystanders = [r for r in self.files if r.tag not in touched and r.count[0] >= 1]
        if not bystanders:
            raise TaskError(
                f"{self.id!r}: no guaranteed bystander file. At least one rule must "
                "have count >= 1 and a tag no expectation references, or the "
                "path-set gate cannot detect collateral damage."
            )
        return self

    def check_no_leak(self, seeds: tuple[int, ...] = (1, 2, 3, 7, 11, 23)) -> list[str]:
        """Return leak warnings: values the instruction reveals but should not.

        River's ANSWER-LEAK is "the instruction reveals the expected answer, so
        the task requires no genuine work". The naive test -- does any expected
        path appear in the instruction -- has false positives: naming the output
        file is a *specification*, not a leak.

        The distinction is whether the value is **generated or fixed**, which we
        can settle by instantiating several seeds: a path or count that is the
        same under every seed was specified by the template, while one that
        varies was computed at instantiation. Only varying values can leak.
        """
        import shutil
        import tempfile

        per_seed_paths: list[set[str]] = []
        per_seed_counts: list[dict[str, int]] = []
        for seed in seeds:
            d = tempfile.mkdtemp(prefix="leakchk-")
            try:
                spec = self.instantiate(seed, d)
                per_seed_paths.append({os.path.basename(p) for p in spec.expected_paths})
                per_seed_counts.append(
                    {t: len(v) for t, v in _tag_counts(self, seed, d).items()})
            finally:
                shutil.rmtree(d, ignore_errors=True)

        fixed_names = set.intersection(*per_seed_paths) if per_seed_paths else set()
        varying_names = set.union(*per_seed_paths) - fixed_names

        warnings = []
        for name in sorted(varying_names):
            if name and name in self.instruction:
                warnings.append(f"generated name {name!r} appears in the instruction")

        words = set(re.findall(r"\b\d+\b", self.instruction))
        for tag in {t for c in per_seed_counts for t in c}:
            values = {c.get(tag, 0) for c in per_seed_counts}
            if len(values) > 1:                       # the count varies -> computed
                leaked = {str(v) for v in values} & words
                if leaked:
                    warnings.append(
                        f"count of tag {tag!r} varies {sorted(values)} yet "
                        f"{sorted(leaked)} appears in the instruction")
        return sorted(set(warnings))

    # --- serialisation ---------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "TaskTemplate":
        ref = d.get("reference") or []
        if isinstance(ref, str):                 # tolerate the single-command form
            ref = [ref]
        return cls(
            id=str(d["id"]),
            instruction=d["instruction"],
            files=[FileRule.from_dict(f) for f in d.get("files", [])],
            expect_deleted=d.get("expect_deleted", []),
            expect_created=d.get("expect_created", []),
            expect_modified=d.get("expect_modified", []),
            checks=list(d.get("checks", [])),
            reference=list(ref),
            max_turns=int(d.get("max_turns", 5)),
        ).validate()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "files": [f.to_dict() for f in self.files],
            "expect_deleted": self.expect_deleted,
            "expect_created": self.expect_created,
            "expect_modified": self.expect_modified,
            "checks": self.checks,
            "reference": self.reference,
            "max_turns": self.max_turns,
        }


def load_templates(path: str) -> list[TaskTemplate]:
    """Load a JSONL template file. Raises TaskError naming the broken line."""
    out, seen = [], set()
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tpl = TaskTemplate.from_dict(json.loads(line))
            except (TaskError, KeyError, ValueError) as e:
                raise TaskError(f"{path}:{lineno}: {e}") from e
            if tpl.id in seen:
                raise TaskError(f"{path}:{lineno}: duplicate template id {tpl.id!r}")
            seen.add(tpl.id)
            out.append(tpl)
    return out


# --- selector resolution -----------------------------------------------------

# Every selector resolves against the facts recorded while the fixture was
# written, so an expectation can depend on anything instantiation actually did.
_SELECTORS = ("tag", "count_of_tag", "names_of_tag", "basenames_of_tag",
              "lines_of_tag", "ints_of_tag", "sum_of_tag", "max_of_tag")


def _resolve(node: Any, facts: dict[str, Any]) -> Any:
    """Walk a structure, replacing selector dicts with concrete values.

    Recursive so a selector can sit anywhere -- including inside a check's
    arguments, which is how `line_count` stays a plain integer comparison and
    `task.py` needs no knowledge of templates at all.
    """
    if isinstance(node, dict):
        keys = [k for k in node if k in _SELECTORS]
        if len(keys) == 1 and len(node) == 1:
            kind, tag = keys[0], node[keys[0]]
            paths = facts["paths"].get(tag, [])
            ints = facts["ints"].get(tag, [])
            if kind in ("tag", "names_of_tag"):
                return list(paths)
            if kind == "count_of_tag":
                return len(paths)
            if kind == "lines_of_tag":
                return facts["lines"].get(tag, 0)
            if kind == "ints_of_tag":
                return list(ints)
            if kind == "sum_of_tag":
                return sum(ints)
            if kind == "max_of_tag":
                return max(ints) if ints else 0
            return [os.path.basename(p) for p in paths]
        return {k: _resolve(v, facts) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, facts) for v in node]
    return node


def _referenced_tags(tpl: TaskTemplate) -> set[str]:
    """Tags any expectation or check depends on."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if (k in _SELECTORS or k == "for_each_tag") and isinstance(v, str):
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for n in (tpl.expect_deleted, tpl.expect_created, tpl.expect_modified, tpl.checks):
        walk(n)
    return found



def _tag_counts(tpl: TaskTemplate, seed: int, dest: str) -> dict[str, list[str]]:
    """Re-derive tag -> paths for an already-instantiated fixture."""
    rng = random.Random(seed)
    pool = WordPool(rng)
    by_tag: dict[str, list[str]] = {}
    for rule in tpl.files:
        n = rng.randint(rule.count[0], rule.count[1])
        for _ in range(n):
            rel = render(rule.name, pool, rng)
            render(rule.content, pool, rng)
            by_tag.setdefault(rule.tag, []).append(rel)
    return by_tag
