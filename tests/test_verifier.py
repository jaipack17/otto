"""Verifier test suite. Run: ./.venv/bin/pytest -q

The assertion is two-sided: the reference solution must score 1.0, and nothing
else may score above 0.0. Drop the first half and an always-zero verifier passes
this whole file.
"""

import os
import shutil

import pytest

from src.verifier.cheats import from_dir, known_cheats, random_cheats, score_action
from src.verifier.task import TaskSpec

TASK = TaskSpec.from_dict({
    "id": "logs-01",
    "instruction": "Delete a.log and sub/b.log, then write their names to removed.txt, one per line.",
    "fixture": "logs",
    "expect_deleted": ["a.log", "sub/b.log"],
    "expect_created": ["removed.txt"],
    "checks": [
        {"kind": "line_count", "path": "removed.txt", "value": 2},
        {"kind": "contains", "path": "removed.txt", "text": "a.log"},
        {"kind": "contains", "path": "removed.txt", "text": "sub/b.log"},
    ],
})


def build(root):
    """Fixture: two old logs to remove, two files that must survive untouched."""
    os.makedirs(f"{root}/sub", exist_ok=True)
    for rel, text in [("a.log", "old\n"), ("sub/b.log", "old\n"),
                      ("fresh.log", "keep\n"), ("app.conf", "port=8080\n")]:
        with open(f"{root}/{rel}", "w") as fh:
            fh.write(text)


def correct(root):
    """The reference solution as Python -- the positive control."""
    os.remove(f"{root}/a.log")
    os.remove(f"{root}/sub/b.log")
    with open(f"{root}/removed.txt", "w") as fh:
        fh.write("a.log\nsub/b.log\n")


def test_reference():
    # Positive control. Without this, an always-zero verifier passes everything below.
    res = score_action(TASK, build, correct)
    assert res.reward == 1.0, res.render()
    assert res.passed and not res.gate_failed


def test_reference_graded():
    # Graded mode must not penalise a fully correct solution.
    assert score_action(TASK, build, correct, graded=True).reward == 1.0


@pytest.mark.parametrize("cheat", known_cheats(TASK, correct=correct), ids=lambda c: c.label)
def test_known_cheat(cheat):
    # Every attack we can name must score exactly zero.
    res = score_action(TASK, build, cheat.apply)
    assert res.reward == 0.0, f"{cheat.label} scored {res.reward}\n{res.render()}"


@pytest.mark.parametrize("cheat", random_cheats(TASK, n=40), ids=lambda c: c.label)
def test_random_cheat(cheat):
    # Mutations we did not think of. Seeded, so any failure is reproducible.
    res = score_action(TASK, build, cheat.apply)
    assert res.reward == 0.0, f"{cheat.label} scored {res.reward}\n{res.render()}"


@pytest.mark.parametrize("cheat", known_cheats(TASK, correct=correct), ids=lambda c: c.label)
def test_gate_graded(cheat):
    # Graded mode relaxes the reward but must never relax the Layer 1 gate:
    # collateral damage gets no partial credit.
    res = score_action(TASK, build, cheat.apply, graded=True)
    if res.gate_failed:
        assert res.reward == 0.0, f"{cheat.label} earned partial credit past the gate"


def test_fixture_untouched(tmp_path):
    # score_action must copy, never mutate the source fixture -- even if the
    # action deletes the entire working copy.
    src = tmp_path / "fixture"
    src.mkdir()
    build(str(src))
    before = sorted(os.listdir(src))
    score_action(TASK, from_dir(str(src)), lambda r: shutil.rmtree(r, ignore_errors=True))
    assert sorted(os.listdir(src)) == before
