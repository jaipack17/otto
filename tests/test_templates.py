"""Template test suite. Run: ./.venv/bin/pytest tests/test_templates.py -q

Two halves, like test_harness.py. The resolver, determinism and gate-rejection
tests are pure. The one test that screens the real template set needs a
container, because the reference and cheat gates execute.
"""

import os
import subprocess

import pytest

from src.harness.sandbox import IMAGE
from src.tasks.screen import screen, screen_all
from src.tasks.template import TaskTemplate, load_templates
from src.verifier.task import TaskError

TEMPLATES = "data/templates/templates.jsonl"
SEEDS = (1, 2, 3, 7, 11)


def _docker_ok() -> bool:
    try:
        return (not subprocess.run(["docker", "info"], capture_output=True,
                                   timeout=30).returncode
                and not subprocess.run(["docker", "image", "inspect", IMAGE],
                                       capture_output=True, timeout=30).returncode)
    except (OSError, subprocess.SubprocessError):
        return False


needs_docker = pytest.mark.skipif(not _docker_ok(), reason="docker unavailable")

ALL = load_templates(TEMPLATES)
IDS = [t.id for t in ALL]


def base(**over):
    """A minimal template that passes every pure gate, for targeted breakage."""
    d = {
        "id": "t", "instruction": "Delete the stale files.",
        "files": [
            {"name": "{word}.log", "content": "old\n", "count": [1, 3], "tag": "stale"},
            {"name": "{word}.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"},
        ],
        "expect_deleted": {"tag": "stale"},
        "checks": [],
        "reference": ["rm *.log"],
        "max_turns": 3,
    }
    d.update(over)
    return d


# ============================================================== the real set

def test_set_loads():
    assert len(ALL) >= 20, f"only {len(ALL)} templates"
    assert len(IDS) == len(set(IDS)), "duplicate template ids"


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_instantiates(tpl, tmp_path):
    # Every seed must produce a fixture and a TaskSpec that itself validates.
    for seed in SEEDS:
        d = tmp_path / f"s{seed}"
        spec = tpl.instantiate(seed, str(d))
        assert spec.expected_paths, "no expected changes"
        assert os.path.isdir(d) and os.listdir(d), "empty fixture"


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_answer_varies(tpl, tmp_path):
    # The point of randomised fixtures: a constant answer is memorisable.
    keys = set()
    for seed in SEEDS:
        spec = tpl.instantiate(seed, str(tmp_path / f"s{seed}"))
        keys.add(repr((sorted(spec.expected_paths), spec.checks)))
    assert len(keys) > 1, "the answer is identical for every seed"


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_no_leak(tpl):
    assert tpl.check_no_leak() == []


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_reference_fits_budget(tpl):
    assert len(tpl.reference) <= tpl.max_turns, (
        f"{len(tpl.reference)} commands but max_turns={tpl.max_turns}")


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_has_bystander(tpl):
    # Without a file that must survive, `rm -rf *` is indistinguishable from a
    # correct solution and the Layer 1 gate has nothing to catch.
    from src.tasks.template import _referenced_tags
    touched = _referenced_tags(tpl)
    assert any(r.tag not in touched and r.count[0] >= 1 for r in tpl.files)


# ================================================================== resolver

def test_determinism(tmp_path):
    tpl = ALL[0]
    a, b = tmp_path / "a", tmp_path / "b"
    sa, sb = tpl.instantiate(42, str(a)), tpl.instantiate(42, str(b))
    assert sorted(sa.expected_paths) == sorted(sb.expected_paths)
    walk = lambda d: sorted((os.path.relpath(os.path.join(r, f), d),
                             open(os.path.join(r, f), "rb").read())
                            for r, _, fs in os.walk(d) for f in fs)
    assert walk(str(a)) == walk(str(b)), "same seed produced different fixtures"


def test_filenames_never_collide(tmp_path):
    # WordPool draws without replacement; a collision would silently shrink the
    # fixture so the expectations disagree with what is on disk.
    tpl = TaskTemplate.from_dict(base(files=[
        {"name": "{word}.log", "content": "old\n", "count": [12, 12], "tag": "stale"},
        {"name": "{word}.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"},
    ]))
    for seed in range(12):
        spec = tpl.instantiate(seed, str(tmp_path / f"s{seed}"))
        assert len(spec.expect_deleted) == 12, "filenames collided"


@pytest.mark.parametrize("selector,expected", [
    ({"count_of_tag": "stale"}, 3),
    ({"lines_of_tag": "stale"}, 6),          # 2 non-empty lines x 3 files
    ({"sum_of_tag": "stale"}, None),         # checked below, value is seed-dependent
    ({"max_of_tag": "stale"}, None),
])
def test_selectors(selector, expected, tmp_path):
    tpl = TaskTemplate.from_dict(base(
        files=[
            {"name": "{word}.log", "content": "old\nport={int}\n", "count": [3, 3], "tag": "stale"},
            {"name": "{word}.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"},
        ],
        expect_created=["out.txt"],
        checks=[{"kind": "contains", "path": "out.txt", "text": selector}],
        reference=["rm *.log", "touch out.txt"],
    ))
    spec = tpl.instantiate(5, str(tmp_path))
    got = spec.checks[0]["text"]
    if expected is not None:
        assert got == expected
    else:
        assert isinstance(got, int) and got > 0


def test_names_and_basenames(tmp_path):
    tpl = TaskTemplate.from_dict(base(
        files=[
            {"name": "sub/{word}.log", "content": "old\n", "count": [2, 2], "tag": "stale"},
            {"name": "{word}.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"},
        ],
        expect_created=["out.txt"],
        checks=[
            {"kind": "contains_each", "path": "out.txt", "texts": {"names_of_tag": "stale"}},
            {"kind": "contains_each", "path": "out.txt", "texts": {"basenames_of_tag": "stale"}},
        ],
        reference=["rm sub/*.log", "touch out.txt"],
    ))
    spec = tpl.instantiate(3, str(tmp_path))
    names, bases = spec.checks[0]["texts"], spec.checks[1]["texts"]
    assert all(n.startswith("sub/") for n in names)
    assert all("/" not in b for b in bases)


def test_for_each_tag_expands(tmp_path):
    tpl = TaskTemplate.from_dict(base(
        files=[
            {"name": "{word}.sh", "content": "#!/bin/sh\n", "count": [3, 3], "tag": "script",
             "mode": "0644"},
            {"name": "{word}.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"},
        ],
        expect_deleted=[], expect_modified={"tag": "script"},
        checks=[{"kind": "mode", "for_each_tag": "script", "value": "0755"}],
        reference=["chmod 0755 *.sh"],
    ))
    spec = tpl.instantiate(1, str(tmp_path))
    assert len(spec.checks) == 3, "one check per generated path"
    assert {c["path"] for c in spec.checks} == set(spec.expect_modified)


# =========================================================== gate rejections

def test_rejects_missing_bystander():
    with pytest.raises(TaskError, match="bystander"):
        TaskTemplate.from_dict(base(files=[
            {"name": "{word}.log", "content": "old\n", "count": [1, 3], "tag": "stale"}]))


def test_rejects_unknown_tag():
    with pytest.raises(TaskError, match="tag"):
        TaskTemplate.from_dict(base(expect_deleted={"tag": "nonexistent"}))


def test_rejects_string_reference():
    # A bare string is coerced to a one-element list, not rejected.
    assert TaskTemplate.from_dict(base(reference="rm *.log")).reference == ["rm *.log"]


def test_rejects_empty_reference():
    with pytest.raises(TaskError, match="reference"):
        TaskTemplate.from_dict(base(reference=[]))


def test_rejects_absolute_path_rule():
    with pytest.raises(TaskError, match="relative"):
        TaskTemplate.from_dict(base(files=[
            {"name": "/etc/{word}", "content": "x", "count": [1, 1], "tag": "stale"},
            {"name": "{word}.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"}]))


def test_screen_rejects_over_budget():
    tpl = TaskTemplate.from_dict(base(reference=["a", "b", "c", "d"], max_turns=2))
    res = screen(tpl, seeds=(1, 2), execute=False)
    assert not res.accepted
    assert any("budget" in r for r in res.rejections), res.rejections


def test_screen_rejects_memorisable():
    # Fixed names and a fixed count: one answer for every seed.
    tpl = TaskTemplate.from_dict(base(files=[
        {"name": "fixed.log", "content": "old\n", "count": [1, 1], "tag": "stale"},
        {"name": "keeper.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"}]))
    res = screen(tpl, seeds=(1, 2, 3), execute=False)
    assert not res.accepted
    assert any("variety" in r for r in res.rejections), res.rejections


def test_screen_rejects_leak():
    tpl = TaskTemplate.from_dict(base(
        instruction="Delete exactly 3 stale files.",
        files=[
            {"name": "{word}.log", "content": "old\n", "count": [2, 4], "tag": "stale"},
            {"name": "{word}.cfg", "content": "k=v\n", "count": [1, 1], "tag": "keep"}]))
    res = screen(tpl, seeds=(1, 2, 3, 7, 11), execute=False)
    leaks = [r for r in res.rejections if "leak" in r]
    assert leaks, res.rejections


def test_screen_without_docker_is_marked_incomplete():
    res = screen(ALL[0], seeds=(1, 2), execute=False)
    assert not res.executed
    assert any("no docker" in w for w in res.warnings)


# ========================================================= the executing gate

@needs_docker
def test_every_template_is_accepted():
    """The whole set must pass every gate, references executed and cheats run.

    This is the test that actually qualifies the task set for training.
    """
    results = screen_all(ALL, seeds=SEEDS, n_random_cheats=10)
    bad = [r for r in results if not r.accepted]
    assert not bad, "\n".join(r.render() for r in bad)


def test_selector_inside_a_list_flattens_next_to_literals():
    from src.tasks.template import _resolve
    facts = {"paths": {"t": ["a/x.tmp", "a/y.tmp"]}, "lines": {}, "ints": {}}
    assert _resolve(["a", {"tag": "t"}, "b"], facts) == ["a", "a/x.tmp", "a/y.tmp", "b"]
    assert _resolve({"tag": "t"}, facts) == ["a/x.tmp", "a/y.tmp"]          # bare form unchanged
    assert _resolve([{"count_of_tag": "t"}], facts) == [2]                  # scalars stay as they are
    assert _resolve([{"kind": "equals", "text": {"count_of_tag": "t"}}], facts) \
        == [{"kind": "equals", "text": 2}]                                  # check specs untouched


def test_content_state_ignores_modes_but_not_content(tmp_path):
    import os
    from src.tasks import screen
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "x.txt").write_text("same")
    os.chmod(a / "x.txt", 0o600)
    os.chmod(b / "x.txt", 0o644)
    assert screen.content_state(str(a)) == screen.content_state(str(b))
    (b / "x.txt").write_text("different")
    assert screen.content_state(str(a)) != screen.content_state(str(b))


def test_recording_captures_the_state_of_the_scored_run(tmp_path):
    import random
    from src.tasks import screen
    state_rng = random.Random(1)

    def stateful_cheat(root):                    # like random_cheats: a different file each call
        open(f"{root}/f{state_rng.randrange(10**6)}", "w").close()

    root = tmp_path / "root"
    root.mkdir()
    action, held = screen.recording(stateful_cheat)
    action(str(root))
    assert set(held["state"]) == {p.name for p in root.iterdir()}   # what ran is what was recorded
    action2, held2 = screen.recording(stateful_cheat)
    action2(str(root))
    assert held2["state"] != held["state"]                           # a second run really differs


def test_recording_still_captures_state_when_the_cheat_fails(tmp_path):
    from src.tasks import screen

    def boom(root):
        open(f"{root}/partial", "w").close()
        raise OSError("refused")

    action, held = screen.recording(boom)
    with pytest.raises(OSError):
        action(str(tmp_path))
    assert "partial" in held["state"]
