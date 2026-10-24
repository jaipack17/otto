"""Harness test suite. Run: ./.venv/bin/pytest tests/test_harness.py -q

Split in two halves. The parser/formatter tests are pure and always run. The
sandbox and episode tests need a docker daemon and the built image, and skip
cleanly without them so the suite still passes on a machine with no container
runtime (Kaggle, for instance).
"""

import os
import subprocess
import time

import pytest

from src.harness.episode import (
    FORMAT_REMINDER,
    Episode,
    copy_fixture,
    format_observation,
    parse_action,
    run_episode,
    run_reference,
    truncate,
)
from src.harness.sandbox import IMAGE, Sandbox, probe_uid
from src.verifier.task import TaskSpec, load_tasks

FIXTURE = "data/fixtures/logs"
TASK = load_tasks("data/tasks/bash_tasks.jsonl")[0]


def _docker_ok(attempts: int = 3) -> bool:
    """Is there a daemon with our image? Retried, because Docker Desktop's
    resource saver pauses its VM when idle and the first command after a pause
    fails while the engine wakes -- which would silently skip every test below."""
    for i in range(attempts):
        try:
            if not subprocess.run(["docker", "info"], capture_output=True,
                                  timeout=30).returncode and \
               not subprocess.run(["docker", "image", "inspect", IMAGE],
                                  capture_output=True, timeout=30).returncode:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
        if i < attempts - 1:
            time.sleep(3)
    return False


_DOCKER = _docker_ok()

# Skipping on a transient docker failure reports green while testing nothing.
# Set REQUIRE_DOCKER=1 (CI, or any run you actually trust) to make that an error.
if os.environ.get("REQUIRE_DOCKER") == "1" and not _DOCKER:
    raise RuntimeError(
        f"REQUIRE_DOCKER=1 but no docker daemon with image {IMAGE!r}; "
        f"build it with: docker build -t {IMAGE} sandbox/"
    )

needs_docker = pytest.mark.skipif(
    not _DOCKER, reason=f"docker daemon or image {IMAGE!r} unavailable"
)


def scripted(*replies):
    """A fake policy that returns canned completions, then <done>."""
    it = iter(replies)
    return lambda messages: next(it, "<done></done>")


# =============================================================== pure: parsing

@pytest.mark.parametrize("raw,expected", [
    ("<think>a</think><command>ls</command>", "ls"),
    ("<think>a</think>\n```bash\nls -la\n```", "ls -la"),
    ("```\npwd\n```", "pwd"),
    ("bash: find . -name '*.log'", "find . -name '*.log'"),
    ("$ pwd", "pwd"),
    ("<command>\n  rm a.log\n</command>", "rm a.log"),
])
def test_parse_ok(raw, expected):
    # Tolerance is the point: TMax found Terminus-2 brittle with small models
    # purely on format grounds.
    _, cmd, done, err = parse_action(raw)
    assert (cmd, done, err) == (expected, False, None)


@pytest.mark.parametrize("raw,fragment", [
    ("I think I should list the files.", "no command found"),
    ("<command>ls</command><command>pwd</command>", "2 commands"),
    ("<think>I need to check whether", "unterminated"),
])
def test_parse_fails(raw, fragment):
    _, cmd, _, err = parse_action(raw)
    assert cmd is None and fragment in err


def test_parse_done():
    think, cmd, done, err = parse_action("<think>finished</think><done></done>")
    assert done and cmd is None and err is None and think == "finished"


def test_parse_ignores_tags_inside_think():
    # A model discussing <command> in its reasoning must not trip the parser.
    _, cmd, _, _ = parse_action(
        "<think>I will use <command>ls</command> next</think><command>pwd</command>")
    assert cmd == "pwd"


def test_parse_extracts_think():
    think, cmd, _, _ = parse_action("<think>look first</think><command>ls</command>")
    assert think == "look first"


# ========================================================== pure: output shape

def test_truncate_passthrough():
    assert truncate("short", 100) == ("short", False)


def test_truncate_keeps_both_ends():
    # Errors arrive last, so a head-only truncation would hide the traceback.
    text = "HEAD" + ("x" * 5000) + "TAIL"
    out, was = truncate(text, 300)
    assert was and len(out) < 500
    assert out.startswith("HEAD") and out.endswith("TAIL") and "elided" in out


def test_format_observation():
    assert format_observation("hi", 0, False, 0.1, False) == "hi\n[exit 0]"
    assert "(no output)" in format_observation("", 0, False, 0.1, False)
    assert "truncated" in format_observation("hi", 0, False, 0.1, True)
    assert "timed out" in format_observation("", -1, True, 3.0, False)


def test_copy_fixture(tmp_path):
    dest = tmp_path / "copy"
    dest.mkdir()
    copy_fixture(FIXTURE, str(dest))
    assert (dest / "a.log").exists() and (dest / "sub" / "b.log").exists()


# ============================================================ docker: sandbox

@needs_docker
def test_shell_state_persists(tmp_path):
    # The whole reason for one container per episode rather than per command.
    copy_fixture(FIXTURE, str(tmp_path))
    with Sandbox(str(tmp_path)) as sb:
        assert sb.exec("pwd").stdout == "/work"
        sb.exec("cd sub")
        assert sb.exec("pwd").stdout == "/work/sub"
        sb.exec("export GREETING=hello")
        assert sb.exec("echo $GREETING").stdout == "hello"


@needs_docker
@pytest.mark.parametrize("cmd,code", [
    ("true", 0), ("ls /nonexistent", 2), ("frobnicate", 127), ("(exit 3)", 3),
])
def test_exit_codes(tmp_path, cmd, code):
    with Sandbox(str(tmp_path)) as sb:
        assert sb.exec(cmd).exit_code == code


@needs_docker
def test_stderr_is_visible(tmp_path):
    # River's error features match on these strings, so the agent must see them.
    with Sandbox(str(tmp_path)) as sb:
        assert "command not found" in sb.exec("frobnicate").stdout


@needs_docker
def test_containment(tmp_path):
    with Sandbox(str(tmp_path)) as sb:
        assert sb.exec("echo hi > /work/ok && echo wrote").stdout == "wrote"
        assert sb.exec("echo hi > /tmp/ok && echo wrote").stdout == "wrote"
        # --read-only blocks the fake-binary hack TMax reports in its D.6.
        assert "Read-only" in sb.exec("echo x > /usr/local/bin/povray").stdout
        assert "Read-only" in sb.exec("echo x > /etc/passwd").stdout
        assert "NONE" in sb.exec("timeout 3 getent hosts example.com || echo NONE").stdout


@needs_docker
def test_timeout_ends_the_episode(tmp_path):
    # A timed-out command keeps running and holds the shell, so the sandbox is
    # marked dead rather than letting every later command stall for its full limit.
    with Sandbox(str(tmp_path), timeout=2.0) as sb:
        res = sb.exec("sleep 30")
        assert res.timed_out and sb.dead
        t0 = time.monotonic()
        assert sb.exec("echo hi").timed_out
        assert time.monotonic() - t0 < 1.0, "dead sandbox should answer immediately"


@needs_docker
def test_late_output_cannot_desync_exit_codes(tmp_path):
    # Per-command sentinel: a backgrounded write must not be mistaken for the
    # next command's sentinel.
    with Sandbox(str(tmp_path)) as sb:
        sb.exec("(sleep 1; echo LATE) & echo now")
        time.sleep(1.5)
        assert sb.exec("(exit 7)").exit_code == 7
        assert sb.exec("echo fine").exit_code == 0


@needs_docker
def test_uid_mapping_lets_verifier_read_agent_files():
    # If this fails, every agent-created file reads as UNREADABLE and the whole
    # verifier is blind. Expected to matter on Linux, not macOS.
    r = probe_uid()
    row = r["modes"]["--user host"]
    assert row["created"] and row["readable"] is True, r


# ============================================================ docker: episode

@needs_docker
def test_reference_solution_passes():
    # Task validation: does this task's own stated solution satisfy its verifier?
    assert run_reference(TASK, FIXTURE).reward == 1.0


@needs_docker
def test_reference_required():
    bare = TaskSpec.from_dict({**TASK.to_dict(), "reference": None})
    with pytest.raises(ValueError):
        run_reference(bare, FIXTURE)


@needs_docker
def test_competent_policy_passes():
    ep = run_episode(TASK, FIXTURE, scripted(
        "<think>look</think><command>ls -R</command>",
        "<command>rm a.log sub/b.log</command>",
        "<command>printf 'a.log\\nsub/b.log\\n' > removed.txt</command>",
        "<done></done>",
    ))
    assert ep.reward == 1.0 and ep.stop_reason == "done", ep.render()


@needs_docker
def test_collateral_damage_fails_the_gate():
    # Every content check would pass; Layer 1 catches it first.
    ep = run_episode(TASK, FIXTURE, scripted(
        "<command>rm -rf /work/*</command>",
        "<command>printf 'a.log\\nsub/b.log\\n' > removed.txt</command>",
        "<done></done>",
    ))
    assert ep.reward == 0.0 and ep.result.gate_failed
    assert "fresh.log" in str(ep.result.unexpected)


@needs_docker
def test_parse_failure_then_recovery():
    ep = run_episode(TASK, FIXTURE, scripted(
        "I'll look around first.",
        "<command>rm a.log sub/b.log</command>",
        "<command>printf 'a.log\\nsub/b.log\\n' > removed.txt</command>",
        "<done></done>",
    ))
    assert ep.reward == 1.0
    assert any(t.content == FORMAT_REMINDER for t in ep.turns)


@needs_docker
def test_consecutive_parse_failures_stop_the_episode():
    ep = run_episode(TASK, FIXTURE, scripted("nope", "still nope"))
    assert ep.stop_reason == "parse_failures" and ep.reward == 0.0


@needs_docker
def test_hang_stops_the_episode():
    ep = run_episode(TASK, FIXTURE, scripted("<command>sleep 60</command>"), timeout=2.0)
    assert ep.stop_reason == "sandbox_dead"


@needs_docker
def test_max_turns_respected():
    ep = run_episode(TASK, FIXTURE, scripted(*["<command>true</command>"] * 20))
    assert ep.stop_reason == "max_turns"
    assert len(ep.commands) == TASK.max_turns


@needs_docker
def test_assistant_messages_are_verbatim():
    """The loss mask depends on this and nothing else guards it.

    If the assistant entries were rebuilt from parsed think/command instead of
    stored raw, phase 2 would tokenize text the model never emitted and mask the
    wrong spans -- training would run, loss would fall, and nothing would improve.
    """
    replies = [
        "<think>look</think><command>ls</command>",
        "  sloppy   <command>pwd</command>  trailing junk ",
        "<done></done>",
    ]
    ep = run_episode(TASK, FIXTURE, scripted(*replies))
    sent = [m["content"] for m in ep.messages if m["role"] == "assistant"]
    assert sent == replies[: len(sent)], sent


@needs_docker
def test_observations_are_user_turns():
    # River's extractor assumes exactly this: parse <command>, treat the next
    # user turn as its observation.
    ep = run_episode(TASK, FIXTURE, scripted(
        "<command>echo one</command>", "<command>echo two</command>", "<done></done>"))
    roles = [m["role"] for m in ep.messages]
    assert roles[:2] == ["system", "user"]
    for i, role in enumerate(roles[2:-1], start=2):
        # after the preamble, roles must alternate assistant/user
        assert role == ("assistant" if i % 2 == 0 else "user"), roles


@needs_docker
def test_source_fixture_is_never_mutated():
    before = sorted(os.listdir(FIXTURE))
    run_episode(TASK, FIXTURE, scripted("<command>rm -rf /work/*</command>", "<done></done>"))
    assert sorted(os.listdir(FIXTURE)) == before


@needs_docker
def test_redirection_errors_are_captured(tmp_path):
    """Regression: bash reports a failed redirection itself, and that diagnostic
    reached the pipe after our sentinel -- so the message was silently dropped.
    Error text is what the agent reacts to and what River's features match on."""
    with Sandbox(str(tmp_path)) as sb:
        res = sb.exec("echo x > /usr/local/bin/povray")
        assert res.exit_code == 1
        assert "Read-only file system" in res.stdout, res.stdout


@needs_docker
def test_stderr_merge_does_not_fork_a_subshell(tmp_path):
    """The 2>&1 wrapper must use a brace group, not parentheses: `( ... )` forks
    a subshell and would silently break cd/export persistence."""
    with Sandbox(str(tmp_path)) as sb:
        sb.exec("mkdir -p sub && cd sub")
        assert sb.exec("pwd").stdout == "/work/sub"
        sb.exec("export MARKER=persisted")
        assert sb.exec("echo $MARKER").stdout == "persisted"


@needs_docker
def test_mode_round_trip_through_bind_mount(tmp_path):
    """Records which file modes survive the host<->container boundary.

    On macOS, Docker Desktop's virtiofs forces the owner-write bit ON: a file
    chmod'ed 0000 inside the container reads as 0600 on the host, 0400 as 0600,
    0444 as 0644. So the verifier cannot observe a non-owner-writable mode, and
    mode-based tasks must use a mode that already has owner-write.

    Expected to behave differently on Linux, where bind mounts are literal. If
    this test starts failing there, that is the signal -- not a regression.
    """
    import stat as statmod

    (tmp_path / "f").write_text("x\n")
    with Sandbox(str(tmp_path)) as sb:
        sb.exec("chmod 0000 f")
    host = statmod.S_IMODE((tmp_path / "f").stat().st_mode)
    assert host & 0o200, (
        f"owner-write was NOT forced on (host mode {host:04o}); this platform "
        "preserves restrictive modes, so mode-based tasks may use them"
    )
