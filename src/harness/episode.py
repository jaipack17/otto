"""The episode loop: let a policy drive the sandbox, then score what it did.

Produces two parallel records of the same episode:

  messages  the conversation EXACTLY as the model saw it. Assistant entries hold
            the model's raw output, never a re-serialization of our parse. This
            is what phase 2 tokenizes, and the assistant entries are precisely
            the spans the loss mask must cover -- reconstructing them from parsed
            fields would train on tokens the model never emitted.
  turns     our structured view: parsed command, reasoning, exit code, timeout,
            truncation. For the monitor and River's behaviour features.

Observations are injected with role "user", which is both the common convention
and exactly what River's extractor assumes ("we parse the <command> and treat the
following user turn as the resulting observation"), so their 12 features apply to
our trajectories without translation.

The policy is a plain callable, so the whole loop runs against a scripted fake
policy with no GPU and no SGLang. Stdlib only.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..verifier.state import diff as state_diff
from ..verifier.state import snapshot
from ..verifier.task import TaskSpec, VerifyResult, verify
from .sandbox import DEFAULT_TIMEOUT, IMAGE, Sandbox

# Takes the conversation so far, returns the model's raw completion text.
Policy = Callable[[list[dict]], str]

SYSTEM_PROMPT = """You are working in a Linux shell. The current directory is /work.

Reply with your reasoning, then exactly ONE command:

<think>why this command</think>
<command>ls -la</command>

You will see the command's output, then you may send another command.
When the task is complete, reply with <done></done> instead of a command."""

# Sent back when we cannot find an action. Short and literal: a 0.6B policy needs
# the format restated, not an explanation of what it did wrong.
FORMAT_REMINDER = """No command found. Reply in exactly this form:
<command>your command here</command>
or <done></done> if the task is finished."""

OBS_LIMIT = 2000           # characters of command output shown to the model
MAX_PARSE_FAILURES = 2     # consecutive unparseable turns before we give up


@dataclass
class Turn:
    """One entry in the episode, in structured form."""

    role: str                            # "assistant" | "tool"
    content: str                         # raw model output, or the observation text
    command: Optional[str] = None        # assistant: the parsed action
    think: Optional[str] = None          # assistant: extracted reasoning
    parse_error: Optional[str] = None    # assistant: why no action was found
    done: bool = False                   # assistant: declared the task finished.
                                         # River's premature_done and gave_up_flag
                                         # features both key on this.
    exit_code: Optional[int] = None      # tool: the command's exit status
    timed_out: bool = False              # tool
    truncated: bool = False              # tool: output was elided


@dataclass
class Episode:
    """A finished episode and its reward."""

    task_id: str
    messages: list[dict] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    result: Optional[VerifyResult] = None
    stop_reason: str = ""                # done | max_turns | sandbox_dead | parse_failures
    diff: str = ""                       # rendered filesystem diff, for diagnosis

    @property
    def reward(self) -> float:
        return 0.0 if self.result is None else self.result.reward

    @property
    def commands(self) -> list[str]:
        return [t.command for t in self.turns if t.command]

    def render(self) -> str:
        out = [f"=== {self.task_id}  stop={self.stop_reason}"]
        for t in self.turns:
            if t.role == "assistant":
                if t.command:
                    out.append(f"  $ {t.command}")
                elif t.done:
                    out.append("  <done>")
                else:
                    out.append(f"  ! unparseable: {t.parse_error}")
            else:
                body = t.content if len(t.content) < 160 else t.content[:160] + "..."
                out.append("    " + body.replace("\n", "\n    "))
        if self.result is not None:
            out.append("  " + self.result.render().replace("\n", "\n  "))
        return "\n".join(out)


# --- parsing -----------------------------------------------------------------

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_COMMAND = re.compile(r"<command>(.*?)</command>", re.DOTALL)
_FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)
_PREFIX = re.compile(r"^\s*(?:bash|sh|\$)\s*:?\s*(.+)$", re.MULTILINE)
_DONE = re.compile(r"<done\s*/?>|</done>", re.IGNORECASE)


def parse_action(raw: str) -> tuple[Optional[str], Optional[str], bool, Optional[str]]:
    """Extract (think, command, done, parse_error) from a raw completion.

    Deliberately forgiving: TMax found Terminus-2 "more brittle with small models"
    purely on format grounds, so we accept the canonical <command> tags, fenced
    code blocks, and a bare `bash:` prefix. For a 0.6B policy, tolerance here is
    worth more than any training trick.

    The reasoning block is stripped BEFORE looking for an action, so a model that
    discusses `<command>` inside its own reasoning does not trip the parser.
    """
    think_match = _THINK.search(raw)
    think = think_match.group(1).strip() if think_match else None

    body = _THINK.sub("", raw)

    # An unterminated reasoning block means the completion was cut off mid-thought;
    # there is no action to find and the rest of `body` is more reasoning.
    if think is None and "<think>" in raw and "</think>" not in raw:
        return raw.split("<think>", 1)[1].strip(), None, False, "unterminated <think>"

    if _DONE.search(body):
        return think, None, True, None

    for pattern, label in ((_COMMAND, "command"), (_FENCE, "fence"), (_PREFIX, "prefix")):
        hits = [m.group(1).strip() for m in pattern.finditer(body)]
        hits = [h for h in hits if h]
        if not hits:
            continue
        if len(hits) > 1:
            # One action per turn: running the first silently would attribute the
            # reward to a command the model may not have meant as its action, and
            # batching them with && tangles credit across the whole batch.
            return think, None, False, f"{len(hits)} commands in one turn ({label})"
        return think, hits[0], False, None

    return think, None, False, "no command found"


def truncate(text: str, limit: int = OBS_LIMIT) -> tuple[str, bool]:
    """Shorten output, keeping both ends. Errors arrive last, so a head-only
    truncation would hide the traceback -- the most informative part of a failure
    and what River's error features key on."""
    if len(text) <= limit:
        return text, False
    head = (limit * 2) // 3
    tail = limit - head
    elided = len(text) - head - tail
    return f"{text[:head]}\n...[{elided} characters elided]...\n{text[-tail:]}", True


def format_observation(
    stdout: str,
    exit_code: int,
    timed_out: bool,
    duration: float,
    truncated: bool,
) -> str:
    """Render one command's result as the model will see it.

    The exit code is always shown, in a fixed position: a consistent format is
    easier for a small model to learn than a conditional one, and River treats a
    non-zero exit status as an error signal the agent is expected to react to.
    """
    if timed_out:
        return f"[timed out after {duration:.0f}s; no output captured]"
    body = stdout if stdout.strip() else "(no output)"
    note = " (output truncated)" if truncated else ""
    return f"{body}\n[exit {exit_code}]{note}"


# --- fixtures ----------------------------------------------------------------

def copy_fixture(src: str, root: str) -> None:
    """Copy a pristine fixture into a throwaway working directory.

    Every episode gets its own copy. Without this, the first destructive rollout
    would consume the fixture and every later episode would start from rubble.
    """
    shutil.copytree(src, root, symlinks=True, dirs_exist_ok=True)


def _force_rmtree(root: str) -> None:
    """Remove a working copy the agent may have made undeletable.

    A rollout that chmod 000's a directory leaves us unable to descend into it, so
    restore our own permissions first or the temp dir leaks on every such episode.
    """
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            try:
                os.chmod(os.path.join(dirpath, d), 0o700)
            except OSError:
                pass
    shutil.rmtree(root, ignore_errors=True)


# --- the loop ----------------------------------------------------------------

def run_episode(
    task: TaskSpec,
    fixture_src: str,
    policy: Policy,
    *,
    image: str = IMAGE,
    timeout: float = DEFAULT_TIMEOUT,
    obs_limit: int = OBS_LIMIT,
    max_parse_failures: int = MAX_PARSE_FAILURES,
    graded: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
) -> Episode:
    """Run one episode and score it.

    Stops on <done>, on max_turns, on a dead sandbox (a hung command), or after
    `max_parse_failures` consecutive unparseable completions. The final snapshot
    is taken AFTER the container is torn down so every write has landed.
    """
    ep = Episode(task_id=task.id)
    root = tempfile.mkdtemp(prefix="episode-")
    try:
        copy_fixture(fixture_src, root)
        before = snapshot(root)

        ep.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.instruction},
        ]
        ep.stop_reason = "max_turns"
        parse_failures = 0

        with Sandbox(root, image=image, timeout=timeout) as sb:
            for _ in range(task.max_turns):
                raw = policy(ep.messages)
                ep.messages.append({"role": "assistant", "content": raw})

                think, command, done, err = parse_action(raw)
                ep.turns.append(Turn("assistant", raw, command=command,
                                     think=think, parse_error=err, done=done))

                if done:
                    ep.stop_reason = "done"
                    break

                if command is None:
                    parse_failures += 1
                    if parse_failures >= max_parse_failures:
                        ep.stop_reason = "parse_failures"
                        break
                    ep.messages.append({"role": "user", "content": FORMAT_REMINDER})
                    ep.turns.append(Turn("tool", FORMAT_REMINDER))
                    continue

                parse_failures = 0
                res = sb.exec(command)
                body, was_truncated = truncate(res.stdout, obs_limit)
                obs = format_observation(body, res.exit_code, res.timed_out,
                                         res.duration, was_truncated)
                ep.messages.append({"role": "user", "content": obs})
                ep.turns.append(Turn("tool", obs, exit_code=res.exit_code,
                                     timed_out=res.timed_out, truncated=was_truncated))

                if sb.dead:
                    ep.stop_reason = "sandbox_dead"
                    break

        # After the container is gone: every write has flushed, and nothing can
        # mutate the tree between snapshot and verification.
        after = snapshot(root)
        ep.result = verify(task, root, before, after, graded=graded)
        ep.diff = state_diff(before, after).render()
        return ep
    finally:
        _force_rmtree(root)


def run_reference(
    task: TaskSpec,
    fixture_src: str,
    *,
    image: str = IMAGE,
    timeout: float = 60.0,
) -> VerifyResult:
    """Execute the task's own shell reference solution and verify it.

    This is task validation, not evaluation: it answers "does this task's stated
    solution satisfy this task's own verifier?" A task that fails here is broken,
    and training on it teaches the model that correct behaviour scores zero.

    TMax deliberately skips this and lets RL soft-filter zero-pass tasks instead.
    At 0.6B we cannot: if almost everything scores zero we could not tell a broken
    task from a task the model simply cannot do.

    The reference gets a longer default timeout than a model turn -- it may be a
    multi-stage pipeline, and it is trusted code.
    """
    if not task.reference:
        raise ValueError(f"{task.id!r}: no reference solution to run")

    root = tempfile.mkdtemp(prefix="reference-")
    try:
        copy_fixture(fixture_src, root)
        before = snapshot(root)
        failed: Optional[str] = None
        with Sandbox(root, image=image, timeout=timeout) as sb:
            # Sequential, in one persistent shell, with && semantics: a reference is
            # trusted code, so a non-zero step means the reference itself is broken
            # and running later steps against its missing output tells us nothing.
            for i, command in enumerate(task.reference):
                res = sb.exec(command)
                if res.timed_out or res.exit_code != 0:
                    failed = (f"reference step {i} ({command!r}) "
                              f"{'timed out' if res.timed_out else f'exited {res.exit_code}'}"
                              f"{': ' + res.stdout.strip()[:200] if res.stdout.strip() else ''}")
                    break
        result = verify(task, root, before, snapshot(root))
        if failed:
            # Surface the broken step: without it a reference that dies on step 0
            # is indistinguishable from one that ran fully and verified wrongly.
            result.failures.insert(0, failed)
        return result
    finally:
        _force_rmtree(root)
