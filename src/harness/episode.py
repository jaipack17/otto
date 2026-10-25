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

SYSTEM_PROMPT = """You are an expert Linux shell agent. Your working directory is /work, and every file you need is inside it.

Tools available: ls, cat, head, tail, find, grep, sed, awk, sort, uniq, wc, cut, tr, xargs, mv, cp, rm, mkdir, chmod. There is no internet, and no Python or compiler.

Reply with exactly ONE command per turn, wrapped in tags:
<command>ls -la</command>
You will see its output and exit status, then you may send the next command.

How to work:
1. Look first. List the files before changing anything.
2. Use paths relative to /work, and write any output file inside /work.
3. If a command fails or the output is unexpected, change your approach instead of repeating it.
4. After you modify files, check the result with ls or cat.

Think concisely. Pinpoint the exact tool needed within 3 sentences."""

# Chosen after drafting three variants (a terse one, a medium one and this detailed one,
# minus its rule about quoting patterns). Constraints that shaped it, each measured:
#   * <done> must never appear here. Offered on turn one, a 0.6B model took it in 19 of 32
#     episodes having run nothing; "never reply <done> first" did not help. DONE_HINT below
#     introduces it only after a command has run.
#   * The conciseness line moved the model's thinking from "hit the 256-token cap" in 14 of
#     16 rollouts to 4 of 16 (a 16-rollout probe, so indicative only).
#   * Length has a price: every SFT example and every RL context re-reads the system prompt.
# The SAME text must be used for training and for evaluation; a policy behaves as it was
# trained to only under the prompt it saw.

# Sent back when we cannot find an action. Short and literal: a 0.6B policy needs
# the format restated, not an explanation of what it did wrong. No mention of <done>:
# this fires after an unparseable FIRST reply, which is exactly the turn it must not
# be offered on.
FORMAT_REMINDER = """No command found. Reply in exactly this form:
<command>your command here</command>"""

# Appended to every observation, so <done> is only ever offered after a command ran.
# The parser still accepts <done> on any turn; we just stop inviting it. Untested risk:
# the model may now answer <done> straight after its first observation (River's
# premature_done) -- the calibration pilot measures that.
DONE_HINT = "\n(When the task is complete, reply <done></done> instead of a command.)"

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

class EpisodeRun:
    """One episode as a state machine.

    The same parsing, execution and verification must serve two callers: the simple
    loop below (a Policy callable, one episode at a time) and the batched,
    token-level rollout in rollout.py (many episodes advanced in lock-step). Two
    copies of this logic would drift apart, and a rollout that scores differently
    from the evaluation loop would make every training number meaningless. So there
    is exactly one, and both drive it:

        run = EpisodeRun(task, fixture_src)
        run.start()
        while not run.finished:
            observation = run.step(raw_completion)   # None once the episode is over
        episode = run.finish()

    `step` returns the text of the next user message (an observation, or the format
    reminder after an unparseable reply) while the episode continues, and None when
    it has just ended. Whatever the ending, that message is still appended to
    `messages`, which keeps the transcript identical to the pre-refactor loop.
    """

    def __init__(
        self,
        task: TaskSpec,
        fixture_src: str,
        *,
        messages: Optional[list[dict]] = None,
        image: str = IMAGE,
        timeout: float = DEFAULT_TIMEOUT,
        obs_limit: int = OBS_LIMIT,
        max_parse_failures: int = MAX_PARSE_FAILURES,
        graded: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.task, self.fixture_src = task, fixture_src
        self.image, self.timeout = image, timeout
        self.obs_limit, self.max_parse_failures = obs_limit, max_parse_failures
        self.graded = graded

        self.ep = Episode(task_id=task.id)
        # A caller that already templated the conversation (TRL hands the rollout its
        # prompts as message lists) passes it in; otherwise build the standard opening.
        self.ep.messages = [dict(m) for m in messages] if messages is not None else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.instruction},
        ]
        self.messages, self.turns = self.ep.messages, self.ep.turns

        self._root: Optional[str] = None
        self._sandbox: Optional[Sandbox] = None
        self._before: Optional[dict] = None
        self._parse_failures = 0
        self._turns_taken = 0
        self._finished = False
        self._result: Optional[Episode] = None

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> "EpisodeRun":
        self._root = tempfile.mkdtemp(prefix="episode-")
        try:
            copy_fixture(self.fixture_src, self._root)
            self._before = snapshot(self._root)
            self._sandbox = Sandbox(self._root, image=self.image, timeout=self.timeout).start()
        except Exception:
            self.close()
            raise
        return self

    @property
    def finished(self) -> bool:
        return self._finished

    def abort(self, reason: str) -> None:
        """End the episode from outside (e.g. the token budget ran out)."""
        if not self._finished:
            self.ep.stop_reason = reason
            self._finished = True

    def _end(self, reason: str) -> None:
        self.ep.stop_reason = reason
        self._finished = True

    def step(self, raw: str) -> Optional[str]:
        """Feed one raw completion. Returns the next user message, or None if over."""
        if self._finished:
            return None
        if self._sandbox is None:
            raise RuntimeError("EpisodeRun.step called before start()")

        self._turns_taken += 1
        self.messages.append({"role": "assistant", "content": raw})
        think, command, done, err = parse_action(raw)
        self.turns.append(Turn("assistant", raw, command=command,
                               think=think, parse_error=err, done=done))

        if done:
            self._end("done")
            return None

        if command is None:
            self._parse_failures += 1
            if self._parse_failures >= self.max_parse_failures:
                self._end("parse_failures")
                return None
            self.messages.append({"role": "user", "content": FORMAT_REMINDER})
            self.turns.append(Turn("tool", FORMAT_REMINDER))
            return self._after_turn(FORMAT_REMINDER)

        self._parse_failures = 0
        res = self._sandbox.exec(command)
        body, was_truncated = truncate(res.stdout, self.obs_limit)
        obs = format_observation(body, res.exit_code, res.timed_out,
                                 res.duration, was_truncated) + DONE_HINT
        self.messages.append({"role": "user", "content": obs})
        self.turns.append(Turn("tool", obs, exit_code=res.exit_code,
                               timed_out=res.timed_out, truncated=was_truncated))
        if self._sandbox.dead:
            self._end("sandbox_dead")
            return None
        return self._after_turn(obs)

    def _after_turn(self, message: str) -> Optional[str]:
        if self._turns_taken >= self.task.max_turns:
            self._end("max_turns")
            return None
        return message

    def finish(self) -> Episode:
        """Tear down, then score. The final snapshot is taken AFTER the container is
        gone so every write has flushed and nothing can mutate the tree between
        snapshot and verification. Idempotent."""
        if self._result is not None:
            return self._result
        if self._root is None or self._before is None:
            raise RuntimeError("EpisodeRun.finish called before start()")
        if not self.ep.stop_reason:
            self.ep.stop_reason = "max_turns"
        self._finished = True
        try:
            if self._sandbox is not None:
                self._sandbox.stop()
                self._sandbox = None
            after = snapshot(self._root)
            self.ep.result = verify(self.task, self._root, self._before, after,
                                    graded=self.graded)
            self.ep.diff = state_diff(self._before, after).render()
            self._result = self.ep
            return self.ep
        finally:
            self.close()

    def close(self) -> None:
        """Release the container and working copy. Safe to call any number of times."""
        if self._sandbox is not None:
            try:
                self._sandbox.stop()
            finally:
                self._sandbox = None
        if self._root is not None:
            _force_rmtree(self._root)
            self._root = None


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
    """Run one episode against a Policy callable and score it.

    Stops on <done>, on max_turns, on a dead sandbox (a hung command), or after
    `max_parse_failures` consecutive unparseable completions.
    """
    run = EpisodeRun(task, fixture_src, image=image, timeout=timeout,
                     obs_limit=obs_limit, max_parse_failures=max_parse_failures,
                     graded=graded, system_prompt=system_prompt)
    try:
        run.start()
        while not run.finished:
            run.step(policy(run.messages))
        return run.finish()
    finally:
        run.close()


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
