"""Container lifecycle and a persistent shell inside it.

One container per EPISODE, not per command. A multi-turn agent that runs
`cd /work/sub` and then `ls` must see the second command run in the new
directory; a fresh `docker run` per command silently discards cd, environment
variables and background processes. TMax describes this as a persistent shell.

Commands are fed to a single long-lived bash over stdin, followed by a sentinel
that carries the exit code:

    <command>
    echo "__END_<random>__$?"

We read until the sentinel appears; everything before it is output. The sentinel
is a fresh UUID per sandbox, because a fixed one could be echoed by the model to
forge its own exit code. It travels over stdin rather than argv, so `ps` inside
the container cannot reveal it either.

Stdlib only.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Optional

IMAGE = "otto-sandbox"

# Per-command wall clock. TMax allows 120s; ours are far smaller tasks and a
# 0.6B policy mostly emits fast commands, so a short limit keeps rollouts cheap.
# ExecResult.timed_out makes it measurable -- raise it if traces show timeouts.
DEFAULT_TIMEOUT = 10.0

# Hard ceiling on a container's life, far above any episode (a few turns of seconds).
MAX_LIFETIME = 3600

# Where each command is staged before being sourced. /tmp is the sandbox's writable tmpfs.
CMD_FILE = "/tmp/.otto_cmd"


@dataclass
class ExecResult:
    """One command's outcome, as the agent will see it."""

    stdout: str            # stdout and stderr merged -- the agent must see errors
    exit_code: int         # -1 when we never got a sentinel back
    timed_out: bool
    duration: float


class SandboxError(RuntimeError):
    """The container or its shell is unusable. Not the agent's fault."""


class Sandbox:
    """A throwaway container with a persistent bash, scoped to one episode.

    Use as a context manager so the container is always torn down:

        with Sandbox(fixture_root) as sb:
            print(sb.exec("ls -la").stdout)
    """

    def __init__(
        self,
        fixture_root: str,
        image: str = IMAGE,
        timeout: float = DEFAULT_TIMEOUT,
        user: Optional[str] = "host",   # "host" | None | "uid:gid"
        memory: str = "512m",
        cpus: str = "1",
        pids: int = 128,
    ) -> None:
        self.fixture_root = os.path.abspath(fixture_root)
        self.image = image
        self.timeout = timeout
        self.user = f"{os.getuid()}:{os.getgid()}" if user == "host" else user
        self.memory, self.cpus, self.pids = memory, cpus, pids

        self._cid: Optional[str] = None
        self._shell: Optional[subprocess.Popen] = None
        # Random per sandbox: a predictable sentinel could be forged by the model.
        self._marker = f"__END_{uuid.uuid4().hex[:12]}_"
        # Heredoc terminator for passing a command to the shell as opaque text. Random
        # per sandbox so no command can contain it by accident or on purpose.
        self._eof = f"__OTTO_EOF_{uuid.uuid4().hex[:12]}__"
        # Per-command counter appended to the marker. A command that finishes late
        # (or a backgrounded one that writes after its sentinel) cannot then be
        # mistaken for the NEXT command's sentinel and desync every exit code.
        self._seq = 0
        # Set once the shell can no longer be trusted. A timed-out command keeps
        # running inside the container and holds the shell, so every later command
        # would queue behind it and time out as well. A hang ends the episode.
        self._dead: Optional[str] = None

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> "Sandbox":
        cmd = [
            "docker", "run", "-d", "--rm",
            "--network", "none",              # no exfiltration, no installs
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(self.pids),   # fork bombs
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--read-only",                    # rootfs immutable; /work stays writable
            "--tmpfs", "/tmp:rw,size=64m",    # somewhere scratch, and HOME
            "--env", "HOME=/tmp",
            "--workdir", "/work",
            "-v", f"{self.fixture_root}:/work:rw",
        ]
        if self.user:
            cmd += ["--user", self.user]
        # The keepalive: without a foreground process the container exits at once and
        # there is nothing to exec into. It must NOT be `sleep infinity`: if the owning
        # process is killed (a crash, a Ctrl-C, a stopped job) nothing runs `docker
        # kill`, and an immortal container is stranded forever. We found three left
        # behind for hours by a calibration run that was stopped mid-episode. With a
        # finite lifetime and --rm, an orphan removes itself.
        cmd += [self.image, "sleep", str(MAX_LIFETIME)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SandboxError(f"docker run failed: {proc.stderr.strip()}")
        self._cid = proc.stdout.strip()

        # -i keeps stdin open so we can keep writing commands to the same bash.
        # stderr is merged into stdout: the agent has to see `command not found`
        # and tracebacks, and River's error features are extracted from them.
        self._shell = subprocess.Popen(
            ["docker", "exec", "-i", self._cid, "bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        return self

    def stop(self) -> None:
        if self._shell is not None:
            try:
                self._shell.stdin.close()
            except OSError:
                pass
            try:
                self._shell.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._shell.kill()
            self._shell = None
        if self._cid is not None:
            # SIGKILL rather than a graceful stop: nothing in the container is
            # worth flushing, and `docker stop` waits 10s for SIGTERM first.
            subprocess.run(["docker", "kill", self._cid], capture_output=True, text=True)
            self._cid = None

    @property
    def dead(self) -> Optional[str]:
        """Why the shell is unusable, or None. The episode loop stops when set."""
        return self._dead

    def __enter__(self) -> "Sandbox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- the one operation that matters -------------------------------------

    def exec(self, command: str, timeout: Optional[float] = None) -> ExecResult:
        """Run one command in the persistent shell and return what the agent sees.

        Never raises on agent behaviour: a command that fails, hangs or kills the
        shell is an episode outcome, not an exception. Only a broken container
        raises (SandboxError), because that is our bug.
        """
        if self._shell is None:
            raise SandboxError("sandbox not started")

        # Once the shell is poisoned every command would burn its full timeout for
        # nothing. Answer immediately instead and let the episode end.
        if self._dead:
            return ExecResult(stdout="", exit_code=-1, timed_out=True, duration=0.0)

        limit = self.timeout if timeout is None else timeout
        started = time.monotonic()

        self._seq += 1
        marker = f"{self._marker}{self._seq}__"

        # The command is written to a file through a QUOTED heredoc, then sourced.
        #
        # Why not send it to the shell directly? Because bash parses what it reads from
        # stdin, and stdin here is our own protocol. A malformed command (an unclosed
        # quote, a stray `done`, unbalanced braces) leaves bash waiting for the rest of
        # a statement, so it swallows our sentinel as more of the command and the
        # episode hangs until the timeout kills it: 7 of 68 early rollouts died this way,
        # reward 0, with the model never told what it got wrong. A heredoc body is opaque
        # to the parser, and `.` reads from a file rather than stdin, so a syntax error
        # becomes an ordinary error message the model can learn from.
        #
        # `.` (source) runs in the CURRENT shell, so cd/export/functions still persist
        # across turns -- a subshell or `bash file` would silently break that.
        #
        # `2>&1` merges stderr inside the group: a failed redirection or a syntax error
        # is reported by bash itself and would otherwise reach the pipe AFTER our
        # sentinel and be lost (an earlier version dropped "Read-only file system").
        #
        # `< /dev/null` gives the command EOF on stdin: without it a bare `cat` swallowed
        # the sentinel and every later command, silently (exit -1, output = our own
        # wrapper text), and `grep pat` with no file hung. Heredocs inside the command
        # are unaffected, their body is part of the command text.
        #
        # A leading newline protects against a partial line left by a previous command.
        payload = (f"\ncat > {CMD_FILE} <<'{self._eof}'\n{command}\n{self._eof}\n"
                   f"{{ . {CMD_FILE}; }} 2>&1 < /dev/null\n"
                   f"echo \"{marker}$?\"\n")
        try:
            self._shell.stdin.write(payload.encode())
            self._shell.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise SandboxError(f"shell died before command: {e}") from e

        raw, timed_out = self._read_until(marker.encode(), started + limit)
        text = raw.decode("utf-8", "replace")

        exit_code = -1
        if timed_out:
            self._dead = f"command exceeded {limit:.0f}s and still holds the shell"
        else:
            # Output is everything before the sentinel; the digits after it are $?.
            head, _, tail = text.rpartition(marker)
            text = head
            try:
                exit_code = int(tail.strip().splitlines()[0])
            except (ValueError, IndexError):
                exit_code = -1

        return ExecResult(
            stdout=text.strip("\n"),
            exit_code=exit_code,
            timed_out=timed_out,
            duration=time.monotonic() - started,
        )

    def _read_until(self, marker: bytes, deadline: float) -> tuple[bytes, bool]:
        """Read stdout until `marker` appears or the deadline passes.

        select() rather than a reader thread: we are POSIX-only (docker), and a
        deadline-bounded select loop has fewer moving parts than a thread plus a
        queue. Returning (partial_output, True) on timeout means a hung command
        is reported as a timed-out turn instead of blocking the rollout forever.
        """
        buf = b""
        fd = self._shell.stdout.fileno()
        while marker not in buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return buf, True
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue                      # loop re-checks the deadline
            chunk = os.read(fd, 65536)
            if not chunk:
                return buf, True              # shell exited without a sentinel
            buf += chunk
        return buf, False


# --- diagnostics -------------------------------------------------------------

def probe_uid(image: str = IMAGE) -> dict:
    """Measure whether --user is needed for the verifier to read agent-made files.

    The container's default user is `agent` (uid 1000); the host user is whoever
    runs training. If Docker does not map that transparently, every file the agent
    creates is unreadable to the verifier -- which would surface as UNREADABLE in
    every snapshot and make everything look modified. Rather than guess, write a
    file from inside the container under each setting and look at it from here.
    """
    out = {"host_uid": os.getuid(), "host_gid": os.getgid(), "modes": {}}
    for label, user in (("default (image user)", None), ("--user host", "host")):
        root = tempfile.mkdtemp(prefix="probe-")
        try:
            with Sandbox(root, image=image, user=user) as sb:
                inside = sb.exec("id -u; id -g; echo hi > /work/probe.txt; echo $?")
            path = f"{root}/probe.txt"
            row = {"inside": inside.stdout.split(), "created": os.path.exists(path)}
            if row["created"]:
                st = os.stat(path)
                row["owner"] = f"{st.st_uid}:{st.st_gid}"
                try:
                    with open(path) as fh:
                        row["readable"] = fh.read().strip() == "hi"
                except OSError as e:
                    row["readable"] = f"NO ({type(e).__name__})"
            out["modes"][label] = row
        except SandboxError as e:
            out["modes"][label] = {"error": str(e)}
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return out
