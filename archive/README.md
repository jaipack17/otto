# archive/

Superseded code, kept only so its reasoning is not lost. Nothing here is imported
or executed by the project.

## `run.sh`

The original single-command sandbox runner: one `docker run` per command, with the
containment flags that `src/harness/sandbox.py` now carries.

**Why it was replaced.** It cannot support a multi-turn agent. Each invocation is a
fresh container, so shell state does not survive between turns -- an agent that runs
`cd /work/sub` and then `ls` would see `ls` execute back in `/work`. `cd`, exported
variables and background processes all vanish. TMax describes the requirement as a
*persistent shell*, so `Sandbox` keeps one container per **episode** and feeds
commands to a single long-lived bash over stdin.

Its containment flags carried over unchanged (`--network none`, `--cap-drop ALL`,
`--security-opt no-new-privileges`, `--pids-limit`, `--memory`, `--cpus`), and
`Sandbox` adds `--read-only` plus a `/tmp` tmpfs so the agent cannot write outside
`/work` -- which structurally blocks the fake-binary hack TMax reports in its §D.6.
