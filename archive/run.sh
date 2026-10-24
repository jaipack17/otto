#!/usr/bin/env bash
# Execute one model-generated command in a throwaway container.
#   ./sandbox/run.sh '<command>' [fixture_dir]
# Prints stdout; exit code is the command's.
set -uo pipefail

CMD="${1:?usage: run.sh '<command>' [fixture_dir]}"
FIXTURE="${2:-}"

ARGS=(
  --rm
  --network none              # no exfiltration, no package installs
  --cap-drop ALL
  --security-opt no-new-privileges
  --pids-limit 128
  --memory 512m
  --cpus 1
)
[ -n "$FIXTURE" ] && ARGS+=( -v "$(cd "$FIXTURE" && pwd):/work:rw" )

timeout 10 docker run "${ARGS[@]}" terminal-rlvr-sandbox bash -c "$CMD"
