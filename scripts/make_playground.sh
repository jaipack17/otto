#!/usr/bin/env bash
# (Re)create a small messy directory to try otto in: playground/
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf playground
mkdir -p playground/{logs,cache/old,notes,scripts}
echo "keep me" > playground/README.txt
for n in 1 2 3; do echo "tmp $n" > playground/cache/file$n.tmp; done
echo "old" > playground/cache/old/stale.tmp
printf 'error: disk\nok\nerror: net\nok\n' > playground/logs/app.log
printf 'ok\nok\n' > playground/logs/db.log
for n in 1 2 3; do echo "tmp $n" > playground/logs/file$n.tmp; done
echo "todo" > playground/notes/todo.txt
printf '#!/bin/sh\necho hi\n' > playground/scripts/run.sh
: > playground/notes/empty1.txt
: > playground/notes/empty2.txt
echo "playground/ ready:"; find playground | sort
