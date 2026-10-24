#!/usr/bin/env bash
# Bring a Lightning AI Studio up to a known-good state, then prove it.
#
# Lightning was chosen over Modal (GPUs there need a payment method even with
# credit) and over Kaggle (no Docker daemon, so src/harness/sandbox.py has
# nothing to talk to). A Studio reportedly has Docker, which means our existing
# Sandbox runs unchanged and we keep --read-only, --network none and the rest.
# This script's first job is to verify that claim rather than trust it.
#
# Idempotent and quick to re-run, because free Studios restart every 4 hours.
#
#   bash setup/lightning_bootstrap.sh            # verify + install + test
#   bash setup/lightning_bootstrap.sh --verify   # just report the environment
#   bash setup/lightning_bootstrap.sh --sglang   # also install SGLang (slow)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ONLY_VERIFY=0; WITH_SGLANG=0
for a in "$@"; do
  case "$a" in
    --verify) ONLY_VERIFY=1 ;;
    --sglang) WITH_SGLANG=1 ;;
    *) echo "unknown flag: $a"; exit 2 ;;
  esac
done
say() { printf '\n=== %s ===\n' "$1"; }

say "machine"
echo "  $(uname -srm)"
echo "  cpus     $(nproc 2>/dev/null || echo '?')"
echo "  memory   $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GiB"}' || echo '?')"
echo "  disk     $(df -h . | awk 'NR==2{print $4" free"}')"
echo "  user     $(id)"

say "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
  echo "  nvidia-smi MISSING -- this Studio has no GPU attached"
fi
python3 - <<'PY' 2>/dev/null || echo "  torch not importable (fine; the training env installs its own)"
import torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
print(f"  torch    {torch.__version__}  cuda={torch.cuda.is_available()}")
if cap:
    print(f"  capability sm_{cap[0]}{cap[1]}  bf16={torch.cuda.is_bf16_supported()}")
    # sm_75 = T4 (Turing): no bf16, no flash-attn 2, SGLang a poor bet.
    # sm_86/89 = A10G/L4 (Ampere/Ada): the whole stack works.
    print("  VERDICT  " + ("T4-class: fp16 only, avoid SGLang here"
                           if cap[0] < 8 else "Ampere+ : bf16 and SGLang OK"))
PY

say "docker -- the reason we chose Lightning"
if docker info >/dev/null 2>&1; then
  echo "  daemon   OK ($(docker version --format '{{.Server.Version}}' 2>/dev/null))"
  if docker run --rm hello-world >/dev/null 2>&1; then
    echo "  run      OK -- containers start, Sandbox works unchanged"
  else
    echo "  run      FAILED -- daemon reachable but cannot start containers."
    echo "           Fall back to a subprocess sandbox (setrlimit + setuid),"
    echo "           which loses --read-only. See docs/TASK_DESIGN.md."
  fi
else
  echo "  daemon   ABSENT -- no Docker here after all."
  echo "           Everything below that needs a container will skip."
fi

if [ "$ONLY_VERIFY" = "1" ]; then say "verify-only; stopping here"; exit 0; fi

say "python toolchain"
command -v uv >/dev/null 2>&1 || {
  echo "  installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
}
[ -d .venv ] || uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -q "auditkit[requests]" curatorkit pytest pyarrow \
  && echo "  client packages installed"
./.venv/bin/python - <<'PY'
for m in ("auditkit", "curatorkit", "pytest", "pyarrow"):
    try:
        mod = __import__(m); print(f"  {m:12s} {getattr(mod,'__version__','OK')}")
    except Exception as e:
        print(f"  {m:12s} MISSING ({type(e).__name__})")
PY

say "sandbox image"
if docker info >/dev/null 2>&1; then
  docker build -q -t otto-sandbox sandbox/ >/dev/null && echo "  built otto-sandbox"
else
  echo "  skipped (no daemon)"
fi

if [ "$WITH_SGLANG" = "1" ]; then
  say "sglang (separate venv -- it never co-resolves with vLLM; see docs/SERVING.md)"
  [ -d .venv-sglang ] || uv venv --python 3.12 .venv-sglang
  VIRTUAL_ENV=.venv-sglang uv pip install -q "sglang[all]==0.5.20"
  ./.venv-sglang/bin/python .venv/lib/python3.11/site-packages/auditkit/compat.py --cuda 2>&1 | sed 's/^/  /'
  echo "  if compat reports the radix-cache bug, run: .venv-sglang/bin/python <compat.py> --patch-sglang"
fi

say "platform questions macOS could not answer"
./.venv/bin/python - <<'PY'
import os, stat, subprocess, tempfile
try:
    subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=True)
except Exception:
    raise SystemExit("  skipped (no daemon)")

from src.harness.sandbox import Sandbox, probe_uid

# 1. UID mapping. Docker Desktop on macOS translated ownership so both modes
#    worked; on Linux a bind mount is literal, so files the agent creates may be
#    owned by the image user and unreadable to the verifier -- which would make
#    every snapshot UNREADABLE and the whole reward blind.
r = probe_uid()
print(f"  host uid:gid = {r['host_uid']}:{r['host_gid']}")
for label, row in r["modes"].items():
    print(f"    {label:22s} owner={row.get('owner')} readable={row.get('readable')}")
need = r["modes"].get("default (image user)", {}).get("readable") is not True
print(f"  --user host REQUIRED here: {need}")

# 2. Mode round-trip. Docker Desktop's virtiofs forced the owner-write bit on,
#    so 0000 read back as 0600 and mode-based tasks had to avoid restrictive
#    modes. A literal bind mount should preserve them.
d = tempfile.mkdtemp(); open(f"{d}/f", "w").write("x\n")
with Sandbox(d) as sb:
    sb.exec("chmod 0000 f")
m = stat.S_IMODE(os.lstat(f"{d}/f").st_mode)
print(f"  chmod 0000 inside -> host sees {m:04o}"
      f"  ({'PRESERVED: restrictive modes usable' if m == 0 else 'owner-write forced on'})")
PY

say "test suite"
if docker info >/dev/null 2>&1; then
  REQUIRE_DOCKER=1 ./.venv/bin/pytest tests/ -q 2>&1 | tail -4
else
  ./.venv/bin/pytest tests/ -q 2>&1 | tail -4
fi

say "task set"
./.venv/bin/python -m src.tasks.screen data/templates/templates.jsonl 2>&1 | tail -3

say "done"
echo "  Next: calibration (8 rollouts per template) -- the step that needs this GPU."
