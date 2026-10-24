#!/usr/bin/env bash
# Local (Mac) environment check. GPU-side checks live in setup/modal_app.py.
# Nothing SGLang-related runs here: SGLang is Linux-only wheels. See docs/SERVING.md.
set -u
echo "=== local toolchain ==="
for t in uv git docker modal kaggle; do
  if command -v "$t" >/dev/null 2>&1; then echo "  $t  OK"; else echo "  $t  MISSING"; fi
done
echo "=== venv packages (CPU-side clients only) ==="
if [ -d .venv ]; then
  ./.venv/bin/python - <<'PY'
for mod in ["auditkit", "curatorkit"]:
    try:
        m = __import__(mod)
        print(f"  {mod:12s} {getattr(m, '__version__', 'OK')}")
    except Exception as e:
        print(f"  {mod:12s} MISSING ({type(e).__name__})")
for mod in ["sglang", "vllm", "torch"]:
    try:
        __import__(mod)
        print(f"  {mod:12s} PRESENT -- should not be in the local venv")
    except ImportError:
        print(f"  {mod:12s} absent (correct)")
PY
  echo "=== auditkit pin/compat report ==="
  ./.venv/bin/python .venv/lib/python3.11/site-packages/auditkit/compat.py 2>&1 | sed 's/^/  /'
else
  echo "  no .venv -- run: uv venv --python 3.11 .venv"
  echo "  then: VIRTUAL_ENV=.venv uv pip install 'auditkit[requests]' curatorkit"
fi
echo "=== docker sandbox image ==="
docker image inspect otto-sandbox >/dev/null 2>&1 \
  && echo "  otto-sandbox  built" \
  || echo "  otto-sandbox  NOT BUILT -> docker build -t otto-sandbox sandbox/"
echo "=== disk ==="
df -h . | tail -1
