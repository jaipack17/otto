#!/usr/bin/env bash
# Training environment for Apple Silicon (PyTorch MPS). Separate from .venv, which is
# deliberately torch-free (the reward path must import anywhere), and from
# .venv-sglang (SGLang and vLLM never co-resolve; see docs/SERVING.md).
#
# `pip install aligntune` fails on macOS: it pins trl[vllm], and vLLM has no macOS
# wheels. Two overrides (setup/aligntune_overrides.txt) fix exactly that and nothing
# else, so everything else AlignTune declares is installed as declared.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv-train ] || uv venv --python 3.11 .venv-train
VIRTUAL_ENV=.venv-train uv pip install --override setup/aligntune_overrides.txt aligntune pytest
.venv-train/bin/python -I -c "import aligntune, trl, torch; print('ok', torch.__version__, 'mps', torch.backends.mps.is_available())"
