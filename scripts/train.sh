#!/usr/bin/env bash
# Usage: scripts/train.sh NAME [train.py options...]
# Runs training with the Mac kept awake (caffeinate -dims), logging to runs/NAME/train.log.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?usage: scripts/train.sh NAME [options]}"; shift
mkdir -p "runs/$NAME"
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec caffeinate -dims ./.venv-train/bin/python -u -m src.training.train --name "$NAME" "$@" \
  > >(tee -a "runs/$NAME/train.log") 2>&1
