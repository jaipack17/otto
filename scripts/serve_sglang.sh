#!/usr/bin/env bash
# Serve the merged model with SGLang (Linux + NVIDIA GPU only; it cannot run on a Mac).
#   scripts/serve_sglang.sh [MODEL_DIR] [PORT]
# Then, from any machine that can reach it:  python -m src.deploy.agent --url http://HOST:PORT/v1
# No --reasoning-parser: the client reads raw text so the model's thinking is never split off.
set -euo pipefail
MODEL=${1:-deploy/otto-0.6b}
PORT=${2:-30000}
exec python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" \
    --host 0.0.0.0 --served-model-name otto --dtype bfloat16
