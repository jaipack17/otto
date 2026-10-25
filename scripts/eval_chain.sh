#!/bin/bash
# Wait for training to exit, then evaluate base and trained models one after the other
# (never concurrently: both need the Mac's memory).  usage: scripts/eval_chain.sh RUN
set -u
RUN=${1:-run3}
cd "$(dirname "$0")/.."
while kill -0 "$(cat runs/$RUN/PID)" 2>/dev/null; do sleep 20; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
if [ -f runs/$RUN/adapter_config.json ]; then AD=runs/$RUN
else AD=runs/$RUN/$(ls runs/$RUN | grep '^checkpoint-' | sort -t- -k2 -n | tail -1); fi
echo "adapter: $AD"
TRAINED=$(.venv-train/bin/python -c "import json;print(','.join(json.load(open('runs/$RUN/RUN_INFO.json'))['train_templates']))")
caffeinate -dims .venv-train/bin/python -u -m src.training.evaluate --tag base --trained-on "$TRAINED" --out runs/$RUN/eval
caffeinate -dims .venv-train/bin/python -u -m src.training.evaluate --tag trained --adapter "$AD" --out runs/$RUN/eval
echo EVAL_DONE
