# Colab T4 notebook. One cell per "# %%" block. Settings first:
#   Runtime -> Change runtime type -> T4 GPU.
# Goal: serve the trained model with SGLang and expose it, so the Mac (Docker + verifier)
# can evaluate against it with src/deploy/eval_remote.py. No Docker is needed here.
# Lines starting "# !" are shell commands: put them in a cell with the "# " removed.

# %% 1. install SGLang in its OWN venv, so it cannot clash with the notebook's preinstalled
#       torch. If it fails on a CUDA mismatch, read the error; do not pin blindly.
# !pip install -q uv
# !uv venv /content/sg --python 3.12
# !VIRTUAL_ENV=/content/sg uv pip install "sglang[srt]" peft
# !nvidia-smi --query-gpu=name,driver_version --format=csv

# %% 2. upload the adapter folder (adapter_model.safetensors + adapter_config.json, ~40 MB)
#       to /content/adapter via the right-hand file panel or a private dataset.
#       Merge in fp16 (a T4 is sm75: no bf16):
# !/content/sg/bin/python - <<'PY'
# import torch
# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)
# PeftModel.from_pretrained(base, "/content/adapter").merge_and_unload().save_pretrained("/content/otto")
# AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained("/content/otto")
# PY

# %% 3. serve on GPU 0. The docs' fallback for T4 is the triton attention backend and the
#       pytorch sampler. Wait for "The server is fired up" in server.log before step 4.
import subprocess, os
server = subprocess.Popen(
    ["/content/sg/bin/python", "-m", "sglang.launch_server",
     "--model-path", "/content/otto", "--port", "30000", "--served-model-name", "otto",
     "--dtype", "float16", "--attention-backend", "triton", "--sampling-backend", "pytorch"],
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    stdout=open("/content/server.log", "w"), stderr=subprocess.STDOUT)

# %% 4. smoke test, then open a tunnel (free, no account) and copy the printed https URL.
# !curl -s localhost:30000/v1/models
# !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cf && chmod +x cf
tunnel = subprocess.Popen(["./cf", "tunnel", "--url", "http://localhost:30000"],
                          stdout=open("/content/tunnel.log", "w"), stderr=subprocess.STDOUT)
# !sleep 8; grep -o 'https://[a-z0-9-]*.trycloudflare.com' /content/tunnel.log | head -1
