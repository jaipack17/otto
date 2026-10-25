# ONE Colab cell (Runtime -> T4 GPU). Paste all of this into a single cell and run it.
# It will ask you to pick two files: adapter_config.json and adapter_model.safetensors
# from runs/run3/checkpoint-40/ on your Mac. At the end it prints the URL to send back.
import os, subprocess, sys, time, urllib.request, re, textwrap

def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"FAILED: {cmd}")
    return r.stdout

print(sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"))

# 1. adapter upload
os.makedirs("/content/adapter", exist_ok=True)
if not {"adapter_config.json", "adapter_model.safetensors"} <= set(os.listdir("/content/adapter")):
    from google.colab import files
    for name, data in files.upload().items():
        open(f"/content/adapter/{name}", "wb").write(data)
assert {"adapter_config.json", "adapter_model.safetensors"} <= set(os.listdir("/content/adapter")), "pick both files"

# 2. SGLang in its own venv (cannot clash with Colab's torch)
sh("pip install -q uv")
VPY = "/content/sg/bin/python"
def has(mod):
    return subprocess.run([VPY, "-c", f"import {mod}"], capture_output=True).returncode == 0
# A venv from an earlier failed attempt may point at the wrong interpreter: start clean.
if not (os.path.exists(VPY) and has("sglang")):
    sh("rm -rf /content/sg")
    # sglang pulls outlines-core 0.1.26, which has no wheel here and compiles from source: needs Rust.
    if not os.path.exists(os.path.expanduser("~/.cargo/bin/rustc")):
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.cargo/bin")
    sh("uv venv /content/sg --python 3.12")
    # --python names the venv's interpreter explicitly, so nothing can land in Colab's own Python.
    sh(f'uv pip install --python {VPY} "sglang[srt]" peft transformers')
if not has("torch"):
    sh(f"uv pip install --python {VPY} torch")
for mod in ("torch", "sglang", "peft", "transformers"):
    assert has(mod), f"{mod} is not importable from {VPY}; the install went elsewhere"
print(sh(f"uv pip list --python {VPY} | grep -i -E '^(torch|sglang|transformers|peft|flashinfer)'"))
print("installed")

# 3. merge adapter -> fp16 model (T4 has no bf16)
open("/content/merge.py", "w").write(textwrap.dedent("""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)
    PeftModel.from_pretrained(base, "/content/adapter").merge_and_unload().save_pretrained("/content/otto")
    AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained("/content/otto")
"""))
sh("/content/sg/bin/python /content/merge.py")
print("merged")

# 4. serve (docs' T4 fallback: triton attention + pytorch sampler)
server = subprocess.Popen(
    ["/content/sg/bin/python", "-m", "sglang.launch_server", "--model-path", "/content/otto",
     "--port", "30000", "--served-model-name", "otto", "--dtype", "float16",
     "--attention-backend", "triton", "--sampling-backend", "pytorch"],
    stdout=open("/content/server.log", "w"), stderr=subprocess.STDOUT)
for _ in range(120):
    if server.poll() is not None:
        print(open("/content/server.log").read()[-4000:]); raise SystemExit("server died; log above")
    try:
        urllib.request.urlopen("http://localhost:30000/health", timeout=2); break
    except Exception:
        time.sleep(5)
else:
    print(open("/content/server.log").read()[-4000:]); raise SystemExit("server never became healthy")
print("server up")

# 5. smoke test through the same two-call pattern the client uses
import json
req = urllib.request.Request("http://localhost:30000/v1/completions", data=json.dumps(
    {"model": "otto", "prompt": "<|im_start|>user\nsay hi<|im_end|>\n<|im_start|>assistant\n",
     "max_tokens": 20}).encode(), headers={"Content-Type": "application/json"})
print("smoke:", json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["text"][:80])

# 6. tunnel
sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cf && chmod +x /content/cf")
tunnel = subprocess.Popen(["/content/cf", "tunnel", "--url", "http://localhost:30000"],
                          stdout=open("/content/tunnel.log", "w"), stderr=subprocess.STDOUT)
for _ in range(30):
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/tunnel.log").read())
    if m: break
    time.sleep(2)
print("\nSEND THIS URL BACK:", (m.group(0) + "/v1") if m else "no tunnel url; see /content/tunnel.log")
