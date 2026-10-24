"""Modal app for the GPU-side work.

Three images, deliberately separate -- see docs/SERVING.md for why.

    sglang_image  SGLang runtime ONLY. Serves an OpenAI-compatible HTTP endpoint.
                  This is the inference engine for every eval and benchmark.
    train_image   TRL/AlignTune + vLLM as the GRPO rollout backend. Never imports sglang.
    eval_image    Thin client: auditkit[requests] + curatorkit. No torch, no engine.

SGLang and vLLM cannot co-resolve in one environment (llguidance and xgrammar clash
on every version pair; a resolver asked for both silently installs the frontend-only
`sglang` with no runtime). Keeping them in separate images is the fix, and the HTTP
boundary means nothing is lost: AuditKit's `api:` backend needs only
auditkit[requests] to evaluate a served SGLang model.

Run order:
    modal run setup/modal_app.py::verify_sglang_env     # SGLang image sane?
    modal run setup/modal_app.py::verify_train_env      # training stack sane?
    modal deploy setup/modal_app.py                     # bring up the SGLang server
    modal run setup/modal_app.py::benchmark_base        # measure base pass rate THROUGH SGLang
"""

import modal

APP_NAME = "terminal-rlvr"

# A10G (24GB, Ampere) supports bf16 and flash-attention, unlike Kaggle's T4 (sm_75).
GPU = "A10G"

# Pinned: compat rule (c) says sglang < 0.5.10 is almost always a silent downgrade
# (0.5.9 is the last release on transformers 4.x). Build on Python 3.12 -- sglang
# ships Linux-only manylinux_2_34 wheels for CPython 3.10-3.13, cp312 being the
# safest current target.
SGLANG_VERSION = "0.5.20"
SERVED_MODEL = "Qwen/Qwen3-0.6B"
SGLANG_PORT = 8000

vol = modal.Volume.from_name("terminal-rlvr-vol", create_if_missing=True)
VOL_PATH = "/vol"

# --- images ------------------------------------------------------------------

# compat.py is stdlib-only by design, so it can be dropped into the SGLang venv
# and run there without installing auditkit alongside the runtime.
sglang_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(f"sglang[all]=={SGLANG_VERSION}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(
        ".venv/lib/python3.11/site-packages/auditkit/compat.py",
        "/opt/compat.py",
        copy=True,
    )
)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate",
        "datasets",
        "peft",
        "trl",
        "vllm==0.8.5",          # rollout backend for GRPO. NOT sglang.
        "huggingface_hub[hf_transfer]",
    )
    # AlignTune pins transformers/trl hard; install last so it wins the resolution.
    .pip_install("aligntune")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("auditkit[requests]", "curatorkit", "huggingface_hub")
)

app = modal.App(APP_NAME, image=eval_image)


# --- SGLang server: the one inference engine ---------------------------------

@app.function(
    image=sglang_image,
    gpu=GPU,
    volumes={VOL_PATH: vol},
    timeout=60 * 60,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=SGLANG_PORT, startup_timeout=900)
def sglang_server():
    """OpenAI-compatible SGLang endpoint. Everything else talks to this over HTTP.

    Patch first: sglang 0.5.20's scheduler calls UnifiedRadixCache.dec_lock_ref(node)
    without its required `params` when the radix cache is disabled, which SGLang does
    for several configurations -- the server loads, then crashes on its first request.
    compat.py --patch-sglang defaults the argument (that path never reads it).
    """
    import subprocess, sys

    subprocess.run([sys.executable, "/opt/compat.py", "--patch-sglang"], check=False)

    subprocess.Popen(
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", SERVED_MODEL,
            "--host", "0.0.0.0",
            "--port", str(SGLANG_PORT),
            "--reasoning-parser", "qwen3",   # expose <think> blocks for src/monitor
            "--download-dir", f"{VOL_PATH}/hf",
        ]
    )


# --- verification ------------------------------------------------------------

@app.function(image=sglang_image, gpu=GPU, timeout=900)
def verify_sglang_env():
    """Confirm the SGLang image is a real runtime, not the frontend-only shim."""
    import subprocess, sys

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    # --cuda makes compat check the torch/CUDA pairing too (sglang >= 0.5.11 is
    # built for CUDA 13, so a mismatched torch runtime is an error worth catching).
    print(subprocess.run([sys.executable, "/opt/compat.py", "--cuda"],
                         capture_output=True, text=True).stdout)

    import torch, sglang
    print(f"torch              {torch.__version__}")
    print(f"sglang             {sglang.__version__}")
    print(f"bf16 supported     {torch.cuda.is_bf16_supported()}")
    print(f"capability         {torch.cuda.get_device_capability(0)}")
    # The frontend-only package has no srt subpackage -- this import is the real test.
    import sglang.srt.server_args  # noqa: F401
    print("sglang.srt         present (real runtime, not frontend-only)")
    print(f"\npython             {sys.version}")


@app.function(image=train_image, gpu=GPU, volumes={VOL_PATH: vol}, timeout=600)
def verify_train_env():
    """Smoke-test the training stack. Must NOT contain sglang."""
    import subprocess, sys

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    import torch
    print(f"torch              {torch.__version__}")
    print(f"cuda available     {torch.cuda.is_available()}")
    print(f"device             {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported     {torch.cuda.is_bf16_supported()}")

    for mod in ["transformers", "trl", "peft", "vllm", "aligntune"]:
        try:
            m = __import__(mod)
            print(f"{mod:18s} {getattr(m, '__version__', 'OK')}")
        except Exception as e:
            print(f"{mod:18s} FAILED -> {type(e).__name__}: {e}")

    try:
        import sglang  # noqa: F401
        print("sglang             PRESENT -- it should not be here, expect a broken resolve")
    except ImportError:
        print("sglang             absent (correct: it lives in sglang_image)")

    print(f"\npython             {sys.version}")


# --- base-model benchmark, served by SGLang ----------------------------------

@app.function(image=eval_image, volumes={VOL_PATH: vol}, timeout=1800)
def benchmark_base(n: int = 50, k: int = 8, temperature: float = 0.7):
    """Measure base-model pass rate on bash tasks, generating THROUGH SGLang.

    Routing this via the served endpoint rather than transformers.generate() is what
    makes SGLang the project's inference engine for benchmarking, which the track
    requires -- and it means the base number and the post-training number come from
    the same serving path, so they stay comparable.

    `k` is the group size: it samples k completions per task, mirroring GRPO's rollout
    group. That is the number that matters, because GRPO only has a gradient when a
    group is mixed. All-k-correct or all-k-wrong groups give zero advantage and zero
    gradient, so what you want is the *fraction of tasks with a mixed group*, not the
    aggregate pass rate. Report both.

    Generation only. Verification happens in the sandbox, on a machine with Docker.
    """
    import json, pathlib

    tasks_path = pathlib.Path(f"{VOL_PATH}/tasks/bash_tasks.jsonl")
    if not tasks_path.exists():
        print(f"No tasks at {tasks_path} -- run the CuratorKit pipeline and upload first.")
        return

    tasks = [json.loads(l) for l in tasks_path.read_text().splitlines()][:n]
    base_url = sglang_server.get_web_url()
    print(f"Loaded {len(tasks)} tasks | k={k} | SGLang at {base_url}")

    from auditkit import evaluate
    from auditkit.sample import Sample
    from auditkit.runspec import RunConfig

    samples = [
        Sample(input=t["instruction"], id=str(t.get("id", i)), target=t.get("reference"))
        for i, t in enumerate(tasks)
    ]
    result = evaluate(
        samples,
        model=f"api:{SERVED_MODEL}",
        api_base=f"{base_url}/v1",
        api_key="EMPTY",            # SGLang ignores it, the client requires one
        config=RunConfig(
            temperature=temperature,
            max_tokens=512,
            num_completions=k,
            concurrency=8,
        ),
    )

    dest = pathlib.Path(f"{VOL_PATH}/rollouts/base_{SERVED_MODEL.replace('/', '_')}.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(
        json.dumps({"sample_id": p.sample_id, "completion": p.raw_output})
        for p in result.predictions
    ))
    vol.commit()
    print(f"Wrote {len(result.predictions)} completions -> {dest}")
    print("Next: verify each in the sandbox (./sandbox/run.sh), then compute per-task")
    print("group outcomes and the fraction of mixed groups.")
