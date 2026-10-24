# Serving architecture: SGLang for inference, nothing else

## The constraint

The track requires **SGLang as the core inference and serving engine for model
deployment, evaluation, or benchmarking**. We also need a rollout backend for GRPO,
and the natural choice there (vLLM, via TRL/AlignTune) cannot live in the same
environment as SGLang.

## Why they can't share an environment

From `auditkit/compat.py`, which exists to catch exactly this:

- SGLang and vLLM pin most of their stack exactly and **never co-resolve**.
  `llguidance` alone blocks every pair (vLLM 0.16 needs `>=1.3,<1.4`; SGLang needs
  `<0.8` or `>=1.7.6`), and the closest pair (sglang 0.5.9 / vllm 0.16) still clashes
  on `xgrammar`.
- A resolver asked for both **appears to succeed** by picking sglang 0.5.2 — the
  frontend-only package, with no runtime. You get a silent no-op instead of an error.
- SGLang ships **Linux-only** wheels (manylinux_2_34 from 0.5.11), CPython 3.10–3.13.
  Build the SGLang environment on **3.12**.
- SGLang `>= 0.5.11` is built for **CUDA 13**; a mismatched torch runtime is an error.
- `sglang 0.5.20` crashes on its first request when the radix cache is disabled
  (`UnifiedRadixCache.dec_lock_ref` called without its required `params`).
  `compat.py --patch-sglang` fixes it; `setup/modal_app.py::sglang_server` applies the
  patch at startup.

## The resolution: put the engine behind HTTP

`compat.py` states the way out directly:

> Serving SGLang over HTTP and evaluating it with the `api:` backend needs only
> `auditkit[requests]` in this env.

So SGLang is never a library dependency of anything. It is a **server**. Three images
in `setup/modal_app.py`, none of which can break the others:

```
                        ┌──────────────────────────────┐
                        │  sglang_image   (py3.12)     │
   evals, benchmarks ──▶│  sglang[all] ONLY            │
   agent rollouts       │  OpenAI-compatible :8000     │
                        └──────────────────────────────┘
                                   ▲        ▲
                        HTTP       │        │   HTTP
                        ┌──────────┘        └──────────┐
          ┌─────────────────────────┐   ┌────────────────────────────┐
          │ eval_image   (py3.12)   │   │ train_image    (py3.11)    │
          │ auditkit[requests]      │   │ torch, trl, peft, vllm,    │
          │ curatorkit              │   │ aligntune                  │
          │ no torch, no engine     │   │ never imports sglang       │
          └─────────────────────────┘   └────────────────────────────┘
```

- **`sglang_image`** — the only place SGLang exists. Serves `Qwen/Qwen3-0.6B` with
  `--reasoning-parser qwen3` so `<think>` blocks come back as a separate field, which
  is what `src/monitor` reads.
- **`eval_image`** — thin client. AuditKit's `api:` backend plus CuratorKit. Points at
  the SGLang URL, needs no GPU.
- **`train_image`** — GRPO only. vLLM here is an *internal rollout detail of the
  trainer*, not the project's inference engine, and it is invisible to everything else.
  `verify_train_env` asserts `sglang` is **absent** from this image.

## What this buys

1. The requirement is met where it counts: deployment, evaluation and benchmarking all
   go through SGLang.
2. Base-model and post-training numbers come from the **same serving path**, so they
   are comparable. A `transformers.generate()` baseline against an SGLang-served
   finetune would not be.
3. Nothing can silently downgrade SGLang to the frontend shim, because nothing else is
   installed next to it.

## Commands

```bash
modal run setup/modal_app.py::verify_sglang_env   # real runtime? CUDA pairing sane?
modal run setup/modal_app.py::verify_train_env    # training stack sane, sglang absent?
modal deploy setup/modal_app.py                   # bring the SGLang server up
modal run setup/modal_app.py::benchmark_base      # base pass rate, through SGLang
```

`verify_sglang_env` imports `sglang.srt.server_args` specifically: the frontend-only
package has no `srt` subpackage, so that import is the test for whether you got a real
runtime.
