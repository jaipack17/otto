# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hackathon project (Syntax Error 2026, **Local LLM track, problem statement 6 — "Reasoning with
verifiable answers"**): train a **sub-1B model to act as a terminal agent** using **RLVR/GRPO**,
where the reward comes from actually executing the model's shell commands in a sandbox and checking
the resulting filesystem state. Plan: single-turn command generation first, then multi-turn
(observe stdout → next command).

The repo is currently a **scaffold** — directory skeleton plus GPU bootstrap. Most of
`src/{tasks,verifier,rewards,monitor}` is still empty.

## Hackathon constraints (non-negotiable, they decide architecture)

- **SGLang must be the core inference/serving engine** for deployment, evaluation or benchmarking.
  It is a submission requirement, not a preference.
- The **Lexsi Labs toolkits** must be used across the post-training pipeline, one per stage:
  - Stage 1 data curation → **CuratorKit** (has a `grpo_rollouts` generation task; verifies each
    generated sample against its source chunk, rejects to `rejected.jsonl`)
  - Stage 2 fine-tuning → **AlignTune** (TRL/Unsloth backends; `--type grpo`, also GSPO/DAPO/Dr. GRPO)
  - Stage 3 evals → **AuditKit** (already installed in `.venv`)
- Docs: <https://docs.sglang.io/>, <https://curatorkit.lexsi.ai/latest/>,
  <https://aligntune.lexsi.ai>, <https://auditkit.lexsi.ai>

## Commands

```bash
# local env check (toolchain, venv packages, docker image, disk)
./scripts/check_env.sh

# build the throwaway execution sandbox
docker build -t otto-sandbox sandbox/

# run one model-generated command against a fixture dir
./sandbox/run.sh 'find . -name "*.log" -delete' data/fixtures/some_task

# GPU side (Modal)
modal run setup/modal_app.py::verify_env
modal run setup/modal_app.py::benchmark_base --model-id Qwen/Qwen3-0.6B

# evals
./.venv/bin/auditkit agent eval <config.json>     # end-to-end agent episodes
./.venv/bin/auditkit agent rescore <episodes>     # re-score saved episodes, no model calls
./.venv/bin/python -m auditkit.compat             # check sglang/vllm/torch pin clashes
```

## Environment gotchas that will cost hours otherwise

- **SGLang and vLLM never co-resolve in one environment** (llguidance and xgrammar conflict). A
  resolver asked for both "succeeds" by silently installing the frontend-only `sglang` package with
  no runtime. Keep **separate venvs / Modal images**: vLLM for GRPO rollouts, SGLang for
  serving + AuditKit evals. `auditkit.compat` exists specifically to catch this — run it after any
  install.
- SGLang ships **Linux-only wheels**, CPython 3.10–3.13, and `>= 0.5.11` is built for **CUDA 13**.
  Build the SGLang venv on **3.12**. Nothing SGLang-related runs on the Mac.
- `sglang 0.5.20` has a scheduler crash when the radix cache is disabled; `auditkit.compat
  --patch-sglang` patches it.
- **Kaggle T4 is Turing (sm_75): no bf16, no flash-attn 2, SGLang effectively unusable.** Use
  Modal A10G/L4 for anything SGLang-touching; Kaggle is fp16 training only. See `docs/MODEL_CHOICE.md`.

## Design decisions already made

- **Policy model: `Qwen/Qwen3-0.6B`**, chosen for its native thinking mode — the `<think>…</think> →
  command` shape comes for free, and the reasoning trace is what the LLM monitor in `src/monitor`
  reads. A model with no trace would gut the monitor concept. Rationale and alternatives in
  `docs/MODEL_CHOICE.md`.
- **Measure base pass rate before committing to a model.** GRPO only produces a gradient when the
  base pass rate sits roughly in the **30–70%** band; at 0/8 or 8/8 every group advantage is zero.
  If both candidates sit near zero, **make the tasks easier** rather than reaching for more
  parameters — task difficulty is the cheaper knob.
- **Every reward is computed by execution in a disposable container**, never by string-matching the
  command. `sandbox/run.sh` enforces `--network none`, `--cap-drop ALL`, `--pids-limit 128`,
  512m/1cpu, non-root, 10s timeout. The agent *will* attempt destructive and reward-hacking
  behaviour — that is an experimental result to record, so the container must be the only thing it
  can damage, and ground-truth checks must live outside anything the rollout can write to.

## Layout intent

| Path | Role |
|---|---|
| `src/tasks` | task generation + fixture construction (CuratorKit stage) |
| `src/verifier` | post-state predicates that decide pass/fail |
| `src/rewards` | reward shaping on top of verifier output, fed to AlignTune/GRPO |
| `src/monitor` | LLM monitor reading `<think>` traces for reward hacking / unsafe intent |
| `sandbox/` | Docker image + single-command runner; the execution boundary |
| `setup/` | Modal app (GPU work) and Kaggle bootstrap |
| `data/tasks`, `data/rollouts` | generated task sets and rollout logs |
