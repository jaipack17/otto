# Model selection (<1B, free-GPU friendly)

## Recommendation

**Primary: `Qwen/Qwen3-0.6B`**

- Native **thinking mode** — the model already emits a reasoning block before its
  answer, which is exactly the `<think>…</think> → command` shape we want. No need
  to teach the format from scratch, so GRPO can spend its budget on *correctness*
  rather than on learning to wrap output in tags.
- Newest of the sub-1B instruct models; stronger than Qwen2.5-0.5B on reasoning.
- Gives the LLM monitor a chain of thought to read. **Without a reasoning trace
  there is nothing to monitor**, which would gut the detector concept.

**Code-specialised alternative: `Qwen/Qwen2.5-Coder-0.5B-Instruct`**

- Pretrained heavily on code and shell; likely higher raw bash accuracy.
- No native thinking mode — you must teach the `<think>` format via the prompt
  and a small format reward.

**Fallback if a free GPU allows (24GB): `Qwen/Qwen2.5-Coder-1.5B-Instruct`**

- Use only if sub-1B base pass rate measures too low to train on.

## Decide by measuring, not by vibes

GRPO produces a gradient only when the base model's pass rate sits roughly in the
**30–70%** band. At 0/8 rollouts correct, every advantage in the group is zero and
nothing is learned; at 8/8 there is no headroom. So:

```bash
modal run setup/modal_app.py::benchmark_base --model-id Qwen/Qwen3-0.6B
modal run setup/modal_app.py::benchmark_base --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct
```

Pick whichever lands in the band. If **both** sit near zero, make the task set
easier (shorter pipelines, fewer predicates) before reaching for a bigger model —
task difficulty is a cheaper knob than parameters.

## Free GPU options

| Platform | GPU | VRAM | bf16 | Notes |
|---|---|---|---|---|
| **Modal** | A10G / L4 | 24GB | Yes | ~$30/mo free credits. Ampere/Ada, so the full stack works. **Preferred.** |
| Kaggle | T4 x2 | 16GB each | **No** | 30h/week. Turing: fp16 only, no flash-attn, SGLang likely unsupported. |
| Colab free | T4 | 16GB | **No** | Disconnects aggressively. Least reliable. |

**Use Modal for anything involving SGLang.** SGLang now requires CUDA 13 and
targets newer architectures; T4 (sm_75) is a poor bet. Kaggle is fine for pure
TRL/vLLM training runs in fp16.

## Sizing for GRPO on 24GB

```
LoRA r=16, fp16/bf16
num_generations        8
max_completion_length  512   (drop to 384 if tight)
vllm_gpu_memory_utilization  0.5
```

Run `aligntune advise estimate` for a VRAM/time estimate on your exact config
rather than trusting these numbers.
