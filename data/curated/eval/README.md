---
tags:
- curatorkit
configs:
- config_name: ppo
  default: true
  description: curatorkit ppo export, sha256 91c0cc3807d6a3119b00d7cb4d2a27416d287bff0c28015c83bc1c82d0f85c10
  data_files:
  - split: train
    path: ppo.jsonl
---

# Dataset Card: otto-eval

## Summary

- **Run timestamp**: 2026-10-24T21:24:55.882099Z
- **Wall clock time**: 0.2s
- **Pipeline config hash**: `b0953abd557004fa`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 170 | 170 | — | 0 | 170 |
| SchemaGate | 170 | 170 | 0 | 0 | 170 |
| TextCleaner | 170 | 170 | — | — | 170 |
| SecretsGate | 170 | 170 | 0 | 0 | 170 |
| PPOExporter | — | 170 | — | — | 170 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 170/0 passed (N/A), 0 rejected
- **SchemaGate**: 170/170 passed (100.0%), 0 rejected
- **SecretsGate**: 170/170 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - No deduplication step recorded

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.22
- **python**: 3.11.15

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `ppo` | train (170) | `prompt`: string |
