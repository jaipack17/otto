# Dataset Card: otto-train

## Summary

- **Run timestamp**: 2026-10-24T21:24:55.714697Z
- **Wall clock time**: 1.0s
- **Pipeline config hash**: `bcba2d068ddfb8a7`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 1020 | 1020 | — | 0 | 1020 |
| SchemaGate | 1020 | 1020 | 0 | 0 | 1020 |
| TextCleaner | 1020 | 1020 | — | — | 1020 |
| SecretsGate | 1020 | 1020 | 0 | 0 | 1020 |
| PPOExporter | — | 1020 | — | — | 1020 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 1020/0 passed (N/A), 0 rejected
- **SchemaGate**: 1020/1020 passed (100.0%), 0 rejected
- **SecretsGate**: 1020/1020 passed (100.0%), 0 rejected


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
