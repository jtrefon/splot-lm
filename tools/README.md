# Research tooling

| Tool | Purpose |
|---|---|
| `ci_cpu.py` | Discover CPU regression suites and validate REP result integrity |
| `experiment.py` | Deterministic training/evaluation, strict resume, FP32/BF16/FP8, AdamW or hybrid Muon, explicit checkpoint cadence; default SPLOT plus explicit `dense-ntp` A0 family |
| `optimizers.py` | Semantic hybrid Muon/AdamW parameter partition |
| `accounting.py` | Physical parameters, active capacity, SPLOT and dense causal contraction MACs; analytical only, not measured hardware FLOPs |
| `fp8_probe.py`, `context_probe.py` | Native GEMM and contiguous-context qualification |
| `data_fixture.py`, `corpus_reference.py`, `bpe_reference.py`, `tokenized_shards.py` | Synthetic reproducible data/tokenizer oracles |
| `production_tokenizer.py`, `pilot_dataset.py` | Bounded real corpus materialization, pinned optional tokenizer backend |
| `pilot_ingestion.py` | Verify independently pinned real-pilot manifest, tokenizer/shards and masks; legacy V32768 or exact actual manifest vocab |
| `pilot_v2.py`, `test_pilot_v2.py` | Standalone synthetic `splot-pilot-blocks-v2` schema, bounded stream verifier, exact token-byte/BPB helper and tamper tests; **not** accepted training input |
| `speculation_trace.py` | Exact D0 baseline, trace v2 reducer, actual model calls and full decode timing; no performance claim |
| `test_dense_harness.py` | Hosted tiny-fixture dense NTP parity, accounting, partition, resume, ingestion and family rejection |
| `rep/validate_result.py`, `rep/compare_resume.py` | Structural result gate and exact trusted local checkpoint comparison |

Use an existing compatible PyTorch runtime; optional tokenizer dependencies have separate version/pin evidence. 
```bash
PYTHONPATH=src:tools python tools/ci_cpu.py
```
For actual runs follow [reviewed task files](../experiments/tasks/README.md), [REP](../EXECUTION.md) and [dataset regeneration instructions](../research/data/pilot-dataset.md). Keep run outputs outside checkout. GPU evidence is local execution, not hosted CI reproduction.
