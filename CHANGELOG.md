# Changelog

## Unreleased — 2026-11-28 research snapshot

### Architecture and implementation
- Dense GQA control; shared 2/4/2 recurrent reference; learned GDN/GQA hybrid; sparse top-2 + shared expert SPLOTLM; native offset-2 MTP and exact D1/D2 verifier. Architecture contract v0.2, Nano/Mini inventories. PRs #15–#26.
- Native FP8 selected projection autograd with BF16/FP32 controls, hybrid Muon/AdamW and finite-state auditing; T0005–T0011.
- Opt-in per-update FP8 weight reuse, pilot loader and separate checkpoint timing. T0014, T0017 attempt-001; independent implementation accepted, real pilot remains pending.

### Experiments and data
- RTX 4090 native GEMMs, synthetic training, exact resume and contiguous contexts through 4096 tokens; T0001–T0012.
- Portable corpus, BPE and shard reference; cross-platform synthetic digests; HF Tokenizers 0.23.3 qualification; real EN/PL pilot corpus and 16,384-token tokenizer materialized. T0004, T0008, T0010, T0013–T0016.
- Structured REP evidence, issue dialogue and hosted CPU regression/integrity checks.

### Pending
- T0017 actual 64-step run and full cache-on resume qualification; T0018 exact MTP observer/benchmark contract. No public pretrained weights, production throughput or language-quality claims.

See [evidence index](experiments/results/INDEX.md) and [current state](PROJECT_STATE.md). This changelog summarizes merged milestones; experiment records retain their original timestamps and outcomes.
