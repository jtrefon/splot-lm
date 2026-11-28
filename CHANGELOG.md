# Changelog

## Unreleased — 2026-11-28 research snapshot

### Architecture and implementation
- Dense GQA control; shared 2/4/2 recurrent reference; learned GDN/GQA hybrid; sparse top-2 + shared expert SPLOTLM; native offset-2 MTP and exact D1/D2 verifier. Architecture contract v0.2, Nano/Mini inventories. PRs #15–#26.
- Native FP8 selected projection autograd with BF16/FP32 controls, hybrid Muon/AdamW and finite-state auditing; T0005–T0011.
- Opt-in per-update FP8 weight reuse, pilot loader and separate checkpoint timing. T0014, T0017 attempt-001; independent implementation accepted, real pilot accepted as bounded feasibility with documented procedural deviations (#92).

- T0018 attempt-002 / #91 merged: exact D0 baseline, opt-in D1/D2 trace v2, actual model calls and full decode timing. No trained-checkpoint throughput claim.

### Experiments and data
- RTX 4090 native GEMMs, synthetic training, exact resume and contiguous contexts through 4096 tokens; T0001–T0012.
- Portable corpus, BPE and shard reference; cross-platform synthetic digests; HF Tokenizers 0.23.3 qualification; real EN/PL pilot corpus and 16,384-token tokenizer materialized. T0004, T0008, T0010, T0013–T0016.
- Structured REP evidence, issue dialogue and hosted CPU regression/integrity checks.

### Research direction
- Foundational bilingual research charter and quality–cost evaluation framework; credible matched controls and scale-aware evaluation. Queue T0019/T0020 after active tasks. v0.3 candidates remain proposals, not adopted architecture.

### Pending
- T0019 corrected same-vocab design accepted (A2 deferred); T0020 12 paired CUDA synthetic updates: FP8 slower, modest reserved-memory reduction, backward dominates. T0021 dense NTP harness integrated (V16384 loader blocker transferred to T0023); T0022 exact control/profile BF16 cases, generic operator attribution only. T0023 attempt-003 passed hosted actual-vocab qualification; two failed REP attempts preserved; agent reports owner-authorized test-only correction. T0024 reset/tail prerequisite passed in T0025; full-model BF16 chunk64 failed tensor/routing gates, candidate source excluded. T0026 prospective protocol accepted with planning corrections; T0027 9/9 matched NTP reference pilot, S lower NLL in all paired comparisons but ~24× slower summed step times; no superiority claim. T0028 standalone v2 verifier passed synthetic tests; coverage hardening required. T0029 exact GDN reproduction, FP32 zero route flips, BF16 rounding/route amplification diagnostic; no backward/resource claim. T0030 v2 hardening and synthetic producer passed; streaming prototype required. T0031 P1/P2 rejected, FP32 scan/chunk forward-backward 8.23x/12.19x with three numerical cases qualified; no optimizer/learning claim. T0032 scoped synthetic scale accepted (streaming gaps remain); T0033 FP32-only chunk64 opt-in qualified, complete-update 5.74x/10.59x, scan default and native FP8 retained. T0034 bounded I/O, T0035 matched learning; QAT pretraining RFC #126 proposed, not adopted. T0017 supplement complete, checkpoint preserved locally; exceptions recorded. No public pretrained weights, production throughput or language-quality claims.

See [evidence index](experiments/results/INDEX.md) and [current state](PROJECT_STATE.md). This changelog summarizes merged milestones; experiment records retain their original timestamps and outcomes.
