# Research roadmap / Plan badań

Snapshot 2026-11-28. Evidence and live issues, not elapsed time or a percentage, determine progress. [Current state](PROJECT_STATE.md) · [Evidence index](experiments/results/INDEX.md) · [Open research](research/BACKLOG.md).

| Workstream | Status | Next acceptance gate |
|---|---|---|
| Foundation and architecture contract | Established | Maintain literature, exact inventory and decisions |
| Dense/recurrent/hybrid/sparse/MTP reference | Implemented, CPU correctness validated | Matched architecture ablations on trained weights |
| Native FP8 and Muon | Validated on Nano / RTX 4090 in synthetic and contiguous-context tests | Real-data stability, repeated measurements and tuned matched controls |
| Data and reproducible training harness | Real pilot corpus materialized; verified ingestion implemented | Matched pilot accepted; bounded real corpus and tokenizer binding next |
| Recurrent weight reuse | Independent qualification accepted, opt-in | Repeat synthetic phase measurements independently |
| Native MTP inference | Exact D1/D2 reference, accepted D0 and trace v2 (#91) | Trained-checkpoint acceptance and latency |
| Expert locality and execution | Sparse reference and locality design | Trace actual routing, test cache policies; only then grouped execution/offload |
| Kernel and memory efficiency | FP32 chunk64 opt-in, qualified complete synthetic updates | Matched FP32 pilot accepted; QAT correctness, attention/logit allocation and precision optimization; scan default |
| Scaling | Nano first; Mini inventory validated | Larger token budgets and Mini only after quality/resource gates |
| Publication | Open repository and versioned evidence | Matched ablations, independent replication, prior-art review, licensing and release decisions |

## Scientific objective
[Foundational charter](research/CHARTER.md) · [Evaluation framework](research/evaluation/README.md). First milestone: credible matched comparison demonstrating equivalent/superior held-out language-model quality at lower cost (or superior quality at equivalent cost), with capability guardrails. Nano is a mechanism-discovery instrument, not a production model.

## Immediate sequence

1. **Mac:** [T0036/#129](experiments/tasks/T0036-real-stream.md) producer preallocation, bounded author-disjoint real corpus and tokenizer binding; #124 partial.
2. **Linux:** [T0037/#130](experiments/tasks/T0037-expert-qat.md) opt-in expert-first W4A32 correctness and packed deployment representation; T0035 matched descriptive pilot accepted.
3. [QAT pretraining RFC #126](research/precision/qat-pretraining.md) proposed expert-first W4A32, PTQ/from-start/late-QAT controls after learning baseline. Native FP8 scan remains available; no BF16/FP8 chunk64 or native FP4 on Ada. Review source acquisition and tokenizer binding separately; trained-checkpoint MTP separately.
4. Advance beyond the bounded pilot to larger lawful corpora only when needed for the architecture experiments. Corpus optimization is deferred, not abandoned.

Vision/audio, adaptive token depth, native FP4 on unsupported hardware, external n-gram tables and large offloaded models remain deferred. No promised dates, production performance or novelty claims.

## Proposed v0.3 decision gate — review requested
[Architecture review EN](research/architecture-review/2026-11-28-en.md) / [PL](research/architecture-review/2026-11-28-pl.md), [#85](https://github.com/jtrefon/splot-lm/issues/85). Preserve accepted v0.2 T0017/T0018 evidence unchanged. Before larger training, review conditional n-gram memory #86, heterogeneous exact residency #6, recurrent topology/granularity #87, MTP and profiling. These are proposals to elevate previously deferred memory/offload research, not implemented or adopted features. No new hardware run or model release authorized.
