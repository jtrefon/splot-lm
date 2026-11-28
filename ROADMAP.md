# Research roadmap / Plan badań

Snapshot 2026-11-28. Evidence and live issues, not elapsed time or a percentage, determine progress. [Current state](PROJECT_STATE.md) · [Evidence index](experiments/results/INDEX.md) · [Open research](research/BACKLOG.md).

| Workstream | Status | Next acceptance gate |
|---|---|---|
| Foundation and architecture contract | Established | Maintain literature, exact inventory and decisions |
| Dense/recurrent/hybrid/sparse/MTP reference | Implemented, CPU correctness validated | Matched architecture ablations on trained weights |
| Native FP8 and Muon | Validated on Nano / RTX 4090 in synthetic and contiguous-context tests | Real-data stability, repeated measurements and tuned matched controls |
| Data and reproducible training harness | Real pilot corpus materialized; verified ingestion implemented | Pilot feasible, provenance deviations recorded; matched learning study next |
| Recurrent weight reuse | Independent qualification accepted, opt-in | Repeat synthetic phase measurements independently |
| Native MTP inference | Exact D1/D2 reference, accepted D0 and trace v2 (#91) | Trained-checkpoint acceptance and latency |
| Expert locality and execution | Sparse reference and locality design | Trace actual routing, test cache policies; only then grouped execution/offload |
| Kernel and memory efficiency | Slow reference only | Profile GDN scan, attention/logit allocation and FP8 quantization overhead; validate optimized paths against reference |
| Scaling | Nano first; Mini inventory validated | Larger token budgets and Mini only after quality/resource gates |
| Publication | Open repository and versioned evidence | Matched ablations, independent replication, prior-art review, licensing and release decisions |

## Scientific objective
[Foundational charter](research/CHARTER.md) · [Evaluation framework](research/evaluation/README.md). First milestone: credible matched comparison demonstrating equivalent/superior held-out language-model quality at lower cost (or superior quality at equivalent cost), with capability guardrails. Nano is a mechanism-discovery instrument, not a production model.

## Immediate sequence

1. **Mac:** T0021 scoped dense NTP harness integrated; [T0023/#101](experiments/tasks/T0023-actual-vocab-pilot.md) completes actual V16384 loader and matched-run preflight, no actual training.
2. **Linux:** T0022 exactness-qualified profiling accepted, module attribution unassigned; [T0024/#102](experiments/tasks/T0024-gdn-chunk-feasibility.md) isolates chunked GDN feasibility before any full-model optimization. FP8 reuse-disabled baseline remains negative on speed.
3. After both accepted, decide bounded matched NTP study and opt-in reference optimization; trained-checkpoint MTP benchmark separately. No larger training before reviewed gates.
4. Advance beyond the bounded pilot to larger lawful corpora only when needed for the architecture experiments. Corpus optimization is deferred, not abandoned.

Vision/audio, adaptive token depth, native FP4 on unsupported hardware, external n-gram tables and large offloaded models remain deferred. No promised dates, production performance or novelty claims.

## Proposed v0.3 decision gate — review requested
[Architecture review EN](research/architecture-review/2026-11-28-en.md) / [PL](research/architecture-review/2026-11-28-pl.md), [#85](https://github.com/jtrefon/splot-lm/issues/85). Preserve accepted v0.2 T0017/T0018 evidence unchanged. Before larger training, review conditional n-gram memory #86, heterogeneous exact residency #6, recurrent topology/granularity #87, MTP and profiling. These are proposals to elevate previously deferred memory/offload research, not implemented or adopted features. No new hardware run or model release authorized.
