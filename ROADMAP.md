# Research roadmap / Plan badań

Snapshot 2026-11-28. Evidence and live issues, not elapsed time or a percentage, determine progress. [Current state](PROJECT_STATE.md) · [Evidence index](experiments/results/INDEX.md) · [Open research](research/BACKLOG.md).

| Workstream | Status | Next acceptance gate |
|---|---|---|
| Foundation and architecture contract | Established | Maintain literature, exact inventory and decisions |
| Dense/recurrent/hybrid/sparse/MTP reference | Implemented, CPU correctness validated | Matched architecture ablations on trained weights |
| Native FP8 and Muon | Validated on Nano / RTX 4090 in synthetic and contiguous-context tests | Real-data stability, repeated measurements and tuned matched controls |
| Data and reproducible training harness | Real pilot corpus materialized; verified ingestion implemented | Exact Linux snapshot and first fixed 64-step run |
| Recurrent weight reuse | Independent qualification accepted, opt-in | Complete T0017 resume and actual pilot, then repeated profiling |
| Native MTP inference | Exact D1/D2 reference implemented | T0018 observer and D0 baseline, then trained-checkpoint acceptance and latency |
| Expert locality and execution | Sparse reference and locality design | Trace actual routing, test cache policies; only then grouped execution/offload |
| Kernel and memory efficiency | Slow reference only | Profile GDN scan, attention/logit allocation and FP8 quantization overhead; validate optimized paths against reference |
| Scaling | Nano first; Mini inventory validated | Larger token budgets and Mini only after quality/resource gates |
| Publication | Open repository and versioned evidence | Matched ablations, independent replication, prior-art review, licensing and release decisions |

## Immediate sequence

1. **Linux:** [T0017 continuation](experiments/tasks/T0017-continuation.md) — missing exact 4-step resume/RNG gate, verified actual data and 64-step Nano FP8/Muon pilot. [#77](https://github.com/jtrefon/splot-lm/issues/77)
2. **Mac:** [T0018](experiments/tasks/T0018-mtp-observability.md) — exact MTP observability, D0 comparator and benchmark contract. [#81](https://github.com/jtrefon/splot-lm/issues/81)
3. Use the reviewed pilot checkpoint for inference and routing experiments; profile bottlenecks before larger training. Prepare full factorial matched precision/optimizer and architecture controls with adequate seeds and token budgets. CUDA 13 qualification is separate (#43).
4. Advance beyond the bounded pilot to larger lawful corpora only when needed for the architecture experiments. Corpus optimization is deferred, not abandoned.

Vision/audio, adaptive token depth, native FP4 on unsupported hardware, external n-gram tables and large offloaded models remain deferred. No promised dates, production performance or novelty claims.

## Proposed v0.3 decision gate — review requested
[Architecture review EN](research/architecture-review/2026-11-28-en.md) / [PL](research/architecture-review/2026-11-28-pl.md), [#85](https://github.com/jtrefon/splot-lm/issues/85). Preserve active v0.2 T0017/T0018 unchanged. Before larger training, review conditional n-gram memory #86, heterogeneous exact residency #6, recurrent topology/granularity #87, MTP and profiling. These are proposals to elevate previously deferred memory/offload research, not implemented or adopted features. No new hardware run or model release authorized.
