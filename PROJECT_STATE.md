# Project state / Stan projektu

Snapshot 2026-11-28, reconciled against main `b91a68843b25624abfdc11c6eb4978d3e4c73e39`. **Live GitHub issues/PRs and merged evidence win over this snapshot.** Read [AGENTS](AGENTS.md), [REP](EXECUTION.md), [roadmap](ROADMAP.md), [task index](experiments/tasks/README.md) and [evidence index](experiments/results/INDEX.md). Historical handovers through 27 November are preserved in [this immutable revision](https://github.com/jtrefon/splot-lm/blob/b91a68843b25624abfdc11c6eb4978d3e4c73e39/PROJECT_STATE.md); do not load all of them at every restart.

## Current accepted capability

- Complete sparse recurrent hybrid text reference, native MTP, exact D1/D2 reference, fixed R1–4; Nano 43,055,076 and Mini 146,987,784 physical parameters (Mini inventory only).
- RTX 4090 cu128 FP32/BF16 controls, native FP8 156 selected projections, hybrid Muon/AdamW, same-mode resume, R1/R2/R4 numerical drift evidence and contiguous T4096 resource qualification. T0001–T0012. No production throughput/quality claim.
- Portable synthetic corpus/shards, train-only BPE, exact cross-platform v1/v2 hashes, verified ingestion and configurable checkpoint cadence. T0014.
- Actual EN/PL pilot acquired on Mac: T0016 #76 closed, PR #80 merged. 8 families, 8,987,423 extracted UTF8 bytes, 16,384 tokenizer vocab, 128/16/16 blocks, 56,201 content tokens. Manifest `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55`; full pins in [T0017 continuation](experiments/tasks/T0017-continuation.md).
- T0017 attempt-001 #79 merged, independent cache and loader accepted, **blocked** for real pilot. T512 quantizations 298→156, exact matched state/loss; one paired timing 1.949→1.633 s is descriptive. Do not close #77 until missing full 4-step resume/RNG gate and real pilot accepted.
- PR #82 merged; both named tasks below are ready for explicit owner launch. No open PR at snapshot.

## Active assignments

| Machine | Task | Next |
|---|---|---|
| Linux RTX4090 | [#77 T0017 attempt-002](https://github.com/jtrefon/splot-lm/issues/77) | [Continuation](experiments/tasks/T0017-continuation.md): verify exact five-file dataset from owner-provided copy or qualified regeneration; complete 4-step 2+2 exact RNG/state/validation; fixed64-step FP8/Muon Nano run. Keep checkpoint local. |
| Mac M4 | [#81 T0018](https://github.com/jtrefon/splot-lm/issues/81) | [Speculation observer](experiments/tasks/T0018-mtp-observability.md): D0/D1/D2 traces, per-offset denominators, exact token/cache/RNG, benchmark contract. |

T0001–T0016 closed. T0017 and T0018 open. Owner authorizes named local execution separately; no automatic next task. Next candidates after review: actual trained-checkpoint MTP evaluation and GDN/attention/quantization profiling; then matched experiments and scale.

## Research priorities and authority

Architecture alignment first, not corpus optimization. Owner authorized acquisition of accessible lawfully usable datasets within reviewed caps (2026-11-27); no repeated item consent, no rights/terms bypass or redistribution approval. Maintain immutable provenance and holdouts. FP8-first mixed precision, BF16/FP32 controls, native FP4 not assumed on Ada. Keep #2/#6/#7/#8/#10/#11/#42/#43 open for broader empirical research, licensing and CUDA13 qualification. No affiliation, novelty, benchmark superiority or public-weight claim.

[Open research issues](research/BACKLOG.md) · [Historical decisions](research/decisions/README.md) · [Public overview](README.md) / [PL](README.pl.md).

## Session continuity
[Start prompt](SESSION_START.md) · [Research coordination protocol](RESEARCH_COORDINATION.md). Public documentation reviewed 2026-11-28 (#83); historical records preserved by immutable links. Refresh status at meaningful milestones, not by accumulating stale handover blocks.

## Architecture decision review (proposed)
[2026-11-28 EN](research/architecture-review/2026-11-28-en.md) / [PL](research/architecture-review/2026-11-28-pl.md), #85, #86 conditional memory and #87 recurrence/topology; #6 tier-aware exact residency. Current assignments #77/#81 unchanged. Conditional GO for v0.2 reference; no freeze for larger pretraining before the reviewed v0.3 gates. No claim that candidate extensions are implemented; owner review pending.

## Scientific direction and queued assignments
Maintainer-approved [charter](research/CHARTER.md) / [PL](research/CHARTER.pl.md): achieve more through better architecture, not greater scale. [Evaluation framework](research/evaluation/README.md): quality–cost improvements against credible matched controls. Commodity hardware is a validation constraint. T0017/T0018 remain unchanged; queued Mac [T0019/#89](experiments/tasks/T0019-baseline-design.md) after T0018 acceptance, Linux [T0020/#90](experiments/tasks/T0020-cuda-cost-baseline.md) after T0017 acceptance; each requires reviewed merge and separate named local owner launch. No v0.3 mechanism adopted by the charter.
