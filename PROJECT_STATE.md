# Project state / Stan projektu

Snapshot 2026-11-28, reconciled against main `f49af5bfb0085cbbc7161d5b3354b8e8fe309696`. **Live GitHub issues/PRs and merged evidence win over this snapshot.** Read [AGENTS](AGENTS.md), [REP](EXECUTION.md), [roadmap](ROADMAP.md), [task index](experiments/tasks/README.md) and [evidence index](experiments/results/INDEX.md). Historical handovers through 27 November are preserved in [this immutable revision](https://github.com/jtrefon/splot-lm/blob/b91a68843b25624abfdc11c6eb4978d3e4c73e39/PROJECT_STATE.md); do not load all of them at every restart.

## Current accepted capability

- Complete sparse recurrent hybrid text reference, native MTP, exact D1/D2 reference, fixed R1–4; Nano 43,055,076 and Mini 146,987,784 physical parameters (Mini inventory only).
- RTX 4090 cu128 FP32/BF16 controls, native FP8 156 selected projections, hybrid Muon/AdamW, same-mode resume, R1/R2/R4 numerical drift evidence and contiguous T4096 resource qualification. T0001–T0012. No production throughput/quality claim.
- Portable synthetic corpus/shards, train-only BPE, exact cross-platform v1/v2 hashes, verified ingestion and configurable checkpoint cadence. T0014.
- Actual EN/PL pilot acquired on Mac: T0016 #76 closed, PR #80 merged. 8 families, 8,987,423 extracted UTF8 bytes, 16,384 tokenizer vocab, 128/16/16 blocks, 56,201 content tokens. Manifest `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55`; full pins in [T0017 continuation](experiments/tasks/T0017-continuation.md).
- T0017 attempt-001 #79 merged, independent cache and loader accepted, **blocked** for real pilot. T512 quantizations 298→156, exact matched state/loss; one paired timing 1.949→1.633 s is descriptive. Do not close #77 until missing full 4-step resume/RNG gate and real pilot accepted.
- T0018 attempt-002 accepted, #91 merged and #81 closed: trace v2 counts actual model calls; decode includes complete first round; 28 stdlib/16 hosted integration tests and 9 independently recomputed synthetic fixtures. Attempt-001 review-rejected and preserved. [Evidence](experiments/results/T0018/attempt-002/summary.md). No trained-checkpoint benchmark.
- T0017 attempt-002 submitted #92: numerical evidence accepted as feasibility (64 steps, 23,534 NTP labels, validation NLL 7.744022, 6,652 cache hits/14,749 logical uses, 89 s); not clean REP acceptance. Supplement records unavailable original wheel bytes, no separate explicit wheel consent, failed wrong-config comparator then unapproved corrective rerun, five missing supplement files and volatile local checkpoint. No extra training requested. #77 remains open; #92 needs supplement completeness, explicit disposition and index reconciliation/CI.

## Active assignments

| Machine | Task | Next |
|---|---|---|
| Linux RTX4090 | [#77 T0017](https://github.com/jtrefon/splot-lm/issues/77) | Non-training supplement completion and owner-authorized private checkpoint preservation; [T0020/#90](experiments/tasks/T0020-cuda-cost-baseline.md) independent synthetic VAL after scheduling amendment merge and named launch. |
| Mac M4 | [#89 T0019](https://github.com/jtrefon/splot-lm/issues/89) | [Matched baseline design](experiments/tasks/T0019-baseline-design.md), reviewed spec merged and dependency #81 closed; ready for owner launch. |

T0001–T0016 and T0018 closed. T0017 remains open. Owner authorizes named local execution separately; no automatic next task. No pilot rerun or new architecture adoption.

## Research priorities and authority

Architecture alignment first, not corpus optimization. Owner authorized acquisition of accessible lawfully usable datasets within reviewed caps (2026-11-27); no repeated item consent, no rights/terms bypass or redistribution approval. Maintain immutable provenance and holdouts. FP8-first mixed precision, BF16/FP32 controls, native FP4 not assumed on Ada. Keep #2/#6/#7/#8/#10/#11/#42/#43 open for broader empirical research, licensing and CUDA13 qualification. No affiliation, novelty, benchmark superiority or public-weight claim.

[Open research issues](research/BACKLOG.md) · [Historical decisions](research/decisions/README.md) · [Public overview](README.md) / [PL](README.pl.md).

## Session continuity
[Start prompt](SESSION_START.md) · [Research coordination protocol](RESEARCH_COORDINATION.md). Public documentation reviewed 2026-11-28 (#83); historical records preserved by immutable links. Refresh status at meaningful milestones, not by accumulating stale handover blocks.

## Architecture decision review (proposed)
[2026-11-28 EN](research/architecture-review/2026-11-28-en.md) / [PL](research/architecture-review/2026-11-28-pl.md), #85, #86 conditional memory and #87 recurrence/topology; #6 tier-aware exact residency. Current assignments #77/#81 unchanged. Conditional GO for v0.2 reference; no freeze for larger pretraining before the reviewed v0.3 gates. No claim that candidate extensions are implemented; owner review pending.

## Scientific direction and queued assignments
Maintainer-approved [charter](research/CHARTER.md) / [PL](research/CHARTER.pl.md): achieve more through better architecture, not greater scale. [Evaluation framework](research/evaluation/README.md): quality–cost improvements against credible matched controls. Commodity hardware is a validation constraint. T0019 is ready after T0018 acceptance. T0020 scheduling-only amendment removes T0017 closure dependency; reviewed merge and separate named local launch required. No v0.3 mechanism adopted by the charter.
