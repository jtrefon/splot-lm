# Project state / Stan projektu

Snapshot 2026-11-28; reconciled against merged T0017 #92 at `452f2091f462eea4de9fa1aacba0333fedd38a18` and this integration PR. **Live issues/PRs and merged evidence win.** Read [AGENTS](AGENTS.md), [REP](EXECUTION.md), [RCP](RESEARCH_COORDINATION.md), [roadmap](ROADMAP.md), [task index](experiments/tasks/README.md), [evidence index](experiments/results/INDEX.md). Previous snapshots preserved in Git.

## Scientific direction
[Charter](research/CHARTER.md) / [PL](research/CHARTER.pl.md): achieve more through better architecture, not greater scale. [Evaluation](research/evaluation/README.md): matched quality–cost evidence and capability guardrails. Commodity hardware is a validation constraint, not the mission. v0.2 reference; v0.3 candidates proposed, no large pretraining freeze.

## Current evidence
- Sparse recurrent hybrid reference, native MTP and exact D1/D2, fixed R1–4; Nano 43,055,076 and Mini 146,987,784 at V32768 (Mini inventory only). RTX4090 cu128 native FP8 156 selected projections, hybrid Muon/AdamW, exact same-mode resume, depth drift and T4096 qualification; no production throughput/quality claims.
- T0016: accepted bilingual corpus 8 work families, actual tokenizer V16384, 128/16/16 blocks, 56,201 content tokens, pinned manifest `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55`. No raw corpus or weights committed.
- T0017 attempt-001 blocked for real pilot but independent implementation accepted. [Attempt-002](experiments/results/T0017/attempt-002/summary.md) #92 merged/#77 closed: exact 4-step resume, 64 real-data updates, 23,534 NTP labels, validation NLL 7.744022 (5,250 labels), 6,652/14,749 reuse hits, 89s wall. Feasibility accepted with [procedural deviations](experiments/results/T0017/review-supplement-001/summary.md) (wheel container missing, no separate install consent, corrective rerun without separate approval); not retrospective authorization or clean REP compliance. [Durable checkpoint](experiments/results/T0017/review-supplement-001/checkpoint-durability.json) private ext4, hash verified locally. No pilot rerun or superiority claim.
- T0018 #91 merged/#81 closed: [attempt-002](experiments/results/T0018/attempt-002/summary.md) exact D0 and trace v2, model-call counts and full decode timing; 28 stdlib/16 hosted integration, nine independently checked fixtures. Attempt-001 review-rejected, preserved. Trained-checkpoint benchmark not executed.
- T0019 [attempt-001](experiments/results/T0019/attempt-001/summary.md) design submitted, partial acceptance only. [Review](research/evaluation/T0019-review.md): V16384 dense candidates compared with V32768 SPLOT; correct denominators and A2 protocol before adoption.

## Active assignments
| Machine | Task | Next |
|---|---|---|
| Mac M4 | [T0019 #89](https://github.com/jtrefon/splot-lm/issues/89) | Attempt-002 bounded ANA correction, same original source pin and scope, after integration merge and explicit owner launch. |
| Linux RTX4090 | [T0020 #90](https://github.com/jtrefon/splot-lm/issues/90) | Independent synthetic repeated phase-cost baseline, reviewed spec merged and explicit owner launch required. |

T0001–T0018 closed. #89/#90 open. Owner authorizes named local execution; no launch with outstanding integration PRs. Research coordinator serializes integration, preserves failed/partial evidence and keeps task issues open for remaining gates. No autonomous retries or new optional installs without specific consent.

## Priorities and continuity
[Architecture review EN](research/architecture-review/2026-11-28-en.md) / [PL](research/architecture-review/2026-11-28-pl.md), #85, #86 conditional memory, #87 recurrence/topology, #6 exact residency, #2 MTP, #7/#42 precision/optimizer, #8 data, #10 benchmarks, #11 publication, #43 isolated CUDA13. No new architecture adoption. Owner authorized lawful data acquisition within reviewed caps, no redistribution or rights bypass; no public weights, institutional affiliation or novelty/superiority claim. [Start prompt](SESSION_START.md) · [Backlog](research/BACKLOG.md) · [Historical decisions](research/decisions/README.md).
