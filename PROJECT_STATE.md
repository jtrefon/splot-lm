# Project state / Stan projektu

Snapshot 2026-11-28; reconciled against previous main `d5cc33c0ba3df21e4763284cc3a1f4cde08a38b9` and integrated #103/#104. **Live issues/PRs and merged evidence win.** Read [AGENTS](AGENTS.md), [REP](EXECUTION.md), [RCP](RESEARCH_COORDINATION.md), [roadmap](ROADMAP.md), [task index](experiments/tasks/README.md), [evidence index](experiments/results/INDEX.md). Previous snapshots preserved in Git.

## Scientific direction
[Charter](research/CHARTER.md) / [PL](research/CHARTER.pl.md): achieve more through better architecture, not greater scale. [Evaluation](research/evaluation/README.md): matched quality–cost evidence and capability guardrails. Commodity hardware is a validation constraint, not the mission. v0.2 reference; v0.3 candidates proposed, no large pretraining freeze.

## Current evidence
- Sparse recurrent hybrid reference, native MTP and exact D1/D2, fixed R1–4; Nano 43,055,076 and Mini 146,987,784 at V32768 (Mini inventory only). RTX4090 cu128 native FP8 156 selected projections, hybrid Muon/AdamW, exact same-mode resume, depth drift and T4096 qualification; no production throughput/quality claims.
- T0016: accepted bilingual corpus 8 work families, actual tokenizer V16384, 128/16/16 blocks, 56,201 content tokens, pinned manifest `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55`. No raw corpus or weights committed.
- T0017 attempt-001 blocked for real pilot but independent implementation accepted. [Attempt-002](experiments/results/T0017/attempt-002/summary.md) #92 merged/#77 closed: exact 4-step resume, 64 real-data updates, 23,534 NTP labels, validation NLL 7.744022 (5,250 labels), 6,652/14,749 reuse hits, 89s wall. Feasibility accepted with [procedural deviations](experiments/results/T0017/review-supplement-001/summary.md) (wheel container missing, no separate install consent, corrective rerun without separate approval); not retrospective authorization or clean REP compliance. [Durable checkpoint](experiments/results/T0017/review-supplement-001/checkpoint-durability.json) private ext4, hash verified locally. No pilot rerun or superiority claim.
- T0018 #91 merged/#81 closed: [attempt-002](experiments/results/T0018/attempt-002/summary.md) exact D0 and trace v2, model-call counts and full decode timing; 28 stdlib/16 hosted integration, nine independently checked fixtures. Attempt-001 review-rejected, preserved. Trained-checkpoint benchmark not executed.
- T0019 attempt-002 corrected same-vocab design, accepted; [review](research/evaluation/T0019-acceptance.md). V16384 NTP base 36,615,780; two dense comparator options, no learning result; A2 deferred.
- T0020 accepted 12 fresh-process synthetic single-update cases on RTX4090: T512 BF16/FP8 median 1.292/1.600s, T2048 4.832/5.167s; backward dominates, FP8 reuse disabled. [Review](research/evaluation/T0020-acceptance.md). Negative speed result, modest memory benefit; no steady-state or quality claim.

- T0021 dense NTP harness integrated, hosted CPU 9/9 tests; [review](research/evaluation/T0021-acceptance.md). Actual V16384 pilot loader and reporting/preflight still blocked, transferred to T0023; no matched training.
- T0022 four BF16 control/profile cases exact losses/gradient/model/optimizer equality; [review](research/evaluation/T0022-acceptance.md). T512/T2048 65,555/241,258 launches; generic labels, module attribution unassigned, profiler+summary severe perturbation. No module dominance claim.

- T0023 attempt-001 loader 6/6, dense suite 12/13, required hosted CI failed on report identity-path assertion. [Review](research/evaluation/T0023-review.md). Failed evidence preserved; implementation excluded from main tree. #101 remains open for scoped attempt-002; no matched training.
- T0024 isolated GDN chunk64 feasibility: 30 reported numerical comparisons and 8 resource cases passed, T512/T2048 median scan/chunk64 0.174688/0.011728s and 0.734590/0.046498s, respectively; [review](research/evaluation/T0024-acceptance.md). Component-only; separate document reset/split qualification absent, #102 remains open pending T0025 prerequisite. No full-model advantage or adoption.

## Active assignments
| Machine | Task | Next |
|---|---|---|
| Mac M4 | [T0023 #101](https://github.com/jtrefon/splot-lm/issues/101) | Attempt-002 test-only corrective continuation, actual-vocab qualification, hosted CPU. |
| Linux RTX4090 | [T0025 #105](https://github.com/jtrefon/splot-lm/issues/105) | Missing T0024 reset/tail prerequisite, then opt-in chunk64 full-model qualification. |

T0001–T0022 closed; #101 and #102 remain open, #105 next. T0023 failed code excluded, evidence preserved. T0024 component evidence accepted but missing prerequisite; no automatic retry. Owner authorizes named local execution only after merged specs and zero open integration PRs. No architecture adoption or full training.

## Priorities and continuity
[Architecture review EN](research/architecture-review/2026-11-28-en.md) / [PL](research/architecture-review/2026-11-28-pl.md), #85, #86 conditional memory, #87 recurrence/topology, #6 exact residency, #2 MTP, #7/#42 precision/optimizer, #8 data, #10 benchmarks, #11 publication, #43 isolated CUDA13. No new architecture adoption. Owner authorized lawful data acquisition within reviewed caps, no redistribution or rights bypass; no public weights, institutional affiliation or novelty/superiority claim. [Start prompt](SESSION_START.md) · [Backlog](research/BACKLOG.md) · [Historical decisions](research/decisions/README.md).
