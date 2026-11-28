# Research backlog / Plan prac

Snapshot 2026-11-28. Live issues are authoritative; no duplicate tracker. Previous chronological notes preserved in [history](https://github.com/jtrefon/splot-lm/blob/b91a68843b25624abfdc11c6eb4978d3e4c73e39/research/BACKLOG.md).

| Issue | Workstream | Current priority / next gate |
|---|---|---|
| [#77](https://github.com/jtrefon/splot-lm/issues/77) | T0017 real FP8/Muon architecture pilot | P0: exact data and resume gates, 64 steps |
| [#81](https://github.com/jtrefon/splot-lm/issues/81) | T0018 MTP observability | P0: exact D0/D1/D2 trace and protocol |
| [#85](https://github.com/jtrefon/splot-lm/issues/85) | Architecture decision gate | P0: review v0.3 candidates before heavy training |
| [#86](https://github.com/jtrefon/splot-lm/issues/86) | Conditional memory RFC | P0: proposed causal memory and matched allocation controls |
| [#87](https://github.com/jtrefon/splot-lm/issues/87) | Recurrent topology/granularity | P0: proposed stability/partial untying/expert pool ablations |
| [#2](https://github.com/jtrefon/splot-lm/issues/2) | Native MTP / recurrent self-speculation | P0: trained-checkpoint inference comparisons |
| [#7](https://github.com/jtrefon/splot-lm/issues/7) | Numerical precision | P1: recurrent stability, profile and optimize reference |
| [#10](https://github.com/jtrefon/splot-lm/issues/10) | Experiment/hardware protocol | P1: profile GDN/attention, matched controls and context scaling |
| [#6](https://github.com/jtrefon/splot-lm/issues/6) | MoE locality | P0 design: actual route traces, tiered expert/row residency and exact cache scheduling, then measured hardware validation |
| [#42](https://github.com/jtrefon/splot-lm/issues/42) | Muon hybrid | P1: matched tuning/precision factorial on real data |
| [#8](https://github.com/jtrefon/splot-lm/issues/8) | Data and tokenizer | P1: pilot usable; scale/quality/provenance as research requires |
| [#43](https://github.com/jtrefon/splot-lm/issues/43) | CUDA 13 | P1: isolated runtime qualification, no disturbance of cu128 |
| [#11](https://github.com/jtrefon/splot-lm/issues/11) | Open science, licensing | P2: explicit research/weights licensing and replication |

Closed foundational work: #3/#4/#5/#9/#13 and T0001–T0016; see [task index](../experiments/tasks/README.md) and [evidence](../experiments/results/INDEX.md). T0017 implementation accepted, overall pilot still open. Scope priorities are proposals, not deadlines or local execution authorization.
