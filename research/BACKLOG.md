# Research backlog / Plan prac

Snapshot 2026-11-28. Live issues are authoritative; no duplicate tracker. Previous chronological notes preserved in [history](https://github.com/jtrefon/splot-lm/blob/b91a68843b25624abfdc11c6eb4978d3e4c73e39/research/BACKLOG.md).

| Issue | Workstream | Current priority / next gate |
|---|---|---|
| [#85](https://github.com/jtrefon/splot-lm/issues/85) | Architecture decision gate | P0: review v0.3 candidates before heavy training |
| [#86](https://github.com/jtrefon/splot-lm/issues/86) | Conditional memory RFC | P0: proposed causal memory and matched allocation controls |
| [#87](https://github.com/jtrefon/splot-lm/issues/87) | Recurrent topology/granularity | P0: proposed stability/partial untying/expert pool ablations |
| [#2](https://github.com/jtrefon/splot-lm/issues/2) | Native MTP / recurrent self-speculation | P0: trained-checkpoint inference comparisons |
| [#7](https://github.com/jtrefon/splot-lm/issues/7) | Numerical precision | P1: native FP8 scan retained; matched learning and precision optimization |
| [#126](https://github.com/jtrefon/splot-lm/issues/126) | QAT pretraining | P1 proposed: expert-first W4A32, deployment format and matched PTQ/from-start/late QAT after T0035 |
| [#10](https://github.com/jtrefon/splot-lm/issues/10) | Experiment/hardware protocol | P1: profile GDN/attention, matched controls and context scaling |
| [#6](https://github.com/jtrefon/splot-lm/issues/6) | MoE locality | P0 design: actual route traces, tiered expert/row residency and exact cache scheduling, then measured hardware validation |
| [#42](https://github.com/jtrefon/splot-lm/issues/42) | Muon hybrid | P1: matched tuning/precision factorial on real data |
| [#8](https://github.com/jtrefon/splot-lm/issues/8) | Data and tokenizer | P1: pilot usable; scale/quality/provenance as research requires |
| [#43](https://github.com/jtrefon/splot-lm/issues/43) | CUDA 13 | P1: isolated runtime qualification, no disturbance of cu128 |
| [#11](https://github.com/jtrefon/splot-lm/issues/11) | Open science, licensing | P2: explicit research/weights licensing and replication |

Closed foundational work: #3/#4/#5/#9/#13 and T0001–T0016; see [task index](../experiments/tasks/README.md) and [evidence](../experiments/results/INDEX.md). T0017 and T0018 closed; T0017 procedural exceptions preserved. Scope priorities are proposals, not deadlines or local execution authorization.

Charter-aligned next task queue: T0036/#129 Mac: producer preallocation, author-disjoint rights-cleared bounded real corpus and train-only tokenizer binding; T0037/#130 Linux: expert-first W4A32 opt-in implementation and synthetic correctness/packed qualification. Both require merged spec, zero open integration PRs and named owner launch. #124 remains open for T0036 correction; #125 closed. See [evaluation](evaluation/README.md).
