# T0003 attempt-001 — passed (full FP32 Nano synthetic epoch)

Task T0003 (VAL), parent #10, issue #35. Instruction commit `371aa551a6d553dc38245f9821dfc43490e871cb`; approved source pin and tested commit `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa`. UTC 2026-11-27T15:13:47Z to 2026-11-27T15:15:49Z (122 seconds).

## Authorization and scope
The owner assigned the new T0003 work in the current local session after T0002 evidence PR #34 and the T0003 specification PR #36 merged; that assignment is the launch basis for attempt-001. Dependency review: T0002 #28 was explicitly accepted in the 2026-11-27T14:33:50Z research review and #34 merged at `49f1660`; T0003 #36 merged at `371aa551a6d553dc38245f9821dfc43490e871cb`. Scope was exactly one full synthetic-v1 epoch: 35 optimizer steps on the unchanged 43,055,076-parameter Nano config (A2/R2/FP32/B1/T96/seed42), built-in validation, then bounded CPU checkpoint/optimizer verification. No source edits, gate/tolerance changes, new packages, real corpus, BF16, Mini, second epoch or retry. The GPU was idle at 4 MiB used; no inference process was stopped or restarted.

## Outcome
**Passed.** All three required checks passed. This is one-epoch synthetic FP32 Nano execution and state-growth evidence; it is not language quality, generalization, long-context, production throughput or all-expert coverage evidence.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| cuda_nano_epoch | passed | `run-nano-epoch.json`, `metrics.jsonl`, `comparison-t0002-step1.json`, `execution.log` |
| checkpoint_validation | passed | `checkpoint-validation.json`, `checkpoint.txt`, `execution.log` |

## Observed results
- Source HEAD was exactly `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa` and the detached source worktree was clean at start and end.
- Environment: Python 3.12.7, torch 2.10.0+cu128, CUDA runtime 12.8, numpy 1.26.4, RTX 4090 (24,564 MiB, driver 580.178.04), GPU 0. Free VRAM was 24,077 MiB before and after; the GPU was idle at 4 MiB used.
- Disk available on the output filesystem: 185,505,796 KiB before and 185,000,652 KiB after the epoch.
- Preflight asserted the exact fixture budget `(blocks, valid, ntp, offset2, validation_ntp) = (35, 2591, 2534, 2477, 376)`. Config file SHA-256 `18ae9e3d7fc2cfc427ceec91de33526f51ccb2dc153ab16f8050ee2a061bad2b`; canonical config digest `0af02669bc89325d1e237876771d1c139faded29dbb5099777e9cd918cd53c90`.
- The epoch completed 35 steps with exit 0 and covered each training block exactly once (`batch_indices` sorted exactly `0..34`). Totals were 2,591 valid tokens, 2,534 supervised NTP labels and 2,477 available offset-2 labels.
- Combined A2 weighted objective (not pure NLL): first `15.798876762390137`, last `5.170285701751709`, minimum `4.6889214515686035`, maximum `15.798876762390137`, median `6.0046515464782715`. Final NTP CE: first `10.538311958312988`, last `3.4113402366638184`, minimum `3.1190764904022217`, median `4.011566162109375`. No loss-decrease threshold was imposed; these are descriptive observations.
- Step time: sum `8.36080655513797` s, median `0.23816414299653843` s, min `0.16602806199807674` s, max `0.4048001210321672` s. The actual command elapsed was `45` s, which includes per-step checkpoint writes and validation; step timing excludes checkpoint I/O and validation.
- CUDA allocator high-water marks across steps: maximum allocated `948460544` bytes (904.5 MiB); maximum reserved `1134559232` bytes (1082.0 MiB). First-step peaks were `881575424` / `1033895936` bytes; final-step peaks were `869773312` / `1134559232` bytes.
- Held-out synthetic validation: 376 NTP labels; NLL sum `1351.106689453125` nats; mean NLL `3.593368854928524` nats/token; perplexity `36.35634911641945`. This is synthetic-template validation, not language quality.
- T0002 step-1 comparison: exact equality on the ten non-timing fields (`step`, batch indices, label counts, objective, final/early/offset-2 CE and balance). Timing and allocator peaks were excluded as specified.
- Optimizer-state growth vs T0002: 198 of 198 model state entries now have optimizer state (T0002: 183), with `86110152` Adam moment elements (exp_avg plus exp_avg_sq). Present optimizer step values range from `1.0` to `35.0`. Entry count and moment elements are state inventory observations, not expert coverage or routing-distribution measurements.
- Checkpoint: `516923467` bytes; SHA-256 `27a6c998beed37bd2c7495edc596e2c190c9ab826a0cfe6eac63754f02192d27`. Local-only; no tensors or checkpoint bytes were committed. T0002 baseline checkpoint was 493,312,343 bytes with 183 state entries; peak allocator baseline was 881,575,424 allocated / 1,033,895,936 reserved bytes.
- Analytical forward MACs recorded as `208304394240` ideal MACs. This is an analytical planning figure, not measured FLOPs or throughput; the harness records `measured_flops` and `energy_joules` as null.

## Metrics
`metrics.jsonl` preserves all 35 per-step rows with batch indices, label counts, objective components, timing and CUDA peaks, plus validation, checkpoint inventory, T0002 comparison and resource observations. The observed step time is reported descriptively; no throughput, latency, energy or hardware-efficiency claim is made.

## First failure
No failure in this attempt; no retry was run. The epoch exited 0 and the dependent checkpoint check exited 0. There is no failed-gate reproduction command to report.

## Limitations
- One epoch on one synthetic fixture only; not convergence, long-training stability, full training feasibility or a second-epoch result.
- Synthetic-v1 short-document byte fixture; T96 is packing capacity, not continuous context length. No language-quality or generalization claim.
- FP32 only; no BF16/FP8, activation checkpointing or low-bit study.
- Nano only; no Mini, matched control or architectural-advantage claim.
- Single machine, single RTX 4090 and one software stack; no cross-device or cross-version determinism claim.
- Step timing excludes checkpoint I/O and validation; command elapsed includes them. No optimized throughput or latency claim.
- Optimizer state entries and moment elements do not measure expert routing coverage; no all-expert or all-parameter gradient claim.
- Checkpoint is local-only and not published; it is not available to other agents.
- No measured FLOPs, energy, PCIe/offload or hardware-efficiency claim; analytical forward MACs are not measured FLOPs.

## Proposed next action
Maintainer review of this evidence PR. T0003 (#35) remains open until review/acceptance; no merge or issue closure is performed here. Stop after T0003. BF16 requires a separate reviewed implementation/protocol and owner authorization; larger-context/benchmark work remains separate under #10/#7.
