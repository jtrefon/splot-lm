# T0002 attempt-001 — passed (one-step FP32 Nano CUDA validation)

Task T0002 (VAL), parent #10, issue #28. Instruction commit `b6b17206944c8bf7a59bebfce847609715fbe5aa` (PR #33 merged); approved source pin and tested commit `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa`. UTC 2026-11-27T14:17:02Z to 2026-11-27T14:18:29Z (87 seconds).

## Authorization and scope
The owner explicitly authorized the named T0002 execution in the current local session after PR #33 merged. Scope was the task exactly as written: one synthetic FP32 Nano AdamW step, A2/R2/B1/T96/seed42, built-in held-out validation, then bounded CPU checkpoint/optimizer verification. No source edits, no gate/tolerance changes, no new packages, no real corpus, no BF16, no Mini, no extended training and no retry. The GPU was idle at 4 MiB used; no inference process was stopped or restarted.

## Outcome
**Passed.** All three required checks passed. This is one-step correctness and resource evidence for the 43,055,076-parameter synthetic Nano candidate; it is not long-training feasibility, language quality or optimized throughput.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| cuda_nano_step | passed | `run-nano.json`, `metrics.jsonl`, `execution.log` |
| checkpoint_validation | passed | `checkpoint-validation.json`, `checkpoint.txt`, `execution.log` |

## Observed results
- Source HEAD was exactly `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa` and the detached source worktree was clean at start and end.
- Environment: Python 3.12.7, torch 2.10.0+cu128, CUDA runtime 12.8, numpy 1.26.4, RTX 4090 (24,564 MiB, driver 580.178.04), GPU 0. Free VRAM was 24,077 MiB before and after the step; the GPU was idle at 4 MiB used.
- Disk available on the output filesystem: 185,993,944 KiB before and 185,511,932 KiB after the checkpoint write.
- Nano inventory: actual and declared parameters both 43,055,076; config file SHA-256 `18ae9e3d7fc2cfc427ceec91de33526f51ccb2dc153ab16f8050ee2a061bad2b`; canonical config digest `0af02669bc89325d1e237876771d1c139faded29dbb5099777e9cd918cd53c90`.
- Step 1: 51 valid tokens, 50 supervised NTP labels, 49 offset-2 labels available. Objective 15.798876762390137; final NTP 10.538311958312988; early NTP 10.537959098815918; offset-2 10.44066333770752; balance 1.5909053087234497 (unitless auxiliary loss).
- Step time 0.42532914300682023 s. Peak CUDA allocated 881,575,424 bytes (840.7 MiB); peak reserved 1,033,895,936 bytes (986.0 MiB). Timing includes finite-gradient checks and the optimizer update but excludes checkpoint I/O and validation.
- Held-out synthetic validation: 376 NTP labels; NLL sum 3871.2344665527344 nats; mean NLL 10.295836347214719 nats/token; perplexity 29,609.079916681367.
- Checkpoint verification: exit 0; weights finite; all present Adam moments finite; all present optimizer step values equal 1; 183 optimizer-state entries. The model state dict has 43,055,076 elements. Unused experts need not have optimizer entries, so this is not a claim that every parameter received a gradient.
- Checkpoint: 493,312,343 bytes; SHA-256 `36e161590bf7b5c8a8dc4d6df8b7bcd92d30744de0b73646b51327217bc3695b`. Local-only; no tensors or checkpoint bytes were committed.
- Analytical forward MACs recorded as 4,100,732,928 ideal MACs. This is an analytical planning figure, not measured FLOPs or throughput; the harness records `measured_flops` and `energy_joules` as null.

## Metrics
`metrics.jsonl` contains the raw per-step observations, validation, checkpoint verification, analytical MAC figure and pre/post GPU/disk observations with explicit units. The observed step time is reported descriptively; no throughput, latency, energy or hardware-efficiency claim is made.

## First failure
No failure in this attempt; no retry was run. There is no failed-gate reproduction command to report. The step exited 0 and the dependent checkpoint check exited 0.

## Limitations
- One optimizer step only; this is not evidence of convergence, long-training stability or full training feasibility.
- Synthetic-v1 byte fixture only; not a production tokenizer, corpus or language-quality evaluation.
- FP32 only; no BF16/FP8, activation-checkpointing or optimizer-memory study.
- Nano only; no Mini or matched-control result.
- Single machine, single RTX 4090 and one software stack; no cross-device or cross-version determinism claim.
- Timing is a single observed step including finite-gradient checks and optimizer work; it is not an optimized throughput or latency benchmark.
- Checkpoint is local-only and not published; it is not available to other agents.
- 183 of 198 model state-dict entries have optimizer state; unused experts need not have entries, and no all-parameter gradient claim is made.
- Validation uses synthetic templates; it is not generalization or production quality evidence.

## Proposed next action
Maintainer review of this evidence PR. T0002 (#28) remains open until review/acceptance; no merge or issue closure is performed here. No automatic next task or additional GPU run. Future precision work remains under #7 and larger benchmark/matched-control work under #10.
