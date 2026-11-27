# T0006 attempt-001 — passed (hybrid Muon/AdamW correctness)

Task T0006 (IMP), issue #48, parents #42/#7/#10. Instruction/base `2503a4ed2b219e2f9ca0d941f8d1e2ed675e5e70`; minimum reviewed base `cf7ae82eb046aa3f01910121bbdd6a253808b00f`; tested implementation commit `3a208567f277545f2688814e72488d7e965eb9de`. UTC 2026-11-27T17:08:22Z to 2026-11-27T17:12:54Z (272 seconds).

## Authorization and scope
The owner explicitly authorized T0006 attempt-001 in the current local session. Scope was the reviewed task: opt-in native Muon/AdamW hybrid for the synthetic harness, unchanged default AdamW, semantic exhaustive/disjoint partition, validated nested checkpoint state, CPU regression and bounded paired CUDA checks. Allowed files were `tools/experiment.py`, new `tools/optimizers.py`, new `tools/test_hybrid_optimizer.py`, two new Muon configs, new documentation and REP evidence/index. No `src/`, existing config, CI, Mac, fixture/data, comparator, topology, precision-policy or checkpoint-cadence change was made. The GPU was idle; no inference process was stopped.

## Outcome
**Passed.** All five required checks passed. Native `torch.optim.Muon` 2.10 plus native AdamW are used for a semantic partition; FP32 parameters and both optimizer states are preserved; the default AdamW path and checkpoint format are unchanged. This is correctness evidence only, not optimizer superiority, convergence or tuning evidence.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `summary.md`, `partition.json` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_optimizer | passed | four run JSONs, `comparison.json`, `checkpoint-state.json`, `cuda-tests.txt`, `metrics.jsonl`, `execution.log` |
| handover | passed | `summary.md`, `checkpoints.txt`, `execution.log` |

## Implementation
- `tools/optimizers.py` classifies by actual owning module and role, deduplicates parameter identity, sorts stable canonical names, asserts exhaustive/disjoint inventory and fails closed on unknown parameters. Muon gets 157 hidden 2-D matrices; AdamW gets the tied embedding, RMSNorm, GDN `a`/`b`, `conv`, `log_A`, `dt_bias`, router and other explicit non-Muon state.
- `MuonAdamW` wraps native `torch.optim.Muon` and native `torch.optim.AdamW`. `zero_grad` clears both; `step` runs Muon then AdamW after the harness's single finite-gradient check; there is no double update.
- The nested adapter state contains adapter version, partition metadata/digest, resolved settings, Muon state and AdamW state. Load validates version, partition names/shapes/order and settings before loading both states; cross-mode/repartitioned states are rejected. Default AdamW checkpoints remain plain AdamW state dicts.
- Native Muon NS intermediate dtype is `bfloat16` from installed `torch/optim/_muon.py`; native momentum buffers are FP32. Parameters and AdamW moments are FP32. The NS implementation was not changed.
- New configs are identical to their FP32 counterparts except the explicit optimizer block. `tools/experiment.py` records the resolved policy, partition and state inventory in `run.json`.

## CPU regression
`tools/ci_cpu.py` passed locally from clean tested SHA `3a208567f277545f2688814e72488d7e965eb9de`: **15 CPU scripts, 86 tests, 0 failures, 0 skips**, plus 5 REP reports. The new focused suite contributes five tests, including native rectangular Muon updates, smoke/Nano partition inventory, tied weights once, router/gates excluded, unused-parameter preservation, FP32 state, mode/partition mismatch rejection and exact 4-vs-2+2 resume. The same suite passed on CUDA with `--device cuda` (5/5).

## Paired CUDA validation
| Run | Steps | Parameters | Validation | Result |
|---|---|---|---|---|
| FP32 Nano AdamW | 1 | 43,055,076 | 376 labels; NLL 3871.2344665527344; mean 10.295836347214719; PPL 29609.079916681367 | exit 0 |
| Muon hybrid Nano | 1 | 43,055,076 | 376 labels; NLL 3756.2002563476562; mean 9.989894298796958; PPL 21804.99386233146 | exit 0 |
| Muon smoke continuous | 4 | 179,952 | 376 labels; mean 5.280603692886677; PPL 196.4884582389272 | exit 0 |
| Muon smoke resumed | 4 | 179,952 | identical to continuous | exit 0 |

The FP32 and Muon Nano runs have exactly equal **pre-update** loss history on all ten fields (step, batch, label counts, objective and components) because the optimizer has not stepped yet. The FP32 run also exactly equals T0002 on those non-timing fields. Post-step validation is reported descriptively: Muon `9.989894298796958` vs FP32 `10.295836347214719` mean NLL; no superiority gate or claim. All runs have 51 valid tokens, 50 NTP labels, 49 offset-2 labels and 376 validation labels.

Muon smoke continuous/resumed comparison passed: model, both optimizer states, RNG, non-timing history and validation are exactly equal (`comparison.json`). The smoke comparator ignores only step timing and allocator peaks.

## Partition and optimizer state
- Partition totals: **Muon 157 parameters / 30,425,088 numel; AdamW 41 / 12,629,988; total 43,055,076**; plan SHA-256 `1bed555467ef9a955079b8260636971dafa5b0b308962e9a18e23a5ffa7bf9ed`. Smoke partition is Muon 157 / 167,936 and AdamW 41 / 12,016.
- FP32 AdamW Nano state: 183 state entries, 366 moment tensors, 80,212,095 state elements / 320,848,380 bytes; all FP32 and finite, step 1.0.
- Muon hybrid Nano state: 142 Muon momentum-buffer entries (some unused experts have no state), 27,475,968 Muon elements / 109,903,872 bytes; 41 AdamW states with 82 moment tensors, 25,260,017 elements / 101,040,068 bytes; total 52,735,985 elements / 210,943,940 bytes. All present tensors are FP32 and finite; AdamW step values are 1.0.
- Checkpoints are local-only:
  - FP32 `fe738fe1f68b90e90b660470755212550c1c22d59fcb191ff461b20710902557`, 493,312,663 bytes.
  - Muon Nano `eaee68cc13ca082a6f8ce15c94455d351cdcc18c4a1ec5af0c5f596bfee51e8c`, 383,356,647 bytes.
  - Muon continuous `c82b62e8ee0562e48b506f5d5b4ab893bcd68edad24c1e756ce2cc0d88407a72`, 1,682,787 bytes.
  - Muon resumed-final `46054a9cdf40a8736e6335ea4f0739eef09a8ad3232dbd35877c2625dc729429`, 1,683,107 bytes.

## Resources and timing
- Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8, numpy 1.26.4; RTX 4090, 24,564 MiB, driver 580.178.04, GPU 0; native `torch.optim.Muon` available.
- Free VRAM 24,077 MiB before and after; disk 183,058,748 KiB before and 182,199,016 KiB after. No inference process was stopped.
- CUDA command elapsed: FP32 Nano 3 s, Muon Nano 3 s, Muon continuous 4 s, split 3 s, resume 2 s; comparator <1 s; focused CUDA tests 6 s. No throughput or performance claim is made.
- FP32 Nano peak allocated/reserved: 881575424 / 1033895936 bytes. Muon Nano peak: 661767680 / 805306368 bytes. These are allocator observations, not a superiority claim.

## Metrics
`metrics.jsonl` preserves the 15 CPU suite results, all step rows, validation, pre-update equality, partition, optimizer state, checkpoint state and resource observations with units.

## First failure
No failure in this attempt and no retry. The default AdamW and BF16 regression paths remained green.

## Limitations
- Correctness-only single step plus 4-step synthetic smoke; no convergence, tuning, scale, language quality or optimizer-superiority claim.
- FP32 hybrid only; BF16-hybrid empirical validation, full-epoch matched comparison, scheduler and tuning require separate reviewed tasks.
- Native Muon NS intermediate is BF16 internally; native momentum and both optimizer states are FP32. This is recorded, not changed.
- Some Muon expert parameters were unused in the one-step run; 142 of 157 Muon state entries are present. No all-expert coverage claim.
- No CUDA 13, external kernels, FP8, quantization, real corpus or Mini change.
- Checkpoints are local-only and not published; no tensors are committed.

## Proposed next action
Maintainer review of this evidence PR. T0006 (#48) remains open until review/acceptance; no merge or closure is performed here. Hosted CPU CI is required on the PR in addition to the local `tools/ci_cpu.py` pass. Later matched full-epoch/tuning and optimizer-by-precision comparisons require separate reviewed tasks; #43 CUDA 13 remains a separate environment qualification.
