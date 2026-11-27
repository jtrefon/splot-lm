# T0005 attempt-001 — passed (BF16 harness with FP32 safeguards)

Task T0005 (IMP), issue #40, parent #7/#10. Instruction commit `0fcd9c6b5b9667af47603a63013b89d83e5f565f`; base `0fcd9c6b5b9667af47603a63013b89d83e5f565f`; minimum reviewed base `625e6a7bfac97344fab6a9b102aab3613a670b1e`; tested implementation commit `e759576f607b97602e9d0407adae4307d3eb38f5`. UTC 2026-11-27T15:51:27Z to 2026-11-27T16:02:31Z (664 seconds).

## Authorization and scope
The owner explicitly named T0005 in the current local session. The reviewed task requires a clean implementation commit and bounds this attempt to the Linux RTX 4090 stack, CPU regressions, one paired Nano step and a BF16 smoke resume. Allowed files were `tools/experiment.py`, the new BF16 tests/configs/documentation, and narrowly scoped `src/splot/dense.py`, `mtp.py`, `gdn.py` or `moe.py` changes only if required. No fixture/data, existing FP32 config, topology, tolerance, checkpoint-cadence, Mac, package, real-corpus, Mini, FP8, quantization or long-context change was made. The GPU was idle; no inference process was stopped.

## Outcome
**Passed.** All five required checks passed. The implementation adds opt-in BF16 autocast around forward/objective/evaluation while keeping FP32 master parameters, FP32 AdamW state, deterministic settings and TF32 disabled. The existing FP32 path is unchanged; one GQA softmax/value safeguard was required.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `summary.md` |
| cpu_tests | passed | `cpu-regression.json`, `execution.log` |
| cuda_precision | passed | four run JSONs, `comparison.json`, `dtype-report.json`, `precision-comparison.json`, `checkpoint-state.json`, `metrics.jsonl`, `execution.log` |
| handover | passed | `summary.md`, `checkpoints.txt`, `execution.log` |

## Implementation
- `tools/experiment.py` now resolves explicit `fp32` or `bf16`, rejects unsupported values/backends, probes an actual eligible projection, records requested/actual policy in identity/environment, and applies BF16 autocast only around forward/objective/evaluation. Backward and optimizer remain outside autocast; no GradScaler.
- `src/splot/dense.py` GQA now runs score contraction, softmax and value contraction inside `torch.autocast(..., enabled=False)`. This was required: the probe showed GQA softmax and value inputs were still BF16 before the fix. No other source file required a change.
- New BF16 configs are identical to their FP32 counterparts except `training.precision`. The policy and tests are documented in `research/implementation/bf16-harness.md`.

## CPU regression
13/13 suites passed with **72 tests**, **0 failures**, **0 skips**, including the new `tools/test_bf16_harness.py` (5 focused tests on the actual SPLOTLM model). The new suite also passed explicitly on CUDA with `--device cuda`.

## Paired CUDA validation
| Run | Steps | Parameters | Validation | Result |
|---|---|---|---|---|
| FP32 Nano | 1 | 43,055,076 | 376 labels; NLL 3871.2344665527344; mean 10.295836347214719; PPL 29609.079916681367 | exit 0 |
| BF16 Nano | 1 | 43,055,076 | 376 labels; NLL 3862.8521728515625; mean 10.273543012903092; PPL 28956.298175657404 | exit 0 |
| BF16 smoke continuous | 4 | 179,952 | 376 labels; mean 5.1851281916841545; PPL 178.59634288842062 | exit 0 |
| BF16 smoke resumed | 4 | 179,952 | identical to continuous | exit 0 |

Nano first-step comparison (predeclared smoke tolerances):

| Component | FP32 | BF16 | Absolute difference | Threshold |
|---|---|---|---|---|
| final NTP CE | 10.538311958312988 | 10.53844928741455 | 0.0001373291015625 | <= 0.10 |
| early NTP CE | 10.537959098815918 | 10.537418365478516 | 0.0005407333374023438 | <= 0.10 |
| offset-2 CE | 10.44066333770752 | 10.440749168395996 | 8.58306884765625e-05 | <= 0.10 |
| composite objective | 15.798876762390137 | 15.798896789550781 | 2.002716064453125e-05 | <= 0.20 |

The FP32 Nano run is exactly equal to T0002 on all ten non-timing fields (`t0002_fp32_non_timing_equal=true`). BF16 balance differed by 0.0003714561462402344 (unitless, descriptive). Route top-k set agreement was 0.992647 with top-1 agreement 1.000000 over 408 assignments; route equality is not a gate.

## Dtype and state verification
- Eligible projection outputs were BF16; logits were BF16.
- GDN matrix state, GDN history, GQA score/value inputs, GQA softmax, router probabilities, RMSNorm means/rsqrt, RoPE cos/sin and CE output were verified FP32. GDN raw history was BF16.
- Both Nano checkpoints contain FP32 model tensors, 183 optimizer-state entries, 366 FP32 Adam moment tensors, all finite, with optimizer step value 1.0.
- Checkpoints are local-only: FP32 Nano `e3a612e64d3b5822050769d3b3a831d6b0dd05a96a4125be7fadc05c976dcd3d` (43055076 params, 493,312,599 bytes); BF16 Nano `6fe7676d1d701b6d6a1e6fc310c13fd9fe5a3fd4144b8ee4579351cb21fdf067` (493,312,663 bytes); BF16 continuous `3402bfcf57e51ac8133f9affbb904d159a052e26f938e81ead0d0c5d6236ce9b` (2,414,859 bytes); BF16 resumed-final `ce155ffaf6e8023897ee9a5b4daa9a40e79a5287f8da3bf838f0039fb1288256` (2,415,115 bytes).

## Resources and timing
- Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8, numpy 1.26.4; RTX 4090, 24,564 MiB, driver 580.178.04, GPU 0; CUDA BF16 supported; CPU autocast probe returned BF16.
- Free VRAM 24,077 MiB before and after; disk 184,028,896 KiB before and 183,060,556 KiB after. No inference process was stopped.
- Command elapsed: FP32 Nano 3 s, BF16 Nano 3 s, BF16 continuous 3 s, split 3 s, resume 3 s, comparator 1 s, focused CUDA tests 3 s, verification 4 s. Step timings are in `metrics.jsonl`; no throughput or performance claim is made.

## Metrics
`metrics.jsonl` preserves CPU suite exits/counts, all step rows and validation, the same-mode comparator, dtype checks, precision differences, route observations, checkpoint state and resource observations with explicit units.

## First failure
No failure in this attempt; no retry. The only implementation correction was the predeclared GQA FP32 softmax/value safeguard after the initial autoprobe showed BF16 softmax; it was committed before all reported tests.

## Limitations
- One paired Nano step and a 4-step BF16 smoke only; no convergence, full-epoch BF16/FP32 comparison, long-training or language-quality claim.
- BF16 autocast is not a claim that all activations or kernels are BF16, nor that BF16 is faster. GQA value contraction is FP32; GDN history remains BF16.
- CPU BF16 is validated by focused tests when the backend supports it; it is not a CUDA substitute.
- Synthetic byte fixture only; not a production tokenizer or corpus.
- No FP8, quantization, activation checkpointing, MPS, Mini, real corpus, checkpoint-cadence or topology change.
- No measured throughput, energy, PCIe/offload or production-hardware claim.
- Checkpoints are local-only and not published; no tensors are committed.

## Proposed next action
Maintainer review of this evidence PR. T0005 (#40) remains open until review/acceptance; no merge or closure is performed here. A matched BF16/FP32 full-epoch validation and checkpoint-cadence measurement require separate reviewed tasks. Muon #42 and CUDA 13 #43 remain separate qualification issues and were not changed.
