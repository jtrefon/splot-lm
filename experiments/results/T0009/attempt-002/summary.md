# T0009 attempt-002 — passed (FP8 audit, finite guards and diagnostics correction)

Task T0009 (IMP), issue #56, parents #7/#10. Instruction/base `441a9ec57ed502d04cff55f59eed417cb5aa1f72`; minimum reviewed base `31a3051d26850d5328912e1f9fc93f64591910f8`; tested correction commit `572364ca6320ede03776c549a1eb7e884661984c`. UTC 2026-11-27T18:24:39Z to 2026-11-27T18:26:18Z (99 seconds). Attempt-001 is preserved unchanged in `experiments/results/T0009/attempt-001/`.

## Authorization and correction scope
The owner authorized the requested T0009 corrections within the approved scope: numerical diagnostics, finite guards and acceptance-test corrections. No acceptance threshold was changed, no Muon path was introduced, and no additional task was executed. The correction commit changes only `src/splot/fp8.py`, `tools/experiment.py`, `tools/test_fp8_autograd.py` and `research/implementation/fp8-autograd.md`; `implementation.files.json` records the scoped diff from the attempt-001 evidence head.

## Corrections
- `_quantize_audit` now returns device-resident input/weight/gradient saturation, underflow, nonzero and nonfinite counters; output/dX/dW set a device-side nonfinite flag. No per-projection host synchronization was added. `fp8_diagnostics` reduces the counters only at a report/step boundary and reports native forward/backward calls, wgrad FP32 calls and padding rows.
- `reset_fp8_audit` clears all per-module counters/audit state before each depth. The depth report now carries per-depth native calls, padding rows, saturation/underflow/nonzero and nonfinite status, not cumulative R1/R2/R4 aggregates.
- `fp8_assert_finite` checks FP32 master parameters, FP8 audit/output flags and gradients at the training forward/backward boundaries and evaluation. Nonfinite input, weight and gradient tests were added; failures raise before update/evaluation.
- `_grad_stats` reports matched-only and full-vector (None as zero) relative L2, matched/full parameter counts, missing BF16/FP8 gradient counts and nonfinite counts.
- `_routing_stats` now asserts identical route keys and selected shapes before comparing. Depth entries require both precisions finite for hidden/logits/losses/gradients, exact label counts, route keys/shapes and the CE gates.
- `--report` now exits nonzero if the depth report status fails; the `report_exit_ok` contract is covered by a CPU test.
- Wgrad dispatch is explicitly FP32. The dequantized-oracle comparison uses the actual effective upstream BF16 gradient received by the Function, not the pre-autograd FP32 tensor. Forward/dgrad outputs remain BF16.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `summary.md` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_fp8 | passed | four run JSONs, `comparison.json`, `depth-report.json`, `linear-report.json`, `checkpoint-state.json`, `cuda-tests.txt`, `metrics.jsonl` |
| handover | passed | `summary.md`, `checkpoints.txt`, `execution.log` |

## Results
- CPU CI: **19 scripts, 150 tests, 0 failures, 0 skips**, 11 REP reports. The new CPU test covers the report-exit contract; CUDA tests add nonfinite input/weight/gradient guards.
- CUDA autograd: 5/5 tests. Linear errors versus FP32 `0.034907–0.059632` (gate 0.15); versus dequantized operands `5.599e-08–0.0017197` (gate 0.025). Wgrad dtype is FP32; profiler records `aten::_scaled_mm` and SM89 kernels with no DtoH/HtoD.
- Per-depth audit (all finite, saturation zero, nonfinite false):
  - smoke R1/R2/R4 input underflow `18/28/60`, weight underflow `0/0/0`, grad underflow `0/0/0`; native forward/backward `312/156`, `596/298`, `1160/580`; padded rows `2422/1211`, `4636/2318`, `9736/4868`.
  - nano R1/R2/R4 input underflow `268/526/1352`, weight underflow `380/758/1466`, grad underflow `0/5/19`; native forward/backward `240/120`, `482/241`, `944/472`; padded rows `2422/1211`, `5116/2558`, `9928/4964`.
- Depth CE gates pass; max smoke component/composite differences `0.001502/0.001521`; max Nano component/composite `0.012225/0.011299`. Labels are exact. Full-vector gradient drift (None as zero): smoke `0.084495/0.109737/0.189558`; Nano `0.174625/0.222157/0.342295`. Matched-only values are equal except Nano R2 (`0.222154` matched vs `0.222157` full, 3 missing FP8 grads). Routing keys/shapes match; exact/top1 agreement smoke `0.9510/0.9804`, `0.9265/0.9657`, `0.8934/0.9424`; Nano `0.9265/0.9706`, `0.9093/0.9608`, `0.8701/0.9534`.
- CLI BF16/FP8 Nano: 43,055,076 parameters, 51 valid/50 NTP/49 offset-2/376 validation labels. CE diffs final `0.007979`, early `0.012225`, offset-2 `0.0001497`, composite `0.011023` nats. FP8 audit from the run: input underflow `2456`, weight underflow `1969`, grad underflow `5`, saturation `0`, nonfinite false, native forward/backward `1420/241`, wgrad FP32 `241`, padded rows `10976/2558`, scales CUDA.
- FP8 smoke 4-step continuous and 2+2 resume exactly match; cross-precision resume raises `ValueError: run identity mismatch`.
- Checkpoints preserve identical model keys, FP32 master/Adam state, step 1.0. New attempt-002 checkpoint hashes are recorded in `checkpoints.txt`; FP8 has 180 states/360 moments, BF16 183/366 (unused experts). Checkpoints remain local-only.
- FP8 step time `0.689 s` vs BF16 `0.420 s`; peaks FP8 `875,259,392/1,107,296,256 B` vs BF16 `881,575,424/1,138,753,536 B`. Guards/audit add overhead; no acceleration or quality claim.

## Metrics
`metrics.jsonl` contains CPU suite results, run steps/validation, per-depth audit/CE/finite/label/routing/gradient observations, linear errors, checkpoint state and resource data with units. Attempt-001 remains untouched.

## First failure
No unexpected failure in attempt-002 and no retry. Expected cross-precision rejection is recorded separately. The earlier review findings were corrected before this attempt’s official runs.

## Limitations
- Correctness integration only; no convergence, pretraining, long-context, throughput or language-quality claim.
- Current per-tensor scaling only; no delayed/persistent scaling, MXFP8/NVFP4, FP8 optimizer/state or FP8 output.
- Some unused experts are not dispatched; coverage is recorded, not an all-expert claim.
- Drift/routing are forward/backward comparisons, not trained-model stability.
- No FP8+Muon, fused/grouped expert kernels, real corpus or production benchmark.
- FP8 output/dgrad are BF16; wgrad is FP32. No Tensor Core instruction claim beyond recorded kernel names.

## Proposed next action
Maintainer review of attempt-002 evidence and PR #59. T0009 remains open; no merge or closure. Hosted CPU CI is required. FP8+Muon, long-context, real-corpus and kernel-performance work require separate reviewed tasks.
