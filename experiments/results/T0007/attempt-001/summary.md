# T0007 attempt-001 — passed (native FP8 scaled-GEMM feasibility)

Task T0007 (IMP), issue #51, parents #7/#10. Instruction/base `d2dfd903d252c1ba48641bf7de79fba42c385af8`; minimum reviewed base `f60bd5622733f3619e761681a20472d8c40f478c`; tested implementation commit `d5ea0f94f3145f366675d4bf95cf1851bf18ebcd`. UTC 2026-11-27T17:37:49Z to 2026-11-27T17:38:58Z (69 seconds).

## Authorization and scope
The owner explicitly authorized T0007 on the local RTX 4090 in the current session. The reviewed task is an isolated operator feasibility probe only. Allowed implementation files were new `tools/fp8_probe.py`, new `tools/test_fp8_probe.py`, new `research/implementation/fp8-probe.md`, and REP evidence/index. No package install, CUDA/driver/toolkit change, model integration, autograd claim, harness/config/CI/data edit or real corpus was involved. The GPU was idle; no inference process was stopped.

## Outcome
**Passed.** Actual native `torch._scaled_mm` dispatch succeeded for E4M3 forward and mixed E5M2 dY x E4M3 dgrad/wgrad on the installed RTX 4090 `sm_89` / torch 2.10.0+cu128 stack. All 11 required cases and 33 forward/dgrad/wgrad GEMMs passed the numerical/dispatch gates. All-zero outputs were exact and nonfinite input was rejected. This is operator feasibility, not FP8 training or a performance claim.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `summary.md` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_fp8 | passed | `fp8-probe.json`, `metrics.jsonl`, `execution.log` |
| handover | passed | `summary.md`, `fp8-probe.json`, `execution.log` |

## Native API and format contract
- Installed schema: `aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor`.
- GPU: `NVIDIA GeForce RTX 4090`, capability `(8, 9)`; CUDA `12.8`; torch `2.10.0+cu128`.
- Float8 dtypes present: `float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz, float8_e8m0fnu`.
- Forward: E4M3FN x E4M3FN, BF16 output, row-major LHS and column-major RHS.
- dgrad/wgrad: E5M2 dY x E4M3 other operand, BF16 output; layout copies recorded where the required column-major/row-major orientation was not free.
- Per-tensor current FP32 scalar scales; quantization `s=max(abs(x))/finfo(dtype).max` (all-zero `s=1`), finite input only, clamp/cast/dequantize with saturation/underflow/zero audit counts.
- Actual input dtypes, shapes, strides, scales, output dtype/stride and layout-copy flags are recorded per case/op. No fake-quant matmul is counted as native.

## Numerical results
- 11 cases: six aligned large shapes, four routed-token-like M=1/3/7/17 padded to 16/16/16/32, and one all-zero case.
- 33/33 forward/dgrad/wgrad cases passed finite and numerical gates.
- Random-case relative Frobenius error versus FP32 ranged `0.036661` to `0.062166` (gate `<= 0.15`).
- Random-case relative Frobenius error versus the FP32 product of dequantized operands ranged `0.001630` to `0.001681` (gate `<= 0.025`).
- All-zero forward/dgrad/wgrad outputs were exactly zero; nonfinite quantization input raised `ValueError: nonfinite input cannot be quantized`.

## Small-M padding
| Real M | Padded M | Padding rows | Overhead |
|---|---|---|---|
| 1 | 16 | 15 | 16.000x |
| 3 | 16 | 13 | 5.333x |
| 7 | 16 | 9 | 2.286x |
| 17 | 32 | 15 | 1.882x |

Zero rows were used; forward/dgrad outputs were trimmed. This is padding overhead evidence, not a full sparse-execution claim.

## Timing observations
All timings are paired 5-warmup/20-synchronized-iteration medians; no speed threshold or superiority claim.

| Mode | Median range |
|---|---|
| Prequantized native FP8 GEMM | 0.0698–0.0719 ms |
| Quantization-inclusive (scale/cast/layout + GEMM) | 0.4723–0.5437 ms |
| BF16 matmul baseline | 0.0377–0.0424 ms |

Prequantized FP8 was not faster than BF16 in this small-shape harness; this is reported descriptively, not as a gate. Prequantized allocator peaks ranged 37,996,032–46,269,952 bytes. The profiler recorded `aten::_scaled_mm` and the native kernel `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5_warpsize2x2x1_tensor16x8x32_algo2_execute_kernel__5x_cublas`. No Tensor Core instruction claim is made.

## CPU regression
`tools/ci_cpu.py` passed from clean tested SHA `d5ea0f94f3145f366675d4bf95cf1851bf18ebcd`: **17 CPU scripts, 121 tests, 0 failures, 0 skips**, plus 8 REP reports. The new `tools/test_fp8_probe.py` contributes six deterministic CPU tests for scaling, all-zero, signed extremes, nonfinite rejection, padding/layout and error arithmetic.

## Metrics
`fp8-probe.json` contains the full per-case/per-op dispatch, quantization audit, error, latency and profiler observations; `metrics.jsonl` gives the flat machine-readable observations with units.

## First failure
No failure in this attempt; no retry. Unsupported operators were not encountered on this stack. No silent fallback or simulated native dispatch was used.

## Limitations
- Operator feasibility only; no FP8 model training, autograd integration, optimizer, checkpoint, convergence or language-quality claim.
- Current per-tensor scaling only; no delayed scaling, MXFP8, NVFP4, block scaling, FP8 optimizer or FP8 checkpoint state.
- BF16 output only; no FP8 output or FP8 accumulation claim.
- Quantization-inclusive latency includes Python/layout overhead; small-shape observations are not production throughput.
- Synthetic independent normal inputs and representative shapes, not a corpus benchmark.
- No package, CUDA/driver/toolkit, model, harness, config, CI or data change.
- No Tensor Core instruction claim beyond the recorded native kernel name.

## Proposed next action
Maintainer review of this evidence PR. T0007 (#51) remains open until review/acceptance; no merge or closure is performed here. FP8 autograd/linear integration requires a separate reviewed task and must preserve the BF16/FP32 controls.
