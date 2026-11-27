# T0009 attempt-001 — passed (native FP8 autograd and SPLOT projection integration)

Task T0009 (IMP), issue #56, parents #7/#10. Instruction/base `441a9ec57ed502d04cff55f59eed417cb5aa1f72`; minimum reviewed base `31a3051d26850d5328912e1f9fc93f64591910f8`; tested implementation commit `3c81e1299f6a7b6a9e332354c4d6996e12999667`. UTC 2026-11-27T17:58:45Z to 2026-11-27T18:05:52Z (427 seconds).

## Authorization and scope
The owner explicitly authorized T0009 on the local RTX 4090 within the reviewed specification and limits. Scope was the native first-order FP8 reference integration for selected actual SPLOT projections, not FP8 pretraining or a final optimized backend. Allowed files were new `src/splot/fp8.py`, new `tools/test_fp8_autograd.py`, new `research/implementation/fp8-autograd.md`, two new FP8 configs and minimal `tools/experiment.py` policy/instrumentation, plus REP evidence/index. No model-equation, optimizer, T0007 probe, existing-config, Mac/data, comparator or CI change was made. No packages were installed and CUDA/driver were unchanged.

## Outcome
**Passed.** Selected actual SPLOT linear projections use a native FP8 `torch._scaled_mm` first-order autograd path with E4M3 forward and E5M2 backward operands, device-resident FP32 current scales, FP32 master parameters/AdamW state and BF16 residuals/other autocast operations. Default FP32/BF16 paths are unchanged. The FP8 smoke model resumes exactly and cross-precision resume is rejected.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `summary.md` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_fp8 | passed | four run JSONs, `comparison.json`, `depth-report.json`, `linear-report.json`, `checkpoint-state.json`, `cuda-tests.txt`, `metrics.jsonl` |
| handover | passed | `summary.md`, `checkpoints.txt`, `execution.log` |

## Implementation
- `FP8Linear` is an `nn.Linear`-compatible adapter preserving the original 156 selected FP32 `Parameter` objects, module names, state_dict keys and recurrent shared references. The semantic optimizer partition is unchanged.
- Selected: GQA/GDN `q/k/v/o`, GDN `z`, dense Block and routed/shared Expert `gate/up/down`, and MTP `offset_projection`. Excluded: GQA output gate, GDN `a`/`b`, router, tied embedding/readout, norms, GDN convolution/scalars, recurrence/softmax/loss. 156 selected, 11 excluded; no duplicate embedding parameter.
- Forward uses E4M3FN quantized input/weight, native row-major × column-major `torch._scaled_mm`, BF16 output. Backward uses E5M2 dY with E4M3 saved operands for dgrad/wgrad; dW output is FP32, dX returns the original input dtype. Once-differentiable only.
- M is flattened and zero-padded to 16, outputs/gradients trimmed; K/N must already be multiples of 16. Zero-M returns empty outputs/gradients without dispatch. Unsupported device/dtype/shape fails explicitly.
- Current scales are device-resident FP32 tensors (`absmax/format.max`, all-zero 1, positive normal clamp); no `.item()`, `.cpu()` or host scalar reconstruction per projection. No persistent scale state.
- FP8 uses AdamW only; FP8+Muon, persistent/delayed scales and optimized kernels remain separate. FP32/BF16 configs and paths are unchanged; FP8 configs differ from BF16 only by `training.precision = "fp8"`.

## CPU regression
`tools/ci_cpu.py` passed locally from clean tested SHA: **19 CPU scripts, 149 tests, 0 failures, 0 skips**, plus 10 REP reports. The new CPU suite covers selection/exclusion, parameter/state_dict/recurrent identity, optimizer-partition invariance, padding/layout arithmetic, CPU FP8 rejection and FP32/BF16 no-FP8 defaults.

## Native autograd results
`tools/test_fp8_autograd.py --device cuda` passed 4/4 focused tests. Four shapes (M=1/17/64, K=384/N=512 and M=64/K=32/N=64) produced finite BF16 outputs, FP32 weight gradients and original-dtype input gradients. Relative Frobenius errors versus FP32 were **0.034907–0.059671** (gate 0.15); versus FP32 products of dequantized operands **0.001654–0.023033** (gate 0.025). All-zero outputs/gradients were exact, zero-M did not dispatch, repeated same-weight gradients summed correctly, and the profiler recorded `aten::_scaled_mm` plus SM89 FP8 kernels with no DtoH/HtoD copy events.

## SPLOT depth drift (BF16 vs FP8)
| Model | R | Max CE abs diff (final/early/offset2) | Composite diff | Hidden L2 | Logit L2 | Grad relative L2 | Routing exact/top1 |
|---|---|---|---|---|---|---|---|
| smoke | 1 | 0.000749 | 0.000771 | 0.0407 | 0.0435 | 0.0845 | 0.9510 / 0.9804 |
| smoke | 2 | 0.001274 | 0.001521 | 0.0606 | 0.0605 | 0.1097 | 0.9265 / 0.9657 |
| smoke | 4 | 0.001502 | 0.001228 | 0.1067 | 0.1071 | 0.1896 | 0.8934 / 0.9424 |
| nano | 1 | 0.012225 | 0.011299 | 0.1341 | 0.1253 | 0.1746 | 0.9265 / 0.9706 |
| nano | 2 | 0.012225 | 0.011023 | 0.1751 | 0.1680 | 0.2222 | 0.9093 / 0.9608 |
| nano | 4 | 0.007709 | 0.001211 | 0.2375 | 0.2343 | 0.3423 | 0.8701 / 0.9534 |

All CE gates passed (component <=0.5 nats, composite <=0.75), all outputs finite, labels identical (50 final/49 offset2). Drift and routing are descriptive only; no stability or quality claim. FP8 dispatch coverage was 100% on smoke and 92.3% on Nano (unused experts not dispatched; no all-expert claim). Scales were CUDA-resident.

## CLI validation
- Both Nano runs: 43,055,076 parameters, 51 valid tokens, 50 NTP labels, 49 offset-2 labels and 376 validation labels. Pre-update CE differences BF16 vs FP8: final `0.007979`, early `0.012225`, offset-2 `0.0001497`, composite `0.011023` nats. Validation is descriptive: BF16 mean NLL `10.273543` / PPL `28956.3`; FP8 mean NLL `10.363955` / PPL `31696.3`.
- FP8 Nano diagnostics: 156 selected, 11 excluded, 1420 forward calls / 241 backward calls over the one-step run, forward/backward coverage 0.9615/0.8846, 10,976 padded forward rows / 2,558 padded backward rows, scales on CUDA, AdamW.
- FP8 smoke 4-step continuous and 2+2 resume both exited 0 with identical validation (`1949.7781066894531` NLL sum, mean `5.185580070982588`, PPL `178.67706511552583`); comparator reports exact model, optimizer, RNG, non-timing history and validation equality.
- Cross-precision resume into the FP8 run was rejected with `ValueError: run identity mismatch` before model/optimizer load (expected, recorded).
- Checkpoints are local-only: BF16 Nano `cf798982d93544118193cef23573840edfe368fac87fd6dca5f55462531e1382` (493,312,727 B); FP8 Nano `beab8bfe561f7143b3c7caa45a8255809e0807e63ab51b2759e7bf4aa65c68aa` (488,608,819 B); FP8 continuous `2983c7496055c7bc2e4dee31815852b87c1f838ab9bafb09cb35658d9e91fe38` (2,432,907 B); FP8 resumed-final `656e2d289663b6b3bc21f6bdc8025bd6814fa971d376ad399222753edbc3ad6a` (2,433,163 B).
- Both checkpoints preserve FP32 model weights and FP32 Adam moments; model keys are identical. FP8 has 180 optimizer states / 360 moments versus BF16 183/366 (unused experts); all finite, step 1.0.
- FP8 step time `0.509 s` vs BF16 `0.406 s`; peaks FP8 `874,552,832/1,149,239,296 B` vs BF16 `881,575,424/1,138,753,536 B`. Descriptive only; no acceleration claim.

## Metrics and resources
`metrics.jsonl` holds CPU suite results, run steps/validation, linear autograd errors, R1/R2/4 drift/routing, checkpoint state, profiler and resource observations with units. Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8, RTX 4090 driver 580.178.04; free VRAM 24,077 MiB before/after; disk 182,195,892 KiB before / 180,749,544 KiB after; no inference process stopped.

## First failure
No unexpected failure in this attempt and no retry. The intentional cross-precision resume was rejected as required.

## Limitations
- Correctness integration only; no convergence, pretraining, long-context, throughput or language-quality claim.
- Current per-tensor scaling only; no delayed/persistent scaling, MXFP8/NVFP4, FP8 optimizer/state or FP8 output.
- Some unused experts are not dispatched; coverage is recorded, not an all-expert claim.
- Drift/routing are forward/backward comparisons, not trained-model stability.
- No FP8+Muon, fused/grouped expert kernels, real corpus or production benchmark.
- FP8 output is BF16; no Tensor Core instruction claim beyond recorded kernel names.

## Proposed next action
Maintainer review of this evidence PR. T0009 (#56) remains open until review/acceptance; no merge or closure is performed here. Hosted CPU CI is required on the PR. FP8+Muon, long-context, real-corpus and kernel-performance work each require separate reviewed tasks.
