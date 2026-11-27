# Native FP8 autograd integration

Status: T0009 implementation, correctness scope only. Task [T0009](../tasks/T0009-fp8-autograd.md), issue [#56](https://github.com/jtrefon/splot-lm/issues/56). This is not FP8 pretraining, a final optimized backend or a quality/performance result.

## Policy

`training.precision = "fp8"` is explicit and CUDA-only. The default `fp32` and `bf16` paths are unchanged and do not construct any FP8 modules.

Selected bias-free projection roles are converted to `FP8Linear` after model initialization:

- GQA `q`, `k`, `v`, `o`;
- GDN `q`, `k`, `v`, `z`, `o`;
- dense Block and routed/shared Expert `gate`, `up`, `down`;
- MTP `offset_projection`.

Excluded from FP8: GQA `output_gate`, GDN `a`/`b` scalar gates, MoE router, tied embedding/readout, all norms, GDN convolution/`log_A`/`dt_bias`, recurrence, softmax, losses and other sensitive reductions. These remain FP32 or BF16 autocast as appropriate.

The adapter is an `nn.Linear` subclass that keeps the original FP32 `Parameter` object, module name and `state_dict` key. It does not reinitialize weights or create duplicate parameters. The Muon/AdamW semantic partition therefore retains the same names, shapes and plan hash; this experiment uses AdamW only.

## Autograd

`FP8Linear` implements a first-order `torch.autograd.Function`:

- forward: per-tensor current FP32 scales, E4M3FN quantized input and weight, native `torch._scaled_mm` with BF16 output, row-major LHS and column-major RHS;
- backward: E5M2 quantized gradient, native dgrad with E4M3 weight and native wgrad with E4M3 saved input, FP32 weight gradient, input gradient restored to the original input dtype;
- once-differentiable only; no higher-order derivative claim;
- leading dimensions flattened to M, zero-padded to a multiple of 16, outputs/gradients trimmed; zero-M returns empty outputs/gradients without dispatch;
- K/N must already be multiples of 16 or the call fails explicitly; no implicit K/N padding;
- no `GradScaler`, no FP8 master weights, no FP8 optimizer state.

Scales are device-resident FP32 tensors computed as `absmax / dtype.max`, with all-zero scale 1 and a positive FP32 normal clamp. The hot path does not use `.item()`, `.cpu()` or Python scalar reconstruction. Per-module counters record calls, padding rows, empty calls and last-scale devices without synchronizing per projection.

## Audit and finite guards

Quantization records device-resident saturation, underflow, nonzero and nonfinite counters for input, weight and gradient roles. Outputs, dX and dW also set a device-side nonfinite flag. These counters are reset before each depth report and reduced only at a report or step boundary; there is no per-projection host synchronization. `fp8_diagnostics` reports native forward/backward calls, wgrad FP32 calls, padding rows, per-role quantization counts and underflow fractions.

`fp8_assert_finite` is called at the training step boundary before optimizer update and after backward, and at evaluation. It checks FP32 master parameters, FP8 quantization/output flags and gradients, raising `FloatingPointError` before an update or evaluation result if any nonfinite value is observed. Nonfinite input/weight/gradient cases are covered by tests.

Forward and dgrad outputs are BF16; the explicit wgrad output is FP32. The dequantized-oracle comparison uses the actual effective upstream BF16 gradient received by the autograd Function, not the pre-autograd FP32 tensor.

## Integration and checkpointing

- `tools/experiment.py` resolves `fp8`, records the selected/excluded module inventory, FP8 runtime probe and dispatch/diagnostic counters, and keeps BF16 autocast around the residual/non-FP8 operations.
- FP8 uses AdamW only. FP8+Muon, persistent/delayed scales and optimized grouped dispatch are separate tasks.
- Existing FP32/BF16 configs, model equations, optimizer implementation, comparator and checkpoint cadence are unchanged.
- New configs are identical to their BF16 counterparts except `training.precision = "fp8"`.
- Parameter identity, state_dict names and recurrent shared references are preserved. Same-mode resume is expected to be exact; cross-precision resume is rejected by the run identity.

## Tests and validation

`tools/test_fp8_autograd.py`:

- default CPU: selection/exclusion, parameter identity, state_dict names, recurrent identity, optimizer partition invariance, padding/layout math, CPU FP8 rejection, FP32/BF16 no-FP8 defaults;
- `--device cuda`: actual forward/dX/dW for M=1/17/64 K=384 N=512 and M=64 K=32 N=64; FP32 and dequantized-operand error gates; exact zero; repeated same-weight gradient sum; actual E4M3/E5M2 kernel/copy profile and scale residency; zero-M behavior;
- `--device cuda --report PATH`: BF16-vs-FP8 R1/R2/R4 smoke/Nano CE differences, hidden/logit/gradient finite checks, exact labels, per-depth native/padding/quantization audit and routing key/shape agreement. Matched-only and full-vector gradient drift are both reported; missing gradients count as zero in the full-vector metric. The report exits nonzero if its status fails.

Requirements: forward/dX/dW relative Frobenius <= 0.15 versus FP32, <= 0.025 versus FP32 products of dequantized operands; exact zero; each SPLOT CE absolute difference <= 0.5 nats and composite <= 0.75; all finite; exact label counts and topology. Drift/routing are descriptive only.

## Limitations

- Correctness integration only; no convergence, pretraining, long-context, throughput or language-quality claim.
- Current per-tensor scaling only; no delayed scaling, MXFP8/NVFP4, FP8 optimizer/state or persistent scale policy.
- FP8 output is BF16; no Tensor Core instruction claim beyond the recorded native kernel names.
- Some unused experts are not dispatched; coverage is recorded, not an all-expert claim.
- FP8+Muon, optimized kernel/layout reuse, real corpus and production benchmarks require separate reviewed tasks.
