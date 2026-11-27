# BF16 harness with FP32 numerical safeguards

Status: T0005 implementation, reviewed smoke scope only. Task [T0005](../tasks/T0005-bf16-harness.md), issue [#40](https://github.com/jtrefon/splot-lm/issues/40). This is not a production-training or performance result.

## Policy

`training.precision` is explicit and opt-in:

- `fp32` keeps the existing path unchanged: no autocast context, FP32 model parameters, FP32 AdamW state and FP32 sensitive arithmetic.
- `bf16` keeps the model and optimizer in FP32 and enables `torch.autocast(device_type=..., dtype=torch.bfloat16)` only around the training objective's forward pass and the evaluation forward pass. Backward and `optimizer.step()` run outside autocast. No `GradScaler` is used.
- Unsupported values and unavailable BF16 backends fail before training; there is no silent fallback. CUDA records `torch.cuda.is_bf16_supported()`. A small eligible projection probe must return BF16; CPU BF16 is accepted only when that probe and the focused model tests actually pass.
- Deterministic algorithms remain enabled and TF32 remains disabled. Requested/actual policy, probe dtype, master dtype and BF16 support are recorded in the run identity/environment; `training.precision` is part of the checkpoint identity, so cross-precision resume is rejected.

## FP32-sensitive boundaries

The existing reference already used explicit FP32 accumulation in most sensitive paths. T0005 added one narrow source fix:

- `src/splot/dense.py` GQA: score contraction, causal mask/softmax and value contraction now run inside `torch.autocast(..., enabled=False)`. GQA projections remain eligible BF16 under the outer autocast; the FP32 path is unchanged.

Verified as FP32 under BF16 autocast:

- FP32 master parameters, gradients and AdamW `exp_avg`/`exp_avg_sq`.
- GDN `delta_scan` matrix recurrence and its contractions (autocast disabled), with FP32 matrix state and BF16 raw history.
- Router linear/softmax, deterministic top-k and FP32 expert accumulation.
- Cross-entropy, RMSNorm reductions, RoPE angle arithmetic and GQA score/softmax/value.
- Composite A2 objective arithmetic and validation NLL.

Eligible projections and other broad activations may be BF16; not every activation is guaranteed BF16. The value contraction was made FP32 as part of the GQA safeguard rather than left BF16.

## Harness and configs

- `tools/experiment.py`: `resolve_precision`, BF16-aware `train`/`evaluate`, identity/environment policy fields, precision-specific limitations and report policy.
- New configs are identical to their FP32 counterparts except `training.precision`:
  - `configs/experiments/splot-smoke-synthetic-bf16.json`
  - `configs/experiments/splot-nano-synthetic-bf16.json`
- Existing FP32 configs, topology, fixture/data, numerical tolerances and checkpoint cadence are unchanged.

## Focused tests

`tools/test_bf16_harness.py` uses the actual `SPLOTLM` model, not only a toy linear layer. It checks:

- actual public projection output dtypes via hooks, GDN matrix state, router probabilities, GQA score/value einsum inputs and softmax dtypes;
- FP32 RMSNorm reduction, RoPE angles and cross-entropy against FP32 references;
- FP32 master parameters/Adam moments, finite losses/gradients, FP32 composite objective arithmetic and BF16 evaluation;
- exact same-mode BF16 checkpoint resume on deterministic CPU and cross-precision resume rejection;
- same initial weights/batch selection and precision resolution/fallback rejection.

Run it with the other CPU regressions:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:tools python tools/test_bf16_harness.py
```

## Bounded CUDA validation

The reviewed validation compares one FP32 and one BF16 Nano step with the same seed/inputs, then checks BF16 smoke continuous/resumed equivalence. Predeclared provisional first-step gates:

- absolute final-NTP CE difference <= 0.10 nats;
- absolute early-NTP CE difference <= 0.10 nats;
- absolute offset-2 CE difference <= 0.10 nats;
- absolute composite objective difference <= 0.20.

Balance and route agreement are descriptive only; top-k discontinuities may occur. These are smoke tolerances, not convergence or precision-superiority criteria. The FP32 run must also match the accepted T0002 non-timing first-step history.

## Limitations

- One-step paired Nano and short BF16 smoke only; no convergence, long-training, throughput, energy or production claim.
- BF16 autocast is not a claim that all activations or kernels are BF16, nor that BF16 is faster on this stack.
- CPU BF16 is validated only by the focused deterministic tests when the backend supports it; CPU BF16 is not a CUDA substitute.
- The synthetic byte fixture is not a production tokenizer or corpus.
- Checkpoints remain local-only; no tensors are committed.
- Matched FP32/BF16 full-epoch validation, checkpoint-cadence measurement, activation checkpointing, FP8 and low-bit work require separate reviewed tasks.
