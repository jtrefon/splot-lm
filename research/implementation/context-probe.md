# Contiguous-context resource probe

Status: T0012 implementation, resource-qualification scope only. Task [T0012](../tasks/T0012-contiguous-context.md), issue [#66](https://github.com/jtrefon/splot-lm/issues/66). This is not pretraining, a long-context quality result, a production throughput benchmark or an optimizer/precision superiority claim.

## Scope

`tools/context_probe.py` is an isolated probe. It does not modify the model, training harness, optimizer, existing configs, CI, data fixture or prior evidence. It reuses the merged T0011 Nano Muon configs, the existing R2 A2 objective, native FP8 projection adapter and Muon/AdamW helpers.

Each process measures exactly one precision/context case. BF16+Muon and FP8+Muon cases are launched sequentially as fresh processes; no state is reused across cases.

## Contiguous input

For a requested length `T`:

- one row with exactly `T` valid tokens;
- `input_ids[0] = BOS(1)` and `input_ids[T-1] = EOS(2)`;
- `T-2` interior content IDs drawn independently and uniformly from `8..32767` with a fixed seed;
- all document IDs `0`, positions `0..T-1`, and all mask entries valid;
- offset-1 labels have exactly `T-1` non-ignore positions and offset-2 labels exactly `T-2`; labels never cross the document end.

The CPU-created input tensor digest, shape and counts are recorded. The probe does not repeat short documents or pad a short fixture to imitate a long context.

## Warm-up and measured update

The process first creates a fresh model/optimizer/input, performs one full untimed warm-up update, releases it and clears the CUDA allocator cache. It then recreates the same CPU-seeded model and input, verifies the initialization and input digests and partition plan match the warm-up, and performs one measured full update.

A separate finite no-grad forward checks logits, hidden states and offset-2 outputs before measurement; the FP8 audit is reset afterwards. The measured step then performs:

1. input transfer;
2. forward including the existing weighted A2 loss;
3. backward;
4. loss/parameter/gradient/FP8 audit checks before the optimizer step;
5. one native Muon/AdamW update;
6. post-update master-weight and optimizer-state finite checks.

No checkpoint is written inside the measured step. The probe does not hold the warm-up and measured models simultaneously.

## Timing and memory

Every phase is synchronized with `torch.cuda.synchronize` and timed with `time.perf_counter`. Reported fields include:

- transfer, prepare/reset, forward, backward, finite/audit checks, optimizer step, post-update checks and end-to-end;
- synchronization seconds per phase and in total;
- the finite/audit portion, including the separately measured audit reduction time;
- tokens divided by end-to-end seconds;
- allocator current/peak allocated and reserved bytes, plus free VRAM before the case, before the measured update, after the update and after cleanup;
- model/optimizer state bytes and dtypes.

Synchronization and checkpoint-free overheads are reported, not subtracted or extrapolated. The single measured update is not a steady-state throughput claim.

## Gates

The probe fails closed if:

- the model is not exactly 43,055,076 parameters;
- labels are not exactly `T/T-1/T-2`;
- warm-up and measured initialization/input/partition identities differ;
- any loss, logit, hidden, offset output, gradient, master weight or present optimizer-state tensor is nonfinite;
- FP8 native forward/backward calls are not positive or the audit reports nonfinite data;
- a phase cannot be reported consistently.

BF16/FP8 first pre-update CE differences are compared by the execution driver against the preregistered `.5` nats per component and `.75` nats total gates. Those are descriptive smoke thresholds, not trained equivalence.

## CLI

```bash
PYTHONPATH=src:tools CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/context_probe.py --device cuda --precision bf16|fp8 \
  --length 128|512|2048|4096 --seed 42 --out "$RUNS/<precision>-<length>.json"
```

Outputs must be outside the repository checkout. The probe always runs the warm-up and always keeps the FP8 audit enabled.

## Limits

- Resource qualification only; no convergence, quality, long-context capability, throughput or superiority claim.
- One measured update per precision/context, not a training curve or a production benchmark.
- Existing FP8 per-tensor scaling and Muon implementation are reused unchanged.
- No real corpus, Mini model, new precision/kernel, CUDA upgrade, package installation or background service.
- The T4096 case is conditional on the T2048 resource and time gates in the reviewed task.
