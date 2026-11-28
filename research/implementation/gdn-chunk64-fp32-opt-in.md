# Opt-in FP32 chunk64 GDN training path

2026-11-28 | T0033 / issue #121 | Experimental; scan remains default

## Scope

`recurrence.gdn_algorithm` is an optional validated `RecurrentConfig` field with
values `scan` (default) and `chunk64`. It changes only the GDN contraction order:

- `scan`: existing recurrent `delta_scan` reference, unchanged default.
- `chunk64`: `delta_chunk64`, composing the existing `delta_chunk` oracle over
  consecutive slices of at most 64 tokens with carried FP32 state and no detach.
- The explicit runtime oracle `GatedDeltaNet.run(..., algorithm='chunk')` remains
  unchanged.

The mode is **FP32-only** and experimental. `tools/experiment.py` rejects
`chunk64` with BF16 or FP8, and `GatedDeltaNet.run` fails closed before mutation
if autocast is active, model activations/weights are not FP32, or carried GDN
state/history is not FP32. `delta_chunk64` itself accepts only FP32/FP64 so the
FP64 tiny oracle remains available in tests.

`delta_chunk64` adds no parameters, buffers, optimizer state or checkpoint
entries. Configs that omit the option keep their legacy config digest, report
identity and checkpoint schema. Explicit configs record `gdn_algorithm` in the
config digest and checkpoint identity; `RecurrentCache` also carries the
algorithm and rejects cross-mode cache reuse. Existing `delta_chunk` remains a
small finite-log-decay dense triangular oracle.

## Status and limitations

T0025's BF16 chunk64 full-model candidate failed its preregistered gates and is
not repaired or adopted here. T0031 rejected BF16 islands and accepted full FP32
forward/backward feasibility. T0033 qualifies only the opt-in FP32 training path
on synthetic data, including complete Muon/AdamW updates. No real-data training,
BF16/FP8 chunk64, production kernel, default adoption, learning-quality or
sustained-throughput claim is made. Later matched real-data learning requires a
separate reviewed task.
