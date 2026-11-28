# Opt-in chunk64 GDN reference path

2026-11-28 | T0025 / issue #105 | Experimental; scan remains default

## Scope

`training.recurrence.gdn_algorithm` is an optional validated `RecurrentConfig`
field with values `scan` (default) and `chunk64`. It changes only the GDN
contraction order in `GatedDeltaNet`:

- `scan`: existing recurrent `delta_scan` reference (unchanged default).
- `chunk64`: `delta_chunk64`, which composes the existing `delta_chunk`
  triangular oracle over consecutive slices of at most 64 tokens and carries
  FP32 state without detaching at boundaries.
- The explicit runtime oracle `GatedDeltaNet.run(..., algorithm='chunk')`
  remains unchanged.

`delta_chunk64` adds no parameters, buffers, optimizer state or checkpoint
entries. Existing configs omit the new field and therefore keep their legacy
config digest, forward behavior, state_dict inventory and checkpoint schema.
The mode is recorded through the ordinary config digest when explicitly set.
FP8 combined with `chunk64` is rejected by the experiment harness pending a
separate qualification.

## Status and limitations

This is an experimental reference path, not a production kernel, default or
recommended mode. The T0025 full-model BF16 candidate failed the preregistered
hidden/logit and routing-equivalence gates at T512; no T2048 or resource
measurement was run. `delta_chunk` remains a small finite-log-decay dense
triangular oracle. No learning-quality, full-model speed, sustained-throughput
or architecture-superiority claim is made.
