# FP8 + Muon compatibility

Status: T0011 implementation, correctness scope only. Task [T0011](../tasks/T0011-fp8-muon.md), issue [#62](https://github.com/jtrefon/splot-lm/issues/62). This is not a convergence, throughput, optimizer-superiority or language-quality result.

## Scope

T0011 enables the combined opt-in path for native FP8 selected projections and the existing native Muon/AdamW hybrid. It does not change model equations, topology, FP8 formats, existing configs, optimizer math, numerical tolerances, the comparator or CI.

The existing FP8 adapter is unchanged: selected bias-free SPLOT projections keep their original FP32 master `Parameter` objects, names and `state_dict` keys. The semantic Muon/AdamW partition is built from the converted model, so the FP8 conversion does not change the partition, parameter identity or tied-embedding treatment. FP8 remains E4M3 forward, E5M2 backward, BF16 forward output, input-dtype dgrad and FP32 wgrad. Muon receives hidden 2-D matrices; AdamW receives the tied embedding, norms, GDN scalars/convolution, router and other explicit non-Muon state.

Policy metadata now records `optimizer_scope = "adamw-or-muon-hybrid"`, `policy_version`, `audit_version`, the actual selected optimizer and an accurate `adamw_only` flag. The flag is false for the hybrid path rather than carrying a stale AdamW-only assertion.

## Optimizer configuration

`build_optimizer` retains the default AdamW path when the `optimizer` key is absent. It rejects:

- an explicit null selector;
- unknown selectors, names or keys;
- malformed settings and non-finite hyperparameters;
- a Muon block supplied with an AdamW selector;
- explicit null Muon or AdamW sub-blocks.

The hybrid path uses the same hyperparameters as the existing FP32 Muon configs. Both Muon momentum buffers and AdamW moments/steps remain FP32; the native installed Muon Newton-Schulz intermediate remains BF16 (`torch.optim.Muon` behavior, recorded as `ns_intermediate_dtype = "bfloat16"`). No optimizer math was added.

## Per-step audit and finite guards

For every FP8 training step:

1. the per-module audit is reset before the forward graph is built;
2. all loss fields, FP8 input/weight/gradient quantization flags and outputs are checked before backward;
3. all parameter gradients and FP8 audit flags are checked after backward;
4. the deterministic per-step audit (native calls, role-specific nonzero/underflow/saturation counts, nonfinite flags, padding and scale residency) is captured without timing;
5. the optimizer steps;
6. all master weights and all present optimizer-state tensors are checked for finiteness **before** the checkpoint is written.

A failure raises `FloatingPointError`, leaves the previous checkpoint untouched and does not append the failed step to history. This is a fail-closed boundary check, not a claim that the two native optimizer updates are atomic.

The guard applies to every FP8 mode and to the hybrid BF16 path. Default FP32 and BF16 AdamW behavior is unchanged. Audit counters are reduced only at step/report boundaries; no per-projection host synchronization was added.

Evaluation resets the FP8 audit, performs its own forward-only aggregate audit and resets it again, so validation audit is reported separately from training totals. Checkpoint top-level schema remains `identity`, `model`, `optimizer`, `rng`, `history`; history rows for FP8 additionally contain a deterministic `audit` object. Same-mode resume compares the full non-timing history, including audit. Cross-precision and cross-optimizer identity/state mismatches are rejected.

## Configurations

New configs are derived from their existing precision counterparts with only the exact Muon optimizer block from the corresponding FP32 Muon config:

- `splot-smoke-synthetic-fp8-muon.json`;
- `splot-nano-synthetic-fp8-muon.json`;
- `splot-nano-synthetic-bf16-muon.json`.

The FP8 AdamW control remains `splot-nano-synthetic-fp8.json`. Existing config bytes are unchanged and checked by SHA-256 in the focused CPU suite.

## Tests

`tools/test_fp8_muon.py` adds CPU coverage for policy/partition identity, malformed selectors, unchanged existing config hashes, derived configs, audit reset/serialization/aggregation, adapter state round-trip, cross-mode rejection, post-update finite guard and prior-checkpoint preservation. CUDA coverage adds native FP8+Muon execution, both FP32 optimizer states, inactive-expert handling, per-step audit/native dispatch, nonfinite post-update injection and exact same-mode resume including audit. Hosted CPU CI does not require a GPU; CUDA tests run with `--device cuda`.

The full execution evidence, matched synthetic epoch results and checkpoint hashes are published under `experiments/results/T0011/attempt-001/`.

## Limits

- Correctness and one-epoch descriptive comparison only; no convergence, scaling, language-quality or optimizer-superiority claim.
- Three runs are not a full factorial design; BF16 AdamW is not included, so no interaction effect or statistical inference.
- Existing per-step audits and repeated checkpoint I/O confound wall time; no speed claim.
- Current per-tensor FP8 scaling only; no delayed/persistent scaling, MXFP8/NVFP4, FP8 optimizer state or FP8 output.
- Some unused experts are not dispatched; coverage is recorded, not an all-expert claim.
- Checkpoints remain local-only; no tensors are committed.
