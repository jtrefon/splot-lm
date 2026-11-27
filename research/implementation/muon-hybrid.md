# Hybrid Muon/AdamW optimizer

Status: T0006 implementation, correctness scope only. Task [T0006](../tasks/T0006-muon-hybrid.md), issue [#48](https://github.com/jtrefon/splot-lm/issues/48). This is not an optimizer-superiority, convergence or production-training result.

## Policy

The default optimizer path is unchanged: if `training.optimizer` is absent, the harness constructs the same `torch.optim.AdamW(model.parameters(), lr=training.lr, weight_decay=training.weight_decay)` used before T0006, and preserves the plain AdamW checkpoint format.

The hybrid path is explicit:

```json
"optimizer": {
  "name": "muon",
  "muon": {
    "lr": 0.003,
    "weight_decay": 0.0,
    "momentum": 0.95,
    "nesterov": true,
    "ns_coefficients": [3.4445, -4.775, 2.0315],
    "ns_steps": 5,
    "eps": 1e-7,
    "adjust_lr_fn": "match_rms_adamw"
  },
  "adamw": {
    "lr": 0.003,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1e-8
  }
}
```

Unknown selectors, unknown settings, malformed values and unavailable native Muon fail closed. There is no custom Newton-Schulz implementation, external dependency, scheduler or learning-rate search.

## Semantic partition

`tools/optimizers.py` partitions by actual owning module and parameter role. It deduplicates parameter identity, uses stable canonical names, sorts groups deterministically and asserts exhaustive/disjoint coverage against the model inventory. Unknown parameters fail closed; there is no ndim-only fallback.

| Group | Smoke parameters / numel | Nano parameters / numel | Roles |
|---|---|---|---|
| Muon (native, 2-D hidden matrices) | 157 / 167,936 | 157 / 30,425,088 | GQA/GDN `q/k/v/o`, GDN `z`, GQA output gate, dense SwiGLU, routed/shared expert FFNs, offset projection |
| AdamW | 41 / 12,016 | 41 / 12,629,988 | Tied embedding/output weight, RMSNorm, GDN `a`/`b` scalar gates, `conv`, `log_A`, `dt_bias`, MoE router, depth embedding, any bias/other explicitly non-Muon state |
| Total | 198 / 179,952 | 198 / 43,055,076 | exact model inventory |

The native `torch.optim.Muon` 2.10 momentum buffer is FP32. Its Newton-Schulz intermediate is BF16 (`torch/optim/_muon.py` casts `grad` to `bfloat16()` for the NS iterations); that native implementation is recorded but not modified. The adapter keeps model parameters, Muon momentum and AdamW moments in FP32.

## Adapter and checkpoints

`MuonAdamW` owns a native `torch.optim.Muon` over the Muon group and a native `torch.optim.AdamW` over the AdamW group. `zero_grad` clears both; `step` runs Muon first and AdamW second after the harness's single finite-gradient check. A failure aborts before the checkpoint write.

The adapter state is:

```text
adapter_version, partition metadata (names/shapes/order/plan digest),
settings, muon state_dict, adamw state_dict
```

`load_state_dict` requires the same adapter version, exact partition metadata and settings, then loads both nested states. Cross-mode or repartitioned checkpoints are rejected. Default AdamW checkpoints remain plain `torch.optim.AdamW` state dicts.

## Configs and files

- `configs/experiments/splot-smoke-synthetic-muon.json`
- `configs/experiments/splot-nano-synthetic-muon.json`

Both are identical to their FP32 counterparts except for the explicit `training.optimizer` block. `tools/experiment.py` builds the optimizer through `tools/optimizers.py`, records the resolved policy/partition and optimizer state elements/bytes in `run.json`, and keeps precision/BF16 policy independent. No `src/`, existing-config, CI, Mac, fixture/data, comparator or checkpoint-cadence file changed.

## Tests and validation

`tools/test_hybrid_optimizer.py` covers the native Muon API on rectangular matrices, complete smoke/Nano partition inventory, tied weights once, router/gates excluded from Muon, unused-parameter preservation, FP32 state, mode/partition mismatch rejection, exact 4-vs-2+2 resume with both optimizer states/RNG/history, and unchanged default AdamW selection for FP32/BF16 configs. It supports `--device cuda` for the actual native Muon/adapter checks.

Bounded CUDA validation compares the existing FP32 AdamW Nano step with the hybrid Nano step from the same seed/initial weights/order. The first-step pre-update losses must be exactly equal because the optimizer has not touched the parameters yet; post-step validation is descriptive only. A 4-step hybrid smoke and 2+2 resume are compared with `tools/rep/compare_resume.py`.

## Limitations

- Correctness-only, one step and short synthetic smoke; no convergence, tuning, scale, quality or optimizer-superiority claim.
- FP32 hybrid only; BF16-hybrid empirical validation, scheduler/tuning and full-epoch comparisons require separate reviewed tasks.
- Native Muon NS arithmetic is BF16 internally; this is recorded, not changed.
- No CUDA 13 runtime, external kernels, FP8, quantization, real corpus or Mini.
- Checkpoints remain local-only; no tensors are committed.
