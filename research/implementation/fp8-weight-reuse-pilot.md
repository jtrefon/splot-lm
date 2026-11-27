# FP8 weight reuse and pilot ingestion

Status: T0017 implementation/qualification. Task [T0017](../tasks/T0017-fp8-weight-reuse-pilot.md), issue [#77](https://github.com/jtrefon/splot-lm/issues/77). The real-data pilot is **not run** because T0016 (#76) has no accepted dataset or evidence; this document covers the independent FP8 reuse and pilot-ingestion qualification only.

## Opt-in FP8 weight reuse

`training.fp8_weight_reuse` is an explicit boolean, default `false`. The existing no-reuse path is unchanged. When enabled, each `FP8Linear` may retain the E4M3 quantized weight, its device FP32 scale and quantization audit stats for one forward/backward/update scope.

The cache key is the physical parameter identity plus `Parameter._version`, device, dtype and shape. A changed version, parameter, device, dtype or shape misses and re-quantizes. Activations and gradients are never cached. The cache is not part of `state_dict`, checkpoints or persistent process state.

Lifecycle:

- `reset_fp8_audit` clears cache and counters before each training step;
- `train` clears the cache after backward/audit capture and before the optimizer update;
- an exception during forward/backward/audit clears the cache before propagating;
- checkpoint restore clears the cache;
- evaluation clears it before and after the evaluation scope and reports `cleared-before-and-after-evaluation`;
- no per-projection host synchronization or stale-version reuse is added.

Audit distinguishes actual quantizations from logical per-use operand stats. Cached hits replay the same logical saturation/underflow/nonzero/nonfinite stats, while `weight_quantizations`, `weight_cache_hits`, `weight_reuse_ratio` and `weight_cache_bytes` record the actual cache behavior. This preserves finite detection and prevents hidden reduction of logical audit coverage.

## Pilot ingestion

The new explicit selector is `pilot-blocks-v1` (not synthetic-v2). It requires:

```bash
--dataset-root /path/outside/checkout \
--manifest-sha256 <64 lowercase hex> \
[--source-snapshot-sha256 <64 lowercase hex>]
```

`tools/pilot_ingestion.py` independently verifies the bounded T0016 contract:

- fixed `manifest.json`, `tokenizer.json`, `train.jsonl`, `validation.jsonl`, optional `test.jsonl` paths; regular non-symlink files only; no extra files;
- canonical JSON, manifest SHA-256, tokenizer SHA-256 without importing a tokenizer backend, source snapshot SHA-256 and family/split disjointness;
- <=64 MiB aggregate, <=160 blocks, fixed 512 length, actual vocabulary 264..16384 against model vocabulary 32768, train 32..128 blocks and validation 2..16 blocks;
- shard path/hash/size/count metadata, concatenated logical SHA-256 and independent block validation: exact keys/arrays, token ranges, BOS/EOS, positions, valid prefix/padding, globally unique document IDs, no cross-document labels, exact offset-1/2 labels;
- train and validation are returned; test is integrity-validated only and never returned for training/evaluation.

The run identity records manifest, tokenizer, source snapshot and logical hashes plus data/config/packed hashes; no dataset path is recorded. The pilot model config uses Nano R2 A2, B1, T512, seed 42, FP8+Muon, checkpoint interval 16 and reuse enabled. Unused model-vocabulary IDs are reported explicitly.

## Tests and limits

`tools/test_pilot_ingestion.py` covers schema/counts, exact labels, split/global document checks, source/model mismatch, canonical/extra-file/tamper rejection and shard hash mismatch using small original synthetic artifacts only. `tools/test_fp8_weight_reuse.py` covers cache lifecycle/invalidation, exact Nano cache-off/on first-step outputs/state, cache-free resume, cross-mode rejection and a standalone T512 paired measurement.

This is not pretraining or a quality/convergence/throughput claim. The real pilot remains blocked on accepted T0016 data and exact local source-snapshot verification; no real corpus, raw tokens, checkpoint tensors or production dependency changes are published.
