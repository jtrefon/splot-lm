# Verified dataset ingestion and checkpoint cadence

Status: T0014 implementation, bounded synthetic bridge only. Task [T0014](../tasks/T0014-dataset-ingestion.md), issue [#71](https://github.com/jtrefon/splot-lm/issues/71). This is not real-corpus acquisition, a production data pipeline, rights approval or a production checkpoint service.

## Opt-in v2 ingestion

The default remains `training.data = "synthetic-v1"` with the existing byte fixture and byte-identical behavior. A new explicit mode, `synthetic-shards-v2`, requires both:

```bash
python tools/experiment.py ... \
  --dataset-root /path/outside/checkout \
  --manifest-sha256 <64 lowercase hex>
```

Supplying either option with `synthetic-v1`, or omitting either with `synthetic-shards-v2`, is rejected. The mode calls the existing v2 `tokenized_shards.load_dataset` with the independently pinned manifest SHA. The loader verifies the complete physical set, canonical bytes, format/version, tokenizer artifact, documents, shard hashes, labels, masks, positions, independently repacked blocks, logical/packed/physical digests and counts.

`load_shard_dataset` additionally requires:

- `splot-tokenized-shards-v2` / version 2;
- the manifest tokenizer's actual vocabulary to equal the model vocabulary;
- the manifest pack length to equal `training.sequence_length`;
- nonempty train and validation selections;
- label keys `"1"`/`"2"` converted to integer offsets without changing document IDs, positions or valid masks.

Only train and validation blocks are returned to the harness. Test blocks are not evaluated or used for training. Ingestion is synthetic-v2 only; it does not infer real-source approval. The dataset root is not placed in identity or reports. Identity records the manifest SHA, tokenizer metadata, logical/packed/physical digests, data/config/packed hashes and the configured interval.

The local v2 fixture is regenerated from the accepted `build_demo_bundle`/`materialize_dataset` API, not copied from Mac. The T0013 canonical demo and manifest hashes must match before dependent CUDA work.

## Checkpoint interval

`train` accepts a positive integer `checkpoint_interval` (default 1) and always writes at the final requested step. The interval is a resolved run setting and is included in identity; resuming a checkpoint with a different interval is rejected. A checkpoint is written only when `(step+1) % interval == 0` or `step+1 == steps`.

Writes retain the existing atomic temporary-file replacement and trusted local single-writer assumption. No fsync or power-loss guarantee is claimed. Under normal single-writer interruption, at most `interval-1` completed updates after the last durable checkpoint can be lost. A failed finite guard occurs before history append and checkpoint write, so it cannot publish a checkpoint or history row beyond the prior saved step.

Resume uses the checkpoint as the progress ledger, not `run.json`. The target is the total step count; the loaded checkpoint history is continued without replaying partial steps. `checkpoint_stats` reports the initial/final logical history length, invocation-only history length, write count, write steps, bytes per last/total write and write seconds. Validation time is measured separately from training step timing.

## Tests and limits

`tools/test_dataset_ingestion.py` covers verified v2 blocks/counts, exact offsets, train/validation selection without test training, manifest/dimension rejection, tampered manifest, interval=1 legacy writes, interval>1 final off-boundary save, interrupted-boundary resume equivalence, failed-guard checkpoint preservation, interval identity mismatch and an interval1-vs-interval2 comparator that ignores only timing and identity.

The smoke shard config uses vocabulary 320, R2 A2 batch 1, 96-token blocks, seed 42 and the same FP8+Muon hyperparameters as the existing smoke config. No model equations, optimizer math, tokenizer shard code, existing configs, comparator or CI were changed.

Limitations: synthetic v2 only; no real corpus, rights inference, Mini model, production ingestion, retention service, fsync/power-loss guarantee or speed claim. Checkpoints remain local-only.
