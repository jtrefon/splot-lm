# T0014 attempt-001 — passed (verified v2 ingestion and checkpoint cadence)

Task T0014 (IMP), issue #71, parents #8/#10. Specification PR #73 merged at reviewed main `cb97b045c965beb614083b54dd7be26393bbe7b8`; minimum reviewed base `3fb9388e782a9da8ca68e1458d9caaeb296c2312`. Tested implementation commit `d6f979cfa1e9763fb4179cd7b902a51fbde68199`. UTC 2026-11-27T20:03:56Z to 2026-11-27T20:07:28Z (212 seconds).

## Outcome

**Passed.** All five required checks passed. The v2 fixture reproduces T0013 exactly, interval-2 checkpointing and exact resume pass, the interval-1 control matches the interval-2 run on non-timing content, and cross-manifest/interval resumes are rejected. This is bounded synthetic bridge evidence only; no real data, production cadence or speed claim.

## Authorization and scope

The owner explicitly authorized T0014 in the current local session within the merged specification and resource limits. The implementation changes only `tools/experiment.py`, new `tools/test_dataset_ingestion.py`, new `configs/experiments/splot-smoke-shards-fp8-muon.json` and new `research/implementation/dataset-ingestion.md`. No shard code, existing config, model, comparator, CI or Mac task file changed.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `data-gate.json`, `v2-fixture.json`, `parameter-count.json`, `summary.md` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_ingestion | passed | five run JSONs, `strict-comparison.json`, `cadence-comparison.json`, `checkpoint-state.json`, `mismatch-rejections.json`, `metrics.jsonl`, `execution.log` |
| handover | passed | `summary.md`, `checkpoints.txt`, `execution.log` |

## Verified v2 data gate

The Linux v2 demo ran twice with CUDA hidden and produced byte-identical 21,209-byte output with SHA-256 `1c899be572c22105cf2d37766b9806a5db6daebd9f0a768ce2f793e369511396`, exactly matching T0013 attempt-001 on Mac. Manifest SHA-256 is `1e2b8fa70aafff32331018488992c40ec82521ad26eed5b58532fdf58442bff0`; logical `5d8d5898c41c36ac21cc7ba4785c8747c0a6ec8b20830419a0a082dfd931b5ba`; packed `7585194a799c1bc839591362929bf4e041bb18b4acaf9d15d677ff951743635f`; physical `3229cf587492afd31465fb3731f796f612dbaa279f05bebcedaabf1278813df6`.

The local fixture has 2 train blocks and 1 validation block; train has 142 valid / 134 NTP / 126 offset-2 labels per epoch and validation has 14 NTP labels. The test block is loaded for complete-set verification but is not used for training or evaluation. The alternate one-block-per-shard materialization has a different physical manifest (`d4728a2e…`), while logical/packed digests remain equal; it is used for cross-manifest rejection.

## CPU tests

`python tools/ci_cpu.py` from clean tested SHA `d6f979cfa1e9763fb4179cd7b902a51fbde68199`: **23 scripts, 204 tests, 0 failures, 0 skips, 17 REP reports**. The new suite contributes 10 tests for verified blocks/counts, manifest/dimension/tamper rejection, interval-1 legacy, interval-2 final off-boundary save, split/resume equivalence, failed-guard checkpoint preservation, interval identity mismatch and the interval1-vs-interval2 comparator.

## Ingestion and checkpoint runs

| Run | Steps | Interval | Valid/NTP/offset2 | Val NTP | Writes | Total write bytes | Write seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| continuous-int2 | 4 | 2 | 284/268/252 | 14 | 2 | 3439110 | 0.026512 |
| split-int2 | 4 | 2 | 284/268/252 | 14 | 1 | 1721187 | 0.013935 |
| continuous-int1 | 4 | 1 | 284/268/252 | 14 | 4 | 6876812 | 0.054081 |
| split-int2-metadata | 2 | 2 | 142/134/126 | 14 | 1 | 1718883 | 0.012715 |
| legacy-v1-smoke | 1 | 1 | 51/50/49 | 376 | 1 | 2406023 | 0.010710 |

- `continuous-int2`: 4 steps = 2 epochs, interval 2, writes at steps 2 and 4, 2 writes.
- `split-int2-metadata`: auxiliary split-only replicate (same config/data/seed/interval) recording the consumed step-2 write count; 1 write, not a retry of any failed gate.
- `split-int2` final invocation: resume from step 2 to 4, writes at step 4 only, 1 write; strict comparator vs continuous is exact on model, optimizer, RNG, non-timing history and validation.
- `continuous-int1`: 4 writes at steps 1, 2, 3, 4. The focused comparator against interval-2 is exact on model/optimizer/RNG, non-timing history and validation; identity is intentionally allowed to differ.
- `legacy-v1-smoke`: one CPU step on the original synthetic-v1 config, 179,952 parameters, interval default 1, showing the legacy path remains intact.

Parameter counts were independently computed from model configs: v2 smoke 181,744; legacy smoke 179,952. All v2 checkpoints have 198 model entries, FP32 model/state tensors, and 195,593 optimizer-state elements / 782,372 bytes (Muon 167,936 / 671,744; AdamW 27,657 / 110,628). FP8 native forward/backward calls are positive (1,936 each over 4 steps; 968 each over 2 steps), and nonfinite is false.

Cross-manifest and cross-interval resume attempts both exit 1 with `ValueError: run identity mismatch`. Checkpoint hashes and byte sizes are in `checkpoints.txt`; all are local-only and not published.

## First failure and development notes

No official run or required gate failed and no official run was retried. The split-only metadata replicate explicitly records the step-2 write that was consumed by the required resume invocation; it is not a retry. Checkpoint file hashes can differ between otherwise identical loaded tensors because `torch.save` serialization is not byte-stable; the strict and focused comparators compare loaded tensors, RNG and non-timing history, not file bytes.

## Deviations

None.

## Limitations

- Bounded synthetic v2 bridge only; no real corpus, rights approval, production ingestion, tokenizer or Mini model.
- One smoke configuration and one seed; no convergence, quality, production throughput or checkpoint-service claim.
- Interval saves are atomic local replacements with the trusted single-writer assumption; no fsync/power-loss guarantee. At most interval-1 completed updates can be lost after the last durable checkpoint under normal interruption.
- Test blocks are verified by the loader but are not used for training/evaluation.
- No timing threshold or speed claim; checkpoint write and validation times are reported separately and are descriptive.

## Proposed next action

Maintainer review of PR evidence and hosted CPU CI. T0014 remains open; no merge or closure. Production dataset rights/format and real pilot ingestion require separate reviewed tasks.
