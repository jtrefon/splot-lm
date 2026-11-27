# T0017 attempt-001 — blocked on T0016; FP8 reuse and pilot-ingestion qualification passed

Task T0017 (IMP), issue #77, parents #7/#10/#42. Specification PR #78 merged at reviewed main `4c4e23e84c2a662d8519d4a2a852116c711f7ee2`; minimum reviewed base `8537c533372fb58a5d5ad5c58436f90e58c030ea`. Tested implementation commit `1c4e3fb8549561a0018305bf55596fe7732642b3`. UTC 2026-11-27T20:34:52Z to 2026-11-27T20:37:16Z (144 seconds).

## Outcome

**Blocked for the real pilot; implementation and qualification passed.** T0016 issue #76 is still open with no accepted dataset or evidence directory, so `real_pilot` is `not_run`; no real corpus was acquired or read. The FP8 per-update weight reuse, pilot-blocks-v1 loader, CPU suite, native CUDA cache qualification, cache-free resume and T512 paired measurement all passed.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `nano-cache.json`, `resume-cache.json`, `t512-pair.json`, `run-pilot-loader-smoke.json`, `summary.md` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_qualification | passed | `cache-tests.txt`, `nano-cache.json`, `resume-cache.json`, `t512-pair.json`, `run-pilot-loader-smoke.json`, `metrics.jsonl` |
| real_pilot | not_run | `real-pilot-dependency.json` |
| handover | passed | `summary.md`, `checkpoints.txt`, `execution.log` |

## Implementation

- `training.fp8_weight_reuse` is an explicit opt-in boolean defaulting to false. The existing no-reuse FP8 path is unchanged.
- The cache stores only E4M3 weight, device FP32 scale and quantization stats for one forward/backward/update scope. It is keyed by physical parameter identity, `_version`, device, dtype and shape; no activations, gradients or persistent/checkpoint state are cached.
- Cache is cleared before each step, after backward/audit capture before optimizer update, on exception, on checkpoint restore and before/after evaluation. No per-projection host sync was added.
- Actual weight quantizations and cache hits are reported separately, while logical per-use saturation/underflow/nonzero/nonfinite audit remains complete. Cache bytes and reuse ratio are reported.
- `pilot-blocks-v1` is a separate explicit loader for the T0016 contract: fixed paths, canonical manifest/tokenizer/shard validation, source-family split checks, bounded 64 MiB/160 blocks/512 tokens, actual vocabulary 264..16384 against model vocabulary 32768, exact masks/BOS/EOS/positions/labels and globally unique document IDs. Train/validation only are returned; test is integrity-validated only.
- The new Nano pilot config is FP8+Muon, B1, R2 A2, T512, seed 42, checkpoint interval 16 and reuse enabled. Unused model-vocabulary IDs are reported (synthetic fixture: actual 264, model 32768, 32,504 unused).

## CPU tests

`python tools/ci_cpu.py` from clean tested SHA `1c4e3fb8549561a0018305bf55596fe7732642b3`: **26 scripts, 214 tests, 0 failures, 0 skips, 19 REP reports**. New tests cover cache lifecycle/version invalidation/reset, pilot schema/counts/labels/global IDs/source/model/hash/extra-file rejection, exact Nano cache-off/on checks and resume.

## CUDA qualification

- Focused `tools/test_fp8_weight_reuse.py --device cuda`: **4/4 passed**.
- Nano T96 cache-off/on: identical losses (`total 15.809919357299805`), native calls 241/241, and logical quantization counts. Cache off: 241 actual weight quantizations, 0 hits, 0 cache bytes. Cache on: 138 actual quantizations, 103 hits, reuse ratio `0.42738589211618255`, cache bytes `26739240`. Model and optimizer state are exactly equal after the update.
- Cache-on 1+1 resume vs 2-step continuous: exact model, optimizer and non-timing history; checkpoint has no cache state; cross-mode resume with reuse disabled is rejected with `ValueError`.
- T512 paired one-update measurement: identical losses/native calls and exact model/optimizer state. Cache off: 298 weight quantizations, 0 hits. Cache on: 156 quantizations, 142 hits, reuse ratio `0.47651006711409394`, cache bytes `30278256`. Peak allocated/reserved: off `1697108992`/`1849688064`, on `1702824960`/`1860173824`. E2E `1.949470` s vs `1.633388` s; descriptive only, no speed claim.
- Synthetic pilot-loader smoke (not real data): 43,055,076 params, 512 valid / 511 NTP labels in one step, 1,022 validation NTP labels, `pilot-blocks-v1` data policy with actual vocab 264/model 32768, cache enabled with 156 quantizations/142 hits, interval-16 final write, 395,172,643-byte checkpoint. Test data was integrity-checked but not used for training.

## Real-pilot dependency

`real_pilot` is **not_run**. T0016 #76 is OPEN and has no accepted dataset/evidence directory; no reviewed manifest, source snapshot or locally verified pilot dataset was available. No real corpus was acquired, read, trained on or published. The qualification work is complete and preserved for continuation after T0016 is accepted and the exact source snapshot is locally verified.

## First failure and development notes

No official qualification run failed or was retried. During development, the cache-resume test initially omitted `resume=True` and raised `FileExistsError`; it was fixed before the tested commit. Checkpoint files are local-only; the synthetic pilot-loader checkpoint hash/size is in `checkpoints.txt`.

## Deviations

None.

## Limitations

- Real pilot not run: T0016 dependency unresolved. No real-data, quality, convergence or production-throughput claim.
- Only one smoke configuration/seed for the T512 cache pair; timing and memory are descriptive, not a speed threshold.
- Cache is per-update and process-local; no persistence, second-order use or multi-writer guarantee.
- Pilot loader accepts bounded synthetic-format artifacts only; it does not establish source rights or production data fitness.

## Proposed next action

Maintainer review of the passing implementation/qualification and the blocked real-pilot dependency. T0017 remains open; no merge or closure. After T0016 is accepted and an exact local snapshot is verified, a separately authorized continuation may run the fixed 64-step pilot. No polling or automatic task transition.
