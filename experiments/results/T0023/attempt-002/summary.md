# T0023 attempt-002 — actual-vocabulary preflight; second hosted failure
2026-11-28 | IMP | Issue #101 | Instructions `a87adbd1e7186926fa1766adc4aa8470b71d8579` | Tested `57629a9ea2ab956ec24e1d05fb153f79b5bd2a64`

**Outcome:** **failed required hosted CI again; stopped without correction or rerun** under the T0023 continuation. Loader implementation and hosted loader tests passed; the V16384 CLI parity test still failed on a report-key path. No merge or acceptance. Attempt-001 is preserved untouched.

## 1. Restoration

The six scoped files were restored from original failed implementation `152da24f4bee0f557334530b21a1fe3a03aee31d` onto current main `a87adbd`. `restoration.files.json` records five files byte-identical to the original and only `tools/test_dense_harness.py` changed. `correction.diff` is the exact 12-insertion/6-deletion test-only change.

- `tools/pilot_ingestion.py`: legacy 32768/default preserved; exact actual manifest vocabulary in [264,16384] allowed; arbitrary padded, noninteger/bool, too-small and mismatched values rejected; equal vocab reports zero unused IDs/empty ranges.
- `tools/experiment.py`: shared `ntp_loss` for dense NTP; explicit-family physical/gradient-present and MTP/dormant inventory; legacy absent-family schema unchanged.
- `tools/test_pilot_ingestion.py`: V16384 boundary/label fixtures, legacy equality, exact actual vocabulary and invalid cases.
- `tools/test_dense_harness.py`: test-only path/parity correction, but still failed.
- Docs updated.

## 2. Hosted failure (second, preserved)

Run [46168369820](https://github.com/jtrefon/splot-lm/actions/runs/46168369820) on `57629a9` **failed**:

- `tools/test_dense_harness.py`: **13 run, 12 passed, 1 error**.
- Failing test: `ActualVocabPilotTests.test_v16384_pilot_cli_both_families`, line 381.
- Error: `KeyError: 'dataset_manifest_sha256'` from `report['data_policy']['dataset_manifest_sha256']`.
- The data policy stores this pin as `report['data_policy']['manifest_sha256']`; the identity environment and source-snapshot paths were corrected, but this second key was wrong.
- `tools/test_pilot_ingestion.py`: **6/6 passed**.
- Full aggregate count unavailable because the runner exited nonzero.

No corrective source edit or rerun was performed. `first-failure.txt` and `hosted-ci.json` preserve the failure; `loader-pass.txt` preserves the passing loader suite.

## 3. Static preflight

`proposed-matrix.json` was recomputed from tested source and configs: 3 arms × seeds 42/43/44, 128 updates/pass, 65,536 positions / 46,204 valid tokens / 46,076 NTP labels per run, one-pass data-order digests, common optimizer settings. It is planning-only, not executed, and does not allocate GPU or preregister quality margins.

Local Mac has no torch; local checks cover compile, JSON configs, restoration comparison, matrix generation and diff checks. No local unit/training pass is claimed.

## 4. Limits and next action

- Required hosted CI failed twice; harness remains failed and the V16384 end-to-end CLI path is unverified.
- Loader tests passed both attempts, but no matched-run preflight acceptance is claimed.
- No local training, install, GPU/MPS, real data, test or checkpoint access.
- Coordinator/owner review is required before any further test-path correction and rerun. Attempt-001 remains preserved; no merge, closure, training or next-task launch.

## 5. Handover

Evidence: summary, result, environment, metrics, execution log, first failure, loader pass, hosted run JSON/text, restoration comparison, exact correction diff, implementation diff/file hashes and proposed matrix.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0023/attempt-002 \
  --task T0023 --source 57629a9ea2ab956ec24e1d05fb153f79b5bd2a64 \
  --checks environment,loader,harness,handover
```

`result.json` records environment/loader/handover passed, harness failed, overall failed. No merge/closure.
