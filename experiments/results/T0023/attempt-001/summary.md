# T0023 attempt-001 — actual-vocabulary pilot ingestion; hosted verification failed
2026-11-28 | IMP | Issue #101 | Instructions `d5cc33c0ba3df21e4763284cc3a1f4cde08a38b9` | Tested `152da24f4bee0f557334530b21a1fe3a03aee31d`

**Outcome:** **failed required hosted CI, stopped without correction or rerun** under the T0023 stop rule. Loader implementation and its hosted tests passed; one new dense-harness CLI test failed on a report-path assertion. No merge or acceptance.
**T0021 tested base:** `75fced45d0cff3888e3606a46a529fb3714e93df`.

## 1. Scope delivered

- `tools/pilot_ingestion.py`: preserves legacy model vocabulary 32768 and now accepts only the exact validated manifest vocabulary in the bounded 264–16384 range. Arbitrary padded sizes, noninteger/bool values, values below the manifest vocabulary and values other than legacy 32768 or exact equality fail closed. Equal vocab reports zero unused IDs and empty ranges; legacy unused ranges and all prior validation remain.
- `tools/test_pilot_ingestion.py`: generated V16384 boundary fixture with upper-bound IDs and valid label shifts; legacy 32768/default equality, exact actual-vocab, invalid-vocab, hash/source/tamper and test-integrity cases. Hosted: **6/6 passed**.
- `tools/experiment.py`: dense NTP now uses shared `ntp_loss`; explicit `model_family` configs report physical and gradient-present parameter tensors/numel and separate MTP/dormant-MTP inventory. Legacy absent-family output remains unchanged.
- `tools/test_dense_harness.py`: actual-vocab V16384 A0 dense/SPLOT CLI parity, shared packed digest/batch/labels, finite objectives, unused ranges and inventory checks; seed 42/43/44 one-pass order and T0016 position/label counts; common optimizer assertion.
- Static `proposed-matrix.json`: 3 arms × 3 seeds, 128 updates/pass, 65,536 positions / 46,204 valid tokens / 46,076 NTP labels per run, config identities and one-pass data-order digests. Proposed only; not executed, no quality margins or GPU allocation.

## 2. Hosted CI failure (first failure preserved)

Run [46166417407](https://github.com/jtrefon/splot-lm/actions/runs/46166417407) on `152da24` **failed**:

- `tools/test_dense_harness.py`: 13 run, 1 error, 12 passed.
- Failing test: `ActualVocabPilotTests.test_v16384_pilot_cli_both_families`, line 379.
- Error: `KeyError: 'dataset_manifest_sha256'`.
- The test read `report['identity']['dataset_manifest_sha256']`; the CLI stores that pin at `report['identity']['environment']['dataset_manifest_sha256']` (and in `data_policy`). This is a new-test assertion error; no loader, model, optimizer or numerical failure was reported.
- `tools/test_pilot_ingestion.py`: **6/6 passed**.
- Full aggregate count is unavailable because the runner exited nonzero; no final `CPU scripts` total is claimed.

Per T0023: no corrective source edit or rerun was performed. The failed run and excerpt are in `first-failure.txt` and `hosted-ci.json`.

## 3. Local checks

Local Mac CPython 3.15.2 has no torch. Local static checks passed: compile of all four changed Python files, JSON parsing of the three configs, proposed-matrix generation and diff check. No local unit/training pass is claimed. The proposed matrix is analytical/static and not an executed training protocol.

## 4. Limits and next action

- Required hosted CI failed; harness check is failed and no scoped acceptance is claimed.
- Loader hosted tests passed, but the end-to-end V16384 CLI path is not verified because its test failed before completing assertions.
- No local training, package install, GPU/MPS, real data, test or checkpoint access.
- Proposed matrix is not a preregistered quality protocol and allocates no GPU.
- Coordinator/owner review is required before any corrective test/path change and rerun. No merge, closure, training or next-task launch.

## 5. Handover

Evidence: summary, result, environment, metrics, sanitized execution log, first failure, loader pass, hosted run JSON/text, implementation diff/file hashes, proposed matrix/script and config identities.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0023/attempt-001 \
  --task T0023 --source 152da24f4bee0f557334530b21a1fe3a03aee31d \
  --checks environment,loader,harness,handover
```

Result status is `failed` because `harness` failed; `environment`, `loader` and the evidence handover are passed. No merge/closure.
