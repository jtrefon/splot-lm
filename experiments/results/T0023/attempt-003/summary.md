# T0023 attempt-003 — actual-vocabulary preflight; hosted pass
2026-11-28 | IMP | Issue #101 | Instructions `a87adbd1e7186926fa1766adc4aa8470b71d8579` | Tested `4a4bb5e4ce0566166c4c8968506c8f89cc3a2bcf`

**Outcome:** **passed required hosted CPU CI** after the owner-authorized one-line test correction. The actual-vocabulary loader and the V16384 end-to-end CLI parity test now pass. Attempt-001 and attempt-002 failures remain preserved. No merge, closure, training or next-task launch.

## 1. Correction

Attempt-002 failed because the new test read `report['data_policy']['dataset_manifest_sha256']`; the data policy field is `manifest_sha256`. Attempt-003 changes only that key in `tools/test_dense_harness.py` (1 insertion / 1 deletion). No production code, identity/schema, gate, tolerance or fixture changed.

The six scoped files remain the original tested implementation from `152da24f4bee0f557334530b21a1fe3a03aee31d`, with only the test correction: five files are byte-identical to the original and `tools/test_dense_harness.py` differs only in report-path and parity assertions. `restoration.files.json` and `correction.diff` record this.

## 2. Hosted result

Run [46170273255](https://github.com/jtrefon/splot-lm/actions/runs/46170273255) **succeeded** on `4a4bb5e4ce0566166c4c8968506c8f89cc3a2bcf`:

- `tools/test_dense_harness.py`: **13/13 passed** in 12.925 s, including `ActualVocabPilotTests.test_v16384_pilot_cli_both_families`.
- `tools/test_pilot_ingestion.py`: **6/6 passed** in 0.303 s.
- Full CPU suite: **30 scripts / 32 REP reports**, exit 0.
- The V16384 test now verifies identity-environment and data-policy manifest/source pins, test-only policy, two finite history rows, all-row batch/valid/label parity, and SPLOT gradient/MTP inventory.

## 3. Static preflight

`proposed-matrix.json` was recomputed from tested source and configs: 3 arms × seeds 42/43/44, 128 updates/pass, 65,536 positions / 46,204 valid tokens / 46,076 NTP labels per run, one-pass data-order digests and common optimizer settings. It is planning-only, not executed, and does not allocate GPU or preregister quality margins.

Local Mac has no torch; local checks cover compile, JSON configs, restoration comparison, matrix generation and diff checks. No local unit/training pass is claimed.

## 4. Limits

- Required hosted CI passed at the tested implementation; the evidence head is validated by a separate required hosted run on the same source.
- No local torch, training, package install, GPU/MPS, real data, test or checkpoint access.
- Hosted tests use tiny synthetic fixtures; no quality, throughput or production-vocabulary training claim.
- The proposed 9-run matrix is static and not an executed or preregistered protocol.

## 5. Handover

Flat evidence: summary, result, environment, metrics, execution log, dense/loader pass excerpts, hosted run JSON/text, restoration comparison, exact correction diff, implementation diff/file hashes and proposed matrix. Attempt-001 and attempt-002 evidence remain untouched.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0023/attempt-003 \
  --task T0023 --source 4a4bb5e4ce0566166c4c8968506c8f89cc3a2bcf \
  --checks environment,loader,harness,handover
```

Output: STRUCTURALLY VALID: passed (not scientific acceptance). One INDEX row is appended; PR #106 and issue #101 carry the handover. No merge, closure, training or next-task launch.
