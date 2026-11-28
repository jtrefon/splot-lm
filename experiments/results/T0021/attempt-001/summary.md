# T0021 attempt-001 — dense NTP comparator integration
2026-11-28 | IMP | Issue #97 | Source base `a771c370efe662f17ddf8c06769cbd06a783388a` | Instructions `2babd431a8c0cb61a0f1129b4aae8c2383d7dbf8`

**Outcome:** passed after one test-only correction. No local training, package install, GPU/MPS, real data, test or checkpoint access.
**Tested implementation:** `75fced45d0cff3888e3606a46a529fb3714e93df`.
**First implementation:** `340e56a418e50b67ad169267ee863dfb94da587a`; its hosted run failed two new-test assertions, corrected in the tested SHA.
**Hosted CPU:** [run 46164883574](https://github.com/jtrefon/splot-lm/actions/runs/46164883574) **success** on `75fced4`; dense suite **9/9**, full **30 scripts / 27 REP reports**. First failure [46164663077](https://github.com/jtrefon/splot-lm/actions/runs/46164663077) on `340e56a` is preserved.

## 1. Implementation

- Explicit optional `model_family`; absence or `splot` preserves the legacy SPLOT path, config hash, identity and report output.
- `dense-ntp` uses `DenseConfig`/`DenseLM` only and rejects recurrence/expert fields, non-A0 modes, `loops != 1`, FP8/reuse and nonzero auxiliary weights.
- Dense objective is final NTP only: `total=final_ntp`, early/offset/balance zero, `offset2_count=0`; labels remain pre-shifted and masked.
- Existing `train`, `evaluate`, checkpoint/resume, data ingestion and semantic Muon/AdamW paths are reused. Dense identity adds `model_family` and `dense_parameter_count`; cross-family resume fails.
- `tools/accounting.py` adds `DenseShape`/`dense_account` and `--dense-config`; no SPLOT accounting behavior changed.
- Configs: `dense-ntp-512x6-q8kv4-ffn2432.json`, `dense-ntp-512x6-q8kv2-ffn2176.json`, and `splot-nano-v16384-a0.json`.

## 2. Analytical counts

| Target | Physical params | Decode 512 | Prefill 512 | Decode 2048 | Prefill 2048 |
|---|---:|---:|---:|---:|---:|
| Dense Q8/KV4, FFN 2432 | 35,527,168 | 38,666,240 | 18,993,381,376 | 48,103,424 | 85,637,201,920 |
| Dense Q8/KV2, FFN 2176 | 32,381,440 | 35,520,512 | 17,382,768,640 | 44,957,696 | 79,194,750,976 |
| SPLOT V16384 A0 | 36,763,620 total / 36,615,780 base | 35,125,248 | 17,381,326,848 | 42,203,136 | 76,773,064,704 |

SPLOT auxiliary head is 147,840; `balance_weight=0.01` is explicit. Dense configs have no dormant MTP head. MACs are ideal contraction counts, not training FLOPs.

## 3. Tests

| Check | Result |
|---|---|
| Local stdlib: py_compile, JSON configs, dense accounting, diff check | passed |
| Hosted `tools/test_dense_harness.py` | 9/9 passed, 8.749 s |
| Hosted full CPU suite | 30 scripts, 27 REP reports, passed |
| Local torch | absent; no local unit/training claim |

Dense tests cover exact accounting, parameter/partition inventory for both candidates, dense objective equality with manual masked NTP, continuous 4 vs 2+2 model/optimizer/RNG/history/validation, cross-family rejection, legacy SPLOT CLI identity, synthetic-v2/pilot ingestion and CPU BF16 capability when supported. No GPU qualification is claimed.

## 4. First failures

- **Hosted:** `340e56a` run 46164663077 failed 2/9 dense tests: partition entries were read with a nonexistent `group` key, and the pilot loader fixture used model vocabulary 264 although the accepted loader pins 32768. Both were defects in the new test, not the dense model/harness. Corrected in `75fced4`; the first failure is preserved in `first-failure.txt`.
- **Local:** the first dense-counts invocation failed with `ModuleNotFoundError: accounting` because `PYTHONPATH` was relative to the evidence directory. The runner now uses absolute `src`/`tools` paths; preserved in `execution-first-failures.log`.

No numerical gate or tolerance was changed.

## 5. Limitations

- No local torch, training, real data, GPU/MPS or test/checkpoint access; hosted CPU tests use tiny synthetic fixtures only.
- Candidate configs are planning artifacts from accepted T0019, not trained or optimized results.
- Dense BF16 capability is reported only if the existing CPU runner supports it; no GPU or production-throughput claim.
- A2/auxiliary comparison remains deferred; dense mode is A0 NTP only.
- Legacy SPLOT behavior is regression-tested but no new SPLOT training is executed.

## 6. Handover

Flat evidence: summary, result, environment, metrics, sanitized execution log, first-failure logs, dense-counts JSON/formulas, implementation diff/file hashes, hosted run metadata/excerpts and config identities.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0021/attempt-001 \
  --task T0021 --source 75fced45d0cff3888e3606a46a529fb3714e93df \
  --checks environment,implementation,unit_tests,handover
```

Output is recorded in `result.json` and the PR. One INDEX row is appended; issue #97 receives the handover. No merge, closure, training or next-task launch.
