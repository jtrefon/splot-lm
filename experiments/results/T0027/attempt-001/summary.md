# T0027 attempt-001 — bounded matched V16384 NTP learning pilot

Task T0027 (VAL), issue #109, parents #85/#8/#10/#87/#101. Instructions/source `003cf8db6126f0aaefdd599ac159f2503457f709` in `research/T0027-matched-ntp-attempt-001`. UTC 2026-11-28T17:41:54Z to 2026-11-28T17:51:01Z (547 s matrix wall). Overall: **passed** — all nine fixed-budget runs completed and passed the finite/data/identity/resource gates. No quality superiority or noninferiority threshold exists in this pilot; the paired differences are descriptive only.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `dataset-identity.json`, `source-register.json`, `config-audit.json`, `execution.log` |
| matched_runs | passed | nine `run-*.json`, nine `audit-*.json`, `run-timings.json`, `checkpoint-inventory.json`, `matrix.json` |
| analysis | passed | `analysis.json`, `metrics.jsonl`, `run-*.json` |
| handover | passed | `summary.md`, `execution.log`, `checkpoint-inventory.json`, `analysis.json` |

## Environment and identity

- GPU 0 NVIDIA GeForce RTX 4090, 24,077 MiB free at every preflight; driver 580.178.04. Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8; deterministic algorithms, TF32 off, CUBLAS workspace `:4096:8`.
- Host RAM available minimum 26,880,626,688 B; private ext4 scratch free minimum 179,000,242,176 B; every run re-checked ≥20,480 MiB GPU, ≥12 GiB RAM and ≥4 GiB scratch.
- Source `003cf8db6126f0aaefdd599ac159f2503457f709` was clean before and after the runs; origin/main had zero open integration PRs at start, and T0023 corrected `4a4bb5e...` is an ancestor. No source changes, packages, downloads or installs.
- Existing `splot-pilot-blocks-v1` snapshot: five files, 2,727,588 bytes, all five SHA-256 values match the accepted T0016/T0023 pins. The copy was produced and verified during accepted T0017 attempt-002 and was copied without regeneration. Manifest `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55`, source `c84bd8a6e94c5d0f4d429e44a43c7bffd9a6dfb9a263b9a3440126e23b4aa087`, tokenizer `104d3927bb55cdd2a911dbe21c8e755264756cedb5663bd39365c3d4cba3864b`, logical `70ba8d35d2ae334a8601f6389dc39208f3636937cb9dbfda639ead7ab30c052c`.
- Actual V16384 with zero unused token IDs; train 128 blocks / 46,204 valid / 46,076 NTP / 45,948 available offset-2 labels; validation 5,250 NTP labels. Test shard was integrity-checked only, never trained or evaluated.
- Configs differ from the reviewed files only in `training.seed` 42/43/44; original/derived SHA-256 values and the exact one-line diffs are in `config-audit.json` and `config-seed-diff.txt`.

## Matrix and per-run results

Counterbalanced order: 42:S,P,C; 43:P,C,S; 44:C,S,P. Each run: 128 updates, B1, T512, BF16 autocast/FP32 master, Muon/AdamW, checkpoint interval 16 (8 writes), no resume.

| Run | Order | Val NLL | Val PPL | Weighted train NLL sum | Wall s | Step s sum | Valid tok/s | Peak alloc MiB | Checkpoint bytes | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S-seed42 | 1 | 7.434416 | 1693.268 | 339731.919 | 135 | 127.061 | 363.6 | 1415.3 | 319,086,611 | a1d34d3fd243… |
| P-seed42 | 2 | 7.499754 | 1807.597 | 341807.143 | 18 | 5.264 | 8777.1 | 707.5 | 317,884,443 | 2a380f5fe741… |
| C-seed42 | 3 | 7.592448 | 1983.162 | 343146.516 | 12 | 5.373 | 8599.6 | 663.0 | 292,718,555 | c0f70ff2b4ad… |
| P-seed43 | 4 | 7.396814 | 1630.781 | 345384.437 | 11 | 5.298 | 8721.2 | 707.8 | 317,884,315 | 98098d96bad8… |
| C-seed43 | 5 | 7.425299 | 1677.901 | 346429.963 | 12 | 5.316 | 8691.9 | 664.7 | 292,718,491 | bae81e0a3b67… |
| S-seed43 | 6 | 7.329211 | 1524.178 | 340376.450 | 135 | 127.052 | 363.7 | 1413.9 | 319,086,547 | 4433a704947b… |
| C-seed44 | 7 | 7.523176 | 1850.436 | 348941.753 | 12 | 5.388 | 8574.7 | 665.5 | 292,718,491 | 806eacb9c1d4… |
| S-seed44 | 8 | 7.478690 | 1769.921 | 342318.239 | 135 | 127.169 | 363.3 | 1412.4 | 319,086,547 | 81b460e2910b… |
| P-seed44 | 9 | 7.519210 | 1843.111 | 344695.974 | 10 | 5.383 | 8583.3 | 707.0 | 317,884,379 | e3ac71f01940… |

All nine exit codes were 0, all histories contained exactly 128 steps, and all 128 train indices per seed matched the T0023 proposed matrix. Every run had identical dataset/packed SHA-256 and the same source commit. No timeout, retry, continuation, test evaluation or source change occurred. Checkpoint paths are withheld; full hashes and bytes are in `checkpoint-inventory.json`.

## Arm summaries

| Arm | Family | Params total | Non-MTP params | Analytical forward MACs | MAC / valid token (M) | Mean step s | Mean valid tok/s | Mean wall s | Mean val s | Max peak alloc MiB | Mean checkpoint MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | SPLOT A0 scan | 36,763,620 | 36,615,780 | 1,555,549,658,112 | 33.667 | 127.094 | 363.5 | 135.0 | 3.3228 | 1415.3 | 319.1 |
| P | dense parameter-matched | 35,527,168 | 35,527,168 | 1,696,698,552,320 | 36.722 | 5.315 | 8693.9 | 13.0 | 0.1118 | 707.8 | 317.9 |
| C | dense forward-MAC-matched | 32,381,440 | 32,381,440 | 1,551,353,335,808 | 33.576 | 5.359 | 8622.0 | 12.0 | 0.1124 | 665.5 | 292.7 |

SPLOT S has 147,840 dormant MTP parameters. Its gradient-present parameter count varied with sparse expert routing on the final batch (35,436,132 / 35,436,132 / 36,025,956 for seeds 42/43/44); this is the physical gradient-present observation, not an active routed count. Dense P/C updated all physical parameters.

## Paired descriptive analysis

Negative difference means SPLOT S had lower validation NLL. These are paired by seed only; the arms have different initial shapes and the same seed shares the data order, not the initialization. n=3, one snapshot; no significance, equivalence, superiority or generalization claim.

| Pair | Seed | S NLL | Comparator NLL | S − comparator | Percent of comparator |
|---|---:|---:|---:|---:|---:|
| S−P | 42 | 7.434416 | 7.499754 | -0.065338 | -0.8712% |
| S−P | 43 | 7.329211 | 7.396814 | -0.067604 | -0.9140% |
| S−P | 44 | 7.478690 | 7.519210 | -0.040520 | -0.5389% |
| S−C | 42 | 7.434416 | 7.592448 | -0.158032 | -2.0814% |
| S−C | 43 | 7.329211 | 7.425299 | -0.096088 | -1.2941% |
| S−C | 44 | 7.478690 | 7.523176 | -0.044486 | -0.5913% |

| Pair | Mean | Median | Min | Max | SD (sample, n−1) |
|---|---:|---:|---:|---:|---:|
| S−P | -0.057821 | -0.065338 | -0.067604 | -0.040520 | 0.015025 |
| S−C | -0.099536 | -0.096088 | -0.158032 | -0.044486 | 0.056851 |

Observed: S had lower validation NLL than P and C in all three paired seeds. The descriptive mean differences were -0.057821 (S−P) and -0.099536 (S−C) nats. This is a three-seed pilot signal only.

## Analytical cost and measured cost

- Analytical forward MAC totals (ideal unpadded document contractions, not actual training FLOPs): S 1,555,549,658,112; P 1,696,698,552,320; C 1,551,353,335,808.
- Analytical gaps: S−P -8.3190%; S−C +0.2705%; P−C +9.3689%.
- Measured mean step-time ratios: S/P 23.91×; S/C 23.72×. Mean wall ratios: S/P 10.38×; S/C 11.25×.
- The large measured gap is specific to this reference GDN scan implementation and these dense CUDA paths. It is not an actual FLOP measurement, an optimized dense comparison, or an architecture-efficiency claim.
- Host RSS was not measured. Energy was not measured and no estimate is made. Checkpoints total 2,789,068,379 bytes across nine private files; 72 serialized writes (8 per run).

## Interpretation and limitations

- This is a descriptive feasibility and initial matched learning/cost signal on one small bilingual one-pass snapshot. It is not a power-qualified confirmation, production model, convergence result, architecture-superiority result or trained optimized baseline.
- Only one pass of 128 updates and three seeds per arm were run. The paired NLL differences are small-sample descriptions, not a quality threshold. There is no test evaluation and no claim about held-out generalization.
- Different parameter counts, initial shapes, routing behavior and analytical control bases remain; data order is paired by seed, initialization is not.
- Analytical MAC estimates exclude backward, padding and hardware effects. Measured step timing excludes checkpoint I/O and validation; wall includes setup/I/O/evaluation.
- Sparse MoE expert parameters not routed on the last batch have no gradient on that update; the physical and gradient-present inventories are reported separately.

## Process notes

No required training run or gate failed and no training invocation was retried. Before the first run, the start-comment UTC label was corrected. During analysis, the read-only audit helper first lacked its private output directory, then had three read-only field/assumption errors corrected (source-commit constant, master-dtype field location, sparse-MoE gradient-present expectation). The unchanged run outputs were re-audited; no source, data, config, seed, tolerance or gate changed. These are analysis-tooling corrections, not protocol deviations.

## Handover

The flat package contains nine sanitized run JSONs, nine per-run audits, exact derived configs and config hashes/diff, dataset/source identity, matrix order, timings, checkpoint hashes/bytes, aggregate analysis and metrics. No checkpoints, tensors, raw corpus, private paths, test metrics, secrets or hostnames are published. T0027 remains open; no merge, closure or further task is performed.

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0027/attempt-001 \
  --task T0027 \
  --source 003cf8db6126f0aaefdd599ac159f2503457f709 \
  --checks environment,matched_runs,analysis,handover
```

Validator output is reported in the PR body after exact-head packaging.
