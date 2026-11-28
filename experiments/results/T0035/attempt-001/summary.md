# T0035 attempt-001 — matched FP32 scan/chunk64 learning pilot

Task T0035 (VAL), issue #125, parents #85/#7/#10/#109/#121. Instruction/source `4f86f62d3fb673483ddf87001331db90e354b985`; branch `research/T0035-fp32-matched-attempt-001`. UTC 2026-11-28T20:27:58Z to 2026-11-28T20:37:24Z (566 s matrix). Overall: **passed** — nine FP32 runs completed; all identity/data/order/count/finite/resource gates passed. Learning and cost results are descriptive only.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `dataset-identity.json`, `source-register.json`, `preflight.json`, `config-audit.json`, `execution.log` |
| matched_runs | passed | nine `run-*.json`, nine `audit-*.json`, `run-timings.json`, `checkpoint-inventory.json` |
| analysis | passed | `analysis.json`, `metrics.jsonl`, `run-*.json` |
| handover | passed | `summary.md`, `execution.log`, `checkpoint-inventory.json`, `dataset-identity.json` |

## Data and matrix

- Exact existing T0016 five-file snapshot, 2,727,588 bytes; manifest/source/tokenizer/logical pins and all file hashes verified; V16384, zero unused IDs; test integrity only. No regeneration or content inspection.
- FP32 A0 B1 T512, 128 updates, LR .003 Muon/AdamW, checkpoint interval 16 (8 writes), one pass of 128 train blocks. Order 42:S,K,C; 43:K,C,S; 44:C,S,K.
- S/K share FP32 precision, 36,763,620 parameters and analytical MACs `1,555,549,658,112`; S/K initialization and partition equal by seed. C is dense FP32 `32,381,440` parameters, MACs `1,551,353,335,808`.
- All nine share commit/data/packed digest; 46,204 valid / 46,076 NTP / 45,948 offset2 labels and 5,250 validation labels; zero unused IDs; no test training/evaluation.

## Per-run results

| Run | Order | Val NLL | Val PPL | Weighted train NLL | Wall s | Step s | Valid tok/s | Peak alloc B | Checkpoint B | Checkpoint SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S-seed42 | 1 | 7.344828 | 1548.169 | 340409.299 | 136 | 125.803 | 367.3 | 1,479,751,680 | 319,086,611 | fa0a565fd98f… |
| K-seed42 | 2 | 7.380902 | 1605.037 | 339224.642 | 33 | 24.420 | 1892.1 | 926,148,096 | 319,086,611 | e913a87c7db9… |
| C-seed42 | 3 | 7.508320 | 1823.149 | 343283.241 | 10 | 5.187 | 8908.1 | 678,867,456 | 292,718,555 | 314e86fcdb5a… |
| K-seed43 | 4 | 7.345804 | 1549.680 | 343646.884 | 30 | 24.419 | 1892.1 | 927,680,000 | 319,086,547 | de3bbe3c49ba… |
| C-seed43 | 5 | 7.456401 | 1730.907 | 345851.965 | 9 | 4.999 | 9242.2 | 678,867,456 | 292,718,491 | 87059f0e26f4… |
| S-seed43 | 6 | 7.345055 | 1548.520 | 340600.736 | 131 | 123.814 | 373.2 | 1,481,282,048 | 319,086,547 | 21c05d6845de… |
| C-seed44 | 7 | 7.527837 | 1859.081 | 348101.025 | 9 | 5.165 | 8945.7 | 678,867,456 | 292,718,491 | c74b5a430c46… |
| S-seed44 | 8 | 7.412854 | 1657.149 | 341022.294 | 134 | 126.300 | 365.8 | 1,480,303,616 | 319,086,547 | 6103343afecd… |
| K-seed44 | 9 | 7.398495 | 1633.524 | 340354.457 | 30 | 24.874 | 1857.5 | 926,706,176 | 319,086,547 | 0d77b25ccad6… |

## Arm summaries

| Arm | Label | Parameters | Non-MTP | Analytical MACs | Mean Val NLL | Mean step s | Mean wall s | Mean valid tok/s | Max peak alloc B | Mean checkpoint MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | splot-fp32-scan | 36,763,620 | 36,615,780 | 1,555,549,658,112 | 7.367579 | 125.305 | 133.7 | 368.8 | 1,481,282,048 | 319.1 |
| K | splot-fp32-chunk64 | 36,763,620 | 36,615,780 | 1,555,549,658,112 | 7.375067 | 24.571 | 31.0 | 1880.6 | 927,680,000 | 319.1 |
| C | dense-fp32-control | 32,381,440 | 32,381,440 | 1,551,353,335,808 | 7.497519 | 5.117 | 9.3 | 9032.0 | 678,867,456 | 292.7 |

## Paired descriptive contrasts

Negative K−comparator means K has lower validation NLL. n=3 paired seeds; no significance, equivalence, power or generalization claim.

| Pair | Seed | K NLL | Comparator NLL | K − comparator |
|---|---:|---:|---:|---:|
| K-S | 42 | 7.380902 | 7.344828 | +0.036073 |
| K-S | 43 | 7.345804 | 7.345055 | +0.000749 |
| K-S | 44 | 7.398495 | 7.412854 | -0.014359 |
| K-C | 42 | 7.380902 | 7.508320 | -0.127419 |
| K-C | 43 | 7.345804 | 7.456401 | -0.110597 |
| K-C | 44 | 7.398495 | 7.527837 | -0.129343 |

- K−S mean `+0.007488`, median `+0.000749`, range `-0.014359..+0.036073`, sample SD `0.025883`. K was worse on 2/3 seeds and better on 1/3; no clear algorithm signal.
- K−C mean `-0.122453`, median `-0.127419`, range `-0.129343..-0.110597`, sample SD `0.010312`.
- S−C mean `-0.129940`, range `-0.163492..-0.111345`.

## Historical T0027 BF16 context

Matched-seed comparison only; different precision, source revision and time. Not acceptance, equivalence or a precision-effect estimate.

| Arm | Seed | FP32 NLL | T0027 BF16 NLL | FP32 − BF16 |
|---|---:|---:|---:|---:|
| S | 42 | 7.344828 | 7.434416 | -0.089587 |
| S | 43 | 7.345055 | 7.329211 | +0.015845 |
| S | 44 | 7.412854 | 7.478690 | -0.065836 |
| C | 42 | 7.508320 | 7.592448 | -0.084128 |
| C | 43 | 7.456401 | 7.425299 | +0.031102 |
| C | 44 | 7.527837 | 7.523176 | +0.004661 |

## Cost and analytical observations

- FP32 S/K share analytical forward MAC total; C is a separate dense control. Measured step-time means: S `125.305s`, K `24.571s`, C `5.117s`. K/S step-time ratio `0.196×` (K faster); K/C `4.802×` (K slower).
- Mean wall: S `133.7s`, K `31.0s`, C `9.3s`. Peak allocated max S/K/C `1,481,282,048` / `927,680,000` / `678,867,456` bytes.
- Checkpoint writes: 8/run, 72 total; private checkpoints total `2,792,674,947` bytes. No raw checkpoints published.
- These are full 128-update one-pass measurements, not optimized-kernel, sustained-throughput, energy or learning-quality claims.

## Learning signal and QAT readiness

- Mean final NTP training NLL over the first eight updates falls to the last eight in every run: S `8.993` → `6.964` (mean change `-2.029`), K `8.993` → `6.983` (`-2.011`), C `8.916` → `7.088` (`-1.829`). This is a one-pass optimization signal, not convergence or held-test learning.
- The only descriptive algorithm contrast is K−S; it is mixed across seeds and does not establish an algorithm advantage or equivalence. K and S have identical parameter/MAC budgets and FP32 precision; C is a different architecture/implementation control.
- No QAT run was performed. The FP32 chunk64 training path completed three paired seeds with finite full updates, so it is feasible to carry into a separately reviewed QAT task; this evidence is not QAT accuracy, throughput or default-adoption evidence.
- No inferential, power-qualified, optimized-dense or architecture-superiority claim is made.

## Limitations

- Nine synthetic-snapshot runs, three seeds, one pass of 128 updates; no convergence, held-test, generalization, energy or production claim.
- Same snapshot reuse is not independent generalization; S/K share data order but differ in contraction order; C differs in shape/objective implementation.
- T0027 BF16 values are historical context only; precision and time are confounded.
- No QAT, FP8, real-data training, automatic retry, source edits or default adoption.

## Handover

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0035/attempt-001 \
  --task T0035 \
  --source 4f86f62d3fb673483ddf87001331db90e354b985 \
  --checks environment,matched_runs,analysis,handover
```

Validator output is reported in the PR body after exact-head packaging. T0035 remains open; no merge, closure, shared-doc edit or next task.
