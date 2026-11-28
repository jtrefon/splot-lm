# T0020 attempt-001 — repeated CUDA training cost and phase baseline

Task T0020 (VAL), issue #90, parents #85/#7/#10/#42. Instruction/main `a771c370efe662f17ddf8c06769cbd06a783388a` (T0020 scheduling amendment merged in #93; T0017 #77 closed, #92 merged). Pinned source `6e8911662ac56df665147cb79e412f0383cf71d8` in a separate clean detached checkout. UTC 2026-11-28T15:52:05Z to 2026-11-28T15:54:55Z. Overall: **passed**.

## Outcome

All 12 required fresh-process cases passed: lengths 512 and 2048, three counterbalanced pairs in orders BF16→FP8, FP8→BF16, BF16→FP8. All preregistered equality, label, finiteness, native-FP8 and CE gates passed. Backward was the dominant measured phase at both lengths. FP8 complete-update times were longer than BF16 in these single-update measurements; this is descriptive only, not a sustained speedup, steady-state throughput or architecture claim.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `source-verification.json`, `execution.log` |
| paired_runs | passed | 12 `case-*.json`, `run-order.json`, `paired-analysis.json`, `metrics.jsonl`, `execution.log` |
| cost_analysis | passed | `paired-analysis.json`, `analyze_cases.py`, `metrics.jsonl`, `summary.md` |
| handover | passed | flat attempt package; validator result in PR body |

## Environment and source

- GPU 0, NVIDIA GeForce RTX 4090, 24564 MiB total / 24077 MiB free at preflight; driver 580.178.04; one owner-selected GPU.
- Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8, RTX 4090 SM89; deterministic algorithms, TF32 off, CUBLAS workspace `:4096:8`.
- Output tmpfs available at capture: 13330108416 B; repo filesystem: 181955522560 B. Every case reported at least 20,480 MiB free before launch (observed 24,077 MiB).
- Source pin `6e8911662ac56df665147cb79e412f0383cf71d8` clean before/after; probe SHA-256 `4e8e96382e01532bf016568cc92af22d5eae8f3e138abf8b0b92c20d19febb5d`; BF16 config `51c493eb8f118fe1f58fd9a775fc0217f7a6cae0aa74a939d159c17ffbeceac9`; FP8 config `5171e50ca004a5b47fd76b3cedaad52ccf174d6b0dd50665b8a17adb5b141160`.
- **FP8 weight reuse is disabled** by the existing probe default (`convert_selected_linears(model)` with no reuse); this was not changed. FP8 cases report zero cache hits/bytes and 298 actual weight quantizations.
- No packages, system changes, real data, checkpoint, Mini, kernel/optimizer changes, background work or retries.

## Actual order and validation

The exact order was 512-p1-BF16, 512-p1-FP8, 512-p2-FP8, 512-p2-BF16, 512-p3-BF16, 512-p3-FP8, then the same sequence at 2048. All 12 exited 0 with `status=passed`.

All cases satisfy: 43,055,076 parameters; expected labels 511/510 (T512) and 2,047/2,046 (T2048); finite losses, logits, hidden, offset logits, gradients/state and FP8 audit; FP32 master/state; identical input and initialization/partition digests within each length and identical initialization/partition across lengths; positive native FP8 dispatch (298 forward and 298 backward calls) and no nonfinite quantization flags.

### Matched pre-update CE differences (BF16 vs FP8)

| Length | final NTP | early NTP | offset2 | total | Gate |
|---:|---:|---:|---:|---:|---|
| 512 | 0.010992050 | 0.007880211 | 0.002113342 | 0.013486862 | passed (component ≤0.5, total ≤0.75) |
| 2048 | 0.000660896 | 0.001372337 | 0.004379272 | 0.000087738 | passed (component ≤0.5, total ≤0.75) |

The three pairs are deterministic repetitions and have identical CE differences.

## Cost results

Median complete-update time with min–max over three pairs:

| Length | Precision | E2E seconds | Forward/loss | Backward | Checks | Optimizer | NTP labels/s | Peak allocated / reserved bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | BF16 | 1.292059586 (1.285945639–1.298393360) | 0.377673131 | 0.810630361 | 0.030232095 | 0.067881468 | 395.492596 | 1762148864 / 1908408320 |
| 512 | FP8 | 1.599761309 (1.582899947–1.651739883) | 0.534788556 | 0.939188797 | 0.053927497 | 0.068808594 | 319.422652 | 1729731584 / 1874853888 |
| 2048 | BF16 | 4.831730738 (4.760163318–4.849120141) | 1.452030888 | 3.278497016 | 0.029944230 | 0.069875948 | 423.657714 | 6717001728 / 7381975040 |
| 2048 | FP8 | 5.166653380 (5.136033584–5.184571693) | 1.623296373 | 3.427242307 | 0.053197517 | 0.069471753 | 396.194567 | 6575379456 / 7189037056 |

Paired FP8/BF16 complete-update ratios and absolute deltas:

| Length | Pair 1 | Pair 2 | Pair 3 | Median ratio (min–max) | Median delta |
|---:|---:|---:|---:|---:|---:|
| 512 | 1.232108 | 1.225098 | 1.284455 | 1.232108 (1.225098–1.284455) | 0.301367949 s |
| 2048 | 1.089158 | 1.062980 | 1.065483 | 1.065483 (1.062980–1.089158) | 0.317533239 s |

Peak memory is single-case CUDA allocator data. Pre-case free was 24,693,506,048 B in every case; free after cleanup was 24592842752 B (T512 BF16) to 24555094016 B (T2048 FP8). Peak reserved was at most 7381975040 B.

### Phase ranking and reconciliation

| Length | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Rank 6 |
|---:|---|---|---|---|---|---|
| 512 | backward 0.875704327 | forward/loss 0.456822036 | optimizer 0.068345031 | checks 0.041980929 | prepare 0.001710390 | transfer 0.000149749 |
| 2048 | backward 3.342020605 | forward/loss 1.533590016 | optimizer 0.069612460 | checks 0.041842937 | prepare 0.001716750 | transfer 0.000155956 |

The named phases sum to the measured end-to-end time within 0.000004355 s; synchronization total is at most 0.000644353 s. Phases are additive within measurement resolution. They are not kernel-level attribution; the prepare phase is larger for FP8 (about 0.0033 s vs 0.00012 s), consistent with FP8 conversion/setup, but no finer attribution is claimed.

## FP8 audit and padding

All FP8 cases report 156 selected modules, 298 native forward calls, 298 native backward calls, and no saturation or nonfinite values. Padding was 1,440 forward rows at T512 and 1,584 at T2048. Weight reuse was disabled, with 298 actual weight quantizations, zero cache hits and zero cache bytes per case. Input/weight/gradient underflow fractions are recorded per case in `paired-analysis.json`; saturation and nonfinite flags are zero/false throughout.

## Interpretation

Observed: backward is the largest phase, followed by forward/loss; optimizer and finite/audit checks are smaller. FP8 complete-update ratios are 1.232× BF16 at T512 and 1.065× at T2048 in these single-update measurements. This does not establish a stable speedup, steady-state throughput, quality difference or architecture superiority; three repeats are descriptive only.

**Next operator-level question:** profile the T2048 backward pass to separate the GDN recurrent scan from attention/logits and expert dispatch. Exactness guard: require pre-update loss and full gradient/state digests equal to the current probe at fixed seed/input, with no model equation, tolerance, precision or optimizer change. No implementation is included here.

## Limitations and execution note

- One measured update per case after an untimed warmup; three repeats are not independent training seeds.
- Synthetic contiguous documents only; no real corpus, checkpoint, Mini, convergence, quality or production-throughput claim.
- Phase timings include explicit synchronization and audit/finite checks; no kernel-level attribution.
- Peak allocated/reserved are single-case CUDA allocator values, not steady-state memory.
- FP8 weight reuse is disabled; no cache-reuse conclusion.
- The first case `512-p1-bf16` completed and passed, but the shell launcher then failed on zsh's reserved `status` variable while writing metadata. Its probe JSON was preserved and not rerun; its exact start time was not captured, so completion is recorded from file mtime `2026-11-28T15:52:28Z`. No command, input, model, precision or gate changed.
- No T4096 or higher length was run; T0020 requires 512 and 2048 only.

## Handover

Flat evidence: 12 sanitized case JSONs, `case-index.json`, `run-order.json`, `paired-analysis.json`, `analyze_cases.py`, environment/source verification, metrics, execution log and summary. No tensors, raw corpus or private paths.

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0020/attempt-001 \
  --task T0020 \
  --source 6e8911662ac56df665147cb79e412f0383cf71d8 \
  --checks environment,paired_runs,cost_analysis,handover
```

T0020 remains open for review; no merge or closure. No further task is started.
