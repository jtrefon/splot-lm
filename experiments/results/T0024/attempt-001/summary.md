# T0024 attempt-001 — chunked GDN scan numerical and resource feasibility

Task T0024 (VAL), issue #102, parents #85/#7/#10/#87. Instruction/main `d5cc33c0ba3df21e4763284cc3a1f4cde08a38b9`; pinned source `6e8911662ac56df665147cb79e412f0383cf71d8` in a clean detached checkout. UTC 2026-11-28T16:38:33Z to 2026-11-28T16:40:52Z. Overall: **passed**.

## Outcome

The existing `delta_chunk` triangular oracle, composed over C16/32/64 slices with carried FP32 state, preserved outputs, final state and all six input gradients against `delta_scan` for every required numerical case: FP64 T33/129, FP32 T129 ordinary and slow-decay, and FP32 T512/T2048 ordinary, seeds 42/43. All **30/30** chunk comparisons passed with 0 failed and 0 skipped; all C16/32/64 regimes were resource-eligible at both T512 and T2048.

Eight CUDA resource cases (scan and C16/32/64 at T512/T2048) passed their pre-timing and per-repeat output/state/gradient gates. In this isolated component measurement, **chunk64 had the lowest median synchronized forward+backward time** at both lengths. This is a component feasibility result only: it does not show that GDN dominates the full model or that a production kernel is faster.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `source-review.json`, `execution.log` |
| numerical | passed | `numerical.json`, `metrics.jsonl`, `execution.log` |
| resource | passed | eight `resource-*.json`, `resource-analysis.json`, `analyze_resources.py`, `metrics.jsonl` |
| handover | passed | flat package; validator result in PR body |

## Environment and source

- GPU 0, NVIDIA GeForce RTX 4090, 24564 MiB total / 24077 MiB free at preflight; driver 580.178.04.
- Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8; deterministic algorithms, TF32 off, CUBLAS workspace `:4096:8`.
- Host RAM available 26777968640 B; scratch free 181499346944 B; limits 8 GiB / 4 GiB exceeded.
- Source pin `6e8911662ac56df665147cb79e412f0383cf71d8` clean before/after; `gdn.py` SHA-256 `46eaade0fc574353a68654b6dfa52f1c93af271a97b1632aa43551ac20e85f7b`; no reference edits.
- Existing `delta_chunk` is an independent dense triangular solve intended for small finite-log-decay oracle use, not a production kernel. It has no built-in chunking; T0024 composes consecutive slices with carried state. Existing tests are small float64/NumPy cases and do not cover FP32 projected-tensor gradients at T512/T2048 or CUDA resource behavior.

## Numerical qualification

Thresholds: FP64 relative L2 and max abs ≤1e-9; FP32 output/state relative L2 ≤1e-4 and max abs ≤1e-4·max(1, reference max abs); FP32 gradients relative L2 ≤5e-4 and max abs ≤1e-5·max(1, reference max abs); denominator max(norm(ref),1e-12). Worst observed FP64 errors were 1.176e-15 relative L2 / 8.327e-17 max abs; worst FP32 errors were 9.406e-07 relative L2 / 1.267e-07 max abs.

Worst output/state errors by case and chunk (all finite, all passed):

| Case | C16 | C32 | C64 |
|---|---|---|---|
| fp64 s42 ordinary T33 | passed rel 4.882e-16 abs 8.674e-18 | passed rel 5.216e-16 abs 1.041e-17 | passed rel 5.021e-16 abs 8.674e-18 |
| fp64 s42 ordinary T129 | passed rel 5.089e-16 abs 1.388e-17 | passed rel 5.405e-16 abs 1.388e-17 | passed rel 6.101e-16 abs 1.561e-17 |
| fp32 s42 ordinary T129 | passed rel 2.733e-07 abs 6.519e-09 | passed rel 2.847e-07 abs 5.588e-09 | passed rel 3.107e-07 abs 5.588e-09 |
| fp32 s42 ordinary T512 | passed rel 2.724e-07 abs 6.519e-09 | passed rel 2.857e-07 abs 7.451e-09 | passed rel 3.098e-07 abs 1.024e-08 |
| fp32 s42 ordinary T2048 | passed rel 2.761e-07 abs 7.451e-09 | passed rel 2.885e-07 abs 6.519e-09 | passed rel 3.134e-07 abs 1.024e-08 |
| fp32 s43 ordinary T129 | passed rel 2.726e-07 abs 5.588e-09 | passed rel 2.847e-07 abs 6.519e-09 | passed rel 3.082e-07 abs 6.519e-09 |
| fp32 s43 ordinary T512 | passed rel 2.746e-07 abs 6.519e-09 | passed rel 2.864e-07 abs 6.519e-09 | passed rel 3.124e-07 abs 7.451e-09 |
| fp32 s43 ordinary T2048 | passed rel 2.749e-07 abs 7.451e-09 | passed rel 2.867e-07 abs 7.916e-09 | passed rel 3.129e-07 abs 7.451e-09 |
| fp32 s42 slow T129 | passed rel 3.433e-07 abs 1.490e-08 | passed rel 3.551e-07 abs 1.118e-08 | passed rel 3.844e-07 abs 1.490e-08 |
| fp32 s43 slow T129 | passed rel 3.468e-07 abs 1.490e-08 | passed rel 3.639e-07 abs 1.490e-08 | passed rel 3.881e-07 abs 1.490e-08 |

Worst gradient errors by case and chunk (all six inputs, all finite, all passed):

| Case | C16 | C32 | C64 |
|---|---|---|---|
| fp64 s42 ordinary T33 | passed rel 1.091e-15 abs 1.429e-21 | passed rel 1.175e-15 abs 1.694e-21 | passed rel 1.176e-15 abs 1.376e-21 |
| fp64 s42 ordinary T129 | passed rel 6.429e-16 abs 1.489e-23 | passed rel 7.018e-16 abs 1.084e-19 | passed rel 9.004e-16 abs 1.220e-19 |
| fp32 s42 ordinary T129 | passed rel 3.837e-07 abs 8.882e-15 | passed rel 3.985e-07 abs 8.882e-15 | passed rel 4.449e-07 abs 1.066e-14 |
| fp32 s42 ordinary T512 | passed rel 3.467e-07 abs 2.665e-15 | passed rel 3.685e-07 abs 3.553e-15 | passed rel 4.262e-07 abs 5.329e-15 |
| fp32 s42 ordinary T2048 | passed rel 3.338e-07 abs 5.551e-16 | passed rel 3.538e-07 abs 5.551e-16 | passed rel 4.048e-07 abs 6.661e-16 |
| fp32 s43 ordinary T129 | passed rel 3.451e-07 abs 8.882e-15 | passed rel 3.701e-07 abs 7.994e-15 | passed rel 4.193e-07 abs 1.688e-14 |
| fp32 s43 ordinary T512 | passed rel 3.288e-07 abs 2.665e-15 | passed rel 3.588e-07 abs 3.553e-15 | passed rel 4.104e-07 abs 3.109e-15 |
| fp32 s43 ordinary T2048 | passed rel 3.688e-07 abs 6.661e-16 | passed rel 3.784e-07 abs 6.661e-16 | passed rel 4.318e-07 abs 8.882e-16 |
| fp32 s42 slow T129 | passed rel 8.244e-07 abs 5.684e-12 | passed rel 8.404e-07 abs 5.002e-12 | passed rel 8.873e-07 abs 5.457e-12 |
| fp32 s43 slow T129 | passed rel 8.763e-07 abs 5.457e-12 | passed rel 8.940e-07 abs 5.457e-12 | passed rel 9.406e-07 abs 5.457e-12 |

Eligibility matrix: T512 C16/C32/C64 = passed/passed/passed; T2048 = passed/passed/passed. No candidate was rejected; no dependent length was skipped.

## Resource measurements

Fresh process per method/length; one warmup and three measured synchronized forward/backward repeats with fresh independent leaf clones; no profiling or compilation. Medians with the peak-memory range:

| T | Method | Forward s | Backward s | Total s | Peak allocated B | Peak reserved B (min–max) | Increment over input baseline alloc / reserved B |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | scan | 0.0529156090342 | 0.121315973985 | 0.17468823801 | 178603008 | 182452224 (182452224–182452224) | 176120832 / 178257920 |
| 512 | chunk16 | 0.0122416279628 | 0.0261606409913 | 0.0384045839892 | 84961280 | 88080384 (88080384–182452224) | 82479104 / 83886080 |
| 512 | chunk32 | 0.00659470196115 | 0.0135438879952 | 0.0201385899563 | 84403200 | 88080384 (88080384–182452224) | 81921024 / 83886080 |
| 512 | chunk64 | 0.00431600201409 | 0.00741150905378 | 0.0117275110679 | 86356992 | 90177536 (90177536–182452224) | 83874816 / 85983232 |
| 2048 | scan | 0.214634081989 | 0.519956279022 | 0.734590361011 | 512197632 | 524288000 (524288000–524288000) | 502563840 / 501219328 |
| 2048 | chunk16 | 0.0473838030011 | 0.105858114024 | 0.154215896036 | 139003904 | 148897792 (148897792–524288000) | 129370112 / 125829120 |
| 2048 | chunk32 | 0.0248472579988 | 0.0532230790122 | 0.078070337011 | 137752576 | 146800640 (146800640–524288000) | 128118784 / 123731968 |
| 2048 | chunk64 | 0.0165759209776 | 0.0299750449485 | 0.0464975289651 | 144189440 | 155189248 (155189248–524288000) | 134555648 / 132120576 |

`scan` vs chunk component ratios (scan / chunk; >1 means chunk lower in this isolated timing):

| T | Method | Forward | Backward | Total | Peak allocated |
|---:|---|---:|---:|---:|---:|
| 512 | chunk16 | 4.323× | 4.637× | 4.549× | 2.102× |
| 512 | chunk32 | 8.024× | 8.957× | 8.674× | 2.116× |
| 512 | chunk64 | 12.260× | 16.369× | 14.896× | 2.068× |
| 2048 | chunk16 | 4.530× | 4.912× | 4.763× | 3.685× |
| 2048 | chunk32 | 8.638× | 9.769× | 9.409× | 3.718× |
| 2048 | chunk64 | 12.949× | 17.346× | 15.798× | 3.552× |

Chunk64 median total was 0.0117275110679 s vs scan 0.17468823801 s at T512 and 0.0464975289651 s vs 0.734590361011 s at T2048. These are isolated component times; the scan path is a Python recurrence and the chunk path is multiple triangular solves, so launch/Python overhead and algorithmic structure differ. No full-model or production-kernel speed claim is made.

All resource gates passed before and after timing; repeat digests matched the gate digest; no autograd graph was reused. A pre-timing reference scan can leave allocator cache reserved: absolute peak allocated is stable across repeats, while first-repeat peak reserved may include retained cache and repeats 2–3 show the method-specific reserved peak. Both absolute peaks and increments over the recorded input baseline are reported; no retry was performed.

## Next integration question

Integrate the numerically qualified chunk composition behind an opt-in GDN algorithm flag in a separate reviewed IMP and compare full Nano forward/backward at T512/T2048 against scan with exact full-model loss/gradient gates and full-model memory/timing. This component result does not justify a production kernel or full-model speed claim by itself.

## Limitations

- Two lengths, one seed and three repeats for resource timing; descriptive component data only.
- The existing `delta_chunk` is a small finite-log-decay oracle with O(C²) triangular solves per chunk, not a production kernel.
- Resource timing includes Python/launch/allocator effects; no kernel-level attribution or steady-state throughput.
- Peak reserved memory has a first-repeat allocator-cache artifact; allocated peaks are stable.
- No full-model, quality, training, sustained-throughput or architecture-superiority claim.
- No reference-source edits, packages, real data, checkpoints, Mini, background work or automatic retries.

## Handover

`gdn_numerical.py` and `gdn_resource.py` are the sanitized drivers; `numerical.json` contains every case/chunk metric and gradient component; eight `resource-*.json` contain per-repeat synchronized times, peaks and gates; `resource-analysis.json` contains the reproducible aggregate. No tensors, raw corpus or private paths are published.

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0024/attempt-001 \
  --task T0024 \
  --source 6e8911662ac56df665147cb79e412f0383cf71d8 \
  --checks environment,numerical,resource,handover
```

T0024 remains open for review; no merge or closure. No further task is started.
