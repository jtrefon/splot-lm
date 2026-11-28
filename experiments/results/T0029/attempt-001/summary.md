# T0029 attempt-001 — GDN mixed-precision and sparse-routing divergence localization

Task T0029 (VAL), issue #113, parents #85/#7/#87/#105/#109. Instructions `edd2036e284a4f7d1d5904b5284971d33c5c0d39`; exact rejected T0025 source `7dac49e083eae0068ff361e7a81b3a4f41055b3f` in a clean detached worktree; evidence branch `research/T0029-gdn-localization-attempt-001`. UTC 2026-11-28T18:26:05Z to 2026-11-28T18:28:28Z. Overall: **passed** — the bounded forward-only diagnostic completed and reproduced T0025 exactly. No model source change, backward pass, optimizer update, training, checkpoint, actual data, installation or candidate adoption.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `source-identity.json`, `preflight.json`, `execution.log` |
| reproduction | passed | `reproduction.json`, `diagnostic-report.json`, `execution.log` |
| localization | passed | `localization.json`, `routing-corrected.json`, `analysis.json`, `diagnostic-report.json` |
| handover | passed | `summary.md`, `execution.log`, `diagnostic-report.json` |

## Source and preflight

- Instruction main `edd2036e284a4f7d1d5904b5284971d33c5c0d39`; exact evaluated source `7dac49e083eae0068ff361e7a81b3a4f41055b3f` in a detached worktree. Source clean before/after, no source edits.
- Config SHA `51c493eb8f118fe1f58fd9a775fc0217f7a6cae0aa74a939d159c17ffbeceac9`; T0025 reference SHA `7af225f12c1353ff6621c13a13f22c983d1c0bca3ba379b23874181fb58bcd4d`; input `c0aa82fc0906b8a1437b5328f8d2396ab4f79e7b1528e8be399ea107da899270`; initialization `ba9bf9d8472a9b940a416e91f3123ef4163913b7b334291540cb3b321fbaca5b`; partition `1bed555467ef9a955079b8260636971dafa5b0b308962e9a18e23a5ffa7bf9ed`; 43,055,076 parameters. All match T0025.
- GPU 0 RTX 4090, 24,077 MiB free, driver 580.178.04; Python 3.12.7/torch 2.10.0+cu128/CUDA 12.8; deterministic algorithms, TF32 off, CUBLAS workspace `:4096:8`.
- Forward-only: zero backward calls, zero optimizer steps, zero checkpoints, zero training runs, zero test evaluations.

## Exact reproduction

Scan losses are exactly the T0025 values; chunk64 losses differ from the published chunk64 values by 0.0; per-key routing mismatch counts and all tensor metrics are exact. Loss components (scan, chunk64, reproduced chunk64 minus published chunk64):

| Component | Scan | Chunk64 | Chunk delta vs published |
|---|---:|---:|---:|
| total | 15.7475681305 | 15.7477397919 | 0 |
| final_ntp | 10.4938745499 | 10.4939994812 | 0 |
| early_ntp | 10.484287262 | 10.4843463898 | 0 |
| offset2 | 10.4876174927 | 10.4877519608 | 0 |
| balance | 1.07171320915 | 1.07164931297 | 0 |

## Corrected routing aggregation

T0025's published `by_pass` grouped by `key[0]` stage `C`; this report uses `key[1]` logical pass and token-weighted aggregation. The corrected overall disagreement is **34/4096 = 0.830078%**.

| Logical pass | Mismatch / tokens | Fraction |
|---|---:|---:|
| 1 | 5/2048 | 0.244141% |
| 2 | 29/2048 | 1.416016% |

| Physical pool | Mismatch / tokens | Fraction |
|---|---:|---:|
| 0 | 4/1024 | 0.390625% |
| 1 | 9/1024 | 0.878906% |
| 2 | 13/1024 | 1.269531% |
| 3 | 8/1024 | 0.781250% |

First top2 route-set mismatch: **block 2, logical pass 1**, 3/512 positions. At those positions, scan top1 remains rank 0 in the candidate; scan top2 becomes rank 2 (third) for all three. The scan top2-minus-top3 margin median is `0.000651509` (min `5.73248e-05`); the candidate margin median is `0.00102213`. The top1-minus-top2 margins are much larger (scan median `0.462643`), so these flips occur at the second-versus-third expert boundary.

## Localization

### Same-input FP32 GDN

All six captured block/pass pairs compare `delta_scan` vs `delta_chunk64` on the **same captured FP32 projected inputs and state**, with autocast disabled:

- Worst output/state relative L2: `3.69601e-07`; worst max abs: `1.19209e-07`. All calls pass the `1e-4` relative / `1e-4*max(1,refmax)` gate.
- FP64 small oracle: **passed** at `1e-9`.
- First non-zero local GDN output difference: block `0`, logical pass `1`, rel `3.6911e-07`.

### Recurrence and BF16 boundary growth

The no-cache forward supplies zero GDN prefix state at each logical pass, so the captured state input is zero at pass 2; within-pass final-state divergence is reported separately. Per pass maxima:

| Logical pass | GDN input max rel | Delta output max rel | Delta final-state max rel | Router-logit max rel |
|---|---:|---:|---:|---:|
| 1 | 0.00197297 | 0.00705795 | 0.00589261 | 0.0156876 |
| 2 | 0.0277091 | 0.047916 | 0.0657868 | 0.0352946 |

Observed first boundaries:

- Local GDN output difference: block `0`/pass `1` (rel `3.6911e-07`).
- First BF16 pre-cast difference: block `0`/pass `1` (rel `3.46154e-07`; 68 BF16-representation differences at that boundary).
- First output-projection difference: block `0`/pass `1` (rel `7.58343e-05`, max abs `0.000488281`).
- First propagated GDN input difference: block `1`/pass `1` (rel `0.000215899`).
- First router input/logit difference: block `0`/pass `1` (router input rel `3.13669e-05`; logits rel `2.98451e-05`).
- First top2 route-set mismatch: block `2`/pass `1` (after the local and cast differences above).

### FP32 full forward

With autocast disabled and fresh identical initialization/input, scan vs chunk64 full forward has tensor max relative L2 `1.0418e-06` and max abs `3.15905e-06`; routing top2 disagreement is **0.000000%**. Loss absolute deltas: total `0`, final `0`, early `9.53674e-07`, offset2 `0`, balance `1.19209e-07`. The local contraction-order difference is therefore tiny without BF16 casts.

### Teacher-forced scan routes (diagnostic)

Using only a scratch wrapper and the same scan-selected top2 IDs, with chunk router probabilities renormalized on those IDs:

- Natural chunk logits max rel-L2: `0.0322755`.
- Fixed-route logits max rel-L2: `0.00859061`; ratio `0.2662`.
- Fixed-route tensor deltas: max rel `0.00919627`, max abs `0.0234375`; all fall within the existing T0025 tensor gates.
- Selected route sets equal scan by construction (`True`); this is **not** production routing and **not** an exact-equivalence claim. The fixed-route loss deltas vs scan are small (total `-0.00022316`).

| Fixed-route tensor | Rel L2 | Max abs | Reference max abs |
|---|---:|---:|---:|
| hidden_1 | 0.00450827 | 0.00399435 | 0.781867 |
| hidden_2 | 0.00779959 | 0.00892383 | 1.09392 |
| logits_1 | 0.00575808 | 0.015625 | 2.20312 |
| logits_2 | 0.00859061 | 0.0234375 | 2.14062 |
| offset2_1 | 0.00631802 | 0.015625 | 2.125 |
| offset2_2 | 0.00919627 | 0.0234375 | 2.09375 |

## Interpretation

Same-input FP32 GDN comparisons are the local contraction-order control; the full-model BF16 comparison includes recurrence, BF16 boundary casts and sparse routing. First top2 route-set mismatch is at block 2, logical pass 1, with 3 of 512 valid positions. Teacher-forced scan-selected routes changed the maximum logits rel-L2 from 0.0322755 to 0.00859061 (ratio 0.266); this is an intervention result, not proof of a first cause.

The observations are consistent with a chain in which a tiny FP32 contraction-order difference first crosses BF16 output-projection/residual/router boundaries, then sparse top2 route flips amplify the hidden/logit divergence across depth. This is not proof that route flips are the first cause: the local FP32 GDN difference and the first BF16 cast difference occur before the first route flip. FP32 full-model routing is exact, while every observed route flip occurs in the BF16 comparison; this is an observational distinction, not proof that a particular cast caused a flip. The fixed-route intervention reduces the divergence but does not remove it; it is a diagnostic, not candidate routing.

Prospective numerical gate for any future candidate (proposed, not applied here): same-input FP32 GDN output/state must pass `1e-4` relative / `1e-4*max(1,refmax)` at every block/pass; token-weighted top2 route mismatch must be zero at every logical pass; natural BF16 hidden/logit/offset2 must meet the existing T0025 tensor gates without relaxation; and both natural and teacher-forced-route paths must meet those gates before T2048/resource qualification.

## Process notes

Three read-only driver defects stopped pre-result aggregation. They are preserved in `driver-failure-1.json` through `driver-failure-3.json` and summarized in `driver-failures.json`: a route-dict shape error, a raw-vs-reshaped `z` shape error, and a missing fourth-core-block router capture. No source, data, config, seed, tolerance or numerical path was changed. The fourth invocation completed; no automatic retry was used and no numerical gate was rerun after a failed gate.

## Limitations

- One synthetic contiguous T512 invocation, one seed, no backward/optimizer step/training; no learning, convergence, quality or production claim.
- No actual data, checkpoints, test evaluation, timing/throughput or energy measurement.
- Route analysis is top2-set/aggregate only; no raw IDs, logits or tensors are published.
- BF16 cast accounting is observational; it does not prove a specific rounding event caused a specific route flip.
- The fixed-route intervention is a teacher-forced diagnostic with scan-selected IDs; it is not production routing and does not establish exact equivalence.

## Handover

Evidence contains the sanitized diagnostic report, exact reproduction metrics, corrected per-pass/pool routing aggregates, same-input FP32/FP64 comparisons, per-pass recurrence/cast boundary statistics, FP32 full-forward comparison, teacher-forced-route diagnostic, preserved pre-result failures and summary. No raw tensors, token IDs, logits, private paths or checkpoints are published. T0029 remains open; no merge, closure, shared-doc edit or next task.

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0029/attempt-001 \
  --task T0029 \
  --source 7dac49e083eae0068ff361e7a81b3a4f41055b3f \
  --checks environment,reproduction,localization,handover
```

Validator output is reported in the PR body after exact-head packaging.
