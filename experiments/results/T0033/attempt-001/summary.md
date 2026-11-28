# T0033 attempt-001 — opt-in FP32 chunk64 training-path qualification

Task T0033 (IMP+VAL), issue #121, parents #85/#7/#87/#105/#117/#109. Base instruction `04b5a8b8962b40cc19f4afdd8216d6270f7f7f29`; tested implementation `a8df3f0488a3b4987acfbe50e4f1791a8fabf133` on `feat/T0033-gdn-fp32-attempt-001`. UTC 2026-11-28T19:56:29Z to 20:00:21Z. Overall: **passed** — implementation, CPU tests, exact FP32 qualification, packed resume, resource gates and complete-update resource cases passed. No real-data training, BF16/FP8 chunk64, installs, sustained training or default adoption.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `source-files.json`, `preflight.json`, `config-identity.json`, `execution.log` |
| implementation | passed | `source-diff.patch`, `implementation.json`, `config-identity.json`, `summary.md` |
| unit_tests | passed | `unit-tests.txt`, `summary.md`, `execution.log` |
| cuda_qualification | passed | `qualification.json`, `packed-resume-cuda.json`, `execution.log` |
| resources | passed | `resource-gate-512.json`, `resource-gate-2048.json`, six `resource-*.json`, `resources-summary.json` |
| handover | passed | `summary.md`, `execution.log`, `analysis.json`, `probe-failure.json` |

## Implementation

- `delta_chunk64` composes the existing `delta_chunk` oracle over at most 64-token slices with carried FP32 state; the explicit `algorithm='chunk'` oracle is unchanged.
- `RecurrentConfig.gdn_algorithm` is optional and defaults to `scan`; explicit values are `scan|chunk64`.
- `GatedDeltaNet` rejects chunk64 before mutation if autocast is active, activations/weights are not FP32, or carried state/history is not FP32. CLI rejects BF16/FP8 chunk64 and active-autocast requests.
- `RecurrentCache` carries the algorithm; cross-mode cache/checkpoint reuse is rejected before state mutation.
- No parameters or state_dict entries were added. The legacy config file SHA-256 `51c493eb8f118fe1f58fd9a775fc0217f7a6cae0aa74a939d159c17ffbeceac9` matches T0031's `51c493eb…`; explicit configs carry `gdn_algorithm` identity. Changed files: 13 (`source-files.json`, `source-diff.patch`).

## CPU tests

Local CPU command exited 0; the hosted CI gate runs on the PR head: **34 scripts, 40 REP reports**. New `tests/test_gdn_chunk64_fp32.py` is **5/5**: FP64/FP32 component output/state/gradient gates, default/inventory, packed document reset/padding, cache/logical-pass separation, and fail-closed autocast/weight/state/BF16 cases. New `tools/test_gdn_fp32_harness.py` is **2/2**: A0/A2 FP32 CLI scan/chunk 4-vs-2+2 exact resume for model/optimizer/RNG/non-timing history/validation, cross-mode rejection with unchanged checkpoint, and BF16/FP8 chunk64 CLI rejection.

## CUDA qualification

Fresh Nano V32768 R2 A2 B1 seed42 T512, seed43 T512, seed42 T2048; FP32 both arms, complete Muon/AdamW update, natural routing. Identity matches T0031 exactly; loss/tensor metric deltas versus T0031 are `0.0`, routing mismatch is zero, and all T0025 loss/tensor/gradient gates pass.

| Length | Seed | Status | Routing | Max loss abs | Max tensor rel | Gradient global rel | Post-update model rel |
|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 42 | passed | 0/4096 | 9.53674e-07 | 1.0418e-06 | 1.83843e-06 | 0.00020297 |
| 512 | 43 | passed | 0/4096 | 1.90735e-06 | 9.78552e-07 | 1.49121e-06 | 0.000204576 |
| 2048 | 42 | passed | 0/16384 | 0 | 1.71017e-06 | 2.98455e-06 | 0.000194336 |

Post-update model differences are finite and small (about `1.9e-4–2.0e-4` relative); optimizer inventories are finite. No bitwise cross-mode requirement is imposed.

### Packed tiny CUDA resume

Packed synthetic FP32 scan and chunk64 both reproduce 4-step continuous training exactly from 2+2 resume for model, optimizer, RNG, non-timing history and validation. Cross-mode chunk64 checkpoint resumed with scan identity is rejected (`run identity mismatch`) with an unchanged checkpoint SHA.

### Resource numerical gates

T512 and T2048 resource-gate processes each passed **3/3** repeats with full forward/tensor/routing/gradient and complete-update checks before any timing.

## Resources (complete update, qualified FP32 path)

Fresh process/method; fresh identical initialization per repeat; one warmup complete update discarded; three measured synchronized complete updates; diagnostics outside timing. No sustained-throughput or learning claim.

| Length | Method | Policy | Forward median s | Backward median s | Optimizer median s | End-to-end median s | Peak alloc B | Peak reserved B | Labels/s median |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | scan | fp32_scan | 0.377519 | 0.807918 | 0.068141 | 1.253690 | 2,182,651,904 | 2,233,466,880 | 407.6 |
| 512 | chunk64 | fp32_chunk64 | 0.065091 | 0.085143 | 0.068160 | 0.218430 | 1,627,993,088 | 1,681,915,904 | 2339.4 |
| 512 | original_bf16_scan | original_bf16_scan | 0.373063 | 0.802280 | 0.064094 | 1.238666 | 2,053,050,880 | 2,145,386,496 | 412.5 |
| 2048 | scan | fp32_scan | 1.452878 | 3.273496 | 0.068475 | 4.794990 | 7,977,949,696 | 8,218,738,688 | 426.9 |
| 2048 | chunk64 | fp32_chunk64 | 0.150719 | 0.234839 | 0.067285 | 0.452887 | 5,762,677,760 | 6,008,340,480 | 4519.9 |
| 2048 | original_bf16_scan | original_bf16_scan | 1.436053 | 3.208592 | 0.065400 | 4.733520 | 7,214,868,480 | 7,614,758,912 | 432.4 |

FP32 scan/chunk64 end-to-end ratios are `5.740×` at T512 and `10.588×` at T2048. BF16 scan is historical context only, not an equivalence or quality comparison. The FP32 chunk64 path is not adopted as default; no production, throughput, energy or architecture claim is made.

## Limitations and process notes

- Synthetic fixed-seed cases only; no real data, learning, convergence, generalization or production claim.
- FP32-only experimental path; BF16/FP8 chunk64 is rejected, not silently downgraded. Scan remains default.
- Resource timings include forward/loss/backward/optimizer only with no real-data checkpoints; no sustained training claim.
- The first probe invocation failed on a read-only optimizer-inventory tooling key (`KeyError: state`) before any gate result. The probe was corrected, committed as `a8df3f0488a3b4987acfbe50e4f1791a8fabf133`, and all qualification/evidence was rerun from that exact SHA. No source model, data, config, precision policy or gate changed; no automatic retry after a failed gate.
- Changed files and exact source diff are published; no real-data checkpoints, raw tensors or routes are published.

## Handover

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0033/attempt-001 \
  --task T0033 \
  --source a8df3f0488a3b4987acfbe50e4f1791a8fabf133 \
  --checks environment,implementation,unit_tests,cuda_qualification,resources,handover
```

Validator output is reported in the PR body after exact-head packaging. T0033 remains open; no merge, closure, shared-doc edit or next task.
