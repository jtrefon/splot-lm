# T0031 attempt-001 — matched-precision GDN scan/chunk64 feasibility

Task T0031 (VAL), issue #117, parents #85/#7/#87/#105/#113/#109. Instructions `145db440cf60bcd575226b475952f2d7129f09c1`; exact rejected T0025 source `7dac49e083eae0068ff361e7a81b3a4f41055b3f` in a clean detached worktree; branch `research/T0031-gdn-precision-attempt-001`. UTC 2026-11-28T19:19:46Z to 19:22:32Z. Overall: **passed** — reproduction exact, P1/P2 rejected, P3 full FP32 selected, all three forward/backward qualification cases passed, and all six resource processes passed. No source change, optimizer update, training, checkpoint, actual data, install or adoption.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `source-identity.json`, `preflight.json`, `execution.log` |
| reproduction | passed | `reproduction.json`, `precision-screen.json`, `execution.log` |
| precision_screen | passed | `precision-screen.json`, `analysis.json`, `execution.log` |
| forward_backward | passed | `forward-backward.json`, `analysis.json`, `execution.log` |
| resources | passed | `resources-summary.json`, six `resource-*.json`, `execution.log` |
| handover | passed | `summary.md`, `execution.log`, `precision-screen.json`, `forward-backward.json`, `resources-summary.json` |

## Source and preflight

- Instruction main `145db440cf60bcd575226b475952f2d7129f09c1`; exact source `7dac49e083eae0068ff361e7a81b3a4f41055b3f`; source clean before/after; no source edits.
- Config `51c493eb8f118fe1f58fd9a775fc0217f7a6cae0aa74a939d159c17ffbeceac9`; T0025 reference `7af225f12c1353ff6621c13a13f22c983d1c0bca3ba379b23874181fb58bcd4d`; input `c0aa82fc0906b8a1437b5328f8d2396ab4f79e7b1528e8be399ea107da899270`; initialization `ba9bf9d8472a9b940a416e91f3123ef4163913b7b334291540cb3b321fbaca5b`; partition `1bed555467ef9a955079b8260636971dafa5b0b308962e9a18e23a5ffa7bf9ed`; 43,055,076 parameters. All match T0025.
- GPU 0 RTX 4090, 24,077 MiB free, driver 580.178.04; Python 3.12.7/torch 2.10.0+cu128/CUDA 12.8; deterministic algorithms, TF32 off, CUBLAS workspace `:4096:8`.
- No optimizer step, training, checkpoint, actual data, install or test evaluation.

## Exact reproduction

Original BF16 T512 seed42 scan/chunk64: identity exact, scan losses exact, chunk64 published loss delta `0.0`, per-key routing counts exact, tensor metrics within `1e-6`. Corrected routing disagreement is `0.830078%` (`34/4096`). No teacher-forced routing was used.

## Precision screen (T512 seed42, natural routes)

| Policy | Within-policy | Routing mismatch | Max loss abs | Max tensor rel | Max abs / scaled limit | Local FP32 | Outcome |
|---|---|---:|---:|---:|---:|---|---|
| P1 | failed | 43/4096 | 0.00132179 | 0.0386015 | 0.196429 | passed |
| P2 | failed | 23/4096 | 0.000645638 | 0.0268881 | 0.139447 | passed |
| P3 | passed | 0/4096 | 9.53674e-07 | 1.0418e-06 | 1.48535e-06 | passed |

- **P1 (FP32 GDN output-projection island only):** rejected — tensor and routing gates failed. Max tensor rel `0.0386015`, max abs/ratio `0.196429`, routing `43/4096`; local same-input FP32 passed.
- **P2 (full FP32 GDN run island):** rejected — tensor and routing gates failed. Max tensor rel `0.0268881`, max abs/ratio `0.139447`, routing `23/4096`; local same-input FP32 passed.
- **P3 (full FP32 control):** selected — all within-policy gates passed, routing `0/4096`, max tensor rel `1.0418e-06`, max loss abs `9.53674e-07`.

Actual GDN boundary dtypes were FP32 in P1/P2 because the surrounding residual path kept hidden tensors FP32; the wrappers are the declared FP32 compute islands and report actual dtypes. They did not silently bypass an original cast. P3 is a full FP32 control, **not** a BF16 replacement.

Policy scan versus original BF16 scan, reported separately (precision is a confound, not acceptance): P1 routing `112/4096`, total loss delta `0.000893593`, max tensor rel `0.0720737`; P2 `161/4096`, `0.00211811`, `0.0876014`; P3 `199/4096`, `0.000759125`, `0.0993951`.

## P3 forward/backward qualification

| Length | Seed | Status | Routing mismatch | Max loss abs | Gradient global rel | Inventory same | Per-tensor |
|---:|---:|---|---:|---:|---:|---|---|
| 512 | 42 | passed | 0/4096 | 9.53674e-07 | 1.83843e-06 | True | True |
| 512 | 43 | passed | 0/4096 | 1.90735e-06 | 1.49121e-06 | True | True |
| 2048 | 42 | passed | 0/16384 | 0 | 2.98455e-06 | True | True |

All three cases passed. Named gradient inventory is equal, all gradients finite, global relative L2 is far below `0.05`, and every per-tensor gate passes. Gradient digests are in `forward-backward.json`; no raw gradients are published.

## Resources (qualified P3 only)

Fresh process per method/length; warmup plus three synchronized forward/loss/backward repeats; no optimizer update. Original BF16 scan is a separate precision-policy context and not a quality-matched comparison.

| Length | Method | Policy | Forward median (min–max) s | Backward median (min–max) s | Peak allocated B | Peak reserved B | Baseline B |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | chunk64 | P3 | 0.054630 (0.054555–0.064657) | 0.082546 (0.082421–0.082857) | 1,401,839,104 | 1,444,937,728 | 205,799,424 |
| 512 | original_bf16_scan | original_bf16 | 0.340041 (0.339822–0.365379) | 0.798228 (0.797192–0.801107) | 1,831,237,632 | 1,910,505,472 | 205,799,424 |
| 512 | scan | P3 | 0.337645 (0.337124–0.365699) | 0.793368 (0.792238–0.795114) | 1,956,497,920 | 1,996,488,704 | 205,799,424 |
| 2048 | chunk64 | P3 | 0.141374 (0.141027–0.159319) | 0.238453 (0.235843–0.239351) | 5,543,209,472 | 6,012,534,784 | 205,862,400 |
| 2048 | original_bf16_scan | original_bf16 | 1.367020 (1.356426–1.435639) | 3.262771 (3.256357–3.294828) | 6,991,203,840 | 7,390,363,648 | 205,862,400 |
| 2048 | scan | P3 | 1.371709 (1.368617–1.434631) | 3.259341 (3.244127–3.261998) | 7,758,296,576 | 8,220,835,840 | 205,862,400 |

P3 chunk64 was faster than P3 scan in this isolated forward/loss/backward measurement, but P3 is a full FP32 control, not a BF16 replacement. No default adoption, throughput or sustained-training claim is made. Cross-policy cost/quality is ambiguous because precision changed.

## Interpretation and limitations

- The precision screen rejects the two narrower islands under the natural-route, same-policy tensor and routing gates. The selected P3 control passes within-policy and all three forward/backward cases.
- P3 differs from the original BF16 scan in precision policy; the `199/4096` routing difference and tensor deltas are precision-confounded and are not equivalence evidence.
- One synthetic fixed-seed case per length; no actual data, learning, convergence, generalization, production or optimizer-update claim.
- P1/P2 failures are preserved; no fixed-route rescue, threshold relaxation or dependent resource timing for rejected policies.
- Two resource shell invocations exited before model construction due to command-line argument errors; no measurement ran. Six successful explicit invocations are recorded; no automatic retry.

## Handover

Evidence contains the exact reproduction, full policy screen, rejected P1/P2 gates, selected P3 forward/backward qualification, six resource reports, sanitized wrapper/driver, source identity and execution log. No raw tensors, token IDs, routes, private paths or checkpoints are published. T0031 remains open; no merge, closure, shared-doc edit or next task.

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0031/attempt-001 \
  --task T0031 \
  --source 7dac49e083eae0068ff361e7a81b3a4f41055b3f \
  --checks environment,reproduction,precision_screen,forward_backward,resources,handover
```

Validator output is reported in the PR body after exact-head packaging.
