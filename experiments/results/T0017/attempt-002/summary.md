# T0017 attempt-002 — first real-data Nano FP8+Muon pilot

Task T0017 (IMP), issue #77, parents #7/#10/#42. Continuation `experiments/tasks/T0017-continuation.md` merged in PR #82 at `b91a68843b25624abfdc11c6eb4978d3e4c73e39`; instruction/base/tested source `d3752eeb77541866e2332ed238def0c4bac50549`; minimum reviewed base `a93141e3160c0e9bb845d5b0a64df69c0e97157b`. UTC 2026-11-28T13:58:59Z to 2026-11-28T14:04:59Z for the measured run, with evidence assembly afterward. Overall: **passed**.

## Outcome

The exact T0016 `splot-pilot-blocks-v1` snapshot was regenerated locally from the committed registry and official sources and matched every accepted identity byte-for-byte: manifest `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55`, source snapshot `c84bd8a6e94c5d0f4d429e44a43c7bffd9a6dfb9a263b9a3440126e23b4aa087`, tokenizer `104d3927bb55cdd2a911dbe21c8e755264756cedb5663bd39365c3d4cba3864b`, logical `70ba8d35d2ae334a8601f6389dc39208f3636937cb9dbfda639ead7ab30c052c`; five files, 2,727,588 bytes. The missing cache-on four-step continuous versus 2+2 resume/RNG gate passed exactly, including checkpoint absence of cache state and cross-mode rejection. The fixed 64-step Nano R2 A2 T512 B1 FP8+Muon seed-42 pilot then completed with exit 0 in 89 s: 43,055,076 parameters, 23,598 valid / 23,534 NTP / 23,470 offset2 labels from 64 distinct train blocks, 5,250 validation NTP, all finite, native FP8 dispatch, 8,097 actual weight quantizations and 6,652 cache hits (reuse ratio 0.4510), and checkpoints at steps 16/32/48/64. Validation NLL was 7.744022 and PPL 2307.735.

This is a first real-data architecture sanity run, not substantive pretraining. It is one seed, one configuration and 64 steps, with no loss-improvement threshold, tuning, test evaluation or quality/convergence/throughput claim.

## Checks

| Check | Status | Evidence | Key observation |
|---|---|---|---|
| environment | passed | `environment.json`, `execution.log` | Clean tested `d3752eeb77541866e2332ed238def0c4bac50549`; Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8, RTX 4090, driver 580.178.04; 24,077 MiB free; deterministic, TF32 off; 14 GiB output space. |
| implementation | passed | `data-identity.json`, `dataset-build-summary.json`, `dataset-verify.json`, `tokenizer-environment.json`, `real-pilot-run.json`, `execution.log` | No source changes; accepted attempt-001 implementation remains ancestor. Regenerated dataset matched all pinned hashes exactly; tokenizer 0.23.3 used in an isolated outside-checkout venv; no training-environment change. |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` | 27 CPU scripts, 231 tests, 0 failures, 0 errors, 0 skips, 21 REP reports. |
| cuda_qualification | passed | `resume-gate.json`, `resume-compare.json`, `cross-mode.json`, `fp8-focused-cuda.txt`, `execution.log` | Focused CUDA 4/4; 4-step continuous vs 2+2 resume exact model/optimizer/RNG/history/validation at 179,952 parameters; 156 quantizations and 142/204/139/204 cache hits per step; no cache in checkpoint; cross-mode rejected with unchanged checkpoint. |
| real_pilot | passed | `real-pilot-run.json`, `real-pilot-audit.json`, `loss-history.json`, `checkpoint-summary.json`, `metrics.jsonl`, `execution.log` | 64 steps, 64 distinct blocks, 23,598 valid / 23,534 NTP / 23,470 offset2, 5,250 validation NTP, 43,055,076 parameters, all finite, native dispatch, 8,097 quantizations / 6,652 hits, four checkpoints; no test evaluation. |
| handover | passed | `summary.md`, `execution.log`, `data-identity.json` | Flat attempt-002 package; exact REP validator reports `passed`; attempt-001 preserved. |

## Data identity and regeneration

The five-file dataset directory is local-only and not in Git. Regeneration used the accepted producer and frozen registry `configs/data/pilot-sources-v0.2.json`, with official permitted HTTPS sources only:

| Item | Observed | Limit |
|---|---:|---:|
| Network bytes | 11,204,726 | 52,428,800 |
| Extracted UTF-8 bytes | 8,987,423 | 10,485,760 |
| Dataset files | 5 | 5 |
| Dataset bytes | 2,727,588 | — |
| Train / validation / test blocks | 128 / 16 / 16 | 128 / 16 / 16 |
| Content / valid / NTP / offset2 tokens | 56,201 / 56,521 / 56,361 / 56,201 | — |

| Identity | SHA-256 |
|---|---|
| Manifest (4,674 B) | `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55` |
| Source snapshot | `c84bd8a6e94c5d0f4d429e44a43c7bffd9a6dfb9a263b9a3440126e23b4aa087` |
| Tokenizer (`tokenizer.json`, 708,869 B) | `104d3927bb55cdd2a911dbe21c8e755264756cedb5663bd39365c3d4cba3864b` |
| Logical concatenation | `70ba8d35d2ae334a8601f6389dc39208f3636937cb9dbfda639ead7ab30c052c` |

The regenerated `manifest.json` bytes equal the accepted `experiments/results/T0016/attempt-001/manifest.json` bytes. Independent loader validation confirmed canonical bytes, all source records, split disjointness, exact masks/BOS/EOS/positions/labels, 160 globally unique document segments, and test integrity validation only. `tokenizers==0.23.3` was installed offline in an isolated venv from the local uv cache of the official PyPI abi3 wheel; no network package download and no change to the cu128 training environment. The local tokenizer import and a fixed-string encode/decode round-trip passed.

## Missing resume/RNG gate

The gate used `configs/experiments/splot-smoke-synthetic-fp8-muon.json` plus only `training.fp8_weight_reuse=true`, the same temporary outside-checkout file for both runs. Original config SHA-256 `370fb892a514f7bc129490112eba35a22d30f2b4f5bc2236d9fb4221f89642be`; temporary config SHA-256 `7c9e604bd47a0391efaac449ac2d8180b428a81470e8f9813b8c6acbc2a6dfd5`; 179,952 parameters, four steps, interval 1.

| Run | Steps | Checkpoint writes | Cache quantizations / hits per step |
|---|---:|---:|---|
| Continuous | 4 | steps 1,2,3,4 | 156 / 142, 156 / 204, 156 / 139, 156 / 204 |
| 2-step then resume to 4 | 4 | steps 3,4 after initial 1,2 | same per-step values, exact equality |

`tools/rep/compare_resume.py --steps 4 --parameters 179952` passed: model, both optimizer states, RNG, identity, non-timing history and validation exact. The checkpoints contain only `history`, `identity`, `model`, `optimizer` and `rng`; no quantized-weight cache. Resuming the cache-on checkpoint with the reuse-disabled config exited 1 with `ValueError: run identity mismatch`; checkpoint and run JSON hashes were unchanged. Focused CUDA tests were 4/4 with no skips.

## Real pilot

Exact command (local paths replaced by placeholders):

```bash
PYTHONPATH=src:tools CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
  timeout --kill-after=15s 1800s python3 tools/experiment.py \
  --config configs/experiments/splot-nano-pilot-fp8-muon.json \
  --dataset-root <local-verified-dataset> \
  --manifest-sha256 dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55 \
  --source-snapshot-sha256 c84bd8a6e94c5d0f4d429e44a43c7bffd9a6dfb9a263b9a3440126e23b4aa087 \
  --out <local-runs>/real-pilot --steps 64 --device cuda
```

Exit 0, wall 89 s, UTC 2026-11-28T14:03:30Z to 2026-11-28T14:04:59Z. Fixed settings: Nano R2 A2, B1, T512, FP8+Muon, seed 42, LR 0.003, checkpoint interval 16, weight reuse true. Model vocabulary 32,768; actual tokenizer vocabulary 16,384; 16,384 unused logits; no remapping.

| Observation | Value |
|---|---:|
| Parameters | 43,055,076 |
| Steps | 64 |
| Distinct train blocks / repeats | 64 / 0 |
| Valid tokens / NTP / offset2 | 23,598 / 23,534 / 23,470 |
| Padding tokens / positions | 9,170 / 32,768 |
| Weighted final / early / offset2 CE | 7.934573370912 / 7.923808572124 / 8.097908486536 |
| Validation NTP / NLL / PPL | 5,250 / 7.744021716890 / 2307.734801253827 |
| Actual weight quantizations / cache hits | 8,097 / 6,652 |
| Cache reuse ratio / max cache bytes | 0.451013628043 / 30,278,256 |
| Native FP8 forward / backward calls | 14,749 / 14,749 |
| Peak allocated / reserved bytes | 1918267904 / 2109734912 |
| Summed step / validation time | 81.042418144876 s / 5.143885655038 s |
| Checkpoint writes / steps | 4 / 16, 32, 48, 64 |
| Checkpoint bytes written / final bytes | 1,580,809,164 / 395,220,707 |
| Checkpoint write time | 0.890651636000 s |
| Final checkpoint SHA-256 | `d011336f23ee82e6a56b594b44343fa4e4232d2cf70c7a0ae41112db400683b9` (local-only) |

The full 64-step loss history and per-step cache/timing values are in `real-pilot-run.json` (`history`), `loss-history.json` and `metrics.jsonl`. Losses decreased descriptively over the 64 steps (first final NTP 10.5035; last 7.5799; weighted final CE 7.934573), but no improvement threshold or convergence claim applies.

FP8 audit: no saturation and no non-finite values. Summed logical-audit counts were weight nonzero 2,824,273,918 with 23,407 underflow, input nonzero 1,657,187,968 with 673,574 underflow, and gradient nonzero 1,623,493,782 with 580 underflow. These are descriptive counts, not tuned thresholds. Evaluation used the safe cache scope `cleared-before-and-after-evaluation`; evaluation diagnostics are recorded in `real-pilot-run.json`.

The final checkpoint is local-only and not published. It contains model, optimizer, RNG and history, with no weight-cache state; model and optimizer tensors were finite. No test data was evaluated; `test_used_for_training=false` and `test_validated_only=true`.

## Development and audit corrections

1. The first gate preflight accidentally used `configs/experiments/splot-nano-synthetic-fp8-muon.json` (43,055,076 parameters) while the preregistered comparator expects 179,952 parameters. The comparator rejected the inventory; the gate was rerun with the exact tiny `splot-smoke-synthetic-fp8-muon.json` plus only weight reuse. No official gate result was changed and no gate was relaxed.
2. The first independent data-audit script assumed document IDs were unique per token; the contract repeats one document ID across a segment. The audit now counts unique segment IDs per block. The loader and measurements were unchanged.
3. The first checkpoint-audit assertion assumed all four writes had equal bytes; actual writes differ slightly with history length. The audit records actual summed bytes and final bytes. No measurement changed.

No official run failed, timed out or was retried. No source file was modified in this attempt.

## Limitations

- Single seed, single configuration and 64 optimizer steps on a 56,201-token pilot; this is an architecture sanity run, not a convergence, quality, scaling or production-throughput result.
- The 16k tokenizer is a candidate; the model vocabulary remains 32,768 with 16,384 unused logits and no remapping. No tokenizer-quality or language-fairness claim.
- Validation is aggregate only; test data is integrity-checked but not evaluated. No per-language validation claim is made here.
- The Polish upstream sources are mutable; this run verified the frozen local identities at regeneration time. Any future change requires re-verification, not substitution.
- The FP8 weight cache is per-update/process-local; it has no persistence, second-order or multi-writer guarantee. Timing and memory are descriptive, not speed thresholds.
- The trained checkpoint is local-only; no redistribution or weights-publication approval is implied.
- The harness `limitations` list in `real-pilot-run.json` retains generic synthetic-fixture wording; the actual selector was `pilot-blocks-v1` and the verified real pilot data are described above.

## Next action

Maintainer review of this attempt and PR. T0017 remains open; no merge or closure is performed. The local checkpoint is retained for the separately authorized MTP/profile follow-up; no T0020 launch or further training is started here.
