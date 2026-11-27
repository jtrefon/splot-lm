# T0011 attempt-001 — passed (FP8 + Muon compatibility and matched synthetic epoch)

Task T0011 (IMP), issue #62, parents #7/#42/#10. Owner authorized T0011 in the current local session. Specification PR #63 was merged at reviewed main `a2776e95341d89b0dd4eff81107d0ad9f00e3283`; minimum reviewed base `9aabfa9dae9d1d7b5713cee09a73e0e4d50b5d88`. Tested implementation commit `4f92b6ea759f4db64bc827d371506972d282108f`. UTC 2026-11-27T18:53:51Z to 2026-11-27T19:01:37Z (466 seconds).

## Outcome

**Passed.** All six required checks passed. This is correctness and one-epoch descriptive evidence only; it is not a convergence, throughput, optimizer-superiority or language-quality result.

## Authorization and scope

The owner explicitly authorized this named task and the merged specification limits. The implementation changes only the allowed files: `src/splot/fp8.py`, `tools/experiment.py`, `tools/optimizers.py`, new `tools/test_fp8_muon.py`, new `research/implementation/fp8-muon.md`, and the three new configs. Existing configs, model equations, FP8 formats, numerical tolerances, comparator, CI, Mac/data files and optimizer math are unchanged. No installs, CUDA upgrade, real data, Mini run, tuning, background jobs or unrelated process termination were performed.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `summary.md` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_compatibility | passed | compatibility run JSONs, `comparison.json`, `cuda-tests.txt`, `checkpoint-state.json`, `cross-mode.json`, `metrics.jsonl` |
| cuda_epoch | passed | three epoch run JSONs, `epoch-analysis.json`, `first-step-routing.json`, `metrics.jsonl`, `checkpoint-state.json` |
| handover | passed | `summary.md`, `checkpoints.txt`, `execution.log` |

## Implementation

- FP8 selected projections and the native Muon/AdamW hybrid can now be enabled together. FP8 conversion still preserves the original FP32 `Parameter` objects, names, `state_dict` keys, tied embedding and semantic partition.
- Policy metadata records `optimizer_scope = "adamw-or-muon-hybrid"`, `policy_version`, `audit_version`, the actual optimizer and an accurate `adamw_only` flag (false for hybrid runs).
- Optimizer parsing rejects explicit null selectors, unknown names/keys, malformed settings, null sub-blocks and a Muon block supplied with AdamW. The absent selector still defaults to AdamW.
- Every FP8 step resets the audit before forward, checks all loss fields and FP8 quantization/output flags before backward, checks all gradients and FP8 flags after backward, then checks all master weights and present optimizer-state tensors after the optimizer step and before checkpoint write. Failures raise `FloatingPointError` and leave the previous checkpoint untouched.
- Evaluation resets the audit and reports its own aggregate separately. FP8 history rows contain deterministic per-step native calls, role-specific nonzero/underflow/saturation counts, nonfinite flags and padding; no timing is in the audit.
- Both Muon momentum and AdamW state remain FP32; native Muon NS intermediate remains BF16. Checkpoint top-level schema remains `identity`, `model`, `optimizer`, `rng`, `history`.

## CPU tests

`python tools/ci_cpu.py` from clean tested SHA `4f92b6ea759f4db64bc827d371506972d282108f`: **20 scripts, 162 tests, 0 failures, 0 skips, 13 REP reports**. The new `tools/test_fp8_muon.py` contributes 7 CPU tests covering config derivation/hashes, policy/partition, malformed selectors, audit reset/serialization/aggregation, adapter state, cross-mode rejection, post-update guards and prior-checkpoint preservation. Existing FP8, Muon, harness, BF16 and REP tests remain green.

## CUDA compatibility

- Focused `tools/test_fp8_muon.py --device cuda`: **4/4 passed**.
- Smoke FP8+Muon 4-step continuous and 2-step then `--resume` to 4: exact `compare_resume.py` result (model, both optimizer states, RNG, non-timing history including audit, validation).
- Continuous/resumed validation: 376 labels, mean NLL `5.2831730538`, PPL `196.9939571344`.
- Training aggregate audit: 1,313 native forward/backward and 1,313 FP32 wgrad calls, 9,828 padded forward/backward rows; input/weight/grad underflow fractions `3.717641e-05`/`4.382479e-06`/`2.997083e-06`; saturation zero, nonfinite false. Evaluation audit separately: 1,626 native forwards, 11,730 padded rows, no backward/grad.
- Cross-precision and cross-optimizer resume both exit 1 with `ValueError: run identity mismatch`.
- Compatibility checkpoints are local-only: continuous `be09d3b112a1d7a26a68e1dc4bad7ac317b5a98796d2fc3cd0b69a47ed7e9c37` (1,698,531 B); resumed `652e4b9afbcd8bece16757e96be1b21fb162ada3a061c0c4b527a74f73e1c737` (1,699,491 B). Loaded model/optimizer/RNG contents are equal; file bytes differ due serialization.

## Matched synthetic epoch

Three sequential fresh runs on the same tested SHA, seed 42, data/config settings and R2/A2/B1/T96, stopped at exactly 35 steps. All have 43,055,076 parameters, 2,591 valid tokens, 2,534 NTP labels, 2,477 offset-2 labels, 376 validation NTP labels and 35 unique batch indices covering 0..34.

| Run | Precision / optimizer | Validation mean NLL | PPL | Token-weighted final CE | Early CE | Offset CE | Composite mean | Wall / step-sum | Checkpoint |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `splot-nano-synthetic-fp8` | FP8 / AdamW | 3.499237 | 33.0902 | 4.491869 | 4.481841 | 4.439471 | 6.817814 | 80 s / 17.709 s | 516,961,355 B `b314dfd7…8145` |
| `splot-nano-synthetic-fp8-muon` | FP8 / Muon | 1.490035 | 4.43725 | 3.719944 | 3.715179 | 4.415700 | 5.805694 | 57 s / 21.420 s | 395,195,619 B `9042b8fe…5f15` |
| `splot-nano-synthetic-bf16-muon` | BF16 / Muon | 1.696633 | 5.45555 | 3.772564 | 3.754481 | 4.486922 | 5.902975 | 47 s / 11.798 s | 395,162,979 B `a66a3e32…8dc87` |

CE values are token-weighted means where denominators are available. The composite column is the mean per-step A2 objective, **not** a pure token NLL. No loss-decrease, convergence, interaction-effect or superiority gate was applied.

First-step gates:
- FP8 AdamW and FP8 Muon first pre-update loss fields are exactly equal: objective `15.8099193573`, final `10.5464286804`, early `10.5496435165`, offset `10.4408988953`, balance `1.5855262279`.
- Their first-step routing is identical: 408 rows, exact/top-1 agreement `1.0`, and audit dicts equal.
- FP8 AdamW first step exactly matches the T0009 attempt-002 baseline on all non-audit first-step fields.
- BF16 Muon vs FP8 first-step differences are final `0.007979`, early `0.012225`, offset `0.0001497`, objective `0.011023` nats; all within the preregistered .5/.75 gates.
- All epoch losses are finite; all model weights and present optimizer states are FP32 and finite. Muon partition plan `1bed5554…f9ed` is identical for FP8 Muon and BF16 Muon.

FP8 epoch audit:
- AdamW: 7,039 native forward/backward/wgrad calls; 54,582/54,582 padded rows; input/weight/grad underflow fractions `1.804810e-04`/`7.480320e-06`/`1.364012e-05`; saturation zero; nonfinite false. Evaluation: 966 native forwards, 6,498 padded rows, input/weight underflow `4.829435e-04`/`6.727861e-06`, nonfinite false.
- Muon: 9,157 native calls each; 73,686/73,686 padded rows; input/weight/grad underflow `3.282817e-04`/`8.014937e-06`/`2.213135e-06`; saturation zero; nonfinite false. Evaluation: 1,572 native forwards, 11,922 padded rows, input/weight underflow `4.103893e-04`/`7.965510e-06`, nonfinite false.
- Optimizer state: FP8 AdamW 86,110,350 FP32 elements / 344,441,400 B. FP8 Muon total 55,685,105 FP32 elements / 222,740,420 B (Muon 30,425,088 / 121,700,352 B; AdamW 25,260,017 / 101,040,068 B). BF16 Muon has the same state inventory.

## First failure and development notes

- The first focused CUDA run failed in `test_inactive_expert_is_not_updated` because the test helper returned `(inputs, labels), blocks` and the test unpacked it incorrectly. This was a test-harness bug, not an implementation failure. The helper was corrected; the final focused suite passed 4/4. The failed run and traceback are preserved in `cuda-tests.txt` and `execution.log`.
- The first cross-precision rejection setup referenced a non-existent smoke BF16-Muon config and failed with `FileNotFoundError`. A temporary BF16-Muon config differing only in precision was generated outside the checkout; the rejection then produced the expected `ValueError: run identity mismatch`. No numerical gate failed, was retried or was relaxed.
- The implementation test file was amended before the official CPU/CUDA/epoch runs; the tested SHA `4f92b6ea759f4db64bc827d371506972d282108f` is the final clean revision.

## Deviations

None. The development test-harness/setup corrections above were made before the official compatibility and epoch evidence. No source, threshold, config, data, optimizer or methodology deviation remains.

## Limitations

- Correctness and one descriptive synthetic epoch only; no convergence, scaling, language-quality, throughput or optimizer-superiority claim.
- Three runs are not a full factorial design (BF16 AdamW is absent); no interaction effect or statistical inference.
- Existing per-step audits and repeated checkpoint I/O confound wall time; step-sum and wall-time differences must not be interpreted as kernel/optimizer speed.
- Current per-tensor FP8 scaling only; no delayed/persistent scaling, MXFP8/NVFP4, FP8 optimizer state or FP8 output.
- Some unused experts are not dispatched; coverage is recorded, not an all-expert claim.
- Checkpoints are local-only; no tensors are published.

## Proposed next action

Maintainer review of PR evidence and hosted CPU CI. T0011 remains open; no merge or closure. The next real contiguous-context and data-qualified pilot, fused/grouped dispatch and CUDA 13 work require separate reviewed tasks.
