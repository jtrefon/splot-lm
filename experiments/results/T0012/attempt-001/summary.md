# T0012 attempt-001 — passed (contiguous-context FP8/Muon resource qualification)

Task T0012 (IMP), issue #66, parents #7/#10/#42. Specification PR #68 merged at reviewed main `685df410e28a260d45c7cfa2685c6614d64d77e0`; minimum reviewed base `34506006f24a18a0222fb45a61eeaf2f51fd6289`. Tested implementation commit `0ff3ff3bab066324c534326aaadd400b6f1058ad`. UTC 2026-11-27T19:24:59Z to 2026-11-27T19:29:47Z (288 seconds).

## Outcome

**Passed.** All six required checks passed. Required paired cases T128/512/2048 passed, and the conditional T4096 pair was eligible and passed. This is resource qualification only; no pretraining, quality, production-throughput or superiority claim.

## Authorization and scope

The owner explicitly authorized T0012 in the current local session within the merged specification and resource limits. The implementation adds only the isolated `tools/context_probe.py`, `tools/test_context_probe.py` and `research/implementation/context-probe.md`; no existing model, harness, optimizer, config, CI, data or evidence file changed.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| cross_platform | passed | `cross-platform.json`, `t0010-demo-linux.json`, `execution.log` |
| implementation | passed | `implementation.files.json`, `implementation.diff`, `summary.md` |
| cpu_tests | passed | `cpu-ci.json`, `ci-cpu.txt`, `execution.log` |
| cuda_context | passed | `cuda-context.json`, `context-summary.json`, eight `case-*.json`, `metrics.jsonl`, `execution.log` |
| handover | passed | `summary.md`, `execution.log` |

## Cross-platform data gate

With CUDA hidden, `python tools/tokenized_shards.py --demo` ran twice on the tested SHA and produced byte-identical 19,754-byte output with SHA-256 `e329c8fbb2fb34e4f7de66ec7c0ad661786e70725db620bfb6d85fb4743e2dcf`, exactly matching T0010 attempt-001 on Mac. The Linux output also matched the Mac evidence byte-for-byte.

Independent checks passed for manifest SHA-256 `d44629937047fc543f3ce028c7aabf01214f3e9e93547b81262d4f04dd3d47e9`, logical digest `5d8d5898c41c36ac21cc7ba4785c8747c0a6ec8b20830419a0a082dfd931b5ba`, packed logical digest `7585194a799c1bc839591362929bf4e041bb18b4acaf9d15d677ff951743635f`, physical manifest digest `a413fbfd1c34e953b6f105cf70101cd16937440989775d215fd711cfc32abc7a`, and counts 11 documents / 4 blocks / 3 shards / 203 content / 22 special / 225 valid / 159 pad tokens / 214 NTP / 203 offset-2 labels.

## CPU tests

`python tools/ci_cpu.py` from clean tested SHA `0ff3ff3bab066324c534326aaadd400b6f1058ad`: **22 scripts, 188 tests, 0 failures, 0 skips, 15 REP reports**. The new probe test contributes 5 CPU tests covering contiguous boundaries, exact labels, fixed-seed identity, invalid input/device, phase/rate math and paired model/config identity.

## Contiguous cases

The probe constructs one document with exactly T valid tokens, BOS/EOS, interior content IDs 8..32767, all document IDs 0, positions 0..T-1, all valid masks, and exactly T-1/T-2 non-ignore offset labels. All cases report 43,055,076 parameters, identical initialization and input digests within each pair, identical Muon partition plan `1bed555467ef9a955079b8260636971dafa5b0b308962e9a18e23a5ffa7bf9ed`, and FP32 finite model/optimizer state (55,685,105 state elements / 222,740,420 bytes per Muon run).

| T | Precision | First CE final / early / offset | forward / backward / checks / step (s) | E2E (s) | Peak reserved (B) | Free after cleanup (B) | Tokens/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | bf16 | 10.519993 / 10.501455 / 10.532146 | 0.1196 / 0.2242 / 0.0139 / 0.0650 | 0.4379 | 1000341504 | 24626397184 | 292.3 |
| 128 | fp8 | 10.495830 / 10.490265 / 10.529448 | 0.2640 / 0.3505 / 0.0366 / 0.0668 | 0.7361 | 1021313024 | 24622202880 | 173.9 |
| 512 | bf16 | 10.493875 / 10.484287 / 10.487617 | 0.3718 / 0.8005 / 0.0143 / 0.0674 | 1.2696 | 1908408320 | 24592842752 | 403.3 |
| 512 | fp8 | 10.482882 / 10.476407 / 10.485504 | 0.5331 / 0.9491 / 0.0375 / 0.0686 | 1.6069 | 1874853888 | 24588648448 | 318.6 |
| 2048 | bf16 | 10.476771 / 10.477264 / 10.479271 | 1.4530 / 3.2731 / 0.0148 / 0.0697 | 4.8264 | 7381975040 | 24559288320 | 424.3 |
| 2048 | fp8 | 10.477432 / 10.478637 / 10.474892 | 1.6196 / 3.4428 / 0.0381 / 0.0703 | 5.1894 | 7189037056 | 24555094016 | 394.6 |
| 4096 | bf16 | 10.468967 / 10.464988 / 10.483092 | 2.9382 / 6.7651 / 0.0146 / 0.0702 | 9.8039 | 16007561216 | 24257298432 | 417.8 |
| 4096 | fp8 | 10.467954 / 10.464047 / 10.480293 | 3.1332 / 6.8773 / 0.0378 / 0.0690 | 10.1358 | 15699279872 | 24253104128 | 404.1 |

The timing columns include explicit `torch.cuda.synchronize` calls; the synchronization total is recorded per case. These are single measured updates after an untimed warm-up, not steady-state throughput claims.

## BF16/FP8 paired comparison

| T | final CE abs diff | early CE abs diff | offset CE abs diff | objective abs diff | Status |
|---:|---:|---:|---:|---:|---|
| 128 | 0.024163 | 0.011190 | 0.002699 | 0.027662 | passed |
| 512 | 0.010992 | 0.007880 | 0.002113 | 0.013487 | passed |
| 2048 | 0.000661 | 0.001372 | 0.004379 | 0.000088 | passed |
| 4096 | 0.001014 | 0.000940 | 0.002799 | 0.001944 | passed |

All preregistered first pre-update gates pass: component differences <=.5 nats and objective <=.75 nats. All model, logits, hidden, offset, gradient and state checks are finite; FP8 native forward/backward calls are positive and audit nonfinite is false.

FP8 audit (saturation zero, nonfinite false in all cases):
- T128: 292 native forward/backward, 1,584 padded rows; input/weight/grad underflow fractions `8.410124e-05`/`8.340760e-06`/`7.962367e-07`.
- T512: 298 native forward/backward, 1,440 padded rows; `9.758578e-05`/`8.395818e-06`/`2.554404e-07`.
- T2048: 298 native forward/backward, 1,584 padded rows; `1.083054e-04`/`8.395818e-06`/`1.560270e-07`.
- T4096: 298 native forward/backward, 1,152 padded rows; `1.117948e-04`/`8.395818e-06`/`1.311939e-07`.

## T4096 conditional gate

Both T2048 cases passed. At the gate decision, max peak reserved was 7,381,975,040 bytes (7,040 MiB, <=12,288 MiB), minimum free after cleanup was 24,555,094,016 bytes (23,417 MiB, >=20,480 MiB), max T2048 case time was 14.846 s (<120 s), and 7,150.9 s remained (>15 min). T4096 was therefore attempted once in the fixed BF16→FP8 order and both cases passed. Its peak reserved was 16,007,561,216 bytes BF16 and 15,699,279,872 bytes FP8; E2E was 9.804 s and 10.136 s.

## First failure and development notes

Before the tested commit, two development smoke runs failed while writing the probe report: first a `PosixPath` JSON serialization error, then a `torch.dtype` serialization error. Both were probe report-serialization bugs, fixed by converting the config path and autocast dtype to JSON-safe values. The reproduction commands and error text are recorded in `execution.log`; the tested commit is the corrected revision. No official case failed, was retried or had a tolerance changed.

## Deviations

None. The two development smoke failures were corrected before the tested commit and did not affect the official paired cases.

## Limitations

- Resource qualification of one measured update per precision/context; no convergence, quality, long-context capability, production throughput or optimizer/precision superiority claim.
- Synthetic documents only; no real corpus, Mini model or data-rights claim.
- T4096 is optional and was attempted only after the preregistered resource gate; no T8192 or other lengths.
- Existing FP8 per-tensor scaling, Muon implementation and model equations are unchanged.
- Per-phase synchronizations and audit checks are measured, not subtracted; wall/phase differences are not a kernel or optimizer speed claim.

## Proposed next action

Maintainer review of PR evidence and hosted CPU CI. T0012 remains open; no merge or closure. Checkpoint cadence, real-data integration and production pretraining require separate reviewed tasks.
