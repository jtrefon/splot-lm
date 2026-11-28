# T0022 attempt-001 — operator attribution of the BF16 reference backward bottleneck

Task T0022 (VAL), issue #98, parents #85/#7/#10/#42. Instruction/main `2babd431a8c0cb61a0f1129b4aae8c2383d7dbf8`; pinned source `6e8911662ac56df665147cb79e412f0383cf71d8` in a clean detached checkout. UTC 2026-11-28T16:16:54Z to 2026-11-28T16:20:51Z. Overall: **passed**.

## Outcome

Four required fresh-process BF16 cases passed: T512 control, T512 profiler-instrumented, T2048 control, T2048 profiler-instrumented. Control and instrumented cases are **exactly equal** for losses, full gradient inventory/digest, post-update model digest, optimizer-state digest and identity at both lengths. The accepted T0020 BF16 input/initialization/partition identity and exact pre-update losses also match at both lengths.

The profiler could not reliably map generic ATen keys to GDN, attention, expert or optimizer modules. Attribution is therefore bounded and explicitly **unassigned**; no module percentages, operator dominance claim from generic names, or speedup claim is made.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| exactness | passed | four `case-*.json`, `equality-512.json`, `equality-2048.json`, `metrics.jsonl`, `execution.log` |
| attribution | passed | `case-512-profile.json`, `case-2048-profile.json`, `attribution.json`, `analyze_profile.py`, `metrics.jsonl` |
| handover | passed | flat package; bounded aggregates; validator result in PR body |

## Environment and source

- GPU 0, NVIDIA GeForce RTX 4090, 24564 MiB total / 24077 MiB free at preflight; driver 580.178.04.
- Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8; deterministic algorithms, TF32 off, CUBLAS workspace `:4096:8`.
- Host RAM available 26937352192 B; durable scratch free 181953990656 B; all case prechecks exceeded 12 GiB RAM, 8 GiB scratch and 20,480 MiB GPU free.
- Source pin `6e8911662ac56df665147cb79e412f0383cf71d8` clean before/after; probe SHA-256 `4e8e96382e01532bf016568cc92af22d5eae8f3e138abf8b0b92c20d19febb5d`; BF16 config `51c493eb8f118fe1f58fd9a775fc0217f7a6cae0aa74a939d159c17ffbeceac9`; FP8 weight reuse disabled and unchanged.
- No packages, system changes, real data, checkpoints, Mini, kernel/optimizer changes, background work or retries.

## Exactness

| Length | Control/profile losses | Gradient digest | Model digest | Optimizer digest | T0020 reference |
|---:|---|---|---|---|---|
| 512 | final 10.493874549866; early 10.484287261963; offset2 10.487617492676; total 15.747568130493 | `892d9626fe183eaa75a671f37962053e56a01f7b2fa133f46537bd58d73bcfef` | `d613b5589eed7792be48d23d861ff97df42a13e16f7e7a701352072e2f41ce51` | `2f4d9dab17c1cab9c7544a30aa2cb425932c53e18f664b17852bd7fc8d9dd8b0` | identity and losses exact |
| 2048 | final 10.476771354675; early 10.477264404297; offset2 10.479270935059; total 15.726152420044 | `5168bc168a1ae5f0d2de2d28650ed0d6bbb3e68f511f0ae440d670a0f48afdf1` | `1844ec426a90feb50dca2bc5959e8a219bccb317dfca21e06a3c34dca1f77527` | `a076011c9eef4c839f5cc6d02fbefc5e8537751e13873c5c88039fb3e07e7cad` | identity and losses exact |

All cases report 43,055,076 parameters, no missing gradients, FP32 gradients/state, finite losses/state, and exact control/profile equality. The T0020 reference files are `experiments/results/T0020/attempt-001/case-512-1-bf16.json` and `case-2048-1-bf16.json`.

## Bounded attribution

Profiler settings: CPU+CUDA activities, `record_shapes=False`, `profile_memory=False`, `with_stack=False`, units microseconds, profile region limited to `objective.total.backward()` including its synchronize. No trace was exported; only bounded top-30 key averages are retained.

| Length | Total events | Key averages | Control backward | Profiled backward | Profile/control wall ratio | Top self CPU keys | Top self device keys |
|---:|---:|---:|---:|---:|---:|---|---|
| 512 | 630589 | 268 | 0.812423859 s | 21.042650911 s | 25.901× | aten::bmm, cudaLaunchKernel, autograd::engine::evaluate_function: PermuteBackward0, aten::mul, aten::sum | aten::mul, aten::sum, native reduce_kernel, aten::bmm, native elementwise_kernel |
| 2048 | 2391921 | 269 | 3.250816260 s | 79.906601502 s | 24.580× | aten::bmm, cudaLaunchKernel, autograd::engine::evaluate_function: PermuteBackward0, aten::mul, aten::sum | aten::sum, aten::mul, native reduce_kernel, aten::bmm, native elementwise_kernel |

- `cudaLaunchKernel` self CPU: 119584.619 µs over 65555 calls at T512; 448057.403 µs over 241258 calls at T2048.
- Top-30 self CPU sums are 0.940974988 s / 3.684157712 s; top-30 self device sums are 0.204586024 s / 0.796971575 s. These are bounded top-30 values, not full totals, and CPU/device self times can overlap.
Each case JSON retains self and inclusive CPU/device times; rankings use self time and inclusive totals are reported alongside. Synchronization is included in backward wall; host launch cost is shown by `cudaLaunchKernel`. No separate synchronization-kernel attribution is claimed.

- Profiled backward wall time is 25.901× / 24.580× the unprofiled control, demonstrating severe profiler perturbation. It is **not** a performance measurement.

**Module attribution: unassigned.** The top keys are generic profiler/ATen labels (`aten::bmm`, `cudaLaunchKernel`, autograd engine functions, `aten::mul`, `aten::sum`, native reduce/elementwise kernels). No reliable mapping to GDN scan, attention/logits, expert dispatch, optimizer or quantization was available, so no module percentages or dominance claim is inferred. Confidence is low for module attribution and bounded operator-level only.

## Next question

Scoped instrumentation IMP: tag backward operators by module/path (GDN recurrent scan vs attention/logits vs expert dispatch) and rerun this exact four-case equality/profiling harness. Exactness guard: full gradient, model and optimizer digests must equal the current control at both lengths, with no equation, tolerance, precision or optimizer change. Not implemented here.

## Limitations

- Four single-update cases; no steady-state, convergence, quality or production-throughput claim.
- Profiler perturbation is large; instrumented latency is diagnostic only.
- Only top-30 key averages per axis are published; no full trace and no complete event attribution.
- Self CPU/device times overlap and are not additive; no module attribution is inferred from generic ATen names.
- No real data, checkpoint, Mini, package, kernel or reference-source change.

## Handover

`profile_driver.py` and `compare_cases.py` are the sanitized reproducible scripts; `case-*.json` are the four fresh-process cases; `equality-512.json` / `equality-2048.json` hold exactness and T0020-reference gates; `attribution.json` holds bounded aggregates. No tensors or full private paths are published.

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0022/attempt-001 \
  --task T0022 \
  --source 6e8911662ac56df665147cb79e412f0383cf71d8 \
  --checks environment,exactness,attribution,handover
```

T0022 remains open for review; no merge or closure. No further task is started.
