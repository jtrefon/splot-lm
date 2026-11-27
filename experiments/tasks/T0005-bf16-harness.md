# T0005 — BF16 training harness with FP32 numerical safeguards
Issue [#40](https://github.com/jtrefon/splot-lm/issues/40) | Parent #7/#10 | Class IMP
Ready only after reviewed specification merged and explicit Linux owner authorization. T0003 #35 accepted/closed, #37 merged. Independent of Mac T0004. Minimum reviewed base 625e6a7bfac97344fab6a9b102aab3613a670b1e; branch from current reviewed main containing this task. Record actual base, task commit and tested implementation SHA separately.
RTX4090, existing Python 3.12.7 torch 2.10.0+cu128. CPU and one CUDA device only, maximum 90 min foreground, >=20480 MiB free VRAM before CUDA, >=4 GiB disk; each CPU test <=120s, CUDA run <=300s plus 15s grace, owner stop takes precedence. No packages, sudo, real corpus, Mini, FP8, quantization, MPS, long context, topology changes, checkpoint-cadence change or additional tasks. No unrelated process termination without separate consent.

## Objective
Add opt-in BF16 autocast to the actual training/evaluation harness with FP32 master weights and optimizer state, FP32 sensitive arithmetic, and exact same-mode resume. Existing FP32 path and defaults must remain unchanged. Establish correctness not throughput or precision superiority.

## 1. Scope and environment (environment)
Read AGENTS.md, EXECUTION.md, #40/#7, T0003 evidence and existing harness/tests. Verify clean checkout and no overlap, owner authorization, approved GPU/stack, torch.cuda.is_bf16_supported() (record actual result; if unavailable CUDA gate blocked), disk/VRAM. CPU work may proceed without GPU if authorized; no fallback silently labeled CUDA. Create feat/T0005-bf16-attempt-NNN branch and post start. Allowed files: tools/experiment.py, new tools/test_bf16_harness.py, new configs/experiments/splot-smoke-synthetic-bf16.json and splot-nano-synthetic-bf16.json, new research/implementation/bf16-harness.md, and narrowly scoped src/splot/dense.py/mtp.py/gdn.py/moe.py changes ONLY if required to enforce sensitive FP32 operations under autocast; explain exact change and preserve FP64 oracles and FP32 path. No changes to Mac files, fixture/data, existing FP32 configs, numerical tolerances or topology. If other changes needed, stop and ask in #40.

## 2. Implementation (implementation)
- Explicit precision fp32 (existing) or bf16; reject unsupported values/backend, no silent fallback. Use torch.autocast(device_type=device.type,dtype=torch.bfloat16) around forward/objective and evaluation for bf16, not backward/optimizer; no GradScaler required for this BF16 policy. CPU BF16 supported only if actual operations work, fail clearly otherwise. Preserve existing FP32 behavior and deterministic settings, TF32 disabled. Do not cast entire model/optimizer to BF16. Record requested/actual policy in run identity/config, same-mode checkpoint identity strict; reject cross-precision resume. No changes to checkpoint cadence in this task.
- Audit sensitive operations: FP32 master parameters/gradients/Adam moments; GDN matrix recurrence and its contractions outside autocast; FP32 router linear/softmax, deterministic top-k and accumulation; FP32 CE, RMSNorm reductions and RoPE angle arithmetic. GQA score contraction and softmax explicitly FP32 (casting inputs alone is not enough under autocast), preserve existing FP64 oracle; value contraction may be BF16 if documented. No extra casts on FP32 path that change semantics. Record actual dtypes via hooks or focused tests, not assumptions about autocast. Require at least one eligible projection output BF16 and actual GDN matrix/router/reduction FP32. No guarantee all activations BF16. Any discovered mismatch/failure preserved.
- New configs identical to existing counterparts except training.precision. Add focused tests for same initial weights/batch selection, FP32 master parameters/Adam moments, actual eligible BF16 projection and FP32 sensitive operations, finite losses/gradients, FP32 loss arithmetic, precision identity mismatch refusal, same-mode resume exact on deterministic CPU/CUDA where supported. Test actual model, not only toy linear. Compare matched first-step loss components; predeclared provisional numerical gate for Nano first step final/early/offset2 CE each absolute difference <=0.10 nats and composite objective absolute difference <=0.20; balance and route agreement report descriptively (top-k discontinuities), no requirement identical routes or BF16/FP32 tensors. These are smoke tolerances not convergence criteria; failure is evidence, no widening after results. Verify post-step master params and present moments finite and FP32, all present optimizer step values correct.
- Document policy, exact tests, limitations and changes. Commit implementation first; record tested implementation SHA; clean checkout required for CLI, outputs outside repository. No changing code after acceptance tests without new commit and recorded test cycle.

## 3. CPU regression (cpu_tests)
Run from exact clean implementation commit with CUDA hidden and PYTHONPATH=src:tools:
```bash
export PYTHONPATH=src:tools
export CUDA_VISIBLE_DEVICES=""
python tests/numerical/test_recurrent_semantics.py
python tests/numerical/test_hybrid_mixer.py
python tools/test_accounting.py
python tools/test_data_fixture.py
python tests/test_dense.py
python tests/test_recurrent.py
python tests/test_mtp.py
python tests/test_speculative.py
python tests/test_hybrid.py
python tests/test_moe.py
python tools/test_experiment.py
python tools/test_experiment_integration.py
python tools/test_bf16_harness.py
```
Use GNU timeout --kill-after=15s 120s per command (or equivalent bounded foreground); record exit/test counts, no ignored unexpected skips. CPU BF16 test unavailable => document and do not claim CPU validation; existing CPU FP32 tests must pass. Exact CUDA numerical validation still required for full pass.

## 4. Bounded paired CUDA validation (cuda_precision)
Only after CPU FP32 regressions and verified owner-selected GPU. Use same implementation revision, exact committed config, fresh outputs outside checkout, explicit env each command. Select actual verified CUDA index, below uses 0. Require same seeds/inputs/label budgets. Use new empty run directories, no resume across precision. Record actual wall time, config/data digests, per-step loss/memory and dtype assertions.
```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=src:tools
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-nano-synthetic.json --out "$RUNS/fp32" --steps 1 --device cuda
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-nano-synthetic-bf16.json --out "$RUNS/bf16" --steps 1 --device cuda
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-smoke-synthetic-bf16.json --out "$RUNS/bf16-continuous" --steps 4 --device cuda
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-smoke-synthetic-bf16.json --out "$RUNS/bf16-resumed" --steps 2 --device cuda
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-smoke-synthetic-bf16.json --out "$RUNS/bf16-resumed" --steps 4 --device cuda --resume
timeout --kill-after=15s 120s python tools/rep/compare_resume.py "$RUNS/bf16-continuous" "$RUNS/bf16-resumed" --steps 4 --parameters 179952
```
RUNS created outside repository at preflight. Compare Nano FP32 row to T0002 non-timing fields and report any deviation; existing FP32 behavior unchanged is a required gate, so unexpected mismatch blocks acceptance pending diagnosis. Compare paired Nano first-step CE/objective against provisional thresholds in section 2 and record actual absolute differences. Both runs require 43055076 params, 51 valid/50 NTP/49 offset2 and 376 validation NTP labels, finite objectives/gradients/weights/moments/validation, positive timing and allocator stats; dtype policy verified in focused CUDA test (run new test suite with actual device explicitly documented; require no silent skip). Same-mode BF16 smoke comparator exact; cross-precision resume reject verified by test. No performance claim from one step. Stop on first failed dependent gate, no automatic retry. If BF16 deterministic operation fails, preserve operator/stack and report; do not disable determinism or alter model to force pass.

## 5. Evidence and handover (handover)
Check IDs environment,implementation,cpu_tests,cuda_precision,handover. REP IMP: source_commit/tested_commit = actual tested implementation SHA, record base_source_commit and minimum base separately. If blocked before tests, actual verified base and tested_commit null as applicable. Evidence under experiments/results/T0005/attempt-NNN, summary/result/environment/metrics/log and small run JSONs, dtype/numerical test report and comparator output. Include exact commands/exit codes, test totals, scope diff, all measured differences, checkpoint hashes/bytes local-only, limitations. No tensors. Validate exact attempt:
```bash
python tools/rep/validate_result.py "$EVID" --task T0005 --source "$TESTED_IMPLEMENTATION_SHA" --checks environment,implementation,cpu_tests,cuda_precision,handover
```
Commit evidence after implementation commit; append index if no conflict, push PR, comment #40. Do not merge or close; stop. Next matched BF16/FP32 full-epoch validation and checkpoint-cadence measurement require separate reviewed tasks. No production training.
