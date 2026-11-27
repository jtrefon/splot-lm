# T0001 — Linux RTX 4090 correctness and checkpoint validation
Issue [#27](https://github.com/jtrefon/splot-lm/issues/27) | Parent #10 | Class VAL
Specification: ready ONLY after merged on main and owner explicitly launches T0001.
Approved model source: d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa (merged #26).
Instruction revision: actual reviewed main commit containing this file; record separately.
Hardware: owner's Linux RTX 4090 (one GPU), existing approved Python/PyTorch/CUDA environment. FP32 only, synthetic-v1 only. Maximum 60 minutes total foreground execution including tests; owner can stop earlier. No installs, sudo, background jobs, network downloads, real corpus, Nano/Mini training, code changes or tolerance changes.

## Objective
Verify full CPU regressions and actual SPLOT training harness integration, then deterministic CUDA execution of the full 179,952-parameter smoke topology (A2, R2, seed42, sequence length96, batch1) and same-device checkpoint equivalence. This is correctness only, not language quality or speed comparison. Do not claim GPU feasibility for Nano/Mini.

## Mandatory rules
Read AGENTS.md and EXECUTION.md first. Confirm issue/spec and owner authorization; choose next unused attempt, never overwrite. Exact source pin, clean worktree, no fixes/retries. Capture actual commands, UTC times, exit codes, stdout/stderr; sanitize public evidence. Missing dependency/device/authorization => blocked; command failure or violated gate => failed; unexecuted dependent checks => not_run. Stop dependent stages, preserve first failure. If time/owner stop, mark affected check blocked and remaining not_run; retain exit codes. No further tasks without explicit authorization. Evidence branch only, no merge.

## 1. Prepare and verify environment (environment)
From existing repository, fetch origin, inspect #27, confirm reviewed origin/main contains instructions. Refuse dirty working tree (do not stash/reset automatically). Create evidence branch execution/T0001-attempt-001 from origin/main (replace 001 with next unused attempt). Record task_commit before editing files. The source is an older reviewed commit, so keep it separate:
```bash
REPO="$(git rev-parse --show-toplevel)"
SOURCE=d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa
ATTEMPT=001  # use next unused attempt
TASK_COMMIT="$(git rev-parse origin/main)"
WORK="$(mktemp -d "$HOME/splot-T0001-source.XXXXXX")"
RUNS="$(mktemp -d "$HOME/splot-T0001-runs.XXXXXX")"
EVID="$REPO/experiments/results/T0001/attempt-$ATTEMPT"
git worktree add --detach "$WORK" "$SOURCE"
mkdir -p "$EVID"
cd "$WORK"
git rev-parse HEAD
git status --porcelain
```
Ensure HEAD exactly SOURCE and porcelain empty. WORK and RUNS outside repository tree. Use actual absolute REPO/WORK/RUNS/EVID internally; sanitize private path prefixes in published logs. No environment variable dump. Record start/end UTC with date -u. Run and record:
```bash
python --version
python -c 'import platform,torch,numpy; print(platform.system(),platform.machine()); print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),numpy.__version__)'
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
```
Owner selects one RTX4090 index with adequate free memory; if ambiguous or busy, ask/mark blocked. Set CUDA_VISIBLE_DEVICES to that index for GPU runs; record index and model/driver/VRAM, not UUID. Require torch.cuda.is_available() true and correct selected GPU for CUDA. If CPU prerequisites exist but GPU unavailable, complete CPU checks only within owner limit, mark GPU check blocked; explain environment status and independent CPU result. No package installation. Verify interpreter and dependencies consistently for every command. Capture environment.json.

## 2. CPU regressions (cpu_regression)
From WORK, use same interpreter, CUDA hidden, PYTHONPATH=src:tools. Run individually with a 120-second maximum each (GNU timeout --kill-after=15s 120s); record command/exit status and test counts. List below is exhaustive at pinned revision:
```bash
export PYTHONPATH=src:tools
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/numerical/test_recurrent_semantics.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/numerical/test_hybrid_mixer.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tools/test_accounting.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tools/test_data_fixture.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/test_dense.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/test_recurrent.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/test_mtp.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/test_speculative.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/test_hybrid.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tests/test_moe.py
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 120s python tools/test_experiment.py
```
Pass iff all commands exit 0 and each suite reports success; stop on first failure, no threshold widening. Record any skipped suites. Use pipefail or capture exit status before logging; never let tee hide failures.

## 3. Full SPLOT harness integration (cpu_integration)
Only after CPU regression pass:
```bash
CUDA_VISIBLE_DEVICES="" timeout --kill-after=15s 180s python tools/test_experiment_integration.py
```
Pass iff exit 0, test passes including actual 179952 parameter inventory, bitwise resumed CPU parameters and validation. No claimed new measurement until executed.

## 4. Same GPU training runs (cuda_continuous, cuda_split, cuda_resume)
Only after CPU pass and environment GPU gate. Export CUBLAS_WORKSPACE_CONFIG=:4096:8 and CUDA_VISIBLE_DEVICES=<owner-selected actual index> (substitute verified number, not literal brackets). Verify selected name and CUDA availability with Python. Keep source checkout clean; output outside WORK. Use fixed config as committed, no edits. Each run bounded at 5 min plus 15s termination grace; total time ceiling still applies. Execute in order; stop dependent steps on first failure.
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out "$RUNS/continuous" --steps 4 --device cuda
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out "$RUNS/resumed" --steps 2 --device cuda
```
After each successful invocation copy run.json to evidence as run-continuous.json and run-split.json respectively. Record SHA-256 and bytes of each checkpoint before resume, not tensors. Verify each actual run identity source pin, parameters=179952, expected history length, finite losses/validation, actual CUDA device, peak CUDA allocated/reserved fields nonnull and positive, and nonzero supervised labels; exit 0 alone insufficient. The harness checks finite observed gradients and deterministic mode. If any gate fails, record and stop. If split passes, continue:
```bash
timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out "$RUNS/resumed" --steps 4 --device cuda --resume
```
Copy final run.json as run-resumed.json; verify same gates for 4 total steps. --steps is total not additional. Record final checkpoint hashes and sizes, local-only.

## 5. Deterministic resume equivalence (cuda_equivalence)
Only after both 4-step runs succeed. Use helper from reviewed instruction checkout, no model checkout edits:
```bash
timeout --kill-after=15s 120s python "$REPO/tools/rep/compare_resume.py" "$RUNS/continuous" "$RUNS/resumed" --steps 4 --parameters 179952
```
Save actual stdout as comparison.json if successful; errors in execution.log. Pass iff exit 0, model tensor/AdamW/RNG states equal exactly (same device/software), identical non-timing history and validation, identical identity/config/data/label budgets, no changes. Time and allocator peaks explicitly excluded from equality because processes differ. No GPU-vs-CPU bitwise claim. If deterministic GPU algorithms fail, report exact exception; do not disable determinism or change dtype. If comparison fails, record first mismatch path. Verify source HEAD and clean status at end; mismatch invalidates execution, record deviation.

## 6. Publish
Required check IDs: environment, cpu_regression, cpu_integration, cuda_continuous, cuda_split, cuda_resume, cuda_equivalence. Set each status/detail/evidence in result.json; overall status derived by EXECUTION.md. If preflight blocks, still publish actual blocker with required files; tested_commit null if unverified. Do not fabricate metrics.
- Copy safe run JSONs from successful runs (sanitize any paths if necessary, record redactions), include raw step metrics in metrics.jsonl with run label (continuous, split, resumed); split prefix is repeated history, do not sum across runs. Report per-run step/label counts, objective, validation NLL/PPL, allocated/reserved bytes, elapsed seconds; no performance inference. Empty metrics if none.
- summary.md, environment.json, execution.log (safe command/output and first failure), result.json with all evidence SHA-256s; external_artifacts array contains local-only checkpoint names, SHA-256, bytes for available files. No weights in Git.
- From instruction checkout compute files hashes per EXECUTION.md, then:
```bash
cd "$REPO"
python tools/rep/validate_result.py "$EVID" --task T0001 --source d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa --checks environment,cpu_regression,cpu_integration,cuda_continuous,cuda_split,cuda_resume,cuda_equivalence
```
Validator success is structural, not acceptance. Append attempt row to experiments/results/INDEX.md, stage only attempt files and index (execution.log allowed by scoped gitignore exception), inspect diff/redactions, commit named evidence branch, push/open PR referencing #27/#10, comment issue with outcome and PR. Keep worktree and outputs for review; no destructive cleanup. Do not merge or close. Stop after T0001 regardless of outcome. T0002 requires separate review and authorization.
