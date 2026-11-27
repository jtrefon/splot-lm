# T0002 — SPLOT-Nano CUDA resource and short-step validation
Issue [#28](https://github.com/jtrefon/splot-lm/issues/28) | Parent #10; supports #7 | Class VAL
Ready only after this amendment is merged and owner explicitly authorizes T0002. Dependency #27 closed: #31/#32 merged, T0001 attempt-002 accepted. No automatic launch.
Approved model source: d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa (same reviewed model/harness as T0001). Record actual instruction commit separately.
One GPU, RTX 4090, existing approved environment, FP32 deterministic, synthetic-v1 only. Maximum 20 minutes total foreground work, one optimizer step and normal built-in validation, 300s command timeout plus 15s termination grace. Require at least 20,480 MiB free GPU memory immediately before run (conservative preflight guard, not a hard quota or feasibility guarantee). Owner may stop earlier. No installs, sudo, unrelated process termination without separate owner consent, new datasets, BF16, Mini, extended training, source changes, tolerance changes or retry.

## Objective
Establish whether the 43,055,076-parameter Nano candidate completes one AdamW training update on the RTX4090 with finite objective and gradients, valid checkpoint and held-out synthetic evaluation. Capture actual memory/timing for planning #7/#10. One-step success is not long training feasibility or language quality.

## 1. Prepare (environment)
Read AGENTS.md, EXECUTION.md and full issue #28 thread; review T0001 evidence. Follow REP start checklist; choose next unused attempt, post start, branch execution/T0002-attempt-NNN from reviewed origin/main. Confirm clean checkout; no automatic reset/stash. Keep instruction repo and detached source worktree separate, outputs outside source. Example for attempt 001:
```bash
REPO="$(git rev-parse --show-toplevel)"
SOURCE=d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa
ATTEMPT=001
TASK_COMMIT="$(git rev-parse origin/main)"
WORK="$(mktemp -d "$HOME/splot-T0002-source.XXXXXX")"
RUNS="$(mktemp -d "$HOME/splot-T0002-runs.XXXXXX")"
EVID="$REPO/experiments/results/T0002/attempt-$ATTEMPT"
git worktree add --detach "$WORK" "$SOURCE"
mkdir -p "$EVID"
cd "$WORK"
git rev-parse HEAD
git status --porcelain
python --version
python -c 'import torch,numpy; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),numpy.__version__)'
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
```
Verify exact source SHA, clean worktree, available disk (at least 2 GiB for checkpoint/output), installed CUDA PyTorch, owner-selected RTX4090 and free VRAM >=20480 MiB. Record UTC, versions, actual GPU/driver/memory and any differences from T0001. Do not publish identifiers/private paths. If inference occupies GPU, ask owner to free it or report blocked; no autonomous shutdown. Set CUDA_VISIBLE_DEVICES to actual owner-selected numeric index; verify torch sees one correct GPU. Require same approved torch 2.10.0+cu128 environment as T0001, or stop and ask about change. Capture exact commands/exit codes/stdout/stderr, preserve first failure. No rerun inside same attempt.

## 2. Run Nano (cuda_nano_step)
Use exact existing JSON configs/experiments/splot-nano-synthetic.json, V32768, width384, q6/kv2, head64, dense1024/expert512, 2/4/2 blocks, A2 R2 B1 T96 seed42, AdamW lr .003, weight decay 0. The synthetic byte data is intentionally not a production tokenizer. Explicit environment on every invocation; select index above, do not use placeholder:
```bash
export CUDA_VISIBLE_DEVICES=0  # replace 0 only with verified selected GPU index
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=src:tools python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))'
PYTHONPATH=src:tools timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-nano-synthetic.json --out "$RUNS/nano" --steps 1 --device cuda
```
Before launch confirm free memory again. Harness enables deterministic algorithms, disables TF32, checks finite objective and observed gradients, verifies inventory and saves after optimizer step. Do not disable determinism. Pass iff exit 0, exactly one history step, actual parameter count 43055076, pinned commit and CUDA identity, positive valid/NTP labels, finite objective/final/early/offset2/balance and positive finite timing/peak allocated and reserved bytes. OOM or nonfinite is a failed measured gate; missing environment or owner interruption blocked; dependent gates not_run. Preserve stderr and exit. No extrapolation from T0001 memory.

## 3. Check checkpoint and evaluation (checkpoint_validation)
Only after step passes. Copy exact safe run.json to evidence run-nano.json. Run bounded CPU verification of the self-produced checkpoint (no untrusted files):
```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:tools timeout --kill-after=15s 120s python - "$RUNS/nano" <<'PY'
import json, math, sys
from pathlib import Path
import torch
d=Path(sys.argv[1])
r=json.loads((d/'run.json').read_text())
c=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=True)
assert r['parameters']==43055076
assert len(r['history'])==len(c['history'])==1
assert r['history']==c['history'] and r['identity']==c['identity']
assert r['identity']['commit']=='d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa'
assert r['identity']['environment']['device']=='cuda'
assert r['identity']['settings']['precision']=='fp32'
assert sum(x.numel() for x in c['model'].values())==43055076
assert all(torch.isfinite(x).all() for x in c['model'].values())
states=c['optimizer']['state']
assert states and all(float(x['step'])==1 for x in states.values())
assert all(torch.isfinite(x['exp_avg']).all() and torch.isfinite(x['exp_avg_sq']).all() for x in states.values())
v=r['validation']
assert v['ntp_labels']>0 and all(math.isfinite(v[k]) for k in ('nll_sum','mean_nll','perplexity'))
assert math.isclose(v['mean_nll'],v['nll_sum']/v['ntp_labels'],rel_tol=1e-10)
assert math.isclose(v['perplexity'],math.exp(v['mean_nll']),rel_tol=1e-10)
print(json.dumps({'status':'passed','parameters':r['parameters'],'optimizer_state_entries':len(states),'validation':v},indent=2))
PY
```
Save actual stdout as checkpoint-validation.json on success. Unused experts need not have optimizer entries; do not claim all parameters received gradients. Record checkpoint size and SHA256 local-only; no tensors in Git. End source HEAD/clean status and nvidia-smi observation, no automatic cleanup or restart of unrelated services. If source changes, invalidate affected gate, record deviation.

## 4. Evidence and handover
Check IDs exactly environment,cuda_nano_step,checkpoint_validation. Mandatory REP files under experiments/results/T0002/attempt-NNN; include run-nano.json and checkpoint-validation.json when available. Metrics: actual step valid/NTP/offset2 labels, objective components, validation NLL/perplexity/labels, step_seconds, peak allocated/reserved bytes (and MiB if desired), checkpoint size/hash, observed GPU free memory before/after. Timing includes finite-gradient checks and optimizer, excludes checkpoint I/O/validation; no speed or energy claims. Record end UTC, limitations, first failure. If blocked before source verification tested_commit null. Publish evidence even for blocked/failed attempts.
Compute files SHA-256 per EXECUTION.md, then from instruction repo:
```bash
python tools/rep/validate_result.py "$EVID" --task T0002 --source d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa --checks environment,cuda_nano_step,checkpoint_validation
```
Append index row, inspect sanitized diff, commit/push evidence only, open PR referencing #28/#10, comment task issue with outcome/PR. No merge, closure or additional task. Research review precedes next authorization.
