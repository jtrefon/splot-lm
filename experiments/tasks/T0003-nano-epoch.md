# T0003 — Nano FP32 full synthetic epoch and optimizer-state growth
Issue [#35](https://github.com/jtrefon/splot-lm/issues/35) | Parent #10; supports #7/#6 | Class VAL
Ready only after T0002 #28 accepted and its evidence PR #34 merged, this task specification merged, and explicit local owner authorization. No automatic launch.
Approved model source: d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa; record actual instruction commit separately.
One RTX4090, same approved Python 3.12.7 / torch 2.10.0+cu128 environment as T0002; FP32 deterministic only. Maximum 20 minutes foreground, 600s training command timeout plus 15s grace, 180s CPU verification timeout plus 15s grace. Require >=20480 MiB free VRAM and >=4 GiB disk before launch; these are preflight guards, not hard memory caps. Owner can stop earlier. No installs, sudo, source edits, tolerance changes, BF16, Mini, external corpus, second epoch, background job, automatic retry or unrelated process termination without separate owner consent.

## Objective
T0002 validated one short optimizer step and 183 optimizer-state entries. Establish sustained execution across all 35 packed training blocks, observe memory/time distributions as AdamW states accumulate, verify one epoch token budgets and final checkpoint integrity. No language-quality, generalization, long-context, production throughput, all-expert coverage or architectural advantage claim. The fixture contains short documents; T96 is packing capacity, not continuous context length. Per-step checkpoint overwrite incurs substantial cumulative I/O, measured separately from training steps.

## 1. Environment and preflight (environment)
Read AGENTS.md, EXECUTION.md and full #35 thread. Check #28 accepted/closed or explicit maintainer acceptance plus merged #34 evidence. Verify this task merged, owner authorized, attempt number unused. Post start. Follow REP: branch execution/T0003-attempt-NNN from reviewed origin/main, separate detached clean source worktree, outside-repo outputs. Do not stash/reset or overwrite.
```bash
REPO="$(git rev-parse --show-toplevel)"
SOURCE=d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa
ATTEMPT=001  # next unused attempt
TASK_COMMIT="$(git rev-parse origin/main)"
WORK="$(mktemp -d "$HOME/splot-T0003-source.XXXXXX")"
RUNS="$(mktemp -d "$HOME/splot-T0003-runs.XXXXXX")"
EVID="$REPO/experiments/results/T0003/attempt-$ATTEMPT"
git worktree add --detach "$WORK" "$SOURCE"
mkdir -p "$EVID"
cd "$WORK"
git rev-parse HEAD
git status --porcelain
python --version
python -c 'import torch,numpy; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),numpy.__version__)'
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
df -Pk "$RUNS"
sha256sum configs/experiments/splot-nano-synthetic.json
```
Exact source and clean status, approved torch build, owner-selected RTX4090, minimum free VRAM/disk, raw config SHA256 18ae9e3d7fc2cfc427ceec91de33526f51ccb2dc153ab16f8050ee2a061bad2b. Verify one visible selected GPU and fixture budgets before allocating training:
```bash
export CUDA_VISIBLE_DEVICES=0  # substitute verified owner-selected index only
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=src:tools python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))'
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:tools python - <<'PY'
from data_fixture import synthetic_documents, split_documents, pack
s=split_documents(synthetic_documents())
p=pack(s['train'],96)
v=pack(s['validation'],96)
actual=(len(p),sum(sum(b['valid']) for b in p),sum(sum(x!=-100 for x in b['labels'][1]) for b in p),sum(sum(x!=-100 for x in b['labels'][2]) for b in p),sum(sum(x!=-100 for x in b['labels'][1]) for b in v))
print('blocks, valid, ntp, offset2, validation_ntp:',actual)
assert actual==(35,2591,2534,2477,376)
PY
```
Expected from T0001/T0002 fixture manifest and packer (57 train documents,2477 byte tokens, two specials/doc; offset1 len-1 and offset2 len-2). Actual preflight must confirm, not fabricate; mismatch is failed gate, stop. Record UTC, versions, config/data hashes and GPU free memory. No autonomous shutdown of inference. If environment differs, stop and ask in issue.

## 2. Full epoch (cuda_nano_epoch)
Unchanged Nano config: 43,055,076 params, V32768, width384, head64, R2/A2/B1/T96/seed42, AdamW .003, wd0. Verify GPU free memory immediately before launch. Exact 35 steps (one complete epoch), no resume/extension:
```bash
PYTHONPATH=src:tools CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout --kill-after=15s 600s python tools/experiment.py --config configs/experiments/splot-nano-synthetic.json --out "$RUNS/nano-epoch" --steps 35 --device cuda
```
Capture start/end UTC, command wall time, stdout/stderr and exit. No automatic retry. If failure retain actual checkpoint/history if safe (never load unrelated checkpoint). Timeout/owner stop blocked, OOM/nonfinite failure failed; dependent gates not_run. Pass requires exit0, 35 unique one-block batches covering exactly indices 0..34, 2591 valid, 2534 NTP, 2477 available offset2, finite losses and positive finite timing/memory each step, actual CUDA identity and parameter count, unchanged pin/config/fixture. No required loss decrease threshold; report outcome without cherry picking. Harness checks finite observed gradients. Report actual command elapsed separately from synchronized step time (checkpoint writes and validation excluded from step_seconds).

## 3. Checkpoint, state coverage and evaluation (checkpoint_validation)
Only after run passes. Copy exact safe run.json as run-nano-epoch.json. Run this CPU check against trusted own checkpoint:
```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:tools timeout --kill-after=15s 180s python - "$RUNS/nano-epoch" <<'PY'
import json, math, statistics, sys
from pathlib import Path
import torch
p=Path(sys.argv[1]); r=json.loads((p/'run.json').read_text())
c=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True)
h=r['history']; ident=r['identity']
assert r['parameters']==43055076 and ident['commit']=='d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa'
assert ident['environment']['device']=='cuda' and ident['settings']['precision']=='fp32'
assert ident['config_sha256']=='0af02669bc89325d1e237876771d1c139faded29dbb5099777e9cd918cd53c90'
assert ident['data_sha256']=='29f7da71fb1f6808e3f10719ca75174ad1fdb5e06252bdb4c7638d000f5441af'
assert ident['packed_sha256']=='e87b8022b0118b3dcce33b39c59950ca411bb56884b974d7495357520f722ad4'
assert h==c['history'] and ident==c['identity'] and len(h)==35
assert [x['step'] for x in h]==list(range(1,36))
assert all(len(x['batch_indices'])==1 for x in h)
assert sorted(x['batch_indices'][0] for x in h)==list(range(35))
assert (sum(x['valid_tokens'] for x in h),sum(x['ntp_labels'] for x in h),sum(x['offset2_labels_available'] for x in h))==(2591,2534,2477)
assert (r['total_valid_tokens'],r['total_ntp_labels'])==(2591,2534)
for row in h:
    assert all(math.isfinite(row[k]) for k in ('objective','final_ntp','early_ntp','offset2','balance'))
    assert all(math.isfinite(row[k]) and row[k]>0 for k in ('step_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'))
    assert row['peak_cuda_reserved_bytes']>=row['peak_cuda_allocated_bytes'] and row['ntp_labels']>0
assert sum(x.numel() for x in c['model'].values())==43055076
assert all(torch.isfinite(x).all() for x in c['model'].values())
states=c['optimizer']['state']
assert states and all(1<=float(x['step'])<=35 for x in states.values())
assert all(torch.isfinite(x['exp_avg']).all() and torch.isfinite(x['exp_avg_sq']).all() for x in states.values())
v=r['validation']; assert v['ntp_labels']==376
assert all(math.isfinite(v[k]) for k in ('mean_nll','nll_sum','perplexity'))
assert math.isclose(v['mean_nll'],v['nll_sum']/v['ntp_labels'],rel_tol=1e-10)
assert math.isclose(v['perplexity'],math.exp(v['mean_nll']),rel_tol=1e-10)
times=[x['step_seconds'] for x in h]
print(json.dumps({'status':'passed','parameters':r['parameters'],'steps':35,'batch_coverage':35,'valid_tokens':2591,'ntp_labels':2534,'offset2_labels_available':2477,'optimizer_state_entries':len(states),'model_state_entries':len(c['model']),'optimizer_moment_elements':sum(x['exp_avg'].numel()+x['exp_avg_sq'].numel() for x in states.values()),'optimizer_step_min':min(float(x['step']) for x in states.values()),'optimizer_step_max':max(float(x['step']) for x in states.values()),'step_seconds_sum':sum(times),'step_seconds_median':statistics.median(times),'step_seconds_min':min(times),'step_seconds_max':max(times),'peak_allocated_bytes':max(x['peak_cuda_allocated_bytes'] for x in h),'peak_reserved_bytes':max(x['peak_cuda_reserved_bytes'] for x in h),'validation':v},indent=2))
PY
```
Save actual stdout as checkpoint-validation.json. No all-expert coverage gate: state entry count and moment elements are observations, not number of experts; do not infer routing distribution from Adam states. Record size/hash local-only. Compare step 1 against T0002 (same settings/stack): exact non-timing history expected, report actual equality or discrepancies and investigate, no new tolerance/gate after observing. Compare final memory/state inventory with T0002 baseline. Report per-step losses (combined A2 weighted objective is not pure NLL), raw final NTP and validation, no claim of generalization. End source clean verification, pre/post GPU/disk, no automatic cleanup/restart.

## 4. Evidence and handover
Check IDs exactly environment,cuda_nano_epoch,checkpoint_validation. Mandatory REP files at experiments/results/T0003/attempt-NNN; add run-nano-epoch.json, checkpoint-validation.json and checkpoint hash reference when available. Preserve all step rows, metrics units, actual command elapsed, peak and median step time, maximum observed allocator memory, final state entry count/moment elements, observed differences from T0002. State not measured per-step optimizer coverage or actual continuous-context scaling. Record failed/blocked evidence even if no run JSON. No weights in Git. Hash and run from instruction checkout:
```bash
python tools/rep/validate_result.py "$EVID" --task T0003 --source d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa --checks environment,cuda_nano_epoch,checkpoint_validation
```
Append index row, inspect sanitized diff, commit/push evidence only, open PR referencing #35/#10, post handover in #35. Do not merge or close. Stop after T0003; BF16 or larger context requires separate reviewed implementation/protocol and owner authorization.
