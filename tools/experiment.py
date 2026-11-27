"""Synthetic-only experiment harness; no external corpus acquisition."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import torch

from optimizers import build_optimizer, optimizer_state_inventory
from splot.fp8 import convert_selected_linears, fp8_diagnostics, fp8_runtime_probe


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def git_identity(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('run from clean checkout; put outputs outside repository')
    return git('rev-parse', 'HEAD')


def atomic_save(value, path, *, tensor=False):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    if tensor:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_precision(device, precision):
    """Resolve the requested training precision and reject silent fallback."""
    if precision not in ('fp32', 'bf16', 'fp8'):
        raise ValueError('precision must be fp32, bf16 or fp8')
    dev = torch.device(device)
    if precision == 'fp32':
        return {'requested': 'fp32', 'policy': 'fp32', 'autocast_dtype': None,
                'autocast_dtype_name': None, 'probe_dtype': 'not_run', 'bf16_supported': None}
    if dev.type != 'cuda' and precision == 'fp8':
        raise RuntimeError('fp8 requires a CUDA device')
    supported = None
    if dev.type == 'cuda':
        supported = torch.cuda.is_bf16_supported()
        if not supported:
            raise RuntimeError('CUDA BF16 is not supported on this device')
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        probe = torch.nn.functional.linear(torch.ones(2, 4, device=dev), torch.ones(4, 4, device=dev))
    if probe.dtype != torch.bfloat16:
        raise RuntimeError(f'BF16 autocast unavailable on {dev.type}: probe projection returned {probe.dtype}')
    if precision == 'fp8':
        fp8_probe = fp8_runtime_probe(dev)
        if not fp8_probe.get('available'):
            raise RuntimeError('native FP8 unavailable: ' + str(fp8_probe.get('error')))
        return {'requested': 'fp8', 'policy': 'fp8-native-autograd-bf16-residual',
                'autocast_dtype': torch.bfloat16, 'autocast_dtype_name': 'bfloat16',
                'probe_dtype': 'bfloat16', 'bf16_supported': supported,
                'fp8_probe': fp8_probe}
    return {'requested': 'bf16', 'policy': 'bf16-autocast-fp32-master',
            'autocast_dtype': torch.bfloat16, 'autocast_dtype_name': 'bfloat16',
            'probe_dtype': 'bfloat16', 'bf16_supported': supported}


def batch_indices(n, batch_size, step, seed):
    if min(n, batch_size) < 1 or step < 0:
        raise ValueError('invalid schedule')
    batches = math.ceil(n/batch_size)
    epoch, index = divmod(step, batches)
    order = list(range(n)); random.Random(seed+epoch).shuffle(order)
    return order[index*batch_size:(index+1)*batch_size]


def collate(blocks, indices, device):
    rows = [blocks[i] for i in indices]
    inputs = tuple(torch.tensor([r[k] for r in rows], device=device) for k in
                   ('input_ids', 'document_ids', 'position_ids', 'valid'))
    labels = {k: torch.tensor([r['labels'][k] for r in rows], device=device) for k in (1, 2)}
    return inputs, labels


def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def snapshot_rng(device):
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else None}


def restore_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])


def train(model, optimizer, blocks, objective, identity, out, *, steps, batch_size, seed, device,
          resume=False, autocast_dtype=None):
    """One optimizer step per batch. Checkpoint is sole authoritative progress ledger."""
    if autocast_dtype is not None and autocast_dtype != torch.bfloat16:
        raise ValueError('autocast dtype must be bfloat16')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cp = out/'checkpoint.pt'
    if resume:
        # Only resume self-produced trusted checkpoints; Python RNG uses tuples.
        state = torch.load(cp, map_location='cpu', weights_only=True)
        if state['identity'] != identity: raise ValueError('run identity mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng']); history = state['history']
    else:
        if cp.exists() or any(out.iterdir()): raise FileExistsError('new run directory must be empty')
        history = []
    if steps < len(history): raise ValueError('steps is total target, not additional steps')
    model.train()
    for step in range(len(history), steps):
        indices = batch_indices(len(blocks), batch_size, step, seed)
        inputs, labels = collate(blocks, indices, device)
        if not labels[1].ne(-100).any(): raise ValueError('empty NTP batch')
        if torch.device(device).type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        sync(device); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with (torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype)
              if autocast_dtype is not None else nullcontext()):
            o = objective(model, inputs, labels)
        if not torch.isfinite(o.total): raise FloatingPointError('nonfinite objective')
        o.total.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('nonfinite gradient')
        optimizer.step(); sync(device)
        elapsed = time.perf_counter()-start
        row = {'step':step+1, 'batch_indices':indices, 'valid_tokens':int(inputs[3].sum()),
               'ntp_labels':int(labels[1].ne(-100).sum()), 'offset2_labels_available':int(labels[2].ne(-100).sum()),
               'objective':float(o.total.detach()), 'final_ntp':float(o.final_ntp.detach()),
               'early_ntp':float(o.early_ntp.detach()), 'offset2':float(o.offset2.detach()),
               'balance':float(o.balance.detach()), 'step_seconds':elapsed,
               'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
               'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else None}
        history.append(row)
        atomic_save({'identity':identity, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                     'rng':snapshot_rng(device), 'history':history}, cp, tensor=True)
    return history


@torch.no_grad()
def evaluate(model, blocks, predict, *, batch_size, device, autocast_dtype=None):
    if autocast_dtype is not None and autocast_dtype != torch.bfloat16:
        raise ValueError('autocast dtype must be bfloat16')
    model.eval(); nll=0.; count=0
    for i in range(0,len(blocks),batch_size):
        x,y=collate(blocks,list(range(i,min(i+batch_size,len(blocks)))),device)
        with (torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype)
              if autocast_dtype is not None else nullcontext()):
            logits=predict(model,x)
        nll+=float(torch.nn.functional.cross_entropy(logits.float().flatten(0,1),y[1].flatten(),ignore_index=-100,reduction='sum'))
        count+=int(y[1].ne(-100).sum())
    if not count or not math.isfinite(nll): raise ValueError('invalid validation')
    return {'ntp_labels':count,'nll_sum':nll,'mean_nll':nll/count,'perplexity':math.exp(nll/count)}


def logical_macs(blocks, indices, count_document):
    """Sum unpadded document contractions; not actual padded kernel work."""
    return sum(count_document(n) for i in indices
               for n in Counter(d for d in blocks[i]['document_ids'] if d >= 0).values())


def main():
    from splot import DenseConfig, RecurrentConfig, SPLOTLM, training_objective
    from data_fixture import synthetic_documents, split_documents, pack, manifest
    from accounting import Shape, account
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    p.add_argument('--steps',type=int,required=True); p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--resume',action='store_true'); a=p.parse_args()
    if a.steps<1: p.error('steps must be positive')
    root=Path(__file__).resolve().parents[1]; commit=git_identity(root)
    if a.out.resolve().is_relative_to(root): p.error('output must be outside checkout')
    c=json.loads(a.config.read_text()); t=c['training']; d=DenseConfig(**c['dense'])
    rc=RecurrentConfig(dense=d,**c['recurrence'])
    if d.vocab_size<264 or t['data']!='synthetic-v1': p.error('only synthetic-v1 runs supported')
    if t['precision'] not in ('fp32', 'bf16', 'fp8'): p.error('precision must be fp32, bf16 or fp8')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    if t['mode'] not in ('A0','A1','A2') or t['batch_size']<1 or not 1<=t['loops']<=rc.max_loops: p.error('invalid training settings')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if a.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    try:
        policy=resolve_precision(a.device,t['precision'])
    except (ValueError,RuntimeError) as exc:
        p.error(str(exc))
    seed_all(t['seed'])
    splits=split_documents(synthetic_documents()); packed={k:pack(v,t['sequence_length']) for k,v in splits.items() if k!='test'}
    data=manifest()
    model=SPLOTLM(rc,expert_intermediate=c['expert_intermediate']).to(a.device)
    actual=sum(v.numel() for v in model.parameters())
    if actual!=model.parameter_count(): raise RuntimeError('parameter count mismatch')
    fp8_policy=None
    if t['precision']=='fp8':
        try:
            fp8_policy=convert_selected_linears(model)
        except (ValueError,TypeError,RuntimeError) as exc:
            p.error(str(exc))
        if fp8_policy['selected_count'] < 1:
            p.error('no selected FP8 projections')
        if sum(v.numel() for v in model.parameters()) != actual:
            raise RuntimeError('FP8 conversion changed parameter inventory')
    try:
        opt, optimizer_policy = build_optimizer(model,t)
    except (ValueError,TypeError,RuntimeError) as exc:
        p.error(str(exc))
    if t['precision']=='fp8' and optimizer_policy['name'] != 'adamw':
        p.error('FP8 integration supports AdamW only')
    env={'python':platform.python_version(),'torch':str(torch.__version__),'device':a.device,
        'cpu':platform.processor() or platform.machine(),'platform':platform.platform(),'threads':1,
        'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else None,
        'requested_precision':policy['requested'],'precision_policy':policy['policy'],
        'autocast_dtype':policy['autocast_dtype_name'],'autocast_probe_dtype':policy['probe_dtype'],
        'bf16_supported':policy['bf16_supported'],'master_dtype':'float32',
        'optimizer':optimizer_policy['name'],'optimizer_implementation':optimizer_policy['implementation'],
        'fp8':fp8_policy,'fp8_probe':policy.get('fp8_probe'),
        'tf32_enabled':bool(torch.backends.cuda.matmul.allow_tf32) if a.device=='cuda' else False,
        'deterministic_algorithms':True}
    identity={'schema':1,'commit':commit,'config_sha256':digest(c),'data_sha256':digest(data),
              'packed_sha256':digest(packed),'environment':env,'settings':t}
    objective=lambda m,x,y:training_objective(m,x,y,mode=t['mode'],loops=t['loops'],depth_weight=t['depth_weight'],offset_weight=t['offset_weight'],balance_weight=t['balance_weight'])
    history=train(model,opt,packed['train'],objective,identity,a.out,steps=a.steps,batch_size=t['batch_size'],seed=t['seed'],device=a.device,resume=a.resume,autocast_dtype=policy['autocast_dtype'])
    val=evaluate(model,packed['validation'],lambda m,x:m(*x,loops=t['loops']).logits[t['loops']],batch_size=t['batch_size'],device=a.device,autocast_dtype=policy['autocast_dtype'])
    shape=Shape(d=d.width,vocab=d.vocab_size,q_heads=d.q_heads,kv_heads=d.kv_heads,head=d.head_dim,
                gdn_heads=d.q_heads,dense=d.intermediate,expert=c['expert_intermediate'])
    readouts=1 if t['mode']=='A0' else t['loops']
    offsets=min(2,t['loops']) if t['mode']=='A2' else 0
    count_document=lambda n:account(shape,loops=t['loops'],context=n,readouts=readouts,offsets=offsets)['prefill_all_logits_macs']
    macs=sum(logical_macs(packed['train'],r['batch_indices'],count_document) for r in history)
    precision_policy={'requested':policy['requested'],'policy':policy['policy'],
                      'autocast_dtype':policy['autocast_dtype_name'],'probe_dtype':policy['probe_dtype'],
                      'bf16_supported':policy['bf16_supported'],'fp8_probe':policy.get('fp8_probe')}
    optimizer_state=optimizer_state_inventory(opt)
    fp8_diag=fp8_diagnostics(model) if fp8_policy else None
    limitations=['synthetic byte fixture; no production tokenizer','template overlap across splits',
                 ('BF16 autocast with FP32 master weights/state; not optimized' if t['precision']=='bf16' else
                  ('Native FP8 selected projections with BF16 residual; not optimized' if t['precision']=='fp8' else 'FP32 reference, not optimized')),
                 'step timing excludes checkpoint I/O and validation',
                 'MAC estimate counts ideal unpadded contractions, not backward, padding or actual hardware FLOPs']
    if optimizer_policy['name']=='muon':
        limitations.append('Muon/AdamW hybrid correctness only; no optimizer superiority or convergence claim.')
    if t['precision']=='fp8':
        limitations.append('FP8 first-order projection integration only; no pretraining, throughput, convergence or quality claim.')
    report={'identity':identity,'config':c,'data_manifest':data,'parameters':actual,'history':history,'validation':val,
            'precision_policy':precision_policy,'optimizer_policy':optimizer_policy,'optimizer_state':optimizer_state,
            'fp8_policy':fp8_policy,'fp8_diagnostics':fp8_diag,
            'total_valid_tokens':sum(r['valid_tokens'] for r in history),'total_ntp_labels':sum(r['ntp_labels'] for r in history),
            'analytical_forward_macs':macs,'measured_flops':None,'energy_joules':None,'limitations':limitations}
    atomic_save(report,a.out/'run.json'); print(json.dumps({k:report[k] for k in ('parameters','total_valid_tokens','total_ntp_labels','validation')},indent=2))


if __name__=='__main__': main()
