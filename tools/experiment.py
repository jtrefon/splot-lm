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

from optimizers import (assert_optimizer_state_finite, build_optimizer,
                        is_hybrid_optimizer, optimizer_state_inventory)
from splot.fp8 import (aggregate_fp8_diagnostics, clear_fp8_weight_cache,
                       convert_selected_linears, fp8_assert_finite, fp8_diagnostics,
                       fp8_runtime_probe, has_fp8, reset_fp8_audit)

SHARD_FORMAT = 'splot-tokenized-shards-v2'


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


def load_shard_dataset(root, manifest_sha256, *, expected_vocab, expected_length):
    """Load one verified synthetic v2 manifest and expose train/validation only."""
    from tokenized_shards import load_dataset
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 or any(
            character not in '0123456789abcdef' for character in manifest_sha256):
        raise ValueError('manifest SHA-256 must be 64 lowercase hex characters')
    dataset = load_dataset(Path(root), manifest_sha256)
    manifest = dataset['manifest']
    if manifest.get('format') != SHARD_FORMAT or manifest.get('version') != 2:
        raise ValueError('only splot-tokenized-shards-v2 is supported')
    tokenizer = manifest['tokenizer']
    if tokenizer['actual_vocab_size'] != expected_vocab:
        raise ValueError('dataset tokenizer vocabulary does not match model vocabulary')
    if manifest['pack_config']['length'] != expected_length:
        raise ValueError('dataset block length does not match sequence length')
    blocks = {}
    for split in ('train', 'validation'):
        selected = []
        for block in dataset['blocks']:
            if block['split'] != split:
                continue
            converted = dict(block)
            converted['labels'] = {int(key): list(value)
                                   for key, value in block['labels'].items()}
            selected.append(converted)
        if not selected:
            raise ValueError(f'dataset has no {split} blocks')
        blocks[split] = selected
    summary = {
        'format': manifest['format'],
        'version': manifest['version'],
        'manifest_sha256': manifest_sha256,
        'tokenizer': dict(tokenizer),
        'logical_digest': manifest['logical_digest'],
        'packed_logical_digest': manifest['packed_logical_digest'],
        'physical_manifest_digest': manifest['physical_manifest_digest'],
        'counts': manifest['counts'],
        'pack_config': dict(manifest['pack_config']),
        'source_identity': dict(manifest['source_identity']),
        'training_splits': ['train', 'validation'],
    }
    return blocks, summary


def annotate_fp8_policy(fp8_policy, optimizer_policy):
    if fp8_policy is not None:
        fp8_policy['optimizer'] = optimizer_policy['name']
        fp8_policy['adamw_only'] = optimizer_policy['name'] == 'adamw'
    return fp8_policy


def assert_post_update_finite(model, optimizer, stage='post-update'):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f'nonfinite master weight {name} at {stage}')
    assert_optimizer_state_finite(optimizer, stage)


def assert_loss_finite(objective, stage='forward'):
    for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance'):
        value = getattr(objective, name, None)
        if value is not None and not torch.isfinite(value).all():
            raise FloatingPointError(f'nonfinite {name} at {stage}')


def snapshot_rng(device):
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else None}


def restore_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])


def train(model, optimizer, blocks, objective, identity, out, *, steps, batch_size, seed, device,
          resume=False, autocast_dtype=None, post_update_check=False,
          checkpoint_interval=1, checkpoint_stats=None):
    """One optimizer step per batch. Checkpoint is sole authoritative progress ledger."""
    if autocast_dtype is not None and autocast_dtype != torch.bfloat16:
        raise ValueError('autocast dtype must be bfloat16')
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError('checkpoint interval must be a positive integer')
    if checkpoint_stats is None:
        checkpoint_stats = {}
    checkpoint_stats.update({'interval': checkpoint_interval, 'writes': 0,
                             'write_seconds': 0., 'bytes_total': 0,
                             'bytes_last': 0, 'write_steps': []})
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cp = out/'checkpoint.pt'
    if resume:
        # Only resume self-produced trusted checkpoints; Python RNG uses tuples.
        state = torch.load(cp, map_location='cpu', weights_only=True)
        if state['identity'] != identity: raise ValueError('run identity mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng']); history = state['history']
        if has_fp8(model): clear_fp8_weight_cache(model)
    else:
        if cp.exists() or any(out.iterdir()): raise FileExistsError('new run directory must be empty')
        history = []
    checkpoint_stats['initial_history_steps'] = len(history)
    if steps < len(history): raise ValueError('steps is total target, not additional steps')
    fp8_enabled = has_fp8(model)
    post_update_check = bool(post_update_check or fp8_enabled or is_hybrid_optimizer(optimizer))
    model.train()
    for step in range(len(history), steps):
        indices = batch_indices(len(blocks), batch_size, step, seed)
        inputs, labels = collate(blocks, indices, device)
        if not labels[1].ne(-100).any(): raise ValueError('empty NTP batch')
        if torch.device(device).type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        if fp8_enabled: reset_fp8_audit(model)
        sync(device); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        audit = None
        try:
            with (torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype)
                  if autocast_dtype is not None else nullcontext()):
                o = objective(model, inputs, labels)
            if not torch.isfinite(o.total): raise FloatingPointError('nonfinite objective')
            if fp8_enabled:
                assert_loss_finite(o, 'forward')
                fp8_assert_finite(model, 'forward')
            o.total.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('nonfinite gradient')
            if fp8_enabled: fp8_assert_finite(model, 'backward')
            audit = fp8_diagnostics(model) if fp8_enabled else None
        except Exception:
            if fp8_enabled: clear_fp8_weight_cache(model)
            raise
        if fp8_enabled: clear_fp8_weight_cache(model)
        optimizer.step(); sync(device)
        if post_update_check:
            assert_post_update_finite(model, optimizer, f'post-update step {step+1}')
        elapsed = time.perf_counter()-start
        row = {'step':step+1, 'batch_indices':indices, 'valid_tokens':int(inputs[3].sum()),
               'ntp_labels':int(labels[1].ne(-100).sum()), 'offset2_labels_available':int(labels[2].ne(-100).sum()),
               'objective':float(o.total.detach()), 'final_ntp':float(o.final_ntp.detach()),
               'early_ntp':float(o.early_ntp.detach()), 'offset2':float(o.offset2.detach()),
               'balance':float(o.balance.detach()), 'step_seconds':elapsed,
               'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
               'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else None}
        if audit is not None:
            row['audit'] = audit
        history.append(row)
        checkpoint_stats['final_history_steps'] = len(history)
        if (step+1) % checkpoint_interval == 0 or step+1 == steps:
            write_started = time.perf_counter()
            atomic_save({'identity':identity, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                         'rng':snapshot_rng(device), 'history':history}, cp, tensor=True)
            checkpoint_stats['write_seconds'] += time.perf_counter() - write_started
            checkpoint_stats['writes'] += 1
            checkpoint_stats['bytes_last'] = cp.stat().st_size
            checkpoint_stats['bytes_total'] += checkpoint_stats['bytes_last']
            checkpoint_stats['write_steps'].append(step+1)
    checkpoint_stats['final_history_steps'] = len(history)
    return history


@torch.no_grad()
def evaluate(model, blocks, predict, *, batch_size, device, autocast_dtype=None):
    if autocast_dtype is not None and autocast_dtype != torch.bfloat16:
        raise ValueError('autocast dtype must be bfloat16')
    fp8_enabled = has_fp8(model)
    if fp8_enabled:
        reset_fp8_audit(model)
        clear_fp8_weight_cache(model)
    model.eval(); nll=0.; count=0
    for i in range(0,len(blocks),batch_size):
        x,y=collate(blocks,list(range(i,min(i+batch_size,len(blocks)))),device)
        with (torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype)
              if autocast_dtype is not None else nullcontext()):
            logits=predict(model,x)
        if fp8_enabled:
            fp8_assert_finite(model, 'evaluation')
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite evaluation logits')
        nll+=float(torch.nn.functional.cross_entropy(logits.float().flatten(0,1),y[1].flatten(),ignore_index=-100,reduction='sum'))
        count+=int(y[1].ne(-100).sum())
    if not count or not math.isfinite(nll): raise ValueError('invalid validation')
    result = {'ntp_labels':count,'nll_sum':nll,'mean_nll':nll/count,'perplexity':math.exp(nll/count)}
    if fp8_enabled:
        result['fp8_audit'] = fp8_diagnostics(model)
        result['fp8_cache_scope'] = 'cleared-before-and-after-evaluation'
        clear_fp8_weight_cache(model)
        reset_fp8_audit(model)
    return result


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
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dataset-root',type=Path,default=None)
    p.add_argument('--manifest-sha256',default=None)
    p.add_argument('--source-snapshot-sha256',default=None)
    p.add_argument('--checkpoint-interval',type=int,default=None); a=p.parse_args()
    if a.steps<1: p.error('steps must be positive')
    root=Path(__file__).resolve().parents[1]; commit=git_identity(root)
    if a.out.resolve().is_relative_to(root): p.error('output must be outside checkout')
    c=json.loads(a.config.read_text()); t=c['training']; d=DenseConfig(**c['dense'])
    rc=RecurrentConfig(dense=d,**c['recurrence'])
    if t['data'] not in ('synthetic-v1','synthetic-shards-v2','pilot-blocks-v1'):
        p.error('data must be synthetic-v1, synthetic-shards-v2 or pilot-blocks-v1')
    dataset_modes = ('synthetic-shards-v2','pilot-blocks-v1')
    if t['data']=='synthetic-v1' and (a.dataset_root is not None or a.manifest_sha256 is not None
                                    or a.source_snapshot_sha256 is not None):
        p.error('dataset options are only valid for shard/pilot modes')
    if t['data'] in dataset_modes and (a.dataset_root is None or a.manifest_sha256 is None):
        p.error(f"{t['data']} requires --dataset-root and --manifest-sha256")
    if t['data']=='synthetic-shards-v2' and a.source_snapshot_sha256 is not None:
        p.error('--source-snapshot-sha256 is only valid for pilot-blocks-v1')
    reuse = t.get('fp8_weight_reuse', False)
    if not isinstance(reuse, bool):
        p.error('fp8_weight_reuse must be boolean')
    interval = a.checkpoint_interval if a.checkpoint_interval is not None else t.get('checkpoint_interval', 1)
    if type(interval) is not int or interval < 1:
        p.error('checkpoint interval must be a positive integer')
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
    dataset_manifest_sha256 = None
    dataset_source_snapshot_sha256 = None
    if t['data']=='synthetic-v1':
        splits=split_documents(synthetic_documents()); packed={k:pack(v,t['sequence_length']) for k,v in splits.items() if k!='test'}
        data=manifest()
    elif t['data']=='synthetic-shards-v2':
        try:
            packed, data = load_shard_dataset(
                a.dataset_root, a.manifest_sha256,
                expected_vocab=d.vocab_size, expected_length=t['sequence_length'])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            p.error(f'dataset ingestion failed: {exc}')
        dataset_manifest_sha256 = a.manifest_sha256
        packed = {'train': packed['train'], 'validation': packed['validation']}
        if not packed['train'] or not packed['validation']:
            p.error('dataset must contain train and validation blocks')
    else:
        from pilot_ingestion import load_pilot_dataset
        try:
            loaded = load_pilot_dataset(
                a.dataset_root, a.manifest_sha256,
                expected_source_snapshot_sha256=a.source_snapshot_sha256,
                expected_model_vocab=d.vocab_size)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            p.error(f'pilot ingestion failed: {exc}')
        packed = {'train': loaded['blocks']['train'],
                  'validation': loaded['blocks']['validation']}
        data = loaded['manifest']
        dataset_manifest_sha256 = a.manifest_sha256
        dataset_source_snapshot_sha256 = data['source_snapshot_sha256']
        if not packed['train'] or not packed['validation']:
            p.error('pilot dataset must contain train and validation blocks')
    model=SPLOTLM(rc,expert_intermediate=c['expert_intermediate']).to(a.device)
    actual=sum(v.numel() for v in model.parameters())
    if actual!=model.parameter_count(): raise RuntimeError('parameter count mismatch')
    fp8_policy=None
    if t['precision']=='fp8':
        try:
            fp8_policy=convert_selected_linears(model, weight_reuse=reuse)
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
    annotate_fp8_policy(fp8_policy, optimizer_policy)
    env={'python':platform.python_version(),'torch':str(torch.__version__),'device':a.device,
        'cpu':platform.processor() or platform.machine(),'platform':platform.platform(),'threads':1,
        'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else None,
        'requested_precision':policy['requested'],'precision_policy':policy['policy'],
        'autocast_dtype':policy['autocast_dtype_name'],'autocast_probe_dtype':policy['probe_dtype'],
        'bf16_supported':policy['bf16_supported'],'master_dtype':'float32',
        'optimizer':optimizer_policy['name'],'optimizer_implementation':optimizer_policy['implementation'],
        'fp8':fp8_policy,'fp8_probe':policy.get('fp8_probe'),
        'data_selector':t['data'],'dataset_manifest_sha256':dataset_manifest_sha256,
        'dataset_source_snapshot_sha256':dataset_source_snapshot_sha256,
        'dataset_tokenizer_sha256':data.get('tokenizer_sha256') if t['data']=='pilot-blocks-v1' else None,
        'tokenizer_actual_vocab_size':(d.vocab_size if t['data']=='synthetic-shards-v2'
                                        else data.get('vocab_size') if t['data']=='pilot-blocks-v1' else None),
        'fp8_weight_reuse':reuse,
        'checkpoint_interval':interval,
        'tf32_enabled':bool(torch.backends.cuda.matmul.allow_tf32) if a.device=='cuda' else False,
        'deterministic_algorithms':True}
    identity={'schema':1,'commit':commit,'config_sha256':digest(c),'data_sha256':digest(data),
              'packed_sha256':digest(packed),'environment':env,'settings':t,
              'checkpoint_interval':interval,'fp8_weight_reuse':reuse,
              'dataset_source_snapshot_sha256':dataset_source_snapshot_sha256}
    objective=lambda m,x,y:training_objective(m,x,y,mode=t['mode'],loops=t['loops'],depth_weight=t['depth_weight'],offset_weight=t['offset_weight'],balance_weight=t['balance_weight'])
    post_update_check = t['precision']=='fp8' or optimizer_policy['name']=='muon'
    checkpoint_stats={}
    history=train(model,opt,packed['train'],objective,identity,a.out,steps=a.steps,batch_size=t['batch_size'],seed=t['seed'],device=a.device,resume=a.resume,autocast_dtype=policy['autocast_dtype'],post_update_check=post_update_check,checkpoint_interval=interval,checkpoint_stats=checkpoint_stats)
    validation_started=time.perf_counter()
    val=evaluate(model,packed['validation'],lambda m,x:m(*x,loops=t['loops']).logits[t['loops']],batch_size=t['batch_size'],device=a.device,autocast_dtype=policy['autocast_dtype'])
    validation_seconds=time.perf_counter()-validation_started
    evaluation_fp8_audit=val.pop('fp8_audit',None)
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
    fp8_diag=aggregate_fp8_diagnostics([row.get('audit') for row in history]) if fp8_policy else None
    if t['data']=='synthetic-shards-v2':
        tokenizer_vocab = data.get('tokenizer', {}).get('actual_vocab_size')
    elif t['data']=='pilot-blocks-v1':
        tokenizer_vocab = data.get('vocab_size')
    else:
        tokenizer_vocab = None
    data_policy = {
        'mode': t['data'],
        'manifest_sha256': dataset_manifest_sha256,
        'source_snapshot_sha256': dataset_source_snapshot_sha256,
        'tokenizer_sha256': data.get('tokenizer_sha256',
                                     data.get('tokenizer', {}).get('artifact_sha256')),
        'tokenizer_actual_vocab_size': tokenizer_vocab,
        'model_vocab_size': d.vocab_size,
        'unused_token_id_count': (d.vocab_size - tokenizer_vocab
                                  if tokenizer_vocab is not None else None),
        'unused_token_id_ranges': data.get('unused_token_id_ranges'),
        'logical_digest': data.get('logical_sha256', data.get('logical_digest')),
        'packed_logical_digest': data.get('packed_logical_digest'),
        'test_used_for_training': False,
        'test_validated_only': data.get('test_validated_only', False),
    }
    checkpoint_policy = {
        'interval': interval,
        'default': 1,
        'final_step_always_saved': True,
        'atomic_replace': True,
        'retention_service': False,
    }
    invocation = {
        'mode': 'resume' if a.resume else 'fresh',
        'initial_history_steps': checkpoint_stats.get('initial_history_steps'),
        'final_history_steps': checkpoint_stats.get('final_history_steps'),
        'invocation_history_steps': (checkpoint_stats.get('final_history_steps', 0)
                                     - checkpoint_stats.get('initial_history_steps', 0)),
    }
    limitations=['synthetic byte fixture; no production tokenizer','template overlap across splits',
                 ('BF16 autocast with FP32 master weights/state; not optimized' if t['precision']=='bf16' else
                  ('Native FP8 selected projections with BF16 residual; not optimized' if t['precision']=='fp8' else 'FP32 reference, not optimized')),
                 'step timing excludes checkpoint I/O and validation',
                 'MAC estimate counts ideal unpadded contractions, not backward, padding or actual hardware FLOPs']
    if optimizer_policy['name']=='muon':
        limitations.append('Muon/AdamW hybrid correctness only; no optimizer superiority or convergence claim.')
    if t['precision']=='fp8':
        limitations.append('FP8 first-order projection integration only; no pretraining, throughput, convergence or quality claim.')
    if t['precision']=='fp8' and optimizer_policy['name']=='muon':
        limitations.append('Combined FP8 + Muon correctness only; no interaction, convergence or optimizer-superiority claim.')
    if t['data']=='synthetic-shards-v2':
        limitations.append('Verified synthetic v2 shard ingestion only; no real-source approval, production dataset or test evaluation.')
    if t['data']=='pilot-blocks-v1':
        limitations.append('Bounded pilot-blocks-v1 ingestion only; the source snapshot must be separately verified against reviewed T0016 pins; test data is integrity-checked only.')
    if reuse:
        limitations.append('FP8 weight reuse is per-update only; no persistent cache or second-order claim.')
    if interval > 1:
        limitations.append('Checkpoint interval reduces durable write frequency; interruption can lose up to interval-1 updates since the prior checkpoint under the trusted single-writer assumption.')
    report={'identity':identity,'config':c,'data_manifest':data,'data_policy':data_policy,
            'parameters':actual,'history':history,'validation':val,'validation_seconds':validation_seconds,
            'checkpoint_policy':checkpoint_policy,'checkpoint_stats':checkpoint_stats,'invocation':invocation,
            'precision_policy':precision_policy,'optimizer_policy':optimizer_policy,'optimizer_state':optimizer_state,
            'fp8_policy':fp8_policy,'fp8_diagnostics':fp8_diag,
            'fp8_evaluation_diagnostics':evaluation_fp8_audit,
            'total_valid_tokens':sum(r['valid_tokens'] for r in history),'total_ntp_labels':sum(r['ntp_labels'] for r in history),
            'analytical_forward_macs':macs,'measured_flops':None,'energy_joules':None,'limitations':limitations}
    atomic_save(report,a.out/'run.json'); print(json.dumps({k:report[k] for k in ('parameters','total_valid_tokens','total_ntp_labels','validation')},indent=2))


if __name__=='__main__': main()
