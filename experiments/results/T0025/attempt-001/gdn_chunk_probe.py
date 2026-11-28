"""T0025 full-model scan vs opt-in chunk64 numerical/resource probe.

Numerical mode runs scan and chunk64 in one process with fresh models and
identical seed/input/partition, then applies the preregistered full-model
gates. Resource mode runs one method per process with one warmup and three
measured complete optimizer updates, using a scan reference gate for chunk64.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch

from context_probe import (_move_inputs, _partition_summary, _precheck,
                           build_contiguous_case, load_config, model_from,
                           tensor_digest)
from experiment import assert_post_update_finite, seed_all
from optimizers import build_optimizer, optimizer_state_inventory
from splot import ntp_loss

MODEL_PARAMETERS = 43055076


def all_tensors_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_tensors_finite(item) for item in value)
    return True


def losses_and_total(out, labels, cfg):
    t = cfg['training']
    final = ntp_loss(out.logits[2], labels[1])
    early = ntp_loss(out.logits[1], labels[1])
    aux = torch.stack((ntp_loss(out.offset2[1], labels[2]),
                       ntp_loss(out.offset2[2], labels[2]))).mean()
    balance = out.balance_loss
    total = final + t['depth_weight'] * early + t['offset_weight'] * aux + t['balance_weight'] * balance
    return total, {'total': float(total.detach()), 'final_ntp': float(final.detach()),
                   'early_ntp': float(early.detach()), 'offset2': float(aux.detach()),
                   'balance': float(balance.detach())}


def gradient_inventory(model):
    named = []
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        else:
            named.append((name, parameter.grad))
    return {
        'present': [name for name, _ in named],
        'missing': missing,
        'digest': tensor_digest(named),
        'elements': sum(value.numel() for _, value in named),
        'finite': all(bool(torch.isfinite(value).all()) for _, value in named),
    }


def memory_state():
    free, total = torch.cuda.mem_get_info()
    return {'free_bytes': free, 'total_bytes': total,
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def compare_tensor(candidate, reference):
    delta = candidate.double() - reference.double()
    rel = delta.norm().item() / max(reference.double().norm().item(), 1e-12)
    max_abs = delta.abs().max().item()
    ref_max = reference.double().abs().max().item()
    finite = bool(torch.isfinite(candidate).all().item())
    return {'rel_l2': rel, 'max_abs': max_abs, 'reference_max_abs': ref_max,
            'relative_ok': rel <= 0.02, 'max_abs_ok': max_abs <= 0.05 * max(1.0, ref_max),
            'finite': finite,
            'status': 'passed' if rel <= 0.02 and max_abs <= 0.05 * max(1.0, ref_max) and finite else 'failed'}


def routing_metrics(scan_routing, chunk_routing):
    scan = {key: route for key, route in scan_routing}
    chunk = {key: route for key, route in chunk_routing}
    if set(scan) != set(chunk):
        return {'status': 'failed', 'error': 'routing key set differs',
                'disagreement_fraction': 1.0, 'by_pass': {}, 'by_pool': {}}
    by_key = {}
    by_pass = {}
    by_pool = {}
    all_exact = True
    for key in sorted(scan):
        a = scan[key].selected.detach().cpu()
        b = chunk[key].selected.detach().cpu()
        if a.shape != b.shape:
            return {'status': 'failed', 'error': 'routing shape differs', 'by_pass': {}, 'by_pool': {}}
        exact = 0
        for row_a, row_b in zip(a.tolist(), b.tolist()):
            exact += int(set(row_a) == set(row_b))
        fraction = 1.0 - exact / max(1, len(a))
        all_exact = all_exact and fraction == 0.0
        by_key[str(key)] = {'tokens': len(a), 'disagreement_fraction': fraction,
                            'exact_sets': exact == len(a)}
        by_pass[str(key[0])] = by_pass.get(str(key[0]), []) + [fraction]
        by_pool[str(key[2])] = by_pool.get(str(key[2]), []) + [fraction]
    return {
        'status': 'passed' if all_exact else 'failed',
        'disagreement_fraction': max([row['disagreement_fraction'] for row in by_key.values()], default=0.0),
        'by_key': by_key,
        'by_pass': {key: sum(values) / len(values) for key, values in by_pass.items()},
        'by_pool': {key: sum(values) / len(values) for key, values in by_pool.items()},
    }


def compare_gradients(scan, chunk):
    names = sorted(set(scan) & set(chunk))
    inventory_same = sorted(scan) == sorted(chunk)
    global_num = 0.0
    global_den = 0.0
    per = {}
    for name in names:
        delta = chunk[name].double() - scan[name].double()
        global_num += float(delta.square().sum())
        global_den += float(scan[name].double().square().sum())
        rel = delta.norm().item() / max(scan[name].double().norm().item(), 1e-12)
        abs_l2 = delta.norm().item()
        per[name] = {'rel_l2': rel, 'abs_l2': abs_l2,
                     'status': 'passed' if rel <= 0.10 or abs_l2 <= 1e-7 else 'failed'}
    global_rel = math.sqrt(global_num / max(global_den, 1e-12))
    return {
        'inventory_same': inventory_same,
        'global_rel_l2': global_rel,
        'global_ok': global_rel <= 0.05,
        'per_tensor': per,
        'all_per_tensor_ok': all(row['status'] == 'passed' for row in per.values()),
        'status': 'passed' if inventory_same and global_rel <= 0.05 and
                  all(row['status'] == 'passed' for row in per.values()) else 'failed',
    }


def setup_model(cfg, device):
    seed_all(42)
    model = model_from(cfg)
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != MODEL_PARAMETERS:
        raise RuntimeError(f'parameter count mismatch: {actual}')
    initialization_sha256 = tensor_digest(model.named_parameters())
    model = model.to(device)
    optimizer, optimizer_policy = build_optimizer(model, cfg['training'])
    if sum(parameter.numel() for parameter in model.parameters()) != actual:
        raise RuntimeError('model parameter inventory changed')
    return {'model': model, 'optimizer': optimizer,
            'optimizer_policy': optimizer_policy,
            'initialization_sha256': initialization_sha256,
            'parameters': actual}


def build_state(cfg, length, device):
    setup = setup_model(cfg, device)
    inputs, labels, case = build_contiguous_case(length, 42)
    precheck = _precheck(setup['model'], inputs, labels, device, False)
    return setup, inputs, labels, case, precheck


def warmup_update(cfg, length, device):
    setup, inputs, labels, case, precheck = build_state(cfg, length, device)
    model, optimizer = setup['model'], setup['optimizer']
    moved_inputs, moved_labels = _move_inputs(inputs, labels, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
        total, losses = losses_and_total(out, moved_labels, cfg)
    total.backward()
    assert_post_update_finite(model, optimizer, 'warmup pre-update')
    optimizer.step()
    assert_post_update_finite(model, optimizer, 'warmup post-update')
    finite = all_tensors_finite(model.state_dict()) and all_tensors_finite(optimizer.state_dict())
    del setup, inputs, labels, case, precheck, model, optimizer, out, total
    gc.collect(); torch.cuda.empty_cache()
    return {'status': 'passed' if finite else 'failed', 'losses': losses}


def run_update(cfg, length, device, timed=False):
    setup, inputs, labels, case, precheck = build_state(cfg, length, device)
    model, optimizer = setup['model'], setup['optimizer']
    baseline = memory_state()
    torch.cuda.reset_peak_memory_stats()
    phases = {}
    sync = lambda: torch.cuda.synchronize(device) if torch.device(device).type == 'cuda' else None
    started = time.perf_counter()
    phase = time.perf_counter()
    moved_inputs, moved_labels = _move_inputs(inputs, labels, device)
    sync(); phases['transfer'] = time.perf_counter() - phase
    phase = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    phases['prepare'] = time.perf_counter() - phase
    phase = time.perf_counter()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
        total, losses = losses_and_total(out, moved_labels, cfg)
    sync(); phases['forward'] = time.perf_counter() - phase
    phase = time.perf_counter()
    total.backward()
    sync(); phases['backward'] = time.perf_counter() - phase
    phase = time.perf_counter()
    gradients = gradient_inventory(model)
    finite = all_tensors_finite(model.state_dict()) and all_tensors_finite(optimizer.state_dict())
    sync(); phases['checks'] = time.perf_counter() - phase
    phase = time.perf_counter()
    optimizer.step()
    sync(); phases['optimizer'] = time.perf_counter() - phase
    phase = time.perf_counter()
    assert_post_update_finite(model, optimizer, 'measured post-update')
    sync(); phases['post_update'] = time.perf_counter() - phase
    phases['end_to_end'] = time.perf_counter() - started
    result = {
        'setup': setup,
        'input': case,
        'precheck': precheck,
        'losses': losses,
        'gradients': gradients,
        'model_finite': all_tensors_finite(model.state_dict()),
        'optimizer_finite': all_tensors_finite(optimizer.state_dict()),
        'optimizer_state': optimizer_state_inventory(optimizer),
        'baseline': baseline,
        'phases': phases if timed else None,
        'out': out,
        'model': model,
        'optimizer': optimizer,
        'moved_labels': moved_labels,
    }
    return result


def release(result):
    keys = ('out', 'model', 'optimizer', 'moved_labels', 'setup', 'input', 'precheck')
    for key in keys:
        result.pop(key, None)
    gc.collect()
    torch.cuda.empty_cache()


def baseline_check(scan, reference_path):
    if not reference_path:
        return {'status': 'not_run', 'reason': 'no T0022 reference provided'}
    ref = json.loads(Path(reference_path).read_text())
    identity = ref.get('identity', ref.get('warmup', {}))
    expected = {'input_sha256': identity.get('input_sha256'),
                'initialization_sha256': identity.get('initialization_sha256'),
                'partition_plan_sha256': identity.get('partition_plan_sha256')}
    actual = scan['identity']
    return {
        'status': 'passed' if actual == expected and scan['losses'] == ref['losses'] else 'failed',
        'identity_match': actual == expected,
        'losses_match': scan['losses'] == ref['losses'],
        'reference': reference_path.name,
    }


def numerical(length, out, reference_path):
    scan_cfg = load_config('bf16')[1]
    scan_cfg['recurrence']['gdn_algorithm'] = 'scan'
    chunk_cfg = load_config('bf16')[1]
    chunk_cfg['recurrence']['gdn_algorithm'] = 'chunk64'
    device = 'cuda'
    scan = run_update(scan_cfg, length, device, timed=False)
    scan_identity = {
        'input_sha256': scan['input']['input_sha256'],
        'initialization_sha256': scan['setup']['initialization_sha256'],
        'partition_plan_sha256': _partition_summary(scan['setup']['optimizer_policy'])['plan_sha256'],
    }
    scan['identity'] = scan_identity
    scan_grads = {name: value.detach().clone() for name, value in
                  zip(scan['gradients']['present'], [g for n, g in scan['model'].named_parameters() if g.grad is not None])}
    scan_out = scan['out']
    scan_losses = scan['losses']
    scan_finite = {'model': scan['model_finite'], 'optimizer': scan['optimizer_finite'],
                   'gradients': scan['gradients']['finite']}
    scan_baseline = baseline_check(scan, reference_path)
    scan_routing = scan_out.routing
    del scan['model'], scan['optimizer'], scan['setup']
    gc.collect(); torch.cuda.empty_cache()

    chunk = run_update(chunk_cfg, length, device, timed=False)
    chunk_identity = {
        'input_sha256': chunk['input']['input_sha256'],
        'initialization_sha256': chunk['setup']['initialization_sha256'],
        'partition_plan_sha256': _partition_summary(chunk['setup']['optimizer_policy'])['plan_sha256'],
    }
    chunk_grads = {name: value.detach() for name, value in
                   zip(chunk['gradients']['present'], [g for n, g in chunk['model'].named_parameters() if g.grad is not None])}
    identity_match = scan_identity == chunk_identity
    loss_metrics = {name: {'abs_diff': abs(scan_losses[name] - chunk['losses'][name]),
                           'status': 'passed' if abs(scan_losses[name] - chunk['losses'][name]) <= 0.02 else 'failed'}
                    for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance')}
    tensor_metrics = {}
    for label, ref, cand in (
        ('logits_1', scan_out.logits[1], chunk['out'].logits[1]),
        ('logits_2', scan_out.logits[2], chunk['out'].logits[2]),
        ('hidden_1', scan_out.hidden[1], chunk['out'].hidden[1]),
        ('hidden_2', scan_out.hidden[2], chunk['out'].hidden[2]),
        ('offset2_1', scan_out.offset2[1], chunk['out'].offset2[1]),
        ('offset2_2', scan_out.offset2[2], chunk['out'].offset2[2]),
    ):
        tensor_metrics[label] = compare_tensor(cand, ref)
    grad = compare_gradients(scan_grads, chunk_grads)
    missing_same = scan['gradients']['missing'] == chunk['gradients']['missing']
    grad['missing_inventory_same'] = missing_same
    grad['missing'] = {'scan': scan['gradients']['missing'], 'chunk64': chunk['gradients']['missing']}
    if not missing_same:
        grad['status'] = 'failed'
    routing = routing_metrics(scan_routing, chunk['out'].routing)
    finite_ok = (all(scan_finite.values()) and chunk['model_finite'] and
                 chunk['optimizer_finite'] and chunk['gradients']['finite'])
    gate_ok = (scan_baseline['status'] == 'passed' and identity_match and
               all(row['status'] == 'passed' for row in loss_metrics.values()) and
               all(row['status'] == 'passed' for row in tensor_metrics.values()) and
               grad['status'] == 'passed' and routing['status'] == 'passed' and finite_ok)
    report = {
        'status': 'passed' if gate_ok else 'failed',
        'mode': 'numerical', 'length': length, 'device': device,
        'source_commit': None,
        'scan_identity': scan_identity, 'chunk64_identity': chunk_identity,
        'identity_match': identity_match,
        'baseline': scan_baseline,
        'losses': {'scan': scan_losses, 'chunk64': chunk['losses'], 'components': loss_metrics},
        'tensors': tensor_metrics,
        'gradients': grad,
        'routing': routing,
        'finite': {'scan': scan_finite, 'chunk64': {'model': chunk['model_finite'],
                                                    'optimizer': chunk['optimizer_finite'],
                                                    'gradients': chunk['gradients']['finite']}},
        'limitations': [
            'One BF16 synthetic update per mode; no learning-quality or convergence claim.',
            'Candidate chunk64 is a different contraction order; cross-mode bitwise equality is not required.',
            'Routing equality is top2 sets per valid position; no raw IDs are published.',
        ],
    }
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'length': length,
                      'identity_match': identity_match,
                      'baseline': scan_baseline['status'],
                      'loss_gate': all(r['status'] == 'passed' for r in loss_metrics.values()),
                      'tensor_gate': all(r['status'] == 'passed' for r in tensor_metrics.values()),
                      'gradient_gate': grad['status'], 'routing_gate': routing['status'],
                      'finite': finite_ok}, indent=2, sort_keys=True))
    return 0 if report['status'] == 'passed' else 1


def resource(length, algorithm, out):
    device = 'cuda'
    cfg = load_config('bf16')[1]
    cfg['recurrence']['gdn_algorithm'] = algorithm
    warm = warmup_update(cfg, length, device)
    if warm['status'] != 'passed':
        raise RuntimeError('warmup failed')
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    scan_cfg = load_config('bf16')[1]
    scan_cfg['recurrence']['gdn_algorithm'] = 'scan'
    reference = None
    if algorithm == 'chunk64':
        reference = run_update(scan_cfg, length, device, timed=False)
        ref_grads = {name: value.detach().clone() for name, value in
                     zip(reference['gradients']['present'],
                         [g for n, g in reference['model'].named_parameters() if g.grad is not None])}
        ref_out = reference['out']
        ref_losses = reference['losses']
        ref_identity = {
            'input_sha256': reference['input']['input_sha256'],
            'initialization_sha256': reference['setup']['initialization_sha256'],
            'partition_plan_sha256': _partition_summary(reference['setup']['optimizer_policy'])['plan_sha256'],
        }
    else:
        ref_grads = ref_out = ref_losses = ref_identity = None
    repeats = []
    for repeat in range(1, 4):
        result = run_update(cfg, length, device, timed=True)
        identity = {'input_sha256': result['input']['input_sha256'],
                    'initialization_sha256': result['setup']['initialization_sha256'],
                    'partition_plan_sha256': _partition_summary(result['setup']['optimizer_policy'])['plan_sha256']}
        if ref_identity is not None and identity != ref_identity:
            raise ValueError('repeat initialization/input/partition mismatch')
        baseline = result['baseline']
        peak = memory_state()
        phases = result['phases']
        gate = None
        if algorithm == 'chunk64':
            grad = compare_gradients(ref_grads, {name: value.detach() for name, value in
                                                zip(result['gradients']['present'],
                                                    [g for n, g in result['model'].named_parameters() if g.grad is not None])})
            loss_metrics = {name: {'abs_diff': abs(ref_losses[name] - result['losses'][name]),
                                   'status': 'passed' if abs(ref_losses[name] - result['losses'][name]) <= 0.02 else 'failed'}
                            for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance')}
            tensors = {}
            for label, ref, cand in (
                ('logits_1', ref_out.logits[1], result['out'].logits[1]),
                ('logits_2', ref_out.logits[2], result['out'].logits[2]),
                ('hidden_1', ref_out.hidden[1], result['out'].hidden[1]),
                ('hidden_2', ref_out.hidden[2], result['out'].hidden[2]),
                ('offset2_1', ref_out.offset2[1], result['out'].offset2[1]),
                ('offset2_2', ref_out.offset2[2], result['out'].offset2[2]),
            ):
                tensors[label] = compare_tensor(cand, ref)
            routing = routing_metrics(ref_out.routing, result['out'].routing)
            grad['missing_inventory_same'] = (result['gradients']['missing'] ==
                                              reference['gradients']['missing'])
            if not grad['missing_inventory_same']:
                grad['status'] = 'failed'
            gate = {'losses': loss_metrics, 'tensors': tensors, 'gradients': grad,
                    'routing': routing,
                    'status': 'passed' if (all(r['status'] == 'passed' for r in loss_metrics.values())
                                           and all(r['status'] == 'passed' for r in tensors.values())
                                           and grad['status'] == 'passed' and routing['status'] == 'passed') else 'failed'}
        repeats.append({
            'repeat': repeat,
            'identity': identity,
            'phases': phases,
            'peak_allocated_bytes': peak['peak_allocated_bytes'],
            'peak_reserved_bytes': peak['peak_reserved_bytes'],
            'baseline_allocated_bytes': baseline['allocated_bytes'],
            'increment_allocated_bytes': peak['peak_allocated_bytes'] - baseline['allocated_bytes'],
            'increment_reserved_bytes': peak['peak_reserved_bytes'] - baseline['reserved_bytes'],
            'losses': result['losses'],
            'finite': result['model_finite'] and result['optimizer_finite'] and result['gradients']['finite'],
            'gate': gate,
        })
        del result, baseline, peak
        gc.collect(); torch.cuda.empty_cache()
    ok = all(row['finite'] and (row['gate'] is None or row['gate']['status'] == 'passed') for row in repeats)
    report = {'status': 'passed' if ok else 'failed', 'mode': 'resource',
              'length': length, 'algorithm': algorithm, 'seed': 42,
              'reference_algorithm': 'scan' if algorithm == 'chunk64' else None,
              'repeats': repeats,
              'limitations': ['Three measured complete updates per method/length; descriptive only.',
                              'No sustained throughput or full-model production claim.']}
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'length': length, 'algorithm': algorithm,
                      'repeats': len(repeats)}, indent=2))
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('numerical', 'resource'), required=True)
    parser.add_argument('--length', type=int, choices=(512, 2048), required=True)
    parser.add_argument('--algorithm', choices=('scan', 'chunk64'), default='scan')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--t0022-reference', type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == 'numerical':
        return numerical(args.length, args.out, args.t0022_reference)
    return resource(args.length, args.algorithm, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
