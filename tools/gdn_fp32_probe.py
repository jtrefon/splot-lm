"""T0033 opt-in FP32 chunk64 CUDA qualification and resource probe.

Forward/backward plus complete Muon/AdamW updates only. No real data, sustained
training, BF16/FP8 chunk64, forced routing or automatic retry. Raw tensors and
routes are never written; only digests and aggregate gate statistics leave the
process.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from context_probe import (build_contiguous_case, load_config, model_from,
                           tensor_digest, _move_inputs, _partition_summary)
from data_fixture import synthetic_documents, split_documents, pack
from experiment import (assert_post_update_finite, digest, evaluate,
                        seed_all, train)
from splot import ntp_loss
from optimizers import build_optimizer, optimizer_state_inventory
from splot import training_objective

SOURCE = '7dac49e083eae0068ff361e7a81b3a4f41055b3f'
EXPECTED_INIT = 'ba9bf9d8472a9b940a416e91f3123ef4163913b7b334291540cb3b321fbaca5b'
EXPECTED_INPUT_512_42 = 'c0aa82fc0906b8a1437b5328f8d2396ab4f79e7b1528e8be399ea107da899270'
EXPECTED_PARTITION = '1bed555467ef9a955079b8260636971dafa5b0b308962e9a18e23a5ffa7bf9ed'
EXPECTED_PARAMETERS = 43055076
FP32_SCAN = 'configs/experiments/splot-nano-synthetic-fp32-scan.json'
FP32_CHUNK = 'configs/experiments/splot-nano-synthetic-fp32-chunk64.json'
BF16_SCAN = 'configs/experiments/splot-nano-synthetic-bf16-muon.json'
SMOKE_SCAN = 'configs/experiments/splot-smoke-synthetic-fp32-scan.json'
SMOKE_CHUNK = 'configs/experiments/splot-smoke-synthetic-fp32-chunk64.json'
T0031_REF = 'experiments/results/T0031/attempt-001/forward-backward.json'


def losses_and_total(out, labels, cfg):
    t = cfg['training']
    final = ntp_loss(out.logits[2], labels[1])
    early = ntp_loss(out.logits[1], labels[1])
    aux = torch.stack((ntp_loss(out.offset2[1], labels[2]),
                       ntp_loss(out.offset2[2], labels[2]))).mean()
    balance = out.balance_loss
    total = (final + t['depth_weight'] * early + t['offset_weight'] * aux
             + t['balance_weight'] * balance)
    return total, {'total': float(total.detach()),
                   'final_ntp': float(final.detach()),
                   'early_ntp': float(early.detach()),
                   'offset2': float(aux.detach()),
                   'balance': float(balance.detach())}


def finite(tensor):
    return bool(torch.isfinite(tensor).all().item())


def stats_pair(reference, candidate):
    ref = reference.detach().float()
    cand = candidate.detach().float()
    delta = cand.double() - ref.double()
    return {
        'elements': int(ref.numel()),
        'rel_l2': delta.norm().item() / max(ref.double().norm().item(), 1e-12),
        'max_abs': delta.abs().max().item(),
        'reference_max_abs': ref.double().abs().max().item(),
        'finite': finite(ref) and finite(cand),
    }


def tensor_gate(row):
    return (row['rel_l2'] <= 0.02
            and row['max_abs'] <= 0.05 * max(1.0, row['reference_max_abs'])
            and row['finite'])


def route_counts(scan_routes, cand_routes):
    by_key = {}
    total = mismatches = 0
    for key in scan_routes:
        a = scan_routes[key]['selected']
        b = cand_routes[key]['selected']
        same = (torch.sort(a, dim=-1).values == torch.sort(b, dim=-1).values).all(-1)
        n = int(same.numel())
        m = int((~same).sum())
        by_key[str(key)] = {'tokens': n, 'mismatch_tokens': m,
                            'disagreement_fraction': m / max(1, n)}
        total += n
        mismatches += m
    return {'tokens': total, 'mismatch_tokens': mismatches,
            'disagreement_fraction': mismatches / max(1, total),
            'by_key': by_key}


def compare_outputs(reference, candidate):
    losses = {name: abs(reference['losses'][name] - candidate['losses'][name])
              for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance')}
    tensors = {}
    for group in ('logits', 'hidden', 'offset2'):
        for depth in sorted(reference['output'][group]):
            tensors[f'{group}_{depth}'] = stats_pair(
                reference['output'][group][depth],
                candidate['output'][group][depth])
    routing = route_counts(reference['routes'], candidate['routes'])
    loss_ok = all(v <= 0.02 for v in losses.values())
    tensor_ok = all(tensor_gate(row) for row in tensors.values())
    routing_ok = routing['mismatch_tokens'] == 0
    finite_ok = reference['finite'] and candidate['finite'] and all(
        row['finite'] for row in tensors.values())
    failed = []
    if not loss_ok: failed.append('losses')
    if not tensor_ok: failed.append('tensors')
    if not routing_ok: failed.append('routing')
    if not finite_ok: failed.append('finite')
    return {'status': 'passed' if not failed else 'failed',
            'losses': losses, 'tensors': tensors, 'routing': routing,
            'gates': {'losses': loss_ok, 'tensors': tensor_ok,
                      'routing': routing_ok, 'finite': finite_ok},
            'failed_gates': failed}


def compare_gradients(scan, chunk):
    names_scan = scan['gradient_inventory']['names']
    names_chunk = chunk['gradient_inventory']['names']
    inventory_same = names_scan == names_chunk
    per = {}
    global_num = global_den = 0.0
    all_ok = True
    for name in sorted(set(names_scan) & set(names_chunk)):
        ref = scan['gradients'][name].double()
        cand = chunk['gradients'][name].double()
        delta = cand - ref
        rel = delta.norm().item() / max(ref.norm().item(), 1e-12)
        abs_l2 = delta.norm().item()
        ok = finite(ref) and finite(cand) and (rel <= 0.10 or abs_l2 <= 1e-7)
        per[name] = {'rel_l2': rel, 'abs_l2': abs_l2,
                     'status': 'passed' if ok else 'failed'}
        global_num += float(delta.square().sum())
        global_den += float(ref.square().sum())
        all_ok = all_ok and ok
    global_rel = math.sqrt(global_num / max(global_den, 1e-12))
    return {'status': 'passed' if inventory_same and all_ok
            and global_rel <= 0.05 else 'failed',
            'inventory_same': inventory_same,
            'scan_inventory': scan['gradient_inventory'],
            'chunk64_inventory': chunk['gradient_inventory'],
            'scan_digest': scan['gradient_digest'],
            'chunk64_digest': chunk['gradient_digest'],
            'global_rel_l2': global_rel,
            'global_ok': global_rel <= 0.05,
            'per_tensor': per,
            'all_per_tensor_ok': all_ok,
            'finite': scan['gradient_inventory']['finite'] and chunk['gradient_inventory']['finite']}


def compare_post_model(scan, chunk):
    names = sorted(set(scan) & set(chunk))
    per = {}
    global_num = global_den = 0.0
    finite_ok = True
    for name in names:
        ref = scan[name].double()
        cand = chunk[name].double()
        delta = cand - ref
        rel = delta.norm().item() / max(ref.norm().item(), 1e-12)
        max_abs = delta.abs().max().item()
        per[name] = {'rel_l2': rel, 'max_abs': max_abs,
                     'finite': finite(ref) and finite(cand)}
        global_num += float(delta.square().sum())
        global_den += float(ref.square().sum())
        finite_ok = finite_ok and per[name]['finite']
    return {'status': 'passed' if finite_ok else 'failed',
            'names_same': sorted(set(scan)) == sorted(set(chunk)),
            'global_rel_l2': math.sqrt(global_num / max(global_den, 1e-12)),
            'per_tensor': per, 'finite': finite_ok}


def build_fresh(cfg, length, seed):
    seed_all(seed)
    model = model_from(cfg)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f'parameter count mismatch: {parameters}')
    init = tensor_digest(model.named_parameters())
    model = model.to('cuda')
    optimizer, optimizer_policy = build_optimizer(model, cfg['training'])
    inputs, labels, case = build_contiguous_case(length, seed)
    identity = {
        'input_sha256': case['input_sha256'],
        'initialization_sha256': init,
        'partition_plan_sha256': _partition_summary(optimizer_policy)['plan_sha256'],
    }
    return {'model': model, 'optimizer': optimizer, 'inputs': inputs,
            'labels': labels, 'case': case, 'identity': identity,
            'parameters': parameters}


def run_pass(cfg, length, seed, *, backward=False, step=False):
    built = build_fresh(cfg, length, seed)
    model = built['model']
    optimizer = built['optimizer']
    moved_inputs, moved_labels = _move_inputs(built['inputs'], built['labels'], 'cuda')
    optimizer.zero_grad(set_to_none=True)
    autocast = cfg['training'].get('precision') != 'fp32'
    context = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if autocast else contextlib.nullcontext())
    started = time.perf_counter()
    with context:
        out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
        total, losses = losses_and_total(out, moved_labels, cfg)
    if backward:
        total.backward()
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    grads = {}
    grad_inv = None
    grad_digest = None
    if backward:
        named = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        grads = {name: value.detach() for name, value in named}
        grad_inv = {'names': [name for name, _ in named],
                    'parameter_tensors': len(named),
                    'numel': sum(value.numel() for _, value in named),
                    'finite': all(finite(value) for _, value in named)}
        grad_digest = tensor_digest(named)
    post_model = None
    post_digest = None
    optimizer_inventory = None
    optimizer_finite = None
    if step:
        optimizer.step()
        torch.cuda.synchronize()
        assert_post_update_finite(model, optimizer, 'T0033 post-update')
        post_model = {name: p.detach().clone() for name, p in model.named_parameters()}
        post_digest = tensor_digest(model.named_parameters())
        optimizer_inventory = optimizer_state_inventory(optimizer)
        optimizer_finite = all(finite(value) for state in optimizer.state_dict()['state'].values()
                               for value in state.values() if torch.is_tensor(value))
    routes_order = []
    routes = {}
    for key, route in out.routing:
        routes_order.append(key)
        routes[key] = {'selected': route.selected.detach().cpu()}
    output = {
        'logits': {depth: value.detach() for depth, value in out.logits.items()},
        'hidden': {depth: value.detach() for depth, value in out.hidden.items()},
        'offset2': {depth: value.detach() for depth, value in out.offset2.items()},
    }
    model_finite = all(finite(p) for p in model.parameters())
    result = {
        'identity': built['identity'], 'parameters': built['parameters'],
        'losses': losses, 'output': output, 'routes_order': routes_order,
        'routes': routes, 'finite': model_finite and all(
            math.isfinite(v) for v in losses.values()) and all(
            finite(v) for value in output.values() for v in value.values()),
        'gradients': grads, 'gradient_inventory': grad_inv,
        'gradient_digest': grad_digest,
        'post_model': post_model, 'post_model_digest': post_digest,
        'optimizer_inventory': optimizer_inventory,
        'optimizer_finite': optimizer_finite,
        'forward_seconds': forward_seconds,
        'wall_seconds': time.perf_counter() - started,
    }
    del out, total
    del built['model'], built['optimizer'], built['inputs'], built['labels']
    gc.collect()
    torch.cuda.empty_cache()
    return result


def reference_case(reference, length, seed):
    for case in reference['cases']:
        if case['length'] == length and case['seed'] == seed:
            return case
    raise KeyError(f'no T0031 reference for {length}/{seed}')


def pair_gates(length, seed, reference=None):
    scan_cfg = json.loads(Path(FP32_SCAN).read_text())
    chunk_cfg = json.loads(Path(FP32_CHUNK).read_text())
    scan = run_pass(scan_cfg, length, seed, backward=True, step=True)
    chunk = run_pass(chunk_cfg, length, seed, backward=True, step=True)
    within = compare_outputs(scan, chunk)
    grad = compare_gradients(scan, chunk)
    post = compare_post_model(scan['post_model'], chunk['post_model'])
    ref = reference_case(reference, length, seed) if reference else None
    identity_ok = scan['identity'] == chunk['identity'] and (
        ref is None or scan['identity'] == ref['identity'])
    ref_ok = True
    ref_deltas = None
    if ref is not None:
        ref_forward = ref['forward_gates']
        loss_delta = max(abs(within['losses'][name] - ref_forward['losses'][name])
                         for name in within['losses'])
        tensor_delta = max(
            max(abs(within['tensors'][name]['rel_l2'] - ref_forward['tensors'][name]['rel_l2']),
                abs(within['tensors'][name]['max_abs'] - ref_forward['tensors'][name]['max_abs']))
            for name in within['tensors'])
        ref_ok = loss_delta <= 1e-6 and tensor_delta <= 1e-6 and \
            within['routing']['mismatch_tokens'] == 0
        ref_deltas = {'loss_max_abs_delta': loss_delta,
                      'tensor_max_abs_delta': tensor_delta,
                      'routing_mismatch_tokens': within['routing']['mismatch_tokens']}
    status = (within['status'] == 'passed' and grad['status'] == 'passed'
              and post['status'] == 'passed' and post['names_same']
              and identity_ok and ref_ok and scan['finite'] and chunk['finite']
              and scan['optimizer_finite'] and chunk['optimizer_finite'])
    return {'length': length, 'seed': seed,
            'status': 'passed' if status else 'failed',
            'identity': scan['identity'], 'identity_equal': identity_ok,
            'forward_gates': within, 'gradients': grad,
            'post_update': post,
            'post_digests': {'scan': scan['post_model_digest'],
                             'chunk64': chunk['post_model_digest']},
            'optimizer': {'scan': scan['optimizer_inventory'],
                          'chunk64': chunk['optimizer_inventory'],
                          'scan_finite': scan['optimizer_finite'],
                          'chunk64_finite': chunk['optimizer_finite']},
            'reference': ref_deltas}


def qualify(args):
    reference = json.loads(Path(args.t0031_reference).read_text())
    cases = [(512, 42), (512, 43), (2048, 42)]
    report = {'status': 'blocked', 'source': SOURCE, 'cases': [], 'stop_reason': None}
    for length, seed in cases:
        case = pair_gates(length, seed, reference)
        report['cases'].append(case)
        if case['status'] != 'passed':
            report['status'] = 'failed'
            report['stop_reason'] = f'first required failure at {length}/{seed}'
            break
    if report['status'] != 'failed':
        report['status'] = 'passed'
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'],
                      'cases': [c['status'] for c in report['cases']],
                      'stop_reason': report['stop_reason']}, indent=2))
    return 0 if report['status'] == 'passed' else 2


def packed_resume(args):
    cfg_scan = json.loads(Path(SMOKE_SCAN).read_text())
    cfg_chunk = json.loads(Path(SMOKE_CHUNK).read_text())
    blocks = pack(split_documents(synthetic_documents())['train'], 96)
    root = Path(args.out).parent / 'packed-resume-work'
    root.mkdir(parents=True, exist_ok=True)
    report = {'status': 'blocked', 'cases': [], 'cross_mode': None}

    def one(cfg, algorithm, path, steps, resume=False):
        identity = {'schema': 1, 'fixture': 't0033-packed-fp32',
                    'algorithm': algorithm, 'config': digest(cfg), 'seed': 42}
        seed_all(42 if not resume else 99)
        model = model_from(cfg).to('cuda')
        optimizer, _ = build_optimizer(model, cfg['training'])
        objective = lambda m, x, y: training_objective(m, x, y, mode=cfg['training']['mode'],
                                                        loops=cfg['training']['loops'],
                                                        depth_weight=cfg['training']['depth_weight'],
                                                        offset_weight=cfg['training']['offset_weight'],
                                                        balance_weight=cfg['training']['balance_weight'])
        history = train(model, optimizer, blocks, objective, identity, path,
                        steps=steps, batch_size=1, seed=42, device='cuda',
                        resume=resume, post_update_check=True, autocast_dtype=None,
                        checkpoint_interval=1)
        val = evaluate(model, blocks, lambda m, x: m(*x, loops=2).logits[2],
                       batch_size=1, device='cuda')
        return model, optimizer, history, val

    def state(path):
        return torch.load(Path(path) / 'checkpoint.pt', map_location='cpu', weights_only=True)

    def equal(a, b):
        if torch.is_tensor(a):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    for algorithm, cfg in (('scan', cfg_scan), ('chunk64', cfg_chunk)):
        cont = root / f'{algorithm}-continuous'
        split = root / f'{algorithm}-split'
        m, o, h, v = one(cfg, algorithm, cont, 4)
        one(cfg, algorithm, split, 2)
        cp_before = hashlib.sha256((split / 'checkpoint.pt').read_bytes()).hexdigest()
        m2, o2, h2, v2 = one(cfg, algorithm, split, 4, resume=True)
        a, b = state(cont), state(split)
        ok = all(equal(a[k], b[k]) for k in ('model', 'optimizer', 'rng'))
        ignore = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}
        hist_ok = ([{k: x for k, x in row.items() if k not in ignore} for row in h] ==
                   [{k: x for k, x in row.items() if k not in ignore} for row in h2])
        val_ok = v == v2
        report['cases'].append({'algorithm': algorithm,
                                'status': 'passed' if ok and hist_ok and val_ok else 'failed',
                                'model_optimizer_rng_exact': ok,
                                'history_exact_non_timing': hist_ok,
                                'validation_exact': val_ok,
                                'checkpoint_changed': cp_before != hashlib.sha256(
                                    (split / 'checkpoint.pt').read_bytes()).hexdigest()})
    # Cross-mode must reject before checkpoint mutation.
    chunk_split = root / 'chunk64-cross'
    one(cfg_chunk, 'chunk64', chunk_split, 2)
    before = hashlib.sha256((chunk_split / 'checkpoint.pt').read_bytes()).hexdigest()
    try:
        one(cfg_scan, 'scan', chunk_split, 4, resume=True)
        cross = {'status': 'failed', 'reason': 'scan resume unexpectedly accepted chunk64 checkpoint'}
    except ValueError as exc:
        after = hashlib.sha256((chunk_split / 'checkpoint.pt').read_bytes()).hexdigest()
        cross = {'status': 'passed', 'error': str(exc), 'checkpoint_unchanged': before == after}
    report['cross_mode'] = cross
    status = all(c['status'] == 'passed' for c in report['cases']) and cross['status'] == 'passed'
    report['status'] = 'passed' if status else 'failed'
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'cases': report['cases'],
                      'cross_mode': cross}, indent=2))
    return 0 if status else 2


def resource_gate(args):
    reference = json.loads(Path(args.t0031_reference).read_text())
    report = {'status': 'passed', 'length': args.length, 'repeats': []}
    for repeat in range(1, 4):
        case = pair_gates(args.length, 42, reference)
        report['repeats'].append({'repeat': repeat, 'status': case['status'],
                                  'forward_gates': case['forward_gates'],
                                  'gradients': case['gradients'],
                                  'post_update': case['post_update']})
        if case['status'] != 'passed':
            report['status'] = 'failed'
            break
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'length': args.length,
                      'repeats': [r['status'] for r in report['repeats']]}, indent=2))
    return 0 if report['status'] == 'passed' else 2


def resource(args):
    if args.method == 'scan':
        cfg = json.loads(Path(FP32_SCAN).read_text())
        policy = 'fp32_scan'
    elif args.method == 'chunk64':
        cfg = json.loads(Path(FP32_CHUNK).read_text())
        policy = 'fp32_chunk64'
    elif args.method == 'original_bf16_scan':
        cfg = json.loads(Path(BF16_SCAN).read_text())
        policy = 'original_bf16_scan'
    else:
        raise ValueError('unknown method')
    report = {'status': 'passed', 'length': args.length, 'method': args.method,
              'policy': policy, 'repeats': []}
    for repeat in range(1, 4):
        built = build_fresh(cfg, args.length, 42)
        model, optimizer = built['model'], built['optimizer']
        mi, ml = _move_inputs(built['inputs'], built['labels'], 'cuda')
        autocast = cfg['training'].get('precision') != 'fp32'
        context = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                   if autocast else contextlib.nullcontext())
        baseline = int(torch.cuda.memory_allocated())
        # Warmup complete update, discarded.
        optimizer.zero_grad(set_to_none=True)
        with context:
            out = model(*mi, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
            total, warm_losses = losses_and_total(out, ml, cfg)
        total.backward(); optimizer.step(); torch.cuda.synchronize()
        assert_post_update_finite(model, optimizer, 'T0033 resource warmup')
        optimizer.zero_grad(set_to_none=True)
        del out, total
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        phases = {}
        started = time.perf_counter()
        phase = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        phases['prepare'] = time.perf_counter() - phase
        phase = time.perf_counter()
        with context:
            out = model(*mi, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
            total, losses = losses_and_total(out, ml, cfg)
        torch.cuda.synchronize(); phases['forward'] = time.perf_counter() - phase
        phase = time.perf_counter()
        total.backward(); torch.cuda.synchronize(); phases['backward'] = time.perf_counter() - phase
        phase = time.perf_counter()
        optimizer.step(); torch.cuda.synchronize(); phases['optimizer'] = time.perf_counter() - phase
        phases['end_to_end'] = time.perf_counter() - started
        # Diagnostics are deliberately outside the measured end-to-end window.
        diagnostic_started = time.perf_counter()
        assert_post_update_finite(model, optimizer, 'T0033 measured post-update')
        finite_ok = all(math.isfinite(v) for v in losses.values())
        grads_finite = all(finite(p.grad) for p in model.parameters() if p.grad is not None)
        diagnostics_seconds = time.perf_counter() - diagnostic_started
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_res = int(torch.cuda.max_memory_reserved())
        grad_digest = tensor_digest([(n, p.grad) for n, p in model.named_parameters()
                                     if p.grad is not None])
        model_digest = tensor_digest(model.named_parameters())
        opt_inv = optimizer_state_inventory(optimizer)
        report['repeats'].append({
            'repeat': repeat, 'baseline_allocated_bytes': baseline,
            'phases': phases, 'peak_allocated_bytes': peak_alloc,
            'peak_reserved_bytes': peak_res,
            'labels': int(ml[1].ne(-100).sum()),
            'labels_per_second': int(ml[1].ne(-100).sum()) / phases['end_to_end'],
            'losses': losses, 'losses_finite': finite_ok,
            'gradients_finite': grads_finite, 'diagnostics_seconds': diagnostics_seconds,
            'gradient_digest': grad_digest, 'model_digest': model_digest,
            'optimizer_inventory': opt_inv,
        })
        if not finite_ok or not grads_finite:
            report['status'] = 'failed'
            break
        del out, total, model, optimizer
        gc.collect(); torch.cuda.empty_cache()
    def summary(values):
        return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    report['summary'] = {
        name: summary([r['phases'][name] for r in report['repeats']])
        for name in ('prepare', 'forward', 'backward', 'optimizer', 'end_to_end')}
    report['summary']['peak_allocated_bytes'] = max(r['peak_allocated_bytes'] for r in report['repeats'])
    report['summary']['peak_reserved_bytes'] = max(r['peak_reserved_bytes'] for r in report['repeats'])
    report['no_sustained_training_claim'] = True
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'method': args.method,
                      'length': args.length, 'summary': report['summary']}, indent=2))
    return 0 if report['status'] == 'passed' else 2


def check(args):
    scan = json.loads(Path(FP32_SCAN).read_text())
    chunk = json.loads(Path(FP32_CHUNK).read_text())
    seed_all(42)
    scan_init = tensor_digest(model_from(scan).named_parameters())
    seed_all(42)
    chunk_init = tensor_digest(model_from(chunk).named_parameters())
    seed_all(42)
    model = model_from(scan)
    optimizer, policy = build_optimizer(model, scan['training'])
    inputs, labels, case = build_contiguous_case(512, 42)
    result = {'status': 'passed' if (
        scan_init == chunk_init == EXPECTED_INIT
        and case['input_sha256'] == EXPECTED_INPUT_512_42
        and _partition_summary(policy)['plan_sha256'] == EXPECTED_PARTITION
        and sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS)
        else 'failed',
        'scan_config_sha256': digest(scan), 'chunk64_config_sha256': digest(chunk),
        'scan_init': scan_init, 'chunk64_init': chunk_init,
        'input_sha256': case['input_sha256'],
        'partition_plan_sha256': _partition_summary(policy)['plan_sha256'],
        'parameters': sum(p.numel() for p in model.parameters())}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'passed' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'qualify', 'packed', 'resource-gate', 'resource'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--t0031-reference', type=Path, required=True)
    parser.add_argument('--length', type=int, choices=(512, 2048), default=512)
    parser.add_argument('--method', choices=('scan', 'chunk64', 'original_bf16_scan'), default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == 'check':
        return check(args)
    if args.mode == 'qualify':
        return qualify(args)
    if args.mode == 'packed':
        return packed_resume(args)
    if args.mode == 'resource-gate':
        return resource_gate(args)
    if args.method is None:
        parser.error('--method required for resource')
    return resource(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'DRIVER ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
