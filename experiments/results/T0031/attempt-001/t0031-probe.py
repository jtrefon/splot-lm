#!/usr/bin/env python3
"""T0031 matched-precision GDN scan/chunk64 feasibility driver.

Forward/backward diagnostics only: no optimizer step, training, checkpoint,
source edit, install or automatic retry. Raw tensors/routes are never written;
only digests, dtypes, gate values and aggregate statistics leave the process.
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
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import torch
import torch.nn.functional as F

import splot.gdn as gdnmod
from context_probe import (build_contiguous_case, load_config, model_from,
                           tensor_digest, _move_inputs, _partition_summary,
                           _precheck)
from gdn_chunk_probe import losses_and_total
from experiment import seed_all
from optimizers import build_optimizer
from splot.gdn import GatedDeltaNet

SOURCE = '7dac49e083eae0068ff361e7a81b3a4f41055b3f'
EXPECTED_INIT = 'ba9bf9d8472a9b940a416e91f3123ef4163913b7b334291540cb3b321fbaca5b'
EXPECTED_INPUT_512_42 = 'c0aa82fc0906b8a1437b5328f8d2396ab4f79e7b1528e8be399ea107da899270'
EXPECTED_PARTITION = '1bed555467ef9a955079b8260636971dafa5b0b308962e9a18e23a5ffa7bf9ed'
EXPECTED_PARAMETERS = 43055076
FWD_REL = 0.02
FWD_ABS = 0.05
LOSS_ABS = 0.02
LOCAL_REL = 1e-4
LOCAL_ABS = 1e-4
GRAD_GLOBAL = 0.05
GRAD_TENSOR_REL = 0.10
GRAD_ABS_L2 = 1e-7
ORIG_SCAN = gdnmod.delta_scan
ORIG_CHUNK64 = gdnmod.delta_chunk64
POLICIES = ('P1', 'P2', 'P3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(tensor):
    return bool(torch.isfinite(tensor).all().item())


def all_finite(value):
    if torch.is_tensor(value):
        return finite(value)
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    return True


def stats_pair(reference, candidate):
    ref = reference.detach().float()
    cand = candidate.detach().float()
    delta = cand.double() - ref.double()
    ref_norm = ref.double().norm().item()
    return {
        'elements': int(ref.numel()),
        'rel_l2': delta.norm().item() / max(ref_norm, 1e-12),
        'max_abs': delta.abs().max().item(),
        'reference_max_abs': ref.double().abs().max().item(),
        'finite': finite(ref) and finite(cand),
    }


def tensor_gate(row):
    return (row['rel_l2'] <= FWD_REL
            and row['max_abs'] <= FWD_ABS * max(1.0, row['reference_max_abs'])
            and row['finite'])


def route_counts(scan_routes, cand_routes):
    by_key = {}
    total = 0
    mismatches = 0
    for key in scan_routes:
        a = scan_routes[key]['selected']
        b = cand_routes[key]['selected']
        if a.shape != b.shape:
            raise ValueError(f'routing shape mismatch at {key}')
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
    loss_ok = all(value <= LOSS_ABS for value in losses.values())
    tensor_ok = all(tensor_gate(row) for row in tensors.values())
    routing_ok = routing['mismatch_tokens'] == 0
    finite_ok = (reference['finite'] and candidate['finite']
                 and all(all_finite(v) for v in reference['output'].values())
                 and all(all_finite(v) for v in candidate['output'].values()))
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


class PolicyLog:
    def __init__(self, policy):
        self.policy = policy
        self.signatures = {}

    def record(self, operation, **values):
        key = operation + '|' + json.dumps(values, sort_keys=True)
        row = self.signatures.setdefault(key, dict(values, operation=operation, calls=0))
        row['calls'] += 1

    def summary(self):
        return {'policy': self.policy, 'signatures': list(self.signatures.values())}


class PolicyWrapper:
    def __init__(self, policy):
        self.policy = policy
        self.log = PolicyLog(policy)
        self.restore = []

    def install(self, model):
        if self.policy == 'original':
            self.log.record('original_bf16_forward', model_dtype='float32',
                            autocast='bfloat16', assignment='no precision wrapper')
            return
        if self.policy == 'P3':
            self.log.record('full_model_forward', model_dtype='float32',
                            autocast='disabled', surrounding_assignment='native FP32 full forward')
            return
        for block in model.core:
            attn = getattr(block, 'attn', None)
            if not isinstance(attn, GatedDeltaNet):
                continue
            if self.policy == 'P1':
                linear = attn.o
                original = linear.forward
                def forward(module, x, _orig=original, _log=self.log):
                    in_dtype = x.dtype
                    weight_dtype = module.weight.dtype
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        value = F.linear(x.float(), module.weight.float())
                        if module.bias is not None:
                            value = value + module.bias.float()
                    out = value.to(in_dtype)
                    _log.record('GDN.o', input_dtype=str(in_dtype),
                                weight_dtype=str(weight_dtype),
                                compute_dtype='float32', output_dtype=str(out.dtype),
                                assignment='cast back to the existing GDN input dtype before residual')
                    return out
                linear.forward = types.MethodType(forward, linear)
                self.restore.append((linear, 'forward', original))
            elif self.policy == 'P2':
                original = attn.run
                def run(module, x, previous, *, algorithm=None, _orig=original, _log=self.log):
                    in_dtype = x.dtype
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        y, state = _orig(x.float(), previous, algorithm=algorithm)
                    out = y.to(in_dtype)
                    state_value, history = state
                    _log.record('GatedDeltaNet.run', input_dtype=str(in_dtype),
                                effective_input_dtype='float32',
                                weight_dtype='float32',
                                computed_output_dtype=str(y.dtype),
                                returned_output_dtype=str(out.dtype),
                                state_dtype=str(state_value.dtype),
                                history_dtype=str(history.dtype),
                                assignment='cast y back to surrounding hidden dtype; state/history remain FP32 internal')
                    return out, (state_value, history)
                attn.run = types.MethodType(run, attn)
                self.restore.append((attn, 'run', original))
            else:
                raise ValueError('unknown policy')

    def uninstall(self):
        for module, attribute, original in self.restore:
            setattr(module, attribute, original)
        self.restore = []


class DeltaCapture:
    def __init__(self):
        self.calls = {}
        self.index = 0

    def install(self):
        gdnmod.delta_scan = self._scan
        gdnmod.delta_chunk64 = self._chunk

    def _key(self):
        # One contiguous document and two logical passes in this task: GDN call order is
        # block 0,1,2 for pass 1 and then block 0,1,2 for pass 2.
        key = (self.index % 3, self.index // 3 + 1)
        self.index += 1
        return key

    def _scan(self, q, k, v, g, beta, state):
        key = self._key()
        rec = self.calls.setdefault(key, {})
        for name, value in (('q', q), ('k', k), ('v', v), ('g', g),
                            ('beta', beta), ('state', state)):
            rec[name] = value.detach().cpu()
        out, final = ORIG_SCAN(q, k, v, g, beta, state)
        rec['out'] = out.detach().cpu()
        rec['final'] = final.detach().cpu()
        return out, final

    def _chunk(self, q, k, v, g, beta, state):
        key = self._key()
        rec = self.calls.setdefault(key, {})
        for name, value in (('q', q), ('k', k), ('v', v), ('g', g),
                            ('beta', beta), ('state', state)):
            rec[name] = value.detach().cpu()
        out, final = ORIG_CHUNK64(q, k, v, g, beta, state)
        rec['out'] = out.detach().cpu()
        rec['final'] = final.detach().cpu()
        return out, final

    def uninstall(self):
        gdnmod.delta_scan = ORIG_SCAN
        gdnmod.delta_chunk64 = ORIG_CHUNK64


def local_same_input(capture):
    calls = {}
    worst_rel = 0.0
    worst_abs = 0.0
    first = None
    for key in sorted(capture.calls, key=lambda item: (item[1], item[0])):
        rec = capture.calls[key]
        q, k, v, g, beta, state = (rec[name] for name in
                                    ('q', 'k', 'v', 'g', 'beta', 'state'))
        with torch.no_grad():
            scan_out, scan_final = ORIG_SCAN(q, k, v, g, beta, state)
            chunk_out, chunk_final = ORIG_CHUNK64(q, k, v, g, beta, state)
        out = stats_pair(scan_out, chunk_out)
        final = stats_pair(scan_final, chunk_final)
        ok = (out['rel_l2'] <= LOCAL_REL and out['max_abs'] <= LOCAL_ABS * max(1.0, out['reference_max_abs'])
              and final['rel_l2'] <= LOCAL_REL and final['max_abs'] <= LOCAL_ABS * max(1.0, final['reference_max_abs'])
              and out['finite'] and final['finite'])
        calls[str(key)] = {'block': key[0], 'logical_pass': key[1],
                           'output': out, 'state': final, 'status': 'passed' if ok else 'failed'}
        worst_rel = max(worst_rel, out['rel_l2'], final['rel_l2'])
        worst_abs = max(worst_abs, out['max_abs'], final['max_abs'])
        if first is None and out['rel_l2'] > 1e-12:
            first = {'key': str(key), 'block': key[0], 'logical_pass': key[1],
                     'output_rel_l2': out['rel_l2'], 'output_max_abs': out['max_abs']}
    oracle = {}
    ordered = sorted(capture.calls, key=lambda item: (item[1], item[0]))
    for key in (ordered[0], ordered[-1]):
        rec = capture.calls[key]
        q, k, v, g, beta, state = (rec[name].double() for name in
                                    ('q', 'k', 'v', 'g', 'beta', 'state'))
        with torch.no_grad():
            scan_out, scan_final = ORIG_SCAN(q, k, v, g, beta, state)
            chunk_out, chunk_final = ORIG_CHUNK64(q, k, v, g, beta, state)
        out = stats_pair(scan_out, chunk_out)
        final = stats_pair(scan_final, chunk_final)
        ok = (out['rel_l2'] <= 1e-9 and out['max_abs'] <= 1e-9 * max(1.0, out['reference_max_abs'])
              and final['rel_l2'] <= 1e-9 and final['max_abs'] <= 1e-9 * max(1.0, final['reference_max_abs']))
        oracle[str(key)] = {'output': out, 'state': final, 'status': 'passed' if ok else 'failed'}
    return {'status': 'passed' if all(row['status'] == 'passed' for row in calls.values()) else 'failed',
            'calls': calls, 'worst_rel_l2': worst_rel, 'worst_max_abs': worst_abs,
            'first_nonzero': first,
            'fp64_oracle': {'status': 'passed' if all(row['status'] == 'passed' for row in oracle.values()) else 'failed',
                            'calls': oracle}}


def build_fresh(cfg, length, seed, device='cuda'):
    seed_all(seed)
    model = model_from(cfg)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f'parameter count mismatch: {parameters}')
    init = tensor_digest(model.named_parameters())
    model = model.to(device)
    optimizer, optimizer_policy = build_optimizer(model, cfg['training'])
    inputs, labels, case = build_contiguous_case(length, seed)
    precheck = _precheck(model, inputs, labels, device, False)
    model.train()
    identity = {
        'input_sha256': case['input_sha256'],
        'initialization_sha256': init,
        'partition_plan_sha256': _partition_summary(optimizer_policy)['plan_sha256'],
    }
    return {'model': model, 'optimizer': optimizer, 'inputs': inputs,
            'labels': labels, 'case': case, 'identity': identity,
            'parameters': parameters, 'precheck': precheck}


def run_one(cfg, length, seed, policy, *, capture=False, backward=False):
    built = build_fresh(cfg, length, seed)
    model = built['model']
    optimizer = built['optimizer']
    cap = DeltaCapture() if capture else None
    wrapper = PolicyWrapper(policy)
    if cap is not None:
        cap.install()
    wrapper.install(model)
    moved_inputs, moved_labels = _move_inputs(built['inputs'], built['labels'], 'cuda')
    optimizer.zero_grad(set_to_none=True)
    context = (contextlib.nullcontext() if policy == 'P3' else
               torch.autocast(device_type='cuda', dtype=torch.bfloat16))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with context:
        out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
        total, losses = losses_and_total(out, moved_labels, cfg)
    torch.cuda.synchronize()
    forward_wall = time.perf_counter() - started
    total_value = float(total.detach())
    if backward:
        total.backward()
        torch.cuda.synchronize()
    wall = time.perf_counter() - started
    output = {
        'logits': {depth: value.detach() for depth, value in out.logits.items()},
        'hidden': {depth: value.detach() for depth, value in out.hidden.items()},
        'offset2': {depth: value.detach() for depth, value in out.offset2.items()},
    }
    routes_order = []
    routes = {}
    for key, route in out.routing:
        routes_order.append(key)
        routes[key] = {'selected': route.selected.detach().cpu()}
    model_finite = all(finite(p) for p in model.parameters())
    grads = {}
    grad_inventory = None
    grad_digest = None
    if backward:
        named = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        grads = {name: value.detach() for name, value in named}
        grad_inventory = {
            'names': [name for name, _ in named],
            'parameter_tensors': len(named),
            'numel': sum(value.numel() for _, value in named),
            'finite': all(finite(value) for _, value in named),
        }
        grad_digest = tensor_digest(named)
    result = {
        'identity': built['identity'],
        'parameters': built['parameters'],
        'losses': losses,
        'output': output,
        'routes_order': routes_order,
        'routes': routes,
        'finite': model_finite and all_finite(output) and all(math.isfinite(v) for v in losses.values()),
        'model_finite': model_finite,
        'gradients': grads,
        'gradient_inventory': grad_inventory,
        'gradient_digest': grad_digest,
        'policy_log': wrapper.log.summary(),
        'capture': cap,
        'precheck': built['precheck'],
        'forward_seconds': forward_wall,
        'wall_seconds': wall,
    }
    wrapper.uninstall()
    if cap is not None:
        cap.uninstall()
    del out, total
    del built['model'], built['optimizer'], built['inputs'], built['labels']
    gc.collect()
    torch.cuda.empty_cache()
    return result


def reproduction_check(scan, chunk, reference):
    actual_identity = scan['identity']
    expected_identity = reference['scan_identity']
    identity_exact = actual_identity == expected_identity
    scan_exact = scan['losses'] == reference['losses']['scan']
    chunk_deltas = {name: abs(chunk['losses'][name] - reference['losses']['chunk64'][name])
                    for name in reference['losses']['chunk64']}
    chunk_ok = all(value <= 1e-6 for value in chunk_deltas.values())
    routes = route_counts(scan['routes'], chunk['routes'])
    counts = {}
    count_ok = True
    for key, row in reference['routing']['by_key'].items():
        expected = int(round(row['disagreement_fraction'] * row['tokens']))
        actual = routes['by_key'][key]['mismatch_tokens']
        counts[key] = {'expected': expected, 'actual': actual, 'exact': expected == actual}
        count_ok = count_ok and expected == actual
    tensor_deltas = {}
    tensor_ok = True
    for name, expected in reference['tensors'].items():
        group, depth = name.rsplit('_', 1)
        actual = stats_pair(scan['output'][group][int(depth)],
                            chunk['output'][group][int(depth)])
        tensor_deltas[name] = {
            'actual_rel_l2': actual['rel_l2'],
            'published_rel_l2': expected['rel_l2'],
            'rel_l2_abs_delta': abs(actual['rel_l2'] - expected['rel_l2']),
            'actual_max_abs': actual['max_abs'],
            'published_max_abs': expected['max_abs'],
            'max_abs_delta': abs(actual['max_abs'] - expected['max_abs']),
        }
        tensor_ok = tensor_ok and tensor_deltas[name]['rel_l2_abs_delta'] <= 1e-6 \
            and tensor_deltas[name]['max_abs_delta'] <= 1e-6
    status = identity_exact and scan_exact and chunk_ok and count_ok and tensor_ok \
        and scan['finite'] and chunk['finite']
    return {
        'status': 'passed' if status else 'failed',
        'identity_exact': identity_exact,
        'scan_losses_exact': scan_exact,
        'chunk64_loss_max_abs_delta': max(chunk_deltas.values()),
        'chunk64_loss_deltas': chunk_deltas,
        'per_key_routing_counts_exact': count_ok,
        'per_key_routing_counts': counts,
        'tensor_metrics_within_1e-6': tensor_ok,
        'tensor_metric_deltas': tensor_deltas,
        'scan': {'identity': actual_identity, 'losses': scan['losses'],
                 'finite': scan['finite']},
        'chunk64': {'identity': chunk['identity'], 'losses': chunk['losses'],
                    'finite': chunk['finite']},
        'routing': routes,
        'expected_identity': expected_identity,
        'expected_scan_losses': reference['losses']['scan'],
        'expected_chunk64_losses': reference['losses']['chunk64'],
    }


def policy_screen(cfg_base, reference, policy):
    cfg_scan = copy.deepcopy(cfg_base)
    cfg_scan['recurrence']['gdn_algorithm'] = 'scan'
    cfg_chunk = copy.deepcopy(cfg_base)
    cfg_chunk['recurrence']['gdn_algorithm'] = 'chunk64'
    scan = run_one(cfg_scan, 512, 42, policy, capture=True)
    chunk = run_one(cfg_chunk, 512, 42, policy)
    within = compare_outputs(scan, chunk)
    local = local_same_input(scan['capture'])
    identity_equal = scan['identity'] == chunk['identity']
    status = (within['status'] == 'passed' and local['status'] == 'passed'
              and identity_equal and scan['finite'] and chunk['finite'])
    result = {
        'status': 'passed' if status else 'failed',
        'within_policy': within,
        'same_input_fp32': local,
        'identity_equal': identity_equal,
        'policy_identity': scan['identity'],
        'parameters': scan['parameters'],
        'policy_log_scan': scan['policy_log'],
        'policy_log_chunk': chunk['policy_log'],
        'failed_gates': list(within['failed_gates']),
    }
    if local['status'] != 'passed':
        result['failed_gates'].append('same_input_fp32')
    if not identity_equal:
        result['failed_gates'].append('identity')
    return result, scan


def compare_gradients(scan, chunk):
    names_scan = scan['gradient_inventory']['names']
    names_chunk = chunk['gradient_inventory']['names']
    inventory_same = names_scan == names_chunk
    per = {}
    global_num = 0.0
    global_den = 0.0
    all_ok = True
    for name in sorted(set(names_scan) & set(names_chunk)):
        ref = scan['gradients'][name].double()
        cand = chunk['gradients'][name].double()
        delta = cand - ref
        rel = delta.norm().item() / max(ref.norm().item(), 1e-12)
        abs_l2 = delta.norm().item()
        finite_ok = finite(ref) and finite(cand)
        ok = finite_ok and (rel <= GRAD_TENSOR_REL or abs_l2 <= GRAD_ABS_L2)
        per[name] = {'rel_l2': rel, 'abs_l2': abs_l2, 'finite': finite_ok,
                     'status': 'passed' if ok else 'failed'}
        global_num += float(delta.square().sum())
        global_den += float(ref.square().sum())
        all_ok = all_ok and ok
    global_rel = math.sqrt(global_num / max(global_den, 1e-12))
    status = inventory_same and all_ok and global_rel <= GRAD_GLOBAL \
        and scan['gradient_inventory']['finite'] and chunk['gradient_inventory']['finite']
    return {
        'status': 'passed' if status else 'failed',
        'inventory_same': inventory_same,
        'scan_inventory': scan['gradient_inventory'],
        'chunk64_inventory': chunk['gradient_inventory'],
        'scan_digest': scan['gradient_digest'],
        'chunk64_digest': chunk['gradient_digest'],
        'global_rel_l2': global_rel,
        'global_ok': global_rel <= GRAD_GLOBAL,
        'per_tensor': per,
        'all_per_tensor_ok': all_ok,
        'finite': scan['gradient_inventory']['finite'] and chunk['gradient_inventory']['finite'],
    }


def screen(args):
    reference = json.loads(args.t0025_reference.read_text())
    base_cfg = load_config('bf16')[1]
    base_scan = run_one(copy.deepcopy(base_cfg), 512, 42, 'original')
    base_cfg_chunk = copy.deepcopy(base_cfg)
    base_cfg_chunk['recurrence']['gdn_algorithm'] = 'chunk64'
    base_chunk = run_one(base_cfg_chunk, 512, 42, 'original')
    rep = reproduction_check(base_scan, base_chunk, reference)
    report = {'status': 'blocked', 'reproduction': rep,
              'reproduction_passed': rep['status'] == 'passed',
              'policies': {}, 'selected_policy': None}
    if rep['status'] != 'passed':
        report['status'] = 'blocked'
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'status': report['status'], 'reproduction': rep['status']}, indent=2))
        return 2
    base_scan_output = base_scan
    for policy in POLICIES:
        cfg_scan = copy.deepcopy(base_cfg)
        cfg_scan['recurrence']['gdn_algorithm'] = 'scan'
        cfg_chunk = copy.deepcopy(base_cfg)
        cfg_chunk['recurrence']['gdn_algorithm'] = 'chunk64'
        result, scan = policy_screen(base_cfg, reference, policy)
        if policy == 'P1':
            result['vs_original_bf16_scan'] = compare_outputs(base_scan_output, scan)
        else:
            result['vs_original_bf16_scan'] = compare_outputs(base_scan_output, scan)
        report['policies'][policy] = result
        if result['status'] == 'passed' and report['selected_policy'] is None:
            report['selected_policy'] = policy
        del scan
    del base_chunk
    report['status'] = 'passed' if report['selected_policy'] else 'negative'
    report['selection_order'] = list(POLICIES)
    report['limitations'] = [
        'One synthetic T512 seed42 screen per policy; no learning or generalization claim.',
        'Policy scan versus original BF16 scan is reported descriptively; precision is a confound.',
        'P3 is a full FP32 control, not a BF16 replacement.',
        'No teacher-forced routing was used for acceptance.',
    ]
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'reproduction': rep['status'],
                      'selected_policy': report['selected_policy'],
                      'policies': {p: report['policies'][p]['status'] for p in POLICIES}},
                     indent=2))
    return 0


def forward_backward(args):
    reference = json.loads(args.t0025_reference.read_text())
    base_cfg = load_config('bf16')[1]
    cases = [(512, 42), (512, 43), (2048, 42)]
    report = {'status': 'blocked', 'policy': args.policy, 'cases': [],
              'stop_reason': None}
    for length, seed in cases:
        cfg_scan = copy.deepcopy(base_cfg)
        cfg_scan['recurrence']['gdn_algorithm'] = 'scan'
        cfg_chunk = copy.deepcopy(base_cfg)
        cfg_chunk['recurrence']['gdn_algorithm'] = 'chunk64'
        scan = run_one(cfg_scan, length, seed, args.policy, capture=True, backward=True)
        chunk = run_one(cfg_chunk, length, seed, args.policy, backward=True)
        within = compare_outputs(scan, chunk)
        local = local_same_input(scan['capture'])
        grad = compare_gradients(scan, chunk)
        identity_equal = scan['identity'] == chunk['identity']
        status = (within['status'] == 'passed' and local['status'] == 'passed'
                  and grad['status'] == 'passed' and identity_equal
                  and scan['finite'] and chunk['finite'])
        case = {
            'length': length, 'seed': seed,
            'status': 'passed' if status else 'failed',
            'identity': scan['identity'], 'identity_equal': identity_equal,
            'forward_gates': within, 'same_input_fp32': local,
            'gradients': grad,
            'policy_log_scan': scan['policy_log'],
            'policy_log_chunk': chunk['policy_log'],
        }
        report['cases'].append(case)
        del scan, chunk
        if not status:
            report['status'] = 'failed'
            report['stop_reason'] = f'first required failure at length {length} seed {seed}'
            break
    if report['status'] != 'failed':
        report['status'] = 'passed'
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'policy': args.policy,
                      'cases': [c['status'] for c in report['cases']],
                      'stop_reason': report['stop_reason']}, indent=2))
    return 0 if report['status'] == 'passed' else 2


def resource(args):
    base_cfg = load_config('bf16')[1]
    cfg = copy.deepcopy(base_cfg)
    if args.method == 'scan':
        cfg['recurrence']['gdn_algorithm'] = 'scan'
    elif args.method == 'chunk64':
        cfg['recurrence']['gdn_algorithm'] = 'chunk64'
    elif args.method == 'original_bf16_scan':
        cfg['recurrence']['gdn_algorithm'] = 'scan'
    else:
        raise ValueError('unknown method')
    built = build_fresh(cfg, args.length, 42)
    model = built['model']
    optimizer = built['optimizer']
    policy = 'original_bf16' if args.method == 'original_bf16_scan' else args.policy
    wrapper = PolicyWrapper(policy)
    if policy != 'original_bf16':
        wrapper.install(model)
    moved_inputs, moved_labels = _move_inputs(built['inputs'], built['labels'], 'cuda')
    context = (contextlib.nullcontext() if policy == 'P3' else
               torch.autocast(device_type='cuda', dtype=torch.bfloat16))
    baseline_allocated = int(torch.cuda.memory_allocated())
    optimizer.zero_grad(set_to_none=True)
    with context:
        out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
        total, warm_losses = losses_and_total(out, moved_labels, cfg)
        total.backward()
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    del out, total
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    repeats = []
    for repeat in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with context:
            out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
            total, losses = losses_and_total(out, moved_labels, cfg)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        total.backward()
        torch.cuda.synchronize()
        end_seconds = time.perf_counter() - started
        losses_finite = all(math.isfinite(v) for v in losses.values())
        grads_finite = all(finite(p.grad) for p in model.parameters() if p.grad is not None)
        repeats.append({
            'repeat': repeat,
            'forward_seconds': forward_seconds,
            'backward_seconds': end_seconds - forward_seconds,
            'end_to_end_seconds': end_seconds,
            'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
            'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
            'losses': losses,
            'losses_finite': losses_finite,
            'gradients_finite': grads_finite,
        })
        del out, total
        gc.collect()
    def summary(values):
        return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    result = {
        'status': 'passed' if all(r['losses_finite'] and r['gradients_finite'] for r in repeats) else 'failed',
        'policy': policy, 'method': args.method, 'length': args.length, 'seed': 42,
        'identity': built['identity'], 'parameters': built['parameters'],
        'baseline_allocated_bytes': baseline_allocated,
        'policy_log': wrapper.log.summary(),
        'warmup_losses': warm_losses,
        'repeats': repeats,
        'summary': {
            'forward_seconds': summary([r['forward_seconds'] for r in repeats]),
            'backward_seconds': summary([r['backward_seconds'] for r in repeats]),
            'end_to_end_seconds': summary([r['end_to_end_seconds'] for r in repeats]),
            'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in repeats),
            'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in repeats),
        },
        'no_optimizer_update': True,
        'diagnostics_outside_timing': True,
        'numerical_gate_source': 'selected-policy forward_backward qualification must pass before resource runs',
        'limitations': [
            'Forward/loss/backward only; no optimizer update or sustained training claim.',
            'Fresh-process method timing; no throughput or cross-policy quality claim.',
        ],
    }
    wrapper.uninstall()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'method': args.method,
                      'length': args.length, 'policy': policy,
                      'forward_median': result['summary']['forward_seconds']['median'],
                      'backward_median': result['summary']['backward_seconds']['median']}, indent=2))
    return 0 if result['status'] == 'passed' else 2


def check(args):
    cfg = load_config('bf16')[1]
    inputs, labels, case = build_contiguous_case(512, 42)
    seed_all(42)
    model = model_from(cfg)
    init = tensor_digest(model.named_parameters())
    parameters = sum(p.numel() for p in model.parameters())
    optimizer, policy = build_optimizer(model, cfg['training'])
    partition = _partition_summary(policy)['plan_sha256']
    result = {
        'status': 'passed' if (case['input_sha256'] == EXPECTED_INPUT_512_42
                               and init == EXPECTED_INIT
                               and partition == EXPECTED_PARTITION
                               and parameters == EXPECTED_PARAMETERS
                               and not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
                  else 'failed',
        'source_commit': SOURCE,
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_clean': not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
        'instruction_commit': args.instruction_commit,
        'config_sha256': sha(Path('configs/experiments/splot-nano-synthetic-bf16-muon.json')),
        'reference_sha256': sha(args.t0025_reference),
        'input_sha256': case['input_sha256'],
        'initialization_sha256': init,
        'partition_plan_sha256': partition,
        'parameters': parameters,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'passed' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'screen', 'fb', 'resource'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--t0025-reference', type=Path, required=True)
    parser.add_argument('--instruction-commit', required=True)
    parser.add_argument('--policy', choices=POLICIES, default=None)
    parser.add_argument('--method', choices=('scan', 'chunk64', 'original_bf16_scan'), default=None)
    parser.add_argument('--length', type=int, choices=(512, 2048), default=512)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == 'check':
        return check(args)
    if args.mode == 'screen':
        return screen(args)
    if args.mode == 'fb':
        if args.policy is None:
            parser.error('--policy required for fb')
        return forward_backward(args)
    if args.mode == 'resource':
        if args.policy is None or args.method is None:
            parser.error('--policy and --method required for resource')
        return resource(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'DRIVER ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
