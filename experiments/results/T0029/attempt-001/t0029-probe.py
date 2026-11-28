#!/usr/bin/env python3
"""T0029 bounded GDN mixed-precision / sparse-routing localization.

Read-only diagnostic for the exact rejected T0025 source (7dac49e). It runs
forward passes only: no backward, no optimizer step, no checkpoint, no source
edit and no retry. Raw tensors/IDs/logits are never written; only statistics,
digests and aggregate route/margin summaries leave this process.
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
                           tensor_digest, _move_inputs, _partition_summary)
from gdn_chunk_probe import build_state, losses_and_total
from experiment import seed_all
from optimizers import build_optimizer
from splot.moe import Route
from splot.dense import _accum

SOURCE_COMMIT = '7dac49e083eae0068ff361e7a81b3a4f41055b3f'
EXPECTED_INIT = 'ba9bf9d8472a9b940a416e91f3123ef4163913b7b334291540cb3b321fbaca5b'
EXPECTED_INPUT = 'c0aa82fc0906b8a1437b5328f8d2396ab4f79e7b1528e8be399ea107da899270'
EXPECTED_PARTITION = '1bed555467ef9a955079b8260636971dafa5b0b308962e9a18e23a5ffa7bf9ed'
EXPECTED_PARAMETERS = 43055076
FP32_REL = 1e-4
FP32_ABS = 1e-4
FP64_REL = 1e-9
FP64_ABS = 1e-9
LENGTH = 512
SEED = 42

ORIG_SCAN = gdnmod.delta_scan
ORIG_CHUNK64 = gdnmod.delta_chunk64
CURRENT = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(tensor):
    return bool(torch.isfinite(tensor).all().item())


def summarize(value):
    x = value.detach().float().flatten().cpu()
    if x.numel() == 0:
        return None
    q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float32))
    return {
        'n': int(x.numel()), 'min': float(x.min()), 'p05': float(q[0]),
        'median': float(q[1]), 'mean': float(x.mean()), 'p95': float(q[2]),
        'max': float(x.max()), 'nonpositive': int((x <= 0).sum()),
        'le_1e-6': int((x <= 1e-6).sum()), 'le_1e-4': int((x <= 1e-4).sum()),
        'le_1e-3': int((x <= 1e-3).sum()), 'le_1e-2': int((x <= 1e-2).sum()),
    }


def stats_pair(reference, candidate, *, bf16_cast=False):
    ref = reference.detach()
    cand = candidate.detach()
    rf = ref.float()
    cf = cand.float()
    delta = cf.double() - rf.double()
    ref_norm = rf.double().norm().item()
    row = {
        'elements': int(ref.numel()),
        'rel_l2': delta.norm().item() / max(ref_norm, 1e-12),
        'max_abs': delta.abs().max().item(),
        'reference_max_abs': rf.double().abs().max().item(),
        'finite': finite(ref) and finite(cand),
    }
    if bf16_cast:
        rb = rf.to(torch.bfloat16)
        cb = cf.to(torch.bfloat16)
        differing = rb != cb
        inf = torch.tensor(float('inf'), dtype=torch.bfloat16, device=rb.device)
        ulp = (torch.nextafter(rb, inf) - rb).abs().float().double()
        crossings = differing & (delta.abs() <= ulp)
        row.update({
            'bf16_differing_elements': int(differing.sum()),
            'bf16_differing_fraction': float(differing.float().mean()),
            'bf16_ulp_threshold_crossings': int(crossings.sum()),
            'sign_flips': int(((rf.sign() != cf.sign()) & (rf != 0) & (cf != 0)).sum()),
        })
    return row


def gate(row, rel=FP32_REL, abs_limit=FP32_ABS):
    return (row['rel_l2'] <= rel
            and row['max_abs'] <= abs_limit * max(1.0, row['reference_max_abs'])
            and row['finite'])


def margin_stats(logits):
    logits = logits.float()
    top = torch.topk(logits, 3, dim=-1).values
    return {
        'top1_minus_top2': summarize(top[..., 0] - top[..., 1]),
        'top2_minus_top3': summarize(top[..., 1] - top[..., 2]),
    }


def route_counts(keys, scan_selected, cand_selected):
    by_key = {}
    by_pass = {}
    by_pool = {}
    total = 0
    mismatches = 0
    for key in keys:
        a = scan_selected[key]
        b = cand_selected[key]
        if isinstance(a, dict):
            a = a['selected']
        if isinstance(b, dict):
            b = b['selected']
        if a.shape != b.shape:
            raise ValueError(f'routing shape mismatch at {key}')
        same = (torch.sort(a, dim=-1).values == torch.sort(b, dim=-1).values).all(-1)
        n = int(same.numel())
        m = int((~same).sum())
        by_key[str(key)] = {'tokens': n, 'mismatch_tokens': m,
                            'disagreement_fraction': m / max(1, n),
                            'exact_sets': m == 0}
        for group, bucket in ((key[1], by_pass), (key[2], by_pool)):
            entry = bucket.setdefault(str(group), {'tokens': 0, 'mismatch_tokens': 0})
            entry['tokens'] += n
            entry['mismatch_tokens'] += m
        total += n
        mismatches += m
    for bucket in (by_pass, by_pool):
        for entry in bucket.values():
            entry['disagreement_fraction'] = entry['mismatch_tokens'] / max(1, entry['tokens'])
    return {'tokens': total, 'mismatch_tokens': mismatches,
            'disagreement_fraction': mismatches / max(1, total),
            'by_key': by_key, 'by_pass': by_pass, 'by_pool': by_pool}


def rank_counts(scan_logits, cand_logits, mask):
    scan_top = torch.topk(scan_logits.float(), 2, dim=-1).indices
    order = torch.argsort(cand_logits.float(), dim=-1, descending=True, stable=True)
    rank1 = (order == scan_top[:, 0, None]).float().argmax(-1)
    rank2 = (order == scan_top[:, 1, None]).float().argmax(-1)
    def counts(values):
        out = torch.bincount(values[mask].long(), minlength=8).tolist()
        return {str(i): int(v) for i, v in enumerate(out) if v}
    return {'scan_top1_rank_in_candidate': counts(rank1),
            'scan_top2_rank_in_candidate': counts(rank2)}


def route_analysis(keys, scan_routes, cand_routes, scan_logits, cand_logits):
    base = route_counts(keys, {k: v['selected'] for k, v in scan_routes.items()},
                        {k: v['selected'] for k, v in cand_routes.items()})
    first = None
    for key in keys:
        a = scan_routes[key]['selected']
        b = cand_routes[key]['selected']
        same = (torch.sort(a, dim=-1).values == torch.sort(b, dim=-1).values).all(-1)
        mask = ~same
        if int(mask.sum()):
            first = key
            break
    result = dict(base)
    result['first_mismatch'] = None
    result['by_pass_key1'] = base['by_pass']
    result['by_pool_key2'] = base['by_pool']
    if first is not None:
        a = scan_routes[first]['selected']
        b = cand_routes[first]['selected']
        mask = ~((torch.sort(a, dim=-1).values == torch.sort(b, dim=-1).values).all(-1))
        sl = scan_logits[first].float().cpu()
        cl = cand_logits[first].float().cpu()
        result['first_mismatch'] = {
            'key': str(first), 'block': first[2], 'logical_pass': first[1],
            'tokens': int(mask.numel()), 'mismatch_tokens': int(mask.sum()),
            'disagreement_fraction': float(mask.float().mean()),
            'scan_margins_all': margin_stats(sl),
            'candidate_margins_all': margin_stats(cl),
            'scan_margins_mismatch': margin_stats(sl[mask]) if mask.any() else None,
            'candidate_margins_mismatch': margin_stats(cl[mask]) if mask.any() else None,
            'rank_counts_mismatch': rank_counts(sl, cl, mask),
        }
    return result


class Capture:
    def __init__(self, label):
        self.label = label
        self.calls = {}
        self.moe = {}
        self.handles = []
        self.run_originals = []
        self.apply_original = None
        self.counters = {}

    def install(self, model):
        self.model = model
        self.heads = model.config.dense.q_heads
        self.dim = model.config.dense.head_dim
        gdnmod.delta_scan = self._scan_entry
        gdnmod.delta_chunk64 = self._chunk_entry
        self.apply_original = model._apply_block
        def apply_wrapper(instance, block, x, positions, plans, previous, gamma, valid, key):
            if key[0] == 'C':
                CURRENT['key'] = key
                CURRENT['block'] = key[2]
                CURRENT['logical_pass'] = key[1]
                CURRENT['stage'] = 'C'
                CURRENT['block_input'] = x.detach().cpu()
                CURRENT['gamma'] = float(gamma)
            return self.apply_original(block, x, positions, plans, previous, gamma, valid, key)
        model._apply_block = types.MethodType(apply_wrapper, model)
        for i, block in enumerate(model.core):
            attn = getattr(block, 'attn', None)
            if isinstance(attn, gdnmod.GatedDeltaNet):
                original = attn.run
                def run_wrapper(instance, x, previous, *, algorithm=None,
                                _orig=original, _i=i):
                    self.counters[_i] = self.counters.get(_i, 0) + 1
                    logical_pass = self.counters[_i]
                    CURRENT['block'] = _i
                    CURRENT['logical_pass'] = logical_pass
                    CURRENT['stage'] = 'C'
                    y, state = _orig(x, previous, algorithm=algorithm)
                    key = (_i, logical_pass)
                    rec = self.calls.setdefault(key, {})
                    rec['x'] = x.detach().cpu()
                    rec['y'] = y.detach().cpu()
                    rec['block_input'] = CURRENT.get('block_input')
                    rec['gamma'] = CURRENT.get('gamma')
                    return y, state
                attn.run = types.MethodType(run_wrapper, attn)
                self.run_originals.append((attn, original))
                self.handles.append(attn.z.register_forward_hook(self._z_hook(i)))
                self.handles.append(attn.norm.register_forward_pre_hook(self._norm_pre(i)))
                self.handles.append(attn.norm.register_forward_hook(self._norm_post(i)))
                self.handles.append(attn.o.register_forward_pre_hook(self._o_pre(i)))
                self.handles.append(attn.o.register_forward_hook(self._o_post(i)))
            moe = getattr(block, 'moe', None)
            if moe is not None:
                self.handles.append(moe.register_forward_pre_hook(self._moe_pre(i)))

    def _key(self):
        return (CURRENT.get('block'), CURRENT.get('logical_pass'))

    def _record(self, field, value):
        key = self._key()
        if key[0] is None or key[1] is None:
            return
        self.calls.setdefault(key, {})[field] = value.detach().cpu()

    def _scan_entry(self, q, k, v, g, beta, state):
        key = self._key()
        rec = self.calls.setdefault(key, {})
        for name, value in (('q', q), ('k', k), ('v', v), ('g', g),
                            ('beta', beta), ('state', state)):
            rec[name] = value.detach().cpu()
        out, final = ORIG_SCAN(q, k, v, g, beta, state)
        rec['delta_out'] = out.detach().cpu()
        rec['delta_final'] = final.detach().cpu()
        return out, final

    def _chunk_entry(self, q, k, v, g, beta, state):
        key = self._key()
        rec = self.calls.setdefault(key, {})
        for name, value in (('q', q), ('k', k), ('v', v), ('g', g),
                            ('beta', beta), ('state', state)):
            rec[name] = value.detach().cpu()
        out, final = ORIG_CHUNK64(q, k, v, g, beta, state)
        rec['delta_out'] = out.detach().cpu()
        rec['delta_final'] = final.detach().cpu()
        return out, final

    def _z_hook(self, block_index):
        def hook(module, inputs, output):
            if CURRENT.get('block') == block_index:
                reshaped = F.silu(output.reshape(-1, self.heads, self.dim))
                self._record('z', reshaped)
        return hook

    def _norm_pre(self, block_index):
        def hook(module, inputs):
            if CURRENT.get('block') == block_index:
                self._record('delta_out_hook', inputs[0])
        return hook

    def _norm_post(self, block_index):
        def hook(module, inputs, output):
            if CURRENT.get('block') == block_index:
                self._record('norm_out', output)
        return hook

    def _o_pre(self, block_index):
        def hook(module, inputs):
            if CURRENT.get('block') == block_index:
                self._record('precast_bf16', inputs[0])
        return hook

    def _o_post(self, block_index):
        def hook(module, inputs, output):
            if CURRENT.get('block') == block_index:
                self._record('o_out', output)
        return hook

    def _moe_pre(self, block_index):
        def hook(module, inputs):
            if CURRENT.get('block') != block_index:
                return
            key = self._key()
            rec = self.moe.setdefault(key, {})
            x = inputs[0]
            rec['router_input'] = x.detach().cpu()
            with torch.autocast(device_type=x.device.type, enabled=False):
                flat = x.reshape(-1, x.shape[-1])
                logits = F.linear(_accum(flat), _accum(module.router.weight))
            rec['router_logits'] = logits.detach().cpu()
        return hook

    def uninstall(self):
        gdnmod.delta_scan = ORIG_SCAN
        gdnmod.delta_chunk64 = ORIG_CHUNK64
        self.model._apply_block = self.apply_original
        for attn, original in self.run_originals:
            attn.run = original
        for handle in self.handles:
            handle.remove()
        CURRENT.clear()


def detach_output(out):
    return {
        'logits': {depth: value.detach() for depth, value in out.logits.items()},
        'hidden': {depth: value.detach() for depth, value in out.hidden.items()},
        'offset2': {depth: value.detach() for depth, value in out.offset2.items()},
    }


def extract_routes(routing):
    routes = {}
    order = []
    for key, route in routing:
        order.append(key)
        routes[key] = {
            'indices': route.indices.detach().cpu(),
            'selected': route.selected.detach().cpu(),
        }
    return order, routes


def forward_model(cfg, *, fp32=False, capture=False, keep_setup=False):
    CURRENT.clear()
    setup, inputs, labels, case, precheck = build_state(cfg, LENGTH, 'cuda')
    model = setup['model']
    model.train()
    cap = Capture('fp32' if fp32 else 'bf16') if capture else None
    if cap is not None:
        cap.install(model)
    moved_inputs, moved_labels = _move_inputs(inputs, labels, 'cuda')
    setup['optimizer'].zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = (contextlib.nullcontext() if fp32 else
               torch.autocast(device_type='cuda', dtype=torch.bfloat16))
    with context:
        out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
        total, losses = losses_and_total(out, moved_labels, cfg)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    identity = {
        'input_sha256': case['input_sha256'],
        'initialization_sha256': setup['initialization_sha256'],
        'partition_plan_sha256': _partition_summary(setup['optimizer_policy'])['plan_sha256'],
    }
    routes_order, routes = extract_routes(out.routing)
    output = detach_output(out)
    finite_ok = (all(finite(value) for value in
                     list(output['logits'].values()) + list(output['hidden'].values()) +
                     list(output['offset2'].values()))
                 and all(finite(parameter) for parameter in model.parameters()))
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    if cap is not None:
        cap.uninstall()
    del out, total
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        'identity': identity, 'losses': losses, 'output': output,
        'routes_order': routes_order, 'routes': routes, 'finite': finite_ok,
        'wall_seconds': wall, 'peak_allocated_bytes': peak_allocated,
        'peak_reserved_bytes': peak_reserved, 'capture': cap,
        'precheck': precheck,
    }
    if keep_setup:
        result['setup'] = setup
    else:
        del setup
    return result


def compare_outputs(reference, candidate):
    tensors = {}
    for group in ('logits', 'hidden', 'offset2'):
        for depth in sorted(reference['output'][group]):
            tensors[f'{group}_{depth}'] = stats_pair(
                reference['output'][group][depth], candidate['output'][group][depth])
    losses = {name: abs(reference['losses'][name] - candidate['losses'][name])
              for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance')}
    routing = route_counts(candidate['routes_order'], reference['routes'],
                           candidate['routes'])
    return {'losses': losses, 'tensors': tensors, 'routing': routing}


def local_same_input(cap):
    calls = {}
    worst = {'rel_l2': 0.0, 'max_abs': 0.0}
    first_nonzero = None
    for key in sorted(cap.calls, key=lambda item: (item[1], item[0])):
        rec = cap.calls[key]
        q, k, v, g, beta, state = (rec[name] for name in
                                    ('q', 'k', 'v', 'g', 'beta', 'state'))
        with torch.no_grad():
            scan_out, scan_final = ORIG_SCAN(q, k, v, g, beta, state)
            chunk_out, chunk_final = ORIG_CHUNK64(q, k, v, g, beta, state)
        out = stats_pair(scan_out, chunk_out)
        final = stats_pair(scan_final, chunk_final)
        row = {'block': key[0], 'logical_pass': key[1],
               'output': out, 'state': final,
               'output_ok': gate(out), 'state_ok': gate(final)}
        calls[str(key)] = row
        worst['rel_l2'] = max(worst['rel_l2'], out['rel_l2'], final['rel_l2'])
        worst['max_abs'] = max(worst['max_abs'], out['max_abs'], final['max_abs'])
        if first_nonzero is None and out['rel_l2'] > 1e-12:
            first_nonzero = {'key': str(key), 'block': key[0], 'logical_pass': key[1],
                             'output_rel_l2': out['rel_l2'], 'output_max_abs': out['max_abs']}
    oracle = {}
    for key in (sorted(cap.calls, key=lambda item: (item[1], item[0]))[0],
                sorted(cap.calls, key=lambda item: (item[1], item[0]))[-1]):
        rec = cap.calls[key]
        q, k, v, g, beta, state = (rec[name].double() for name in
                                    ('q', 'k', 'v', 'g', 'beta', 'state'))
        with torch.no_grad():
            scan_out, scan_final = ORIG_SCAN(q, k, v, g, beta, state)
            chunk_out, chunk_final = ORIG_CHUNK64(q, k, v, g, beta, state)
        out = stats_pair(scan_out, chunk_out)
        final = stats_pair(scan_final, chunk_final)
        oracle[str(key)] = {'block': key[0], 'logical_pass': key[1],
                            'output': out, 'state': final,
                            'output_ok': gate(out, FP64_REL, FP64_ABS),
                            'state_ok': gate(final, FP64_REL, FP64_ABS)}
    status = 'passed' if all(row['output_ok'] and row['state_ok'] for row in calls.values()) else 'failed'
    return {'status': status, 'calls': calls, 'worst': worst,
            'first_nonzero': first_nonzero,
            'fp64_oracle': {'status': 'passed' if all(
                row['output_ok'] and row['state_ok'] for row in oracle.values()) else 'failed',
                'calls': oracle, 'tolerance': {'rel_l2': FP64_REL, 'max_abs': FP64_ABS}}}


def propagation(cap_scan, cap_chunk):
    calls = {}
    firsts = {'gdn_input': None, 'delta_out': None, 'precast_bf16': None,
              'o_out': None, 'router_input': None, 'router_logits': None}
    for key in sorted(cap_scan.calls, key=lambda item: (item[1], item[0])):
        s = cap_scan.calls[key]
        c = cap_chunk.calls[key]
        row = {'block': key[0], 'logical_pass': key[1]}
        for field in ('x', 'block_input', 'q', 'k', 'v', 'g', 'beta', 'state',
                      'delta_out', 'delta_final', 'y', 'norm_out', 'z',
                      'precast_bf16', 'o_out', 'router_input'):
            if field in s and field in c:
                cast = field in ('x', 'block_input', 'y', 'z', 'precast_bf16',
                                 'o_out', 'router_input')
                row[field] = stats_pair(s[field], c[field], bf16_cast=cast)
        if 'norm_out' in s and 'z' in s:
            s_pre = s['norm_out'] * s['z'].float()
            c_pre = c['norm_out'] * c['z'].float()
            row['precast_fp32'] = stats_pair(s_pre, c_pre, bf16_cast=True)
        if 'block_input' in s and 'y' in s:
            s_res = (s['block_input'].float() + s['gamma'] * s['y'].float()).to(torch.bfloat16)
            c_res = (c['block_input'].float() + c['gamma'] * c['y'].float()).to(torch.bfloat16)
            row['residual'] = stats_pair(s_res, c_res, bf16_cast=True)
        if 'router_logits' in s and 'router_logits' in c:
            row['router_logits_stats'] = stats_pair(s['router_logits'], c['router_logits'])
            row['scan_margins'] = margin_stats(s['router_logits'])
            row['candidate_margins'] = margin_stats(c['router_logits'])
        calls[str(key)] = row
        field_map = {'gdn_input': 'x', 'delta_out': 'delta_out',
                     'precast_bf16': 'precast_bf16', 'o_out': 'o_out',
                     'router_input': 'router_input',
                     'router_logits': 'router_logits_stats'}
        for label, field in field_map.items():
            if firsts[label] is None and field in row and row[field]['rel_l2'] > 1e-12:
                firsts[label] = {'key': str(key), 'block': key[0],
                                 'logical_pass': key[1], 'rel_l2': row[field]['rel_l2'],
                                 'max_abs': row[field]['max_abs']}
    by_pass = {}
    for key, row in calls.items():
        p = row['logical_pass']
        entry = by_pass.setdefault(str(p), {'delta_out_max_rel_l2': 0.0,
                                            'state_max_rel_l2': 0.0,
                                            'gdn_input_max_rel_l2': 0.0})
        entry['delta_out_max_rel_l2'] = max(entry['delta_out_max_rel_l2'],
                                            row.get('delta_out', {}).get('rel_l2', 0.0))
        entry['state_max_rel_l2'] = max(entry['state_max_rel_l2'],
                                        row.get('state', {}).get('rel_l2', 0.0))
        entry['gdn_input_max_rel_l2'] = max(entry['gdn_input_max_rel_l2'],
                                            row.get('x', {}).get('rel_l2', 0.0))
        entry['delta_final_max_rel_l2'] = max(
            entry.get('delta_final_max_rel_l2', 0.0),
            row.get('delta_final', {}).get('rel_l2', 0.0))
    return {'calls': calls, 'first_nonzero': firsts, 'by_pass': by_pass,
            'moe': moe_propagation(cap_scan, cap_chunk),
            'state_note': ('The no-cache forward supplies zero GDN prefix state at each '
                           'logical pass, so the state-input difference is zero at pass 2; '
                           'within-pass final-state divergence is reported under delta_final.')}


def moe_propagation(cap_scan, cap_chunk):
    calls = {}
    firsts = {'router_input': None, 'router_logits': None}
    by_pass = {}
    for key in sorted(cap_scan.moe, key=lambda item: (item[1], item[0])):
        s = cap_scan.moe[key]
        c = cap_chunk.moe[key]
        row = {'block': key[0], 'logical_pass': key[1],
               'router_input': stats_pair(s['router_input'], c['router_input'],
                                          bf16_cast=True),
               'router_logits': stats_pair(s['router_logits'], c['router_logits']),
               'scan_margins': margin_stats(s['router_logits']),
               'candidate_margins': margin_stats(c['router_logits'])}
        calls[str(key)] = row
        for field in firsts:
            if firsts[field] is None and row[field]['rel_l2'] > 1e-12:
                firsts[field] = {'key': str(key), 'block': key[0],
                                 'logical_pass': key[1],
                                 'rel_l2': row[field]['rel_l2'],
                                 'max_abs': row[field]['max_abs']}
        entry = by_pass.setdefault(str(key[1]), {
            'router_input_max_rel_l2': 0.0, 'router_logits_max_rel_l2': 0.0})
        entry['router_input_max_rel_l2'] = max(entry['router_input_max_rel_l2'],
                                               row['router_input']['rel_l2'])
        entry['router_logits_max_rel_l2'] = max(entry['router_logits_max_rel_l2'],
                                                row['router_logits']['rel_l2'])
    return {'calls': calls, 'first_nonzero': firsts, 'by_pass': by_pass}


def fp32_full_forward(cfg_scan, cfg_chunk):
    scan = forward_model(cfg_scan, fp32=True, capture=False)
    chunk = forward_model(cfg_chunk, fp32=True, capture=False)
    comparison = compare_outputs(scan, chunk)
    status = 'diagnostic_complete' if scan['finite'] and chunk['finite'] else 'failed'
    result = {'status': status, 'scan_losses': scan['losses'],
              'chunk64_losses': chunk['losses'],
              'loss_abs_deltas': comparison['losses'],
              'tensor_deltas': comparison['tensors'],
              'routing': comparison['routing'],
              'identity_match': scan['identity'] == chunk['identity']}
    del scan, chunk
    gc.collect()
    torch.cuda.empty_cache()
    return result


class FixedRouteMismatch(RuntimeError):
    pass


def fixed_route(scan, chunk, cfg_chunk):
    forced = {key: value['selected'].clone() for key, value in scan['routes'].items()}
    forced_indices = {key: value['indices'].clone() for key, value in scan['routes'].items()}
    model = chunk['setup']['model']
    moe_originals = []
    apply_original = model._apply_block

    def fixed_forward(module, x, valid):
        key = CURRENT.get('key')
        if key not in forced:
            raise FixedRouteMismatch(f'missing forced route for {key}')
        flat = x.reshape(-1, x.shape[-1])
        ix = valid.reshape(-1).nonzero().flatten()
        if not torch.equal(ix.cpu(), forced_indices[key]):
            raise FixedRouteMismatch('valid-position order differs from scan')
        h = flat.index_select(0, ix)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(_accum(h), _accum(module.router.weight))
            p = logits.softmax(-1)
            chosen = forced[key].to(x.device)
            if chosen.shape != (ix.numel(), module.topk):
                raise FixedRouteMismatch('forced route shape differs')
            w = p.gather(1, chosen)
            w = w / w.sum(-1, keepdim=True)
        route = Route(ix, p, chosen, w)
        if not ix.numel():
            return torch.zeros_like(x), route
        y = _accum(module.shared(h))
        for e, expert in enumerate(module.experts):
            rows, slots = (chosen == e).nonzero(as_tuple=True)
            if rows.numel():
                contribution = _accum(expert(h.index_select(0, rows))) * w[rows, slots, None]
                y = y.index_add(0, rows, contribution)
        return torch.zeros_like(flat).index_copy(0, ix, y.to(flat.dtype)).reshape_as(x), route

    def apply_wrapper(instance, block, x, positions, plans, previous, gamma, valid, key):
        CURRENT['key'] = key
        return apply_original(block, x, positions, plans, previous, gamma, valid, key)

    try:
        model._apply_block = types.MethodType(apply_wrapper, model)
        for block in model.core:
            moe = getattr(block, 'moe', None)
            if moe is not None:
                moe_originals.append((moe, moe.forward))
                moe.forward = types.MethodType(fixed_forward, moe)
        setup = chunk['setup']
        fixed_inputs, fixed_labels, _ = build_contiguous_case(LENGTH, SEED)
        moved_inputs, moved_labels = _move_inputs(fixed_inputs, fixed_labels, 'cuda')
        setup['optimizer'].zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(*moved_inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2))
            total, losses = losses_and_total(out, moved_labels, cfg_chunk)
        torch.cuda.synchronize()
        routes_order, routes = extract_routes(out.routing)
        output = detach_output(out)
        fixed_result = {'losses': losses, 'output': output,
                        'routes_order': routes_order, 'routes': routes,
                        'finite': all(finite(value) for value in
                                      list(output['logits'].values()) +
                                      list(output['hidden'].values()) +
                                      list(output['offset2'].values()))}
        comparison = compare_outputs(scan, fixed_result)
        selected_equal = all(torch.equal(scan['routes'][key]['selected'],
                                         fixed_result['routes'][key]['selected'])
                             for key in scan['routes'])
        natural = compare_outputs(scan, chunk)
        natural_logits = max(natural['tensors'][name]['rel_l2']
                             for name in ('logits_1', 'logits_2'))
        fixed_logits = max(comparison['tensors'][name]['rel_l2']
                           for name in ('logits_1', 'logits_2'))
        result = {
            'status': 'passed', 'selected_equal_scan_by_construction': selected_equal,
            'losses': losses, 'comparison_to_scan': comparison,
            'finite': fixed_result['finite'],
            'natural_logits_max_rel_l2': natural_logits,
            'fixed_logits_max_rel_l2': fixed_logits,
            'fixed_over_natural_logits_rel_ratio': fixed_logits / max(natural_logits, 1e-12),
            'claim_limit': 'teacher-forced diagnostic only; not production routing and not an exact-equivalence claim',
        }
        del out, total
    except FixedRouteMismatch as exc:
        result = {'status': 'not_run', 'reason': str(exc),
                  'claim_limit': 'teacher-forced diagnostic not run; no route claim'}
    finally:
        model._apply_block = apply_original
        for moe, original in moe_originals:
            moe.forward = original
        CURRENT.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return result


def static_check(args):
    cfg = load_config('bf16')[1]
    inputs, labels, case = build_contiguous_case(LENGTH, SEED)
    seed_all(SEED)
    model = model_from(cfg)
    init = tensor_digest(model.named_parameters())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer, policy = build_optimizer(model, cfg['training'])
    partition = _partition_summary(policy)['plan_sha256']
    result = {
        'status': 'passed' if (case['input_sha256'] == EXPECTED_INPUT and
                               init == EXPECTED_INIT and
                               partition == EXPECTED_PARTITION and
                               parameters == EXPECTED_PARAMETERS and
                               not subprocess.check_output(
                                   ['git', '-C', str(Path.cwd()), 'status', '--porcelain'],
                                   text=True).strip()) else 'failed',
        'source_commit': SOURCE_COMMIT,
        'source_head': subprocess.check_output(
            ['git', '-C', str(Path.cwd()), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_clean': not subprocess.check_output(
            ['git', '-C', str(Path.cwd()), 'status', '--porcelain'],
            text=True).strip(),
        'instruction_commit': args.instruction_commit,
        'config_sha256': sha(Path('configs/experiments/splot-nano-synthetic-bf16-muon.json')),
        'reference_sha256': sha(args.t0025_reference),
        'input_sha256': case['input_sha256'],
        'expected_input_sha256': EXPECTED_INPUT,
        'initialization_sha256': init,
        'expected_initialization_sha256': EXPECTED_INIT,
        'partition_plan_sha256': partition,
        'expected_partition_plan_sha256': EXPECTED_PARTITION,
        'parameters': parameters,
        'expected_parameters': EXPECTED_PARAMETERS,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'passed' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--t0025-reference', type=Path, required=True)
    parser.add_argument('--instruction-commit', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.check:
        return static_check(args)
    started = time.perf_counter()
    report = {
        'schema': 'splot.t0029.localization.v1', 'status': 'blocked',
        'task': 'T0029', 'attempt': 1, 'source_commit': SOURCE_COMMIT,
        'instruction_commit': args.instruction_commit,
        'reference_sha256': sha(args.t0025_reference),
        'config_sha256': sha(Path('configs/experiments/splot-nano-synthetic-bf16-muon.json')),
        'source_clean_before': not subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip(),
        'source_head_before': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
    }
    try:
        reference = json.loads(args.t0025_reference.read_text())
        base = load_config('bf16')[1]
        cfg_scan = copy.deepcopy(base)
        cfg_scan['recurrence']['gdn_algorithm'] = 'scan'
        cfg_chunk = copy.deepcopy(base)
        cfg_chunk['recurrence']['gdn_algorithm'] = 'chunk64'

        scan = forward_model(cfg_scan, capture=True)
        chunk = forward_model(cfg_chunk, capture=True, keep_setup=True)
        scan_ref = reference['losses']['scan']
        chunk_ref = reference['losses']['chunk64']
        identity_expected = reference['scan_identity']
        identity_actual = scan['identity']
        scan_loss_exact = scan['losses'] == scan_ref
        chunk_loss_delta = {name: abs(chunk['losses'][name] - chunk_ref[name])
                            for name in chunk_ref}
        chunk_loss_ok = all(value <= 1e-6 for value in chunk_loss_delta.values())
        expected_counts = {}
        for key, row in reference['routing']['by_key'].items():
            expected_counts[key] = int(round(row['disagreement_fraction'] * row['tokens']))
        actual_route = route_counts(scan['routes_order'], scan['routes'], chunk['routes'])
        count_ok = True
        count_rows = {}
        for key, expected in expected_counts.items():
            actual = actual_route['by_key'][key]['mismatch_tokens']
            count_rows[key] = {'expected_mismatch_tokens': expected,
                               'actual_mismatch_tokens': actual,
                               'exact': expected == actual}
            count_ok = count_ok and expected == actual
        scan_tensor_delta = {}
        for name, expected in reference['tensors'].items():
            group, depth = name.rsplit('_', 1)
            actual = stats_pair(scan['output'][group][int(depth)],
                                chunk['output'][group][int(depth)])
            scan_tensor_delta[name] = {
                'rel_l2_actual': actual['rel_l2'],
                'published_rel_l2': expected['rel_l2'],
                'rel_l2_abs_delta': abs(actual['rel_l2'] - expected['rel_l2']),
                'published_max_abs': expected['max_abs'],
                'actual_max_abs': actual['max_abs'],
                'max_abs_delta': abs(actual['max_abs'] - expected['max_abs']),
            }
        tensor_ok = all(row['rel_l2_abs_delta'] <= 1e-6 and row['max_abs_delta'] <= 1e-6
                        for row in scan_tensor_delta.values())
        reproduction_ok = (identity_actual == identity_expected and scan_loss_exact and
                           chunk_loss_ok and count_ok and tensor_ok and
                           scan['finite'] and chunk['finite'])
        reproduction = {
            'status': 'passed' if reproduction_ok else 'failed',
            'scan': {'identity': identity_actual, 'losses': scan['losses'],
                     'finite': scan['finite']},
            'chunk64': {'identity': chunk['identity'], 'losses': chunk['losses'],
                        'finite': chunk['finite']},
            'reference': {'scan_losses': scan_ref, 'chunk64_losses': chunk_ref,
                          'identity': identity_expected},
            'identity_exact': identity_actual == identity_expected,
            'scan_losses_exact': scan_loss_exact,
            'chunk64_loss_max_abs_delta': max(chunk_loss_delta.values()),
            'chunk64_loss_deltas': chunk_loss_delta,
            'chunk64_loss_within_1e-6': chunk_loss_ok,
            'per_key_routing_counts': count_rows,
            'per_key_routing_counts_exact': count_ok,
            'tensor_metric_deltas': scan_tensor_delta,
            'tensor_metrics_within_1e-6': tensor_ok,
            'routing': actual_route,
        }
        report['reproduction'] = reproduction
        report['status'] = 'blocked'
        if not reproduction_ok:
            report['localization'] = {'status': 'not_run',
                                      'reason': 'reproduction gate failed; no attribution'}
            report['status'] = 'failed'
            report['elapsed_seconds'] = time.perf_counter() - started
            report['source_clean_after'] = not subprocess.check_output(
                ['git', 'status', '--porcelain'], text=True).strip()
            report['source_head_after'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True).strip()
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            print(json.dumps({'status': report['status'],
                              'reproduction': reproduction['status']}, indent=2))
            return 2
        local = local_same_input(scan['capture'])
        prop = propagation(scan['capture'], chunk['capture'])
        route_key = lambda key: (key[2], key[1])
        routing = route_analysis(scan['routes_order'], scan['routes'],
                                 chunk['routes'],
                                 {key: capture_logits(scan['capture'], route_key(key))
                                  for key in scan['routes']},
                                 {key: capture_logits(chunk['capture'], route_key(key))
                                  for key in chunk['routes']})
        fp32 = fp32_full_forward(cfg_scan, cfg_chunk)
        fixed = fixed_route(scan, chunk, cfg_chunk)
        report['localization'] = {
            'status': 'passed',
            'same_input_fp32': local,
            'propagation': prop,
            'routing_divergence': routing,
            'fp32_full_forward': fp32,
            'fixed_route': fixed,
            'candidate_hypothesis': make_hypothesis(local, routing, fixed),
            'prospective_numerical_gate': {
                'status': 'proposed_not_applied',
                'requirements': [
                    'same-input FP32 GDN output and carried state pass rel<=1e-4 and maxabs<=1e-4*max(1,refmax) at every captured block/pass',
                    'token-weighted top2 route-set mismatch is zero across all logical passes',
                    'natural BF16 hidden/logit/offset2 differences satisfy the existing T0025 tensor gates without relaxation',
                    'teacher-forced-route and natural BF16 paths both meet the existing T0025 gates before any T2048/resource qualification',
                ],
            },
        }
        report['status'] = 'passed'
        report['elapsed_seconds'] = time.perf_counter() - started
        report['source_clean_after'] = not subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip()
        report['source_head_after'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
        report['gpu_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        report['gpu_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        report['limitations'] = [
            'One synthetic contiguous T512 invocation and one seed; no training, backward, optimizer step or checkpoint.',
            'No actual data, quality, throughput, energy or resource-speed claim.',
            'Local GDN comparison is same-input FP32; full-model BF16/FP32 comparisons include propagation through different recurrence/routing states.',
            'Teacher-forced routes are a diagnostic intervention, not production routing and not an exact-equivalence claim.',
            'Route-flip timing is observational; it does not by itself identify the first causal numerical origin.',
        ]
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'status': report['status'],
                          'reproduction': reproduction['status'],
                          'local_same_input': local['status'],
                          'route_weighted': routing['disagreement_fraction'],
                          'first_route_mismatch': routing.get('first_mismatch', {}).get('key'),
                          'fixed_route': fixed['status']}, indent=2))
        return 0
    except Exception as exc:
        report['status'] = 'blocked'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['stage'] = 'diagnostic'
        report['elapsed_seconds'] = time.perf_counter() - started
        report['source_clean_after'] = not subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip()
        report['source_head_after'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        traceback.print_exc()
        return 2


def capture_logits(capture, key):
    rec = capture.moe.get(key, {})
    if 'router_logits' not in rec:
        raise KeyError(f'no router logits for {key}')
    return rec['router_logits']


def make_hypothesis(local, routing, fixed):
    first = routing.get('first_mismatch')
    text = ('Same-input FP32 GDN comparisons are the local contraction-order control; '
            'the full-model BF16 comparison includes recurrence, BF16 boundary casts '
            'and sparse routing. ')
    if first is not None:
        text += (f"First top2 route-set mismatch is at block {first['block']}, logical pass "
                 f"{first['logical_pass']}, with {first['mismatch_tokens']} of "
                 f"{first['tokens']} valid positions. ")
    else:
        text += 'No top2 route-set mismatch was observed. '
    if fixed.get('status') == 'passed':
        ratio = fixed.get('fixed_over_natural_logits_rel_ratio')
        text += (f"Teacher-forced scan-selected routes changed the maximum logits rel-L2 "
                 f"from {fixed.get('natural_logits_max_rel_l2'):.6g} to "
                 f"{fixed.get('fixed_logits_max_rel_l2'):.6g} (ratio {ratio:.3f}); "
                 'this is an intervention result, not proof of a first cause. ')
    else:
        text += ('Teacher-forced-route diagnostic was not run; route-flip timing remains '
                 'descriptive only. ')
    return {'status': 'descriptive', 'text': text}


if __name__ == '__main__':
    sys.exit(main())
