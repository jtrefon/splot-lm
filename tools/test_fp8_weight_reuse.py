"""Focused tests for opt-in per-update FP8 weight reuse."""
import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
from torch import nn

from context_probe import build_contiguous_case
from data_fixture import pack, split_documents, synthetic_documents
from experiment import collate, seed_all, train
from optimizers import assert_optimizer_state_finite, build_optimizer, optimizer_state_inventory
from splot import DenseConfig, RecurrentConfig, SPLOTLM, training_objective
from splot.fp8 import (FP8Linear, clear_fp8_weight_cache, convert_selected_linears,
                       fp8_assert_finite, fp8_diagnostics, has_fp8, reset_fp8_audit,
                       set_fp8_weight_reuse)


def config(name):
    return json.loads((Path('configs/experiments') / name).read_text())


def model_from(cfg):
    return SPLOTLM(RecurrentConfig(dense=DenseConfig(**cfg['dense']), **cfg['recurrence']),
                   expert_intermediate=cfg['expert_intermediate'])


def tree_digest(value, h=None):
    if h is None:
        h = hashlib.sha256()
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        h.update(str(tensor.dtype).encode('ascii'))
        h.update(str(tuple(tensor.shape)).encode('ascii'))
        h.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            h.update(str(key).encode('utf-8'))
            tree_digest(value[key], h)
    elif isinstance(value, (list, tuple)):
        h.update(type(value).__name__.encode('ascii'))
        for item in value:
            tree_digest(item, h)
    else:
        h.update(repr(value).encode('utf-8'))
    return h.hexdigest()


def tree_equal(a, b, path='root'):
    if torch.is_tensor(a):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError(f'tensor metadata mismatch: {path}')
        if not torch.equal(a, b):
            raise AssertionError(f'tensor mismatch: {path}')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f'dict keys mismatch: {path}')
        for key in a:
            tree_equal(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f'sequence mismatch: {path}')
        for index, (x, y) in enumerate(zip(a, b)):
            tree_equal(x, y, f'{path}.{index}')
    elif a != b:
        raise AssertionError(f'value mismatch: {path}')


def setup_model(cfg, cache, device='cuda'):
    seed_all(42)
    model = model_from(cfg).to(device)
    policy = convert_selected_linears(model, weight_reuse=cache)
    optimizer, optimizer_policy = build_optimizer(model, cfg['training'])
    return model, optimizer, policy, optimizer_policy


def batch_t96(device):
    blocks = pack(split_documents(synthetic_documents())['train'], 96)
    return collate(blocks, [9], device)


def run_update(model, optimizer, inputs, labels, *, cache, device='cuda', timed=False):
    if has_fp8(model):
        reset_fp8_audit(model)
    optimizer.zero_grad(set_to_none=True)
    sync = lambda: torch.cuda.synchronize(device)
    times = {}
    sync(); started = time.perf_counter()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        objective = training_objective(model, inputs, labels, mode='A2', loops=2)
    sync(); times['forward'] = time.perf_counter() - started
    backward_started = time.perf_counter()
    objective.total.backward()
    sync(); times['backward'] = time.perf_counter() - backward_started
    check_started = time.perf_counter()
    fp8_assert_finite(model, 'pair-check')
    audit = fp8_diagnostics(model)
    clear_fp8_weight_cache(model)
    sync(); times['checks'] = time.perf_counter() - check_started
    step_started = time.perf_counter()
    optimizer.step()
    sync(); times['optimizer_step'] = time.perf_counter() - step_started
    post_started = time.perf_counter()
    assert_optimizer_state_finite(optimizer, 'pair-post')
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError('nonfinite pair parameter')
    sync(); times['post_update_checks'] = time.perf_counter() - post_started
    times['end_to_end'] = time.perf_counter() - started
    return objective, audit, times


def compare_pair(cache_off, cache_on):
    model_off, opt_off, _, _ = cache_off
    model_on, opt_on, _, _ = cache_on
    checks = {
        'initial_parameters_equal': True,
        'pre_update_losses_equal': True,
        'native_calls_equal': True,
        'logical_quantization_equal': True,
        'model_equal': True,
        'optimizer_equal': True,
        'cache_hits_positive': True,
        'cache_quantizations_reduced': True,
    }
    for x, y in zip(model_off.parameters(), model_on.parameters()):
        if not torch.equal(x, y):
            checks['model_equal'] = False
    # Compare the initial state was captured before the update.
    if not torch.equal(cache_off[3]['objective_total'], cache_on[3]['objective_total']):
        checks['pre_update_losses_equal'] = False
    if cache_off[3]['native'] != cache_on[3]['native']:
        checks['native_calls_equal'] = False
    if cache_off[3]['quantization'] != cache_on[3]['quantization']:
        checks['logical_quantization_equal'] = False
    try:
        tree_equal(opt_off.state_dict(), opt_on.state_dict())
    except AssertionError:
        checks['optimizer_equal'] = False
    if cache_on[3]['audit']['weight_cache_hits'] <= 0:
        checks['cache_hits_positive'] = False
    if not (cache_on[3]['audit']['weight_quantizations']
            < cache_off[3]['audit']['weight_quantizations']):
        checks['cache_quantizations_reduced'] = False
    return checks


class CacheLifecycleTests(unittest.TestCase):
    def test_reuse_hit_invalidation_and_clear(self):
        linear = nn.Linear(32, 32, bias=False)
        fp = FP8Linear.from_linear(linear)
        fp.set_weight_reuse(True)
        qw1, sw1, stats1, reused1 = fp.quantize_weight(fp.weight)
        qw2, sw2, stats2, reused2 = fp.quantize_weight(fp.weight)
        self.assertFalse(reused1)
        self.assertTrue(reused2)
        self.assertIs(qw1, qw2)
        self.assertIs(sw1, sw2)
        self.assertGreater(fp.audit_weight_cache_bytes, 0)
        with torch.no_grad():
            fp.weight.add_(0.125)
        qw3, _, _, reused3 = fp.quantize_weight(fp.weight)
        self.assertFalse(reused3)
        self.assertFalse(torch.equal(qw1, qw3))
        fp.clear_weight_cache()
        self.assertIsNone(fp._weight_cache)
        self.assertEqual(fp.audit_weight_cache_bytes, 0)
        fp.set_weight_reuse(False)
        _, _, _, reused4 = fp.quantize_weight(fp.weight)
        self.assertFalse(reused4)
        with self.assertRaises(TypeError):
            fp.set_weight_reuse(1)

    def test_reset_stats_clears_cache_and_counters(self):
        linear = nn.Linear(16, 16, bias=False)
        fp = FP8Linear.from_linear(linear)
        fp.set_weight_reuse(True)
        fp.quantize_weight(fp.weight)
        fp.audit_weight_quantizations = 3
        fp.audit_weight_cache_hits = 2
        fp.reset_stats()
        self.assertIsNone(fp._weight_cache)
        self.assertEqual(fp.audit_weight_quantizations, 0)
        self.assertEqual(fp.audit_weight_cache_hits, 0)
        self.assertEqual(fp.audit_weight_cache_bytes, 0)
        self.assertTrue(fp.weight_reuse)


class NanoCacheTests(unittest.TestCase):
    device = 'cuda'

    def require_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')

    def test_nano_first_step_cache_off_on_exact(self):
        self.require_cuda()
        cfg = config('splot-nano-synthetic-fp8-muon.json')
        off = setup_model(cfg, False)
        on = setup_model(cfg, True)
        inputs, labels = batch_t96(self.device)
        results = []
        for model, opt, _, opt_policy, cache in (
            (*off[:3], off[3], False), (*on[:3], on[3], True)
        ):
            objective, audit, times = run_update(model, opt, inputs, labels, cache=cache,
                                                 device=self.device)
            results.append((model, opt, audit, {
                'objective_total': float(objective.total.detach()),
                'native': (audit['native_forward_calls'], audit['native_backward_calls']),
                'quantization': audit['quantization'],
                'audit': audit,
                'times': times,
            }))
        # Capture exact post-step state before comparing.
        model_off, opt_off, audit_off, info_off = results[0]
        model_on, opt_on, audit_on, info_on = results[1]
        for x, y in zip(model_off.parameters(), model_on.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        tree_equal(opt_off.state_dict(), opt_on.state_dict())
        self.assertEqual(info_off['objective_total'], info_on['objective_total'])
        self.assertEqual(info_off['native'], info_on['native'])
        self.assertEqual(info_off['quantization'], info_on['quantization'])
        self.assertGreater(info_on['audit']['weight_cache_hits'], 0)
        self.assertLess(info_on['audit']['weight_quantizations'],
                        info_off['audit']['weight_quantizations'])
        self.assertFalse(off[2]['weight_reuse'])
        self.assertTrue(on[2]['weight_reuse'])

    def test_cache_resume_absent_and_cross_mode_rejected(self):
        self.require_cuda()
        cfg = config('splot-smoke-synthetic-fp8-muon.json')
        blocks = pack(split_documents(synthetic_documents())['train'], 96)
        objective = lambda m, x, y: training_objective(m, x, y, mode='A2', loops=2)
        identity = {'schema': 1, 'tag': 'cache-resume', 'fp8_weight_reuse': True}

        def run(path, steps, resume=False, cache=True, ident=None):
            seed_all(42 if not resume else 99)
            model = model_from(cfg).to(self.device)
            convert_selected_linears(model, weight_reuse=cache)
            opt = build_optimizer(model, cfg['training'])[0]
            history = train(model, opt, blocks, objective, ident or identity, path,
                            steps=steps, batch_size=1, seed=42, device=self.device,
                            resume=resume, autocast_dtype=torch.bfloat16,
                            checkpoint_interval=1, post_update_check=True)
            return model, opt, history

        with tempfile.TemporaryDirectory(prefix='splot-t0017-resume-') as tmp:
            root = Path(tmp)
            a, ao, ah = run(root / 'continuous', 2)
            _, _, _ = run(root / 'split', 1)
            b, bo, bh = run(root / 'split', 2, resume=True)
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            tree_equal(ao.state_dict(), bo.state_dict())
            ignore = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}
            self.assertEqual([{k: v for k, v in row.items() if k not in ignore} for row in ah],
                             [{k: v for k, v in row.items() if k not in ignore} for row in bh])
            checkpoint = torch.load(root / 'split' / 'checkpoint.pt', map_location='cpu',
                                    weights_only=True)
            self.assertEqual(set(checkpoint), {'identity', 'model', 'optimizer', 'rng', 'history'})
            self.assertFalse(any('weight_cache' in str(key) for key in checkpoint['model']))
            with self.assertRaises(ValueError):
                run(root / 'split', 2, resume=True, cache=False,
                    ident={'schema': 1, 'tag': 'cache-resume', 'fp8_weight_reuse': False})


def run_t512_pair():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg = config('splot-nano-synthetic-fp8-muon.json')
    inputs_cpu, labels_cpu, input_meta = build_contiguous_case(512, 42)
    inputs = tuple(value.to('cuda') for value in inputs_cpu)
    labels = {key: value.to('cuda') for key, value in labels_cpu.items()}
    results = {}
    for cache in (False, True):
        model, optimizer, policy, optimizer_policy = setup_model(cfg, cache)
        initial_digest = tree_digest(model.state_dict())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats('cuda')
        objective, audit, phases = run_update(model, optimizer, inputs, labels,
                                              cache=cache, device='cuda', timed=True)
        state = optimizer_state_inventory(optimizer)
        results['cache_on' if cache else 'cache_off'] = {
            'cache': cache,
            'policy': policy,
            'optimizer_policy': optimizer_policy,
            'input': input_meta,
            'initial_model_digest': initial_digest,
            'model_digest': tree_digest(model.state_dict()),
            'optimizer_digest': tree_digest(optimizer.state_dict()),
            'losses': {name: float(getattr(objective, name).detach())
                       for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance')},
            'audit': audit,
            'phases_seconds': phases,
            'state': state,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated('cuda'),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved('cuda'),
            'parameters': sum(p.numel() for p in model.parameters()),
        }
        del model, optimizer, objective
        torch.cuda.empty_cache()
    off, on = results['cache_off'], results['cache_on']
    checks = {
        'initial_state_equal': off['initial_model_digest'] == on['initial_model_digest'],
        'losses_equal': off['losses'] == on['losses'],
        'native_equal': (off['audit']['native_forward_calls'] == on['audit']['native_forward_calls']
                         and off['audit']['native_backward_calls'] == on['audit']['native_backward_calls']),
        'logical_quantization_equal': off['audit']['quantization'] == on['audit']['quantization'],
        'cache_hits_positive': on['audit']['weight_cache_hits'] > 0,
        'cache_quantizations_reduced': on['audit']['weight_quantizations'] < off['audit']['weight_quantizations'],
        'parameters_equal': off['parameters'] == on['parameters'],
        'model_equal': off['model_digest'] == on['model_digest'],
        'optimizer_equal': off['optimizer_digest'] == on['optimizer_digest'],
    }
    return {'status': 'passed' if all(checks.values()) else 'failed',
            'checks': checks, 'input': input_meta,
            'cache_off': {k: off[k] for k in ('losses', 'audit', 'phases_seconds', 'state',
                                              'initial_model_digest', 'model_digest',
                                              'optimizer_digest', 'peak_allocated_bytes',
                                              'peak_reserved_bytes', 'parameters')},
            'cache_on': {k: on[k] for k in ('losses', 'audit', 'phases_seconds', 'state',
                                            'initial_model_digest', 'model_digest',
                                            'optimizer_digest', 'peak_allocated_bytes',
                                            'peak_reserved_bytes', 'parameters')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--t512-pair', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.t512_pair:
        if args.out is None:
            parser.error('--t512-pair requires --out')
        result = run_t512_pair()
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps({'status': result['status'], 'out': str(args.out)}, sort_keys=True))
        sys.exit(0 if result['status'] == 'passed' else 1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CacheLifecycleTests)
    if args.device == 'cuda':
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NanoCacheTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
