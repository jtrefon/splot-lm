"""Focused native FP8 autograd and SPLOT projection tests."""
import argparse
import copy
import json
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

from data_fixture import synthetic_documents, split_documents, pack
from experiment import collate, seed_all, resolve_precision
from splot import DenseConfig, RecurrentConfig, SPLOTLM, training_objective
from splot.fp8 import (FP8Linear, convert_selected_linears, dequantize,
                       fp8_diagnostics, padding_plan, pad_rows, quantize_device,
                       relative_frobenius)
from optimizers import build_semantic_partition


def config(name):
    return json.loads((Path('configs/experiments') / name).read_text())


def model_from(cfg):
    return SPLOTLM(RecurrentConfig(dense=DenseConfig(**cfg['dense']), **cfg['recurrence']),
                   expert_intermediate=cfg['expert_intermediate'])


def one_batch(device):
    blocks = pack(split_documents(synthetic_documents())['train'], 96)
    return collate(blocks, [9], device), blocks


class FP8CPUTests(unittest.TestCase):
    def test_selection_identity_state_dict_and_partition(self):
        cfg = config('splot-smoke-synthetic-bf16.json')
        torch.manual_seed(42)
        model = model_from(cfg)
        before_ids = {id(p) for p in model.parameters()}
        before_keys = set(model.state_dict())
        before_partition = build_semantic_partition(model)
        core_weight_id = id(model.core[0].attn.q.weight)
        policy = convert_selected_linears(model)
        self.assertEqual(policy['selected_count'], 156)
        self.assertEqual(policy['excluded_count'], 11)
        self.assertEqual(sum(isinstance(m, FP8Linear) for m in model.modules()), 156)
        self.assertEqual(before_ids, {id(p) for p in model.parameters()})
        self.assertEqual(before_keys, set(model.state_dict()))
        self.assertEqual(core_weight_id, id(model.core[0].attn.q.weight))
        self.assertEqual(before_partition.metadata,
                         build_semantic_partition(model).metadata)
        for item in policy['selected']:
            if (item['name'].endswith('attn.q') or item['name'].endswith('attn.k')) and (
                    item['name'].startswith('prelude.') or item['name'].startswith('coda.') or
                    item['name'].startswith('core.3.')):
                self.assertEqual(item['role'], 'gqa_projection')
        excluded = {item['name'] for item in policy['excluded']}
        self.assertIn('core.3.attn.output_gate', excluded)
        self.assertIn('core.0.attn.a', excluded)
        self.assertIn('core.0.attn.b', excluded)
        self.assertIn('core.0.moe.router', excluded)
        self.assertNotIn('output_gate', {item['role'] for item in policy['selected']})

    def test_cpu_fp8_execution_is_rejected(self):
        linear = nn.Linear(16, 16, bias=False)
        module = FP8Linear.from_linear(linear)
        with self.assertRaises(RuntimeError):
            module(torch.randn(2, 16))
        with self.assertRaises(RuntimeError):
            resolve_precision('cpu', 'fp8')

    def test_padding_and_error_helpers(self):
        expected = {1: 16, 3: 16, 7: 16, 17: 32}
        for real, padded in expected.items():
            plan = padding_plan(real)
            self.assertEqual(plan['real_M'], real)
            self.assertEqual(plan['padded_M'], padded)
            self.assertGreaterEqual(plan['padding_fraction'], 0.)
        x = torch.ones(3, 4)
        padded = pad_rows(x, 16)
        self.assertEqual(tuple(padded.shape), (16, 4))
        self.assertTrue(torch.equal(padded[:3], x))
        self.assertEqual(int(torch.count_nonzero(padded[3:]).item()), 0)
        self.assertEqual(relative_frobenius(torch.zeros(2, 2), torch.zeros(2, 2)), 0.)
        self.assertGreater(relative_frobenius(torch.ones(2, 2), torch.zeros(2, 2)), 1e11)

    def test_default_paths_have_no_fp8_modules(self):
        for name in ('splot-smoke-synthetic.json', 'splot-smoke-synthetic-bf16.json'):
            model = model_from(config(name))
            self.assertFalse(any(isinstance(m, FP8Linear) for m in model.modules()))
            self.assertEqual(resolve_precision('cpu', 'fp32')['policy'], 'fp32')


class FP8CUDATests(unittest.TestCase):
    device = 'cuda'

    def _linear(self, m, k, n):
        torch.manual_seed(42)
        linear = nn.Linear(k, n, bias=False).to(self.device)
        fp = FP8Linear.from_linear(linear)
        self.assertIs(fp.weight, linear.weight)
        x = torch.randn(m, k, device=self.device)
        dy = torch.randn(m, n, device=self.device)
        return fp, x, dy

    def _refs(self, x, dy, weight, pad_m):
        xp = pad_rows(x, pad_m)
        dyp = pad_rows(dy, pad_m)
        qx, sx = quantize_device(xp, torch.float8_e4m3fn)
        qw, sw = quantize_device(weight, torch.float8_e4m3fn)
        qdy, sdy = quantize_device(dyp, torch.float8_e5m2)
        xq, wq, dyq = dequantize(qx, sx), dequantize(qw, sw), dequantize(qdy, sdy)
        return {
            'forward': (x @ weight.t(), (xq @ wq.t())[:x.shape[0]]),
            'dgrad': (dy @ weight, (dyq @ wq)[:x.shape[0]]),
            'wgrad': (dy.t() @ x, dyq.t() @ xq),
        }

    def test_linear_autograd_thresholds_and_padding(self):
        for m, k, n in ((1, 384, 512), (17, 384, 512), (64, 384, 512), (64, 32, 64)):
            fp, x, dy = self._linear(m, k, n)
            plan = padding_plan(m)
            refs = self._refs(x, dy, fp.weight.detach(), plan['padded_M'])
            x = x.detach().requires_grad_(True)
            y = fp(x)
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(fp.last_forward_dispatched)
            self.assertEqual(fp.last_forward_scales[0].device.type, 'cuda')
            y.backward(dy)
            self.assertEqual(x.grad.dtype, torch.float32)
            self.assertEqual(fp.weight.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(torch.isfinite(fp.weight.grad).all())
            self.assertLessEqual(relative_frobenius(y, refs['forward'][0]), .15)
            self.assertLessEqual(relative_frobenius(x.grad, refs['dgrad'][0]), .15)
            self.assertLessEqual(relative_frobenius(fp.weight.grad, refs['wgrad'][0]), .15)
            self.assertLessEqual(relative_frobenius(y, refs['forward'][1]), .025)
            self.assertLessEqual(relative_frobenius(x.grad, refs['dgrad'][1]), .025)
            self.assertLessEqual(relative_frobenius(fp.weight.grad, refs['wgrad'][1]), .025)
            self.assertEqual(fp.padded_forward_rows, plan['padding_rows'])
            self.assertEqual(fp.padded_backward_rows, plan['padding_rows'])

    def test_repeated_weight_gradient_and_no_duplicate_parameter(self):
        fp, x1, dy1 = self._linear(17, 384, 512)
        fp.weight.grad = None
        x2 = torch.randn_like(x1)
        dy2 = torch.randn_like(dy1)
        y1, y2 = fp(x1), fp(x2)
        torch.autograd.backward([y1, y2], [dy1, dy2])
        summed = fp.weight.grad.clone()
        self.assertEqual(sum(p is fp.weight for p in fp.parameters()), 1)
        fp.weight.grad = None
        y1 = fp(x1)
        y1.backward(dy1)
        g1 = fp.weight.grad.clone()
        fp.weight.grad = None
        y2 = fp(x2)
        y2.backward(dy2)
        g2 = fp.weight.grad.clone()
        torch.testing.assert_close(summed, g1 + g2, rtol=1e-5, atol=1e-6)

    def test_zero_m_and_exact_zero(self):
        fp, _, _ = self._linear(0, 384, 512)
        x = torch.empty(0, 384, device=self.device, requires_grad=True)
        y = fp(x)
        self.assertEqual(tuple(y.shape), (0, 512))
        y.sum().backward()
        self.assertEqual(tuple(x.grad.shape), (0, 384))
        self.assertFalse(fp.last_forward_dispatched)
        self.assertEqual(fp.empty_forward_calls, 1)
        fp2, x, dy = self._linear(16, 384, 512)
        x = torch.zeros_like(x).requires_grad_(True)
        dy = torch.zeros_like(dy)
        y = fp2(x)
        y.backward(dy)
        self.assertEqual(int(torch.count_nonzero(y).item()), 0)
        self.assertEqual(int(torch.count_nonzero(x.grad).item()), 0)
        self.assertEqual(int(torch.count_nonzero(fp2.weight.grad).item()), 0)

    def test_profiler_and_scale_residency(self):
        fp, x, dy = self._linear(17, 384, 512)
        x = x.detach().requires_grad_(True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            y = fp(x)
            y.backward(dy)
        torch.cuda.synchronize()
        keys = [item.key for item in prof.key_averages()]
        self.assertTrue(any('_scaled_mm' in key or 'sm89_' in key for key in keys), keys)
        self.assertTrue(any('sm89_' in key for key in keys), keys)
        self.assertFalse(any('DtoH' in key for key in keys), keys)
        self.assertEqual(fp.last_forward_scales[0].device.type, 'cuda')
        self.assertEqual(fp.last_backward_scales[0].device.type, 'cuda')


def _grad_stats(bf, fp):
    diff = 0.; ref = 0.; matched = 0; missing = 0
    for a, b in zip(bf.parameters(), fp.parameters()):
        if a.grad is None and b.grad is None:
            continue
        if a.grad is None or b.grad is None:
            missing += 1
            continue
        matched += 1
        diff += float((a.grad.float() - b.grad.float()).pow(2).sum().item())
        ref += float(a.grad.float().pow(2).sum().item())
    return {'matched_parameters': matched, 'missing_gradients': missing,
            'relative_l2': math.sqrt(diff / (ref + 1e-12))}


def _routing_stats(bf_routes, fp_routes):
    bf_routes, fp_routes = dict(bf_routes), dict(fp_routes)
    total = exact = top1 = 0
    for key in sorted(set(bf_routes) & set(fp_routes)):
        a, b = bf_routes[key].selected, fp_routes[key].selected
        for row_a, row_b in zip(a, b):
            total += 1
            exact += set(row_a.tolist()) == set(row_b.tolist())
            top1 += int(row_a[0]) == int(row_b[0])
    return {'tokens': total, 'exact_topk_set_agreement': exact / total if total else None,
            'top1_agreement': top1 / total if total else None}


def run_depth_report(device, path):
    blocks = pack(split_documents(synthetic_documents())['train'], 96)
    x, labels = collate(blocks, [9], device)
    report = {'device': device, 'seed': 42, 'thresholds': {'ce': .5, 'objective': .75},
              'models': {}, 'status': 'passed', 'limitations': [
                  'Forward/backward comparison only; no optimizer step, convergence or quality claim.',
                  'Routing and drift are descriptive; top-k discontinuities are allowed.',
                  'No throughput or long-context claim.']}
    for name, cfg_name in (('smoke', 'splot-smoke-synthetic-bf16.json'),
                           ('nano', 'splot-nano-synthetic-bf16.json')):
        cfg = config(cfg_name)
        seed_all(42)
        base = model_from(cfg)
        bf = copy.deepcopy(base).to(device)
        fp = copy.deepcopy(base).to(device)
        policy = convert_selected_linears(fp)
        report['models'][name] = {'parameters': sum(p.numel() for p in base.parameters()),
                                  'selected_count': policy['selected_count'],
                                  'excluded_count': policy['excluded_count'],
                                  'depths': {}}
        for depth in (1, 2, 4):
            readouts = tuple(range(1, depth + 1))
            for model in (bf, fp):
                model.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                out_bf = bf(*x, loops=depth, readouts=readouts, offset2_depths=(1, depth))
                out_fp = fp(*x, loops=depth, readouts=readouts, offset2_depths=(1, depth))
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                obj_bf = training_objective(bf, x, labels, mode='A2', loops=depth)
                obj_fp = training_objective(fp, x, labels, mode='A2', loops=depth)
            obj_bf.total.backward()
            obj_fp.total.backward()
            entry = {
                'depth': depth,
                'ce': {key: {'bf16': float(getattr(obj_bf, key).detach()),
                              'fp8': float(getattr(obj_fp, key).detach()),
                              'absolute_difference': abs(float(getattr(obj_bf, key).detach())
                                                         - float(getattr(obj_fp, key).detach()))}
                       for key in ('final_ntp', 'early_ntp', 'offset2', 'total')},
                'hidden_l2': relative_frobenius(out_fp.hidden[depth], out_bf.hidden[depth]),
                'logit_l2': relative_frobenius(out_fp.logits[depth], out_bf.logits[depth]),
                'gradients': _grad_stats(bf, fp),
                'routing': _routing_stats(out_bf.routing, out_fp.routing),
                'labels': {'bf16_final': obj_bf.ntp_count, 'fp8_final': obj_fp.ntp_count,
                           'bf16_offset2': obj_bf.offset2_count, 'fp8_offset2': obj_fp.offset2_count},
                'finite': all(torch.isfinite(getattr(obj_fp, key)).all()
                              for key in ('final_ntp', 'early_ntp', 'offset2', 'total')),
            }
            entry['ce_status'] = all(
                abs(entry['ce'][key]['absolute_difference']) <=
                (report['thresholds']['ce'] if key != 'total' else report['thresholds']['objective'])
                for key in ('final_ntp', 'early_ntp', 'offset2', 'total'))
            if not entry['ce_status'] or not entry['finite']:
                report['status'] = 'failed'
            report['models'][name]['depths'][depth] = entry
        report['models'][name]['fp8_diagnostics'] = fp8_diagnostics(fp)
        report['models'][name]['policy'] = policy
        del base, bf, fp
        torch.cuda.empty_cache()
    report['status'] = 'passed' if report['status'] == 'passed' else 'failed'
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        FP8CUDATests if args.device == 'cuda' else FP8CPUTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.device == 'cuda' and args.report is not None and result.wasSuccessful():
        summary = run_depth_report(args.device, args.report)
        print(json.dumps({'depth_report': str(args.report), 'status': summary['status']}, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
