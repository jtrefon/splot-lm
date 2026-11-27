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
                       fp8_assert_finite, fp8_diagnostics, padding_plan, pad_rows,
                       quantize_device, relative_frobenius, reset_fp8_audit)
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

    def test_report_exit_contract(self):
        self.assertTrue(report_exit_ok(True, False, None))
        self.assertTrue(report_exit_ok(True, True, 'passed'))
        self.assertFalse(report_exit_ok(True, True, 'failed'))
        self.assertFalse(report_exit_ok(False, False, None))

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
        dy = torch.randn(m, n, device=self.device).to(torch.bfloat16)
        return fp, x, dy

    def _refs(self, x, dy, weight, pad_m):
        dy = dy.float()
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

    def test_nonfinite_guards(self):
        fp, x, dy = self._linear(16, 384, 512)
        bad = x.clone(); bad[0, 0] = float('nan')
        y = fp(bad)
        with self.assertRaises(FloatingPointError):
            fp8_assert_finite(fp, 'input')
        fp, x, dy = self._linear(16, 384, 512)
        with torch.no_grad():
            fp.weight[0, 0] = float('nan')
        y = fp(x)
        with self.assertRaises(FloatingPointError):
            fp8_assert_finite(fp, 'weight')
        fp, x, dy = self._linear(16, 384, 512)
        bad_grad = dy.clone(); bad_grad[0, 0] = float('nan')
        y = fp(x)
        y.backward(bad_grad)
        with self.assertRaises(FloatingPointError):
            fp8_assert_finite(fp, 'grad')

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


def report_exit_ok(test_success, report_requested, report_status):
    return bool(test_success and (not report_requested or report_status == 'passed'))


def _grad_stats(bf, fp):
    matched_diff = matched_ref = full_diff = full_ref = 0.
    matched = full = missing_bf = missing_fp = nonfinite_bf = nonfinite_fp = 0
    for a, b in zip(bf.parameters(), fp.parameters()):
        ag, bg = a.grad, b.grad
        if ag is None and bg is None:
            continue
        full += 1
        if ag is None:
            missing_bf += 1
            ag_use = torch.zeros_like(bg)
        else:
            ag_use = ag.float()
            nonfinite_bf += int((~torch.isfinite(ag_use)).sum().item())
        if bg is None:
            missing_fp += 1
            bg_use = torch.zeros_like(ag) if ag is not None else torch.zeros_like(b.weight)
        else:
            bg_use = bg.float()
            nonfinite_fp += int((~torch.isfinite(bg_use)).sum().item())
        full_diff += float((ag_use - bg_use).pow(2).sum().item())
        full_ref += float(ag_use.pow(2).sum().item())
        if ag is not None and bg is not None:
            matched += 1
            matched_diff += float((ag_use - bg_use).pow(2).sum().item())
            matched_ref += float(ag_use.pow(2).sum().item())
    return {
        'matched_parameters': matched,
        'full_vector_parameters': full,
        'missing_bf16_gradients': missing_bf,
        'missing_fp8_gradients': missing_fp,
        'nonfinite_bf16_gradients': nonfinite_bf,
        'nonfinite_fp8_gradients': nonfinite_fp,
        'relative_l2_matched': math.sqrt(matched_diff / (matched_ref + 1e-12)),
        'relative_l2_full_vector': math.sqrt(full_diff / (full_ref + 1e-12)),
    }


def _routing_stats(bf_routes, fp_routes):
    bf_routes, fp_routes = dict(bf_routes), dict(fp_routes)
    if set(bf_routes) != set(fp_routes):
        raise AssertionError('routing keys differ between BF16 and FP8')
    total = exact = top1 = 0
    for key in sorted(bf_routes):
        a, b = bf_routes[key].selected, fp_routes[key].selected
        if a.shape != b.shape:
            raise AssertionError(f'routing selected shape differs for {key}: {a.shape} != {b.shape}')
        for row_a, row_b in zip(a, b):
            total += 1
            exact += set(row_a.tolist()) == set(row_b.tolist())
            top1 += int(row_a[0]) == int(row_b[0])
    return {'keys_equal': True, 'shape_equal': True, 'tokens': total,
            'exact_topk_set_agreement': exact / total if total else None,
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
            reset_fp8_audit(fp)
            readouts = tuple(range(1, depth + 1))
            for model in (bf, fp):
                model.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                out_bf = bf(*x, loops=depth, readouts=readouts, offset2_depths=(1, depth))
                out_fp = fp(*x, loops=depth, readouts=readouts, offset2_depths=(1, depth))
            audit_failure = None
            try:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    obj_bf = training_objective(bf, x, labels, mode='A2', loops=depth)
                    obj_fp = training_objective(fp, x, labels, mode='A2', loops=depth)
                obj_bf.total.backward()
                obj_fp.total.backward()
                fp8_assert_finite(fp, f'depth{depth}')
            except FloatingPointError as exc:
                audit_failure = f'{type(exc).__name__}: {exc}'
            gradients = _grad_stats(bf, fp)
            labels_value = {'bf16_final': obj_bf.ntp_count, 'fp8_final': obj_fp.ntp_count,
                            'bf16_offset2': obj_bf.offset2_count, 'fp8_offset2': obj_fp.offset2_count}
            labels_equal = (labels_value['bf16_final'] == labels_value['fp8_final'] and
                            labels_value['bf16_offset2'] == labels_value['fp8_offset2'])
            try:
                routing = _routing_stats(out_bf.routing, out_fp.routing)
            except AssertionError as exc:
                routing = {'status': 'failed', 'error': str(exc)}
            finite = all(torch.isfinite(value).all().item()
                         for value in (out_bf.hidden[depth], out_bf.logits[depth],
                                       out_fp.hidden[depth], out_fp.logits[depth]))
            finite = finite and all(torch.isfinite(getattr(obj, key)).all().item()
                                    for obj in (obj_bf, obj_fp)
                                    for key in ('final_ntp', 'early_ntp', 'offset2', 'total'))
            finite = finite and gradients['nonfinite_bf16_gradients'] == 0
            finite = finite and gradients['nonfinite_fp8_gradients'] == 0
            finite = finite and audit_failure is None and routing.get('status') != 'failed'
            entry = {
                'depth': depth,
                'ce': {key: {'bf16': float(getattr(obj_bf, key).detach()),
                              'fp8': float(getattr(obj_fp, key).detach()),
                              'absolute_difference': abs(float(getattr(obj_bf, key).detach())
                                                         - float(getattr(obj_fp, key).detach()))}
                       for key in ('final_ntp', 'early_ntp', 'offset2', 'total')},
                'hidden_l2': relative_frobenius(out_fp.hidden[depth], out_bf.hidden[depth]),
                'logit_l2': relative_frobenius(out_fp.logits[depth], out_bf.logits[depth]),
                'gradients': gradients,
                'routing': routing,
                'labels': labels_value,
                'labels_equal': labels_equal,
                'finite': finite,
                'fp8_diagnostics': fp8_diagnostics(fp),
            }
            if audit_failure is not None:
                entry['audit_failure'] = audit_failure
            entry['ce_status'] = all(
                abs(entry['ce'][key]['absolute_difference']) <=
                (report['thresholds']['ce'] if key != 'total' else report['thresholds']['objective'])
                for key in ('final_ntp', 'early_ntp', 'offset2', 'total'))
            if not entry['ce_status'] or not entry['finite'] or not entry['labels_equal']:
                report['status'] = 'failed'
            report['models'][name]['depths'][depth] = entry
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
    report_status = None
    if args.device == 'cuda' and args.report is not None and result.wasSuccessful():
        summary = run_depth_report(args.device, args.report)
        report_status = summary['status']
        print(json.dumps({'depth_report': str(args.report), 'status': report_status}, indent=2))
    sys.exit(0 if report_exit_ok(result.wasSuccessful(), args.report is not None, report_status) else 1)
