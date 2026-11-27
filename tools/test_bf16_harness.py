"""Focused BF16 autocast and FP32-safeguard tests for the synthetic harness."""
import math
import tempfile
import unittest
from pathlib import Path

import torch

from data_fixture import synthetic_documents, split_documents, pack
from experiment import batch_indices, evaluate, resolve_precision, seed_all, train
from splot import SPLOTLM, RecurrentConfig, ntp_loss, training_objective
from splot.dense import RMSNorm, rope


def _autocast(device='cpu'):
    return torch.autocast(device_type=device, dtype=torch.bfloat16)


class BF16HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = pack(split_documents(synthetic_documents())['train'], 96)
        block = cls.blocks[0]
        cls.x = tuple(torch.tensor([block[k]]) for k in
                      ('input_ids', 'document_ids', 'position_ids', 'valid'))
        cls.labels = {k: torch.tensor([block['labels'][k]]) for k in (1, 2)}

    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def require_cpu_bf16(self):
        try:
            with _autocast():
                probe = torch.nn.functional.linear(torch.ones(2, 4), torch.ones(4, 4))
            if probe.dtype != torch.bfloat16:
                self.skipTest(f'CPU BF16 autocast returned {probe.dtype}')
        except Exception as exc:
            self.skipTest(f'CPU BF16 autocast unavailable: {exc}')

    def test_gqa_gdn_router_and_reduction_dtypes(self):
        self.require_cpu_bf16()
        torch.manual_seed(42)
        model = SPLOTLM(RecurrentConfig()).float().eval()
        seen = {}
        for name, module in (('prelude_q', model.prelude[0].attn.q),
                             ('core_gqa_q', model.core[3].attn.q)):
            module.register_forward_hook(
                lambda m, i, o, key=name: seen.__setitem__(key, o.dtype))
        captured, softmax_dtypes = [], []
        orig_einsum, orig_softmax = torch.einsum, torch.Tensor.softmax

        def einsum(equation, *operands, **kwargs):
            if equation in ('hgtd,hsd->hgts', 'hgts,hsd->hgtd'):
                captured.append((equation, tuple(x.dtype for x in operands if isinstance(x, torch.Tensor))))
            return orig_einsum(equation, *operands, **kwargs)

        def softmax(self, *args, **kwargs):
            softmax_dtypes.append(self.dtype)
            return orig_softmax(self, *args, **kwargs)

        torch.einsum = einsum
        torch.Tensor.softmax = softmax
        try:
            with torch.no_grad(), _autocast():
                out = model(*self.x, loops=2, readouts=(1, 2), offset2_depths=(1, 2), use_cache=True)
        finally:
            torch.einsum = orig_einsum
            torch.Tensor.softmax = orig_softmax
        self.assertTrue(seen)
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in seen.values()), seen)
        self.assertEqual(out.logits[2].dtype, torch.bfloat16)
        self.assertTrue(out.logits[2].isfinite().all())
        self.assertTrue(out.routing)
        self.assertTrue(all(r.probabilities.dtype == torch.float32 for _, r in out.routing))
        gdn = [(key, rows) for key, rows in out.cache.states if key[0] == 'C' and key[2] < 3]
        self.assertTrue(gdn)
        for _, rows in gdn:
            for matrix, history in rows:
                self.assertEqual(matrix.dtype, torch.float32)
                self.assertEqual(history.dtype, torch.bfloat16)
        score = [dtypes for equation, dtypes in captured if equation == 'hgtd,hsd->hgts']
        value = [dtypes for equation, dtypes in captured if equation == 'hgts,hsd->hgtd']
        self.assertTrue(score)
        self.assertTrue(value)
        self.assertTrue(all(all(dtype == torch.float32 for dtype in dtypes) for dtypes in score))
        self.assertTrue(all(all(dtype == torch.float32 for dtype in dtypes) for dtypes in value))
        self.assertTrue(softmax_dtypes)
        self.assertTrue(all(dtype == torch.float32 for dtype in softmax_dtypes), softmax_dtypes)

    def test_fp32_master_weights_adam_and_evaluation(self):
        self.require_cpu_bf16()
        torch.manual_seed(42)
        model = SPLOTLM(RecurrentConfig()).float().train()
        opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
        with _autocast():
            objective = training_objective(model, self.x, self.labels, mode='A2', loops=2)
        self.assertEqual(objective.total.dtype, torch.float32)
        for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance'):
            self.assertEqual(getattr(objective, name).dtype, torch.float32)
        torch.testing.assert_close(
            objective.total,
            objective.final_ntp + .25 * objective.early_ntp + .25 * objective.offset2 + .01 * objective.balance,
            rtol=1e-6, atol=1e-6)
        objective.total.backward()
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(p.grad is None or (p.grad.dtype == torch.float32 and p.grad.isfinite().all())
                            for p in model.parameters()))
        opt.step()
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        state = [s for s in opt.state.values() if s]
        self.assertTrue(state)
        for entry in state:
            self.assertEqual(entry['exp_avg'].dtype, torch.float32)
            self.assertEqual(entry['exp_avg_sq'].dtype, torch.float32)
            self.assertTrue(entry['exp_avg'].isfinite().all())
            self.assertTrue(entry['exp_avg_sq'].isfinite().all())
            self.assertEqual(float(entry['step']), 1.0)
        with _autocast():
            val = evaluate(model, self.blocks[:1], lambda m, x: m(*x, loops=2).logits[2],
                           batch_size=1, device='cpu', autocast_dtype=torch.bfloat16)
        self.assertGreater(val['ntp_labels'], 0)
        self.assertTrue(math.isfinite(val['mean_nll']) and math.isfinite(val['perplexity']))

    def test_rmsnorm_rope_and_ce_are_fp32(self):
        self.require_cpu_bf16()
        torch.manual_seed(1)
        x = torch.randn(2, 3, 16).to(torch.bfloat16)
        norm = RMSNorm(16, 1e-5)
        means, roots = [], []
        orig_mean, orig_rsqrt = torch.Tensor.mean, torch.rsqrt

        def mean(self, *args, **kwargs):
            means.append(self.dtype)
            return orig_mean(self, *args, **kwargs)

        def rsqrt(self, *args, **kwargs):
            roots.append(self.dtype)
            return orig_rsqrt(self, *args, **kwargs)

        torch.Tensor.mean = mean
        torch.rsqrt = rsqrt
        try:
            with _autocast():
                y = norm(x)
        finally:
            torch.Tensor.mean = orig_mean
            torch.rsqrt = orig_rsqrt
        reference = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)
                     * norm.weight).to(torch.bfloat16)
        self.assertTrue(torch.equal(y, reference))
        self.assertTrue(means and all(dtype == torch.float32 for dtype in means))
        self.assertTrue(roots and all(dtype == torch.float32 for dtype in roots))

        z = torch.randn(2, 5, 8).to(torch.bfloat16)
        positions = torch.arange(5)
        cosines, sines = [], []
        orig_cos, orig_sin = torch.Tensor.cos, torch.Tensor.sin

        def cos(self, *args, **kwargs):
            cosines.append(self.dtype)
            return orig_cos(self, *args, **kwargs)

        def sin(self, *args, **kwargs):
            sines.append(self.dtype)
            return orig_sin(self, *args, **kwargs)

        torch.Tensor.cos = cos
        torch.Tensor.sin = sin
        try:
            with _autocast():
                rotated = rope(z, positions, 1e6)
        finally:
            torch.Tensor.cos = orig_cos
            torch.Tensor.sin = orig_sin
        inv = 1e6 ** (-torch.arange(0, z.shape[-1], 2, dtype=torch.float32) / z.shape[-1])
        angle = positions.to(torch.float32)[:, None] * inv
        a, b = z[..., 0::2].float(), z[..., 1::2].float()
        expected = torch.stack((a * angle.cos() - b * angle.sin(),
                                a * angle.sin() + b * angle.cos()), -1).flatten(-2).to(torch.bfloat16)
        self.assertTrue(torch.equal(rotated, expected))
        self.assertTrue(cosines and all(dtype == torch.float32 for dtype in cosines))
        self.assertTrue(sines and all(dtype == torch.float32 for dtype in sines))

        logits = torch.randn(1, 5, 32).to(torch.bfloat16)
        labels = torch.randint(0, 32, (1, 5))
        with _autocast():
            loss = ntp_loss(logits, labels)
        expected_loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, 32), labels.reshape(-1), reduction='sum') / labels.numel()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.equal(loss, expected_loss))

    def test_bf16_resume_exact_and_cross_precision_rejected(self):
        self.require_cpu_bf16()
        identity_bf16 = {'schema': 1, 'settings': {'precision': 'bf16'}, 'environment': {'device': 'cpu'}}
        identity_fp32 = {'schema': 1, 'settings': {'precision': 'fp32'}, 'environment': {'device': 'cpu'}}
        objective = lambda m, x, y: training_objective(m, x, y, mode='A2', loops=2)

        def run(path, steps, resume=False, identity=None, autocast_dtype=torch.bfloat16):
            seed_all(42 if not resume else 99)
            model = SPLOTLM(RecurrentConfig()).float()
            opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
            history = train(model, opt, self.blocks, objective, identity or identity_bf16, out / path,
                            steps=steps, batch_size=1, seed=42, device='cpu', resume=resume,
                            autocast_dtype=autocast_dtype)
            return model, opt, history

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            a, ao, ah = run('continuous', 2)
            run('split', 1)
            b, bo, bh = run('split', 2, resume=True)
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            for x, y in zip(ao.state.values(), bo.state.values()):
                for key in x:
                    torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
            ignore = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}
            self.assertEqual([{k: v for k, v in row.items() if k not in ignore} for row in ah],
                             [{k: v for k, v in row.items() if k not in ignore} for row in bh])
            with self.assertRaises(ValueError):
                run('split', 2, resume=True, identity=identity_fp32, autocast_dtype=None)

    def test_same_seed_batch_and_precision_resolution(self):
        self.require_cpu_bf16()
        seed_all(42)
        a = SPLOTLM(RecurrentConfig())
        seed_all(42)
        b = SPLOTLM(RecurrentConfig())
        for x, y in zip(a.state_dict().values(), b.state_dict().values()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(batch_indices(len(self.blocks), 1, 0, 42),
                         batch_indices(len(self.blocks), 1, 0, 42))
        fp32 = resolve_precision('cpu', 'fp32')
        self.assertIsNone(fp32['autocast_dtype'])
        bf16 = resolve_precision('cpu', 'bf16')
        self.assertEqual(bf16['autocast_dtype'], torch.bfloat16)
        self.assertEqual(bf16['probe_dtype'], 'bfloat16')
        with self.assertRaises(ValueError):
            resolve_precision('cpu', 'fp16')


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main(verbosity=2)
