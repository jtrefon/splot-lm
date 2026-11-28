import math
import unittest

import torch

from splot import DenseConfig, RecurrentConfig, SPLOTLM
from splot.gdn import GatedDeltaNet, delta_chunk64, delta_scan

DTYPES = (torch.float64, torch.float32)


def normalized_inputs(length, dtype, seed=42):
    generator = torch.Generator(device='cpu').manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=dtype)

    q = randn((length, 6, 64))
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12) / 8
    k = randn((length, 6, 64))
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = 0.1 * randn((length, 6, 64))
    g = torch.empty((length, 6), dtype=dtype).uniform_(-0.1, -0.001, generator=generator)
    beta = torch.empty((length, 6), dtype=dtype).uniform_(0.1, 0.9, generator=generator)
    state = 0.01 * randn((6, 64, 64))
    return tuple(value.requires_grad_(True) for value in (q, k, v, g, beta, state))


def compare_tensor(test, candidate, reference, relative, max_abs_scale):
    delta = (candidate.double() - reference.double())
    rel = delta.norm().item() / max(reference.double().norm().item(), 1e-12)
    max_abs = delta.abs().max().item()
    test.assertLessEqual(rel, relative)
    test.assertLessEqual(max_abs, max_abs_scale * max(1.0, reference.double().abs().max().item()))
    test.assertTrue(bool(torch.isfinite(candidate).all()))


def run_gdn(fn, values):
    q, k, v, g, beta, state = values
    out, final = fn(q, k, v, g, beta, state)
    loss = out.square().mean() + final.square().mean()
    grads = torch.autograd.grad(loss, values)
    return out.detach(), final.detach(), tuple(value.detach() for value in grads)


def tiny_config(algorithm='scan'):
    dense = DenseConfig(vocab_size=32768, width=32, layers=8, q_heads=4,
                        kv_heads=2, head_dim=8, intermediate=64)
    return RecurrentConfig(dense=dense, gdn_algorithm=algorithm)


def tiny_model(algorithm='scan'):
    torch.manual_seed(42)
    return SPLOTLM(tiny_config(algorithm), expert_intermediate=32)


def packed_case(length=96):
    doc0, doc1 = 37, 52
    ids = torch.zeros(1, length, dtype=torch.long)
    docs = torch.full((1, length), -1, dtype=torch.long)
    pos = torch.zeros(1, length, dtype=torch.long)
    valid = torch.zeros(1, length, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    ids[0, :doc0] = torch.randint(8, 32768, (doc0,), generator=generator)
    ids[0, doc0:doc0+doc1] = torch.randint(8, 32768, (doc1,), generator=generator)
    docs[0, :doc0] = 0
    docs[0, doc0:doc0+doc1] = 1
    pos[0, :doc0] = torch.arange(doc0)
    pos[0, doc0:doc0+doc1] = torch.arange(doc1)
    valid[0, :doc0+doc1] = True
    return ids, docs, pos, valid, (0, doc0), (doc0, doc0+doc1)


def model_forward(model, inputs, **kwargs):
    with torch.no_grad():
        return model(*inputs, loops=2, readouts=(1, 2), offset2_depths=(1, 2), **kwargs)


class DeltaChunk64Tests(unittest.TestCase):
    def test_chunk64_matches_scan_and_gradients(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                values = normalized_inputs(65, dtype)
                scan = run_gdn(delta_scan, tuple(value.detach().clone().requires_grad_(True) for value in values))
                chunk = run_gdn(delta_chunk64, tuple(value.detach().clone().requires_grad_(True) for value in values))
                if dtype == torch.float64:
                    out_rel, grad_rel = 1e-9, 1e-9
                    out_abs, grad_abs = 1e-9, 1e-9
                else:
                    out_rel, grad_rel = 1e-4, 5e-4
                    out_abs, grad_abs = 1e-4, 1e-5
                compare_tensor(self, chunk[0], scan[0], out_rel, out_abs)
                compare_tensor(self, chunk[1], scan[1], out_rel, out_abs)
                for candidate, reference in zip(chunk[2], scan[2]):
                    compare_tensor(self, candidate, reference, grad_rel, grad_abs)

    def test_config_default_and_inventory(self):
        default_dense = tiny_config('scan').dense
        self.assertEqual(RecurrentConfig(dense=default_dense).gdn_algorithm, 'scan')
        with self.assertRaises(ValueError):
            RecurrentConfig(dense=default_dense, gdn_algorithm='chunk')
        with self.assertRaises(ValueError):
            GatedDeltaNet(DenseConfig(), algorithm='unknown')
        scan = tiny_model('scan')
        chunk = tiny_model('chunk64')
        self.assertEqual(scan.core[0].attn.algorithm, 'scan')
        self.assertEqual(chunk.core[0].attn.algorithm, 'chunk64')
        self.assertEqual(sum(p.numel() for p in scan.parameters()),
                         sum(p.numel() for p in chunk.parameters()))
        self.assertEqual(set(scan.state_dict()), set(chunk.state_dict()))

    def test_packed_rows_reset_and_padding(self):
        ids, docs, pos, valid, doc0, doc1 = packed_case()
        inputs = (ids, docs, pos, valid)
        scan_model = tiny_model('scan').eval()
        chunk_model = tiny_model('chunk64').eval()
        scan = model_forward(scan_model, inputs)
        chunk = model_forward(chunk_model, inputs)
        for depth in (1, 2):
            compare_tensor(self, chunk.logits[depth], scan.logits[depth], 1e-5, 1e-5)
            compare_tensor(self, chunk.hidden[depth], scan.hidden[depth], 1e-5, 1e-5)
        # Each packed document must equal a fresh standalone run for that document.
        for name, indexes in (('doc0', range(*doc0)), ('doc1', range(*doc1))):
            standalone = tuple(value[:, indexes] for value in inputs)
            a = model_forward(tiny_model('scan').eval(), standalone)
            b = model_forward(tiny_model('chunk64').eval(), standalone)
            compare_tensor(self, b.logits[2], a.logits[2], 1e-5, 1e-5)
            compare_tensor(self, chunk.logits[2][:, indexes], scan.logits[2][:, indexes], 1e-5, 1e-5)
            compare_tensor(self, chunk.logits[2][:, indexes], b.logits[2], 1e-5, 1e-5)

    def test_prefix_cache_and_logical_passes(self):
        ids, docs, pos, valid = packed_case()[0:4]
        ids, docs, pos, valid = ids[:, :17], docs[:, :17], pos[:, :17], valid[:, :17]
        model = tiny_model('chunk64').eval()
        full = model_forward(model, (ids, docs, pos, valid))
        cache = None
        pieces = []
        for i in range(17):
            step = (ids[:, i:i+1], docs[:, i:i+1], pos[:, i:i+1], valid[:, i:i+1])
            out = model_forward(model, step, cache=cache, use_cache=True)
            cache = out.cache
            pieces.append(out.logits[2])
        incremental = torch.cat(pieces, dim=1)
        compare_tensor(self, incremental, full.logits[2], 1e-5, 1e-5)
        gdn_keys = [key for key, _ in cache.states if key[0] == 'C' and key[2] < 3]
        self.assertEqual(len(gdn_keys), 6)
        self.assertEqual(len({id(state) for key, state in cache.states if key[0] == 'C' and key[2] < 3}), 6)
        other = tiny_model('chunk64').eval()
        with self.assertRaises(ValueError):
            model_forward(other, (ids[:, :1], docs[:, :1], pos[:, :1], valid[:, :1]),
                         cache=cache, use_cache=True)


if __name__ == '__main__':
    unittest.main()
