import unittest
import torch
from splot import DenseConfig, DenseLM, ntp_loss


def batch():
    x = torch.tensor([[1, 10, 11, 2, 1, 12, 2, 0], [1, 20, 21, 22, 2, 0, 0, 0]])
    docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1], [0, 0, 0, 0, 0, -1, -1, -1]])
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0], [0, 1, 2, 3, 4, 0, 0, 0]])
    return x, docs, pos, docs.ge(0)


class DenseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DenseLM(DenseConfig()).double().eval()
        self.data = batch()

    def equal(self, a, b):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_count_and_tied_output(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), self.model.config.parameter_count())
        self.assertEqual(sum(p is self.model.embedding.weight for p in self.model.parameters()), 1)
        with torch.no_grad():
            self.model.embedding.weight.zero_()
            y, _ = self.model(*self.data)
        self.equal(y, torch.zeros_like(y))

    def test_future_and_document_isolation(self):
        x, doc, pos, valid = self.data
        y, _ = self.model(*self.data)
        altered = x.clone(); altered[0, 2:4] = 50
        z, _ = self.model(altered, doc, pos, valid)
        self.equal(y[0, :2], z[0, :2]); self.equal(y[0, 4:], z[0, 4:]); self.equal(y[1], z[1])
        self.equal(y[~valid], torch.zeros_like(y[~valid]))

    def test_cached_chunks_and_document_reset(self):
        with torch.no_grad():
            full, _ = self.model(*self.data)
            for boundaries in ((0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 3, 6, 8)):
                cache, outputs = None, []
                for a, b in zip(boundaries, boundaries[1:]):
                    y, cache = self.model(*(t[:, a:b] for t in self.data), cache=cache, use_cache=True)
                    outputs.append(y)
                self.equal(full, torch.cat(outputs, 1))
                self.assertEqual([r.length for r in cache.rows], [3, 5])
                for layer in cache.layers:
                    self.assertEqual(layer[0][0].shape, (2, 3, 8))
                    self.assertEqual(layer[1][0].shape, (2, 5, 8))

    def test_cache_branch_and_padding_noop(self):
        with torch.no_grad():
            _, cache = self.model(*(t[:, :3] for t in self.data), use_cache=True)
            saved = cache.layers[0][0][0].clone()
            x = self.data[0][:, 3:4].clone(); x[0, 0] = 60
            self.model(x, *(t[:, 3:4] for t in self.data[1:]), cache=cache, use_cache=True)
            self.equal(saved, cache.layers[0][0][0])
            pad = torch.zeros((2, 1), dtype=torch.long)
            y, same = self.model(pad, pad-1, pad, pad.bool(), cache=cache, use_cache=True)
            self.equal(y, torch.zeros_like(y)); self.assertEqual(same.rows, cache.rows)
            self.assertIs(same.layers[0][0][0], cache.layers[0][0][0])
            expected, _ = self.model(*(t[:, :4] for t in self.data))
            actual, _ = self.model(*(t[:, 3:4] for t in self.data), cache=cache, use_cache=True)
            self.equal(expected[:, -1:], actual)

    def test_loss_gradients_and_empty(self):
        logits, _ = self.model(*self.data)
        labels = torch.full(self.data[0].shape, -100)
        labels[:, :-1] = torch.where(self.data[1][:, :-1].eq(self.data[1][:, 1:]) & self.data[3][:, :-1], self.data[0][:, 1:], -100)
        loss = ntp_loss(logits, labels)
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in self.model.parameters()))
        self.assertGreater(self.model.embedding.weight.grad.norm().item(), 0)
        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(*self.data)
        zero = ntp_loss(logits, torch.full_like(labels, -100)); zero.backward()
        self.assertEqual(zero.item(), 0)
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero().item() == 0 for p in self.model.parameters()))

    def test_invalid_metadata_and_cache_contract(self):
        x, doc, pos, valid = self.data
        with self.assertRaises(ValueError): self.model(x, doc, pos+1, valid)
        with self.assertRaises(ValueError): self.model(*self.data, use_cache=True)
        with torch.no_grad():
            _, c = self.model(*self.data, use_cache=True)
            other = DenseLM(DenseConfig()).double().eval()
            with self.assertRaises(ValueError): other(*self.data, cache=c, use_cache=True)
        with self.assertRaises(ValueError): DenseConfig(kv_heads=3)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
