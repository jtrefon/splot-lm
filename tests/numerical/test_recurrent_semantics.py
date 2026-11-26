"""CPU float64 toy oracle; NOT a GQA/GDN/LM implementation."""
import copy
import unittest
import torch
from torch import nn


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(3, 3, bias=False, dtype=torch.float64)

    def step(self, x, state):
        s = 0.6 * state + self.w(x)
        return x + 0.2 * torch.tanh(s), s

    def scan(self, xs):
        s = torch.zeros_like(xs[0])
        ys = []
        for x in xs:
            y, s = self.step(x, s)
            ys.append(y)
        return torch.stack(ys)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelude = Mixer()
        self.core = nn.ModuleList([Mixer(), Mixer()])
        self.coda = Mixer()

    def full(self, xs, depth, untied=None):
        z = self.prelude.scan(xs)
        h, outputs = z, []
        for r in range(1, depth + 1):
            if r > 1:
                h = (1 - 0.1 / r) * h + (0.1 / r) * z
            for b, block in enumerate(self.core if untied is None else untied[r - 1]):
                h = block.scan(h)
            outputs.append(self.coda.scan(h))
        return torch.stack(outputs)

    def token(self, x, depth, cache):
        def apply(block, key, value):
            y, state = block.step(value, cache.get(key, torch.zeros_like(value)))
            cache[key] = state
            return y

        z = apply(self.prelude, ('P', 0), x)
        h, outputs = z, []
        for r in range(1, depth + 1):
            if r > 1:
                h = (1 - 0.1 / r) * h + (0.1 / r) * z
            for b, block in enumerate(self.core):
                h = apply(block, ('R', r, b), h)
            outputs.append(apply(self.coda, ('C', r), h))
        return torch.stack(outputs)


class Semantics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Toy()
        self.x = torch.randn(5, 3, dtype=torch.float64)

    def equal(self, a, b):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_stage_major_vs_token_major(self):
        for depth in (1, 2, 4):
            cache = {}
            actual = torch.stack([self.model.token(x, depth, cache) for x in self.x], dim=1)
            self.equal(actual, self.model.full(self.x, depth))
            self.assertEqual(len(cache), 1 + 3 * depth)

    def test_prefix_and_depth_invariance(self):
        full = self.model.full(self.x, 4)
        self.equal(full[:, :3], self.model.full(self.x[:3], 4))
        self.equal(full[:2], self.model.full(self.x, 2))

    def test_rollback(self):
        with torch.no_grad():
            cache = {}
            for x in self.x[:2]:
                self.model.token(x, 4, cache)
            snapshot = {k: v.clone() for k, v in cache.items()}
            self.model.token(self.x[2] + 9, 4, cache)
            self.model.token(self.x[3] - 5, 4, cache)
            replay = dict(snapshot)
            y = self.model.token(self.x[2], 4, replay)
            self.equal(y, self.model.full(self.x[:3], 4)[:, -1])

    def test_document_reset(self):
        cache = {}
        for x in self.x[:2]:
            self.model.token(x, 4, cache)
        cache.clear()
        y = torch.stack([self.model.token(x, 4, cache) for x in self.x[2:]], dim=1)
        self.equal(y, self.model.full(self.x[2:], 4))

    def test_shared_gradient_equals_sum_of_untied_copies(self):
        model = self.model
        untied = [copy.deepcopy(model.core) for _ in range(4)]
        model.full(self.x, 4).square().mean().backward()
        shared = [b.w.weight.grad.clone() for b in model.core]
        model.zero_grad(set_to_none=True)
        model.full(self.x, 4, untied=untied).square().mean().backward()
        for b in range(2):
            self.equal(shared[b], sum(layers[b].w.weight.grad for layers in untied))


if __name__ == '__main__':
    unittest.main(verbosity=2)
