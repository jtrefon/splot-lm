"""Independent meta-tensor parameter inventory and accounting invariants."""
import unittest
import torch
from torch import nn
from accounting import Shape, account


def inventory(s, mtp):
    d, q, kv, g = s.d, s.q_heads*s.head, s.kv_heads*s.head, s.gdn_heads*s.head
    tensors = nn.ParameterList()
    def add(*shape):
        tensors.append(nn.Parameter(torch.empty(*shape, device='meta')))
    def linear(i, o):
        add(o, i)
    def ffn(m):
        linear(d, m); linear(d, m); linear(m, d)
    def gqa(gated):
        linear(d, q); linear(d, kv); linear(d, kv); linear(q, d)
        if gated: linear(d, q)
    def gdn():
        for _ in range(4): linear(d, g)
        linear(g, d)
        linear(d, s.gdn_heads); linear(d, s.gdn_heads)
        add(3*g, 1, s.conv)
        add(s.gdn_heads); add(s.gdn_heads); add(s.head)
    add(s.vocab, d)
    for _ in range(4):
        add(d); add(d); gqa(False); ffn(s.dense)
    for b in range(4):
        add(d); add(d)
        if b < 3: gdn()
        else: gqa(True)
        linear(d, s.experts)
        for _ in range(s.experts+1): ffn(s.expert)
    add(d)
    if mtp: linear(d, d); add(d)
    return sum(p.numel() for p in tensors)


class Accounting(unittest.TestCase):
    def test_meta_inventory(self):
        for s in (Shape(), Shape(d=384, q_heads=6, kv_heads=2, gdn_heads=6, dense=1024, expert=512)):
            for mtp in (False, True):
                self.assertEqual(inventory(s, mtp), account(s, mtp=mtp)['physical'])

    def test_mini_counts(self):
        a = account()
        self.assertEqual(a['base'], 146397192)
        self.assertEqual(a['physical'], 146987784)
        self.assertEqual(a['auxiliary'], 590592)

    def test_depth_and_active_bounds(self):
        a, b = account(loops=1), account(loops=4)
        self.assertEqual(a['physical'], b['physical'])
        self.assertEqual(a['target_unique_active_min'], a['target_unique_active_max'])
        self.assertEqual(b['target_unique_active_max'], b['base'])
        self.assertEqual(b['logical_blocks'], 20)
        self.assertGreater(b['decode_macs'], a['decode_macs'])

    def test_readout_cost(self):
        a, b = account(loops=4), account(loops=4, readouts=4, offsets=2)
        self.assertEqual(a['physical'], b['physical'])
        self.assertEqual(b['cache_bytes']-a['cache_bytes'], 12*1024**2)
        self.assertEqual(b['logical_blocks'], 26)
        self.assertGreater(b['decode_macs'], a['decode_macs'])

    def test_prefill_sum(self):
        a = account(loops=2, context=7, readouts=2, offsets=2)
        self.assertEqual(a['prefill_all_logits_macs'], sum(account(loops=2, context=t, readouts=2, offsets=2)['decode_macs'] for t in range(1, 8)))

    def test_invalid_config(self):
        for kw in ({'loops': 0}, {'readouts': 3}, {'context': 0}, {'mtp': False, 'offsets': 1}):
            with self.assertRaises(ValueError): account(**kw)
        with self.assertRaises(ValueError): account(Shape(kv_heads=5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
