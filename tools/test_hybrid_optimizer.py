"""Focused native Muon/AdamW partition, state and resume tests."""
import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from data_fixture import synthetic_documents, split_documents, pack
from experiment import seed_all, train
from optimizers import (MuonAdamW, build_optimizer, build_semantic_partition,
                        optimizer_state_inventory)
from splot import DenseConfig, RecurrentConfig, SPLOTLM, training_objective


def config(name):
    return json.loads((Path('configs/experiments') / name).read_text())


def model_from(cfg):
    return SPLOTLM(RecurrentConfig(dense=DenseConfig(**cfg['dense']), **cfg['recurrence']),
                   expert_intermediate=cfg['expert_intermediate'])


def group_of(partition, name):
    for group in ('muon', 'adamw'):
        if any(entry['name'] == name for entry in partition.metadata['groups'][group]['entries']):
            return group
    raise KeyError(name)


def assert_tree_equal(test, a, b, path='root'):
    if torch.is_tensor(a):
        test.assertIsInstance(b, torch.Tensor, path)
        test.assertEqual(a.dtype, b.dtype, path)
        test.assertTrue(torch.equal(a, b), path)
    elif isinstance(a, dict):
        test.assertIsInstance(b, dict, path)
        test.assertEqual(a.keys(), b.keys(), path)
        for key in a:
            assert_tree_equal(test, a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        test.assertIsInstance(b, (list, tuple), path)
        test.assertEqual(len(a), len(b), path)
        for index, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(test, x, y, f'{path}.{index}')
    else:
        test.assertEqual(a, b, path)


class HybridOptimizerTests(unittest.TestCase):
    device = 'cpu'

    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def blocks(self):
        return pack(split_documents(synthetic_documents())['train'], 96)

    def batch(self):
        block = self.blocks()[0]
        x = tuple(torch.tensor([block[k]], device=self.device) for k in
                  ('input_ids', 'document_ids', 'position_ids', 'valid'))
        y = {k: torch.tensor([block['labels'][k]], device=self.device) for k in (1, 2)}
        return x, y

    def test_semantic_partition_smoke_and_nano(self):
        for name, expected, expected_nano in (
            ('splot-smoke-synthetic.json', 179952, False),
            ('splot-nano-synthetic.json', 43055076, True),
        ):
            cfg = config(name)
            model = model_from(cfg)
            partition = build_semantic_partition(model)
            self.assertEqual(partition.metadata['inventory_numel'], expected)
            self.assertEqual(partition.metadata['inventory_numel'],
                             sum(p.numel() for p in model.parameters()))
            self.assertEqual(partition.metadata['groups']['muon']['numel'] +
                             partition.metadata['groups']['adamw']['numel'], expected)
            self.assertEqual(len(partition.muon) + len(partition.adamw),
                             partition.metadata['parameter_count'])
            self.assertEqual(len({id(p) for p in partition.muon + partition.adamw}),
                             len(partition.muon) + len(partition.adamw))
            self.assertTrue(all(p.ndim == 2 for p in partition.muon))
            self.assertEqual(partition.metadata['ns_intermediate_dtype'], 'bfloat16')
            self.assertEqual(len(partition.metadata['plan_sha256']), 64)
            self.assertEqual(sum(1 for e in partition.metadata['groups']['adamw']['entries']
                                 if e['name'] == 'embedding.weight'), 1)
            for expected_group, needle in (
                ('adamw', 'embedding.weight'), ('adamw', 'norm.weight'),
                ('adamw', 'offset_norm.weight'), ('muon', 'offset_projection.weight'),
                ('adamw', 'core.0.attn.conv'), ('adamw', 'core.0.attn.log_A'),
                ('adamw', 'core.0.attn.dt_bias'), ('adamw', 'core.0.attn.a.weight'),
                ('adamw', 'core.0.attn.b.weight'), ('adamw', 'core.0.moe.router.weight'),
                ('muon', 'core.0.attn.q.weight'), ('muon', 'core.0.attn.k.weight'),
                ('muon', 'core.0.attn.v.weight'), ('muon', 'core.0.attn.z.weight'),
                ('muon', 'core.0.attn.o.weight'), ('muon', 'core.3.attn.output_gate.weight'),
                ('muon', 'core.0.moe.experts.0.gate.weight'),
                ('muon', 'core.0.moe.shared.up.weight'),
                ('muon', 'prelude.0.gate.weight'), ('muon', 'coda.0.attn.q.weight'),
            ):
                self.assertEqual(group_of(partition, needle), expected_group, needle)
            if expected_nano:
                self.assertEqual(partition.metadata['groups']['muon']['numel'] > 0, True)
            del model

    def test_native_muon_rectangular_api(self):
        self.assertTrue(hasattr(torch.optim, 'Muon'))
        layer = nn.Linear(7, 13, bias=False).to(self.device)
        opt = torch.optim.Muon(
            layer.parameters(), lr=.003, weight_decay=0, momentum=.95, nesterov=True,
            ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=5, eps=1e-7,
            adjust_lr_fn='match_rms_adamw')
        before = layer.weight.detach().clone()
        layer(torch.randn(2, 7, device=self.device)).sum().backward()
        opt.step()
        self.assertFalse(torch.equal(before, layer.weight))
        self.assertTrue(layer.weight.isfinite().all())
        state = opt.state_dict()['state']
        self.assertTrue(state)
        self.assertTrue(all(value['momentum_buffer'].dtype == torch.float32
                            and value['momentum_buffer'].isfinite().all()
                            for value in state.values()))

    def test_adapter_state_and_unused_parameter(self):
        cfg = config('splot-smoke-synthetic-muon.json')
        torch.manual_seed(42)
        model = model_from(cfg).to(self.device)
        opt, policy = build_optimizer(model, cfg['training'])
        self.assertIsInstance(opt, MuonAdamW)
        self.assertEqual(policy['name'], 'muon')
        self.assertEqual(policy['partition']['inventory_numel'], 179952)
        x, y = self.batch()
        opt.zero_grad(set_to_none=True)
        objective = training_objective(model, x, y, mode='A2', loops=2)
        objective.total.backward()
        opt.step()
        self.assertTrue(all(p.isfinite().all() for p in model.parameters()))
        inventory = optimizer_state_inventory(opt)
        self.assertGreater(inventory['muon']['elements'], 0)
        self.assertGreater(inventory['adamw']['elements'], 0)
        self.assertGreater(inventory['total']['bytes'], 0)

        torch.manual_seed(43)
        model = model_from(cfg).to(self.device)
        opt, _ = build_optimizer(model, cfg['training'])
        x, y = self.batch()
        opt.zero_grad(set_to_none=True)
        objective = training_objective(model, x, y, mode='A2', loops=2)
        objective.total.backward()
        inactive = model.core[0].moe.experts[0].gate.weight
        before = inactive.detach().clone()
        inactive.grad = None
        opt.step()
        self.assertTrue(torch.equal(before, inactive))
        self.assertNotIn(inactive, opt.muon.state)

    def test_resume_exact_and_mode_mismatch(self):
        cfg = config('splot-smoke-synthetic-muon.json')
        blocks = self.blocks()
        identity = {'schema': 1, 'settings': {'optimizer': {'name': 'muon'}},
                    'environment': {'device': self.device}}
        objective = lambda m, x, y: training_objective(m, x, y, mode='A2', loops=2)

        def run(path, steps, resume=False):
            seed_all(42 if not resume else 99)
            model = model_from(cfg).to(self.device)
            opt, _ = build_optimizer(model, cfg['training'])
            history = train(model, opt, blocks, objective, identity, path,
                            steps=steps, batch_size=1, seed=42, device=self.device,
                            resume=resume)
            return model, opt, history

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            a, ao, ah = run(out/'continuous', 4)
            run(out/'split', 2)
            b, bo, bh = run(out/'split', 4, True)
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            assert_tree_equal(self, ao.state_dict(), bo.state_dict())
            ignore = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}
            self.assertEqual([{k: v for k, v in row.items() if k not in ignore} for row in ah],
                             [{k: v for k, v in row.items() if k not in ignore} for row in bh])

            default_model = model_from(cfg).to(self.device)
            default_opt = torch.optim.AdamW(default_model.parameters(), lr=.003, weight_decay=0)
            with self.assertRaises(ValueError):
                ao.load_state_dict(default_opt.state_dict())
            state = copy.deepcopy(ao.state_dict())
            state['partition']['groups']['muon']['entries'][0]['name'] = 'corrupted'
            with self.assertRaises(ValueError):
                ao.load_state_dict(state)

    def test_default_adamw_selection_fp32_and_bf16(self):
        for name in ('splot-smoke-synthetic.json', 'splot-smoke-synthetic-bf16.json'):
            cfg = config(name)
            model = model_from(cfg)
            opt, policy = build_optimizer(model, cfg['training'])
            self.assertIsInstance(opt, torch.optim.AdamW)
            self.assertEqual(policy['name'], 'adamw')
            self.assertIn('state', opt.state_dict())
            self.assertIn('param_groups', opt.state_dict())
            del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    HybridOptimizerTests.device = args.device
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
