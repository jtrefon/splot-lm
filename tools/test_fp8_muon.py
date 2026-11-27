"""Focused FP8 + native Muon/AdamW compatibility, audit and resume tests."""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from data_fixture import synthetic_documents, split_documents, pack
from experiment import (annotate_fp8_policy, assert_post_update_finite, collate,
                        seed_all, train)
from optimizers import (MuonAdamW, assert_optimizer_state_finite, build_optimizer,
                        build_semantic_partition, optimizer_state_inventory)
from splot import DenseConfig, RecurrentConfig, SPLOTLM, training_objective
from splot.fp8 import (FP8Linear, aggregate_fp8_diagnostics,
                       convert_selected_linears, fp8_diagnostics, reset_fp8_audit)


EXISTING_CONFIG_HASHES = {
    'splot-smoke-synthetic.json': 'd3a1b3e2eca2bb95d817962fbce54ee35943a0bada06d7bb82a1d0ea352369f6',
    'splot-smoke-synthetic-bf16.json': 'fdd142b04686d80ca7308847b2a270b2da98463ca1d50182f2a9da84b7bd0360',
    'splot-smoke-synthetic-fp8.json': '615b617cb9f5b7ccc30a4b16e73aec4d6bfa91ef9118c148edf1645c4572bc39',
    'splot-smoke-synthetic-muon.json': 'd0b4838495e0b04a45ab0be414da272430a0fb60ac71aa7d344355ff10b81479',
    'splot-nano-synthetic.json': '18ae9e3d7fc2cfc427ceec91de33526f51ccb2dc153ab16f8050ee2a061bad2b',
    'splot-nano-synthetic-bf16.json': '5be8fc61fe3cc3af0a2a06e2bbe1c2c16cdc97b569c0dfd22fc890647686a343',
    'splot-nano-synthetic-fp8.json': '5dff101067902f737fb81588095e86622edbaa1c57fb9734ecc0eb1582916522',
    'splot-nano-synthetic-muon.json': '4c0630dd583fb541580a19a02c7948d29a302a556399fea85d17f27dcabf32f2',
}
NEW_CONFIGS = {
    'splot-smoke-synthetic-fp8-muon.json': ('splot-smoke-synthetic-fp8.json',
                                            'splot-smoke-synthetic-muon.json'),
    'splot-nano-synthetic-fp8-muon.json': ('splot-nano-synthetic-fp8.json',
                                           'splot-nano-synthetic-muon.json'),
    'splot-nano-synthetic-bf16-muon.json': ('splot-nano-synthetic-bf16.json',
                                            'splot-nano-synthetic-muon.json'),
}


def config(name):
    return json.loads((Path('configs/experiments') / name).read_text())


def model_from(cfg):
    return SPLOTLM(RecurrentConfig(dense=DenseConfig(**cfg['dense']), **cfg['recurrence']),
                   expert_intermediate=cfg['expert_intermediate'])


def tiny_blocks():
    return [{'input_ids': [i, 1, 0], 'document_ids': [0, 0, -1],
             'position_ids': [0, 1, 0], 'valid': [True, True, False],
             'labels': {1: [1, -100, -100], 2: [-100, -100, -100]}}
            for i in range(2)]


def tiny_model():
    return nn.Sequential(nn.Embedding(5, 4), nn.Linear(4, 5))


def tiny_objective(model, inputs, labels):
    loss = torch.nn.functional.cross_entropy(
        model(inputs[0]).flatten(0, 1), labels[1].flatten(), ignore_index=-100)
    zero = loss.new_zeros(())
    return SimpleNamespace(total=loss, final_ntp=loss, early_ntp=zero,
                           offset2=zero, balance=zero)


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


class FP8MuonCPUTests(unittest.TestCase):
    device = 'cpu'

    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_existing_configs_unchanged_and_new_configs_derived(self):
        for name, expected in EXISTING_CONFIG_HASHES.items():
            actual = hashlib.sha256((Path('configs/experiments') / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)
        for name, (base_name, muon_name) in NEW_CONFIGS.items():
            new = config(name)
            base = config(base_name)
            muon = config(muon_name)
            optimizer = new['training'].pop('optimizer')
            self.assertEqual(new, base, name)
            self.assertEqual(optimizer, muon['training']['optimizer'], name)

    def test_policy_partition_and_metadata(self):
        cfg = config('splot-smoke-synthetic-fp8-muon.json')
        torch.manual_seed(42)
        model = model_from(cfg)
        before_ids = {id(p) for p in model.parameters()}
        before_keys = set(model.state_dict())
        before_partition = build_semantic_partition(model)
        policy = convert_selected_linears(model)
        opt, optimizer_policy = build_optimizer(model, cfg['training'])
        annotate_fp8_policy(policy, optimizer_policy)
        self.assertIsInstance(opt, MuonAdamW)
        self.assertEqual(optimizer_policy['name'], 'muon')
        self.assertEqual(policy['selected_count'], 156)
        self.assertEqual(policy['excluded_count'], 11)
        self.assertEqual(policy['policy_version'], 1)
        self.assertEqual(policy['audit_version'], 1)
        self.assertEqual(policy['optimizer_scope'], 'adamw-or-muon-hybrid')
        self.assertEqual(policy['optimizer'], 'muon')
        self.assertIs(policy['adamw_only'], False)
        self.assertEqual(before_ids, {id(p) for p in model.parameters()})
        self.assertEqual(before_keys, set(model.state_dict()))
        self.assertEqual(before_partition.metadata,
                         build_semantic_partition(model).metadata)
        self.assertEqual(sum(isinstance(m, FP8Linear) for m in model.modules()), 156)
        self.assertEqual(opt.partition.metadata['inventory_numel'], 179952)
        self.assertEqual(opt.partition.metadata['groups']['muon']['numel'] +
                         opt.partition.metadata['groups']['adamw']['numel'], 179952)
        self.assertEqual(len({id(p) for p in opt.partition.muon + opt.partition.adamw}),
                         len(opt.partition.muon) + len(opt.partition.adamw))
        self.assertEqual(opt.partition.metadata['ns_intermediate_dtype'], 'bfloat16')
        self.assertEqual(sum(1 for e in opt.partition.metadata['groups']['adamw']['entries']
                             if e['name'] == 'embedding.weight'), 1)
        self.assertEqual(opt.settings['muon']['lr'], .003)
        self.assertEqual(opt.settings['adamw']['lr'], .003)
        inventory = optimizer_state_inventory(opt)
        self.assertEqual(inventory['adapter_version'], 1)
        self.assertEqual(inventory['total']['elements'], 0)
        self.assertTrue(inventory['total']['finite'])

    def test_malformed_optimizer_selectors(self):
        cfg = config('splot-smoke-synthetic.json')
        model = model_from(cfg)
        for spec in (None, {'name': None}, {'name': 'bogus'}, [],
                     {'name': 'adamw', 'muon': {}},
                     {'name': 'adamw', 'muon': None},
                     {'name': 'muon', 'adamw': None},
                     {'name': 'muon', 'unknown': 1}):
            training = dict(cfg['training'], optimizer=spec)
            with self.assertRaises(ValueError, msg=repr(spec)):
                build_optimizer(model, training)
        opt, policy = build_optimizer(model, cfg['training'])
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(policy['source'], 'default')

    def test_audit_reset_serialization_and_aggregation(self):
        cfg = config('splot-smoke-synthetic-fp8.json')
        model = model_from(cfg)
        convert_selected_linears(model)
        module = next(m for m in model.modules() if isinstance(m, FP8Linear))
        module.forward_calls = 3
        module.backward_calls = 2
        module.padded_forward_rows = 5
        module.padded_backward_rows = 4
        module.audit_native_forward_calls = 3
        module.audit_native_backward_calls = 2
        module.audit_wgrad_fp32_calls = 2
        module.audit_input_nonzero = torch.tensor(4)
        module.audit_input_underflow = torch.tensor(1)
        module.audit_input_saturation = torch.tensor(0)
        module.audit_weight_nonzero = torch.tensor(8)
        module.audit_weight_underflow = torch.tensor(2)
        module.audit_weight_saturation = torch.tensor(0)
        module.audit_grad_nonzero = torch.tensor(6)
        module.audit_grad_underflow = torch.tensor(0)
        module.audit_grad_saturation = torch.tensor(0)
        module.audit_nonfinite = torch.tensor(False)
        module.last_forward_padded_m = 16
        diag = fp8_diagnostics(model)
        self.assertEqual(diag['forward_calls'], 3)
        self.assertEqual(diag['quantization']['input_underflow'], 1)
        self.assertEqual(diag['quantization']['input_underflow_fraction'], .25)
        json.dumps(diag, sort_keys=True, allow_nan=False)
        aggregate = aggregate_fp8_diagnostics([diag, diag])
        self.assertEqual(aggregate['steps'], 2)
        self.assertEqual(aggregate['forward_calls'], 6)
        self.assertEqual(aggregate['quantization']['input_underflow'], 2)
        json.dumps(aggregate, sort_keys=True, allow_nan=False)
        reset_fp8_audit(model)
        cleared = fp8_diagnostics(model)
        self.assertEqual(cleared['forward_calls'], 0)
        self.assertEqual(cleared['quantization']['input_underflow'], 0)
        self.assertFalse(cleared['quantization']['nonfinite'])
        self.assertEqual(cleared['last_forward_padded_m'], [])

    def test_adapter_state_roundtrip_and_cross_mode(self):
        cfg = config('splot-smoke-synthetic-fp8-muon.json')
        torch.manual_seed(42)
        model = model_from(cfg)
        convert_selected_linears(model)
        opt, policy = build_optimizer(model, cfg['training'])
        state = copy.deepcopy(opt.state_dict())
        self.assertEqual(set(state), {'adapter_version', 'partition', 'settings', 'muon', 'adamw'})
        self.assertEqual(state['adapter_version'], 1)
        self.assertEqual(state['settings'], opt.settings)
        self.assertEqual(state['partition'], opt.partition.metadata)
        model2 = model_from(cfg)
        convert_selected_linears(model2)
        opt2, _ = build_optimizer(model2, cfg['training'])
        opt2.load_state_dict(state)
        self.assertEqual(opt2.state_dict()['partition'], state['partition'])
        plain = torch.optim.AdamW(model2.parameters(), lr=.003, weight_decay=0)
        with self.assertRaises(ValueError):
            opt2.load_state_dict(plain.state_dict())

    def test_post_update_guard_preserves_prior_checkpoint(self):
        class CorruptAdamW(torch.optim.AdamW):
            def __init__(self, params, model, target):
                super().__init__(params, lr=.01)
                self.model = model
                self.target = target

            def step(self, closure=None):
                result = super().step(closure)
                with torch.no_grad():
                    if self.target == 'weight':
                        self.model[1].weight[0, 0] = float('nan')
                    else:
                        for entry in self.state.values():
                            for value in entry.values():
                                if torch.is_tensor(value) and value.is_floating_point():
                                    value.view(-1)[0] = float('nan')
                                    return result
                return result

        def run(path, steps, *, resume=False, target=None):
            seed_all(42 if not resume else 99)
            model = tiny_model()
            if target is None:
                opt = torch.optim.AdamW(model.parameters(), lr=.01)
            else:
                opt = CorruptAdamW(model.parameters(), model, target)
            history = train(model, opt, tiny_blocks(), tiny_objective, {'id': 7}, path,
                            steps=steps, batch_size=1, seed=42, device='cpu',
                            resume=resume, post_update_check=True)
            return model, opt, history

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for target in ('weight', 'state'):
                out = root / target
                run(out, 1)
                checkpoint = out / 'checkpoint.pt'
                before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                with self.assertRaises(FloatingPointError):
                    run(out, 2, resume=True, target=target)
                after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                self.assertEqual(before, after)
                self.assertEqual(len(torch.load(checkpoint, map_location='cpu',
                                                weights_only=True)['history']), 1)
                self.assertFalse((out / 'run.json').exists())

    def test_identity_mismatch_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_all(42)
            model = tiny_model()
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            train(model, opt, tiny_blocks(), tiny_objective, {'id': 1}, root / 'run',
                  steps=1, batch_size=1, seed=42, device='cpu')
            seed_all(99)
            model = tiny_model()
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            with self.assertRaises(ValueError):
                train(model, opt, tiny_blocks(), tiny_objective, {'id': 2}, root / 'run',
                      steps=2, batch_size=1, seed=42, device='cpu', resume=True)


class FP8MuonCUDATests(unittest.TestCase):
    device = 'cuda'

    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def model_and_opt(self, name='splot-smoke-synthetic-fp8-muon.json', seed=42):
        cfg = config(name)
        seed_all(seed)
        model = model_from(cfg).to(self.device)
        policy = convert_selected_linears(model)
        opt, optimizer_policy = build_optimizer(model, cfg['training'])
        annotate_fp8_policy(policy, optimizer_policy)
        return cfg, model, opt, policy, optimizer_policy

    def batch(self, index=0):
        blocks = pack(split_documents(synthetic_documents())['train'], 96)
        return collate(blocks, [index], self.device)

    def test_fp8_muon_step_both_states_and_native_dispatch(self):
        cfg, model, opt, policy, optimizer_policy = self.model_and_opt()
        self.assertIsInstance(opt, MuonAdamW)
        self.assertEqual(policy['selected_count'], 156)
        self.assertEqual(policy['adamw_only'], False)
        blocks = pack(split_documents(synthetic_documents())['train'], 96)
        with tempfile.TemporaryDirectory() as tmp:
            history = train(model, opt, blocks, lambda m, x, y: training_objective(
                m, x, y, mode='A2', loops=2), {'schema': 1, 'tag': 'fp8-muon'},
                Path(tmp) / 'step', steps=1, batch_size=1, seed=42,
                device=self.device, autocast_dtype=torch.bfloat16,
                post_update_check=True)
        self.assertEqual(len(history), 1)
        audit = history[0]['audit']
        self.assertGreater(audit['native_forward_calls'], 0)
        self.assertGreater(audit['native_backward_calls'], 0)
        self.assertGreater(audit['wgrad_fp32_calls'], 0)
        self.assertTrue(audit['scales_resident_cuda'])
        self.assertFalse(audit['quantization']['nonfinite'])
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(opt.muon.state)
        self.assertTrue(opt.adamw.state)
        for state in (opt.muon.state, opt.adamw.state):
            for entry in state.values():
                for value in entry.values():
                    if torch.is_tensor(value):
                        self.assertEqual(value.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(value).all())
        inventory = optimizer_state_inventory(opt)
        self.assertGreater(inventory['muon']['bytes'], 0)
        self.assertGreater(inventory['adamw']['bytes'], 0)
        self.assertEqual(inventory['total']['dtypes'], ['float32'])
        self.assertTrue(inventory['total']['finite'])
        assert_optimizer_state_finite(opt)
        assert_post_update_finite(model, opt)

    def test_inactive_expert_is_not_updated(self):
        _, model, opt, _, _ = self.model_and_opt()
        x, y = self.batch(0)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            objective = training_objective(model, x, y, mode='A2', loops=2)
        objective.total.backward()
        candidates = [(name, p) for name, p in model.named_parameters()
                      if '.experts.' in name and name.endswith('.gate.weight')
                      and p.grad is None]
        if not candidates:
            name, inactive = next((name, p) for name, p in model.named_parameters()
                                  if '.experts.' in name and name.endswith('.gate.weight'))
            inactive.grad = None
        else:
            _, inactive = candidates[0]
        before = inactive.detach().clone()
        opt.step()
        assert_post_update_finite(model, opt)
        self.assertTrue(torch.equal(before, inactive))
        self.assertNotIn(inactive, opt.muon.state)

    def test_nonfinite_post_update_injection_preserves_checkpoint(self):
        def make_opt(model, cfg):
            return build_optimizer(model, cfg['training'])[0]

        def corrupt(opt, model, target):
            original = opt.step
            def step(closure=None):
                result = original(closure)
                with torch.no_grad():
                    if target == 'weight':
                        model.core[0].moe.experts[0].gate.weight[0, 0] = float('nan')
                    else:
                        states = [opt.adamw.state, opt.muon.state]
                        for state in states:
                            for entry in state.values():
                                for value in entry.values():
                                    if torch.is_tensor(value) and value.is_floating_point():
                                        value.view(-1)[0] = float('nan')
                                        return result
                return result
            opt.step = step
            return opt

        cfg = config('splot-smoke-synthetic-fp8-muon.json')
        blocks = pack(split_documents(synthetic_documents())['train'], 96)
        identity = {'schema': 1, 'tag': 'injection'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for target in ('weight', 'state'):
                seed_all(42)
                model = model_from(cfg).to(self.device)
                convert_selected_linears(model)
                opt = make_opt(model, cfg)
                train(model, opt, blocks, lambda m, x, y: training_objective(
                    m, x, y, mode='A2', loops=2), identity, root / f'{target}-normal',
                    steps=1, batch_size=1, seed=42, device=self.device,
                    autocast_dtype=torch.bfloat16, post_update_check=True)
                source = root / f'{target}-normal' / 'checkpoint.pt'
                before = hashlib.sha256(source.read_bytes()).hexdigest()
                resumed = root / f'{target}-resume'
                resumed.mkdir()
                (resumed / 'checkpoint.pt').write_bytes(source.read_bytes())
                seed_all(99)
                model = model_from(cfg).to(self.device)
                convert_selected_linears(model)
                opt = corrupt(make_opt(model, cfg), model, target)
                with self.assertRaises(FloatingPointError):
                    train(model, opt, blocks, lambda m, x, y: training_objective(
                        m, x, y, mode='A2', loops=2), identity, resumed,
                        steps=2, batch_size=1, seed=42, device=self.device,
                        autocast_dtype=torch.bfloat16, resume=True)
                after = hashlib.sha256((resumed / 'checkpoint.pt').read_bytes()).hexdigest()
                self.assertEqual(before, after)
                self.assertEqual(len(torch.load(resumed / 'checkpoint.pt',
                                                map_location='cpu', weights_only=True)['history']), 1)

    def test_fp8_muon_resume_exact_history_and_audit(self):
        cfg = config('splot-smoke-synthetic-fp8-muon.json')
        blocks = pack(split_documents(synthetic_documents())['train'], 96)
        identity = {'schema': 1, 'tag': 'resume'}
        objective = lambda m, x, y: training_objective(m, x, y, mode='A2', loops=2)

        def run(path, steps, resume=False):
            seed_all(42 if not resume else 99)
            model = model_from(cfg).to(self.device)
            convert_selected_linears(model)
            opt = build_optimizer(model, cfg['training'])[0]
            history = train(model, opt, blocks, objective, identity, path,
                            steps=steps, batch_size=1, seed=42, device=self.device,
                            autocast_dtype=torch.bfloat16, resume=resume)
            return model, opt, history

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, ao, ah = run(root / 'continuous', 2)
            run(root / 'split', 1)
            b, bo, bh = run(root / 'split', 2, True)
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            tree_equal(ao.state_dict(), bo.state_dict())
            ignore = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}
            self.assertEqual([{k: v for k, v in row.items() if k not in ignore} for row in ah],
                             [{k: v for k, v in row.items() if k not in ignore} for row in bh])
            self.assertIn('audit', ah[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        FP8MuonCUDATests if args.device == 'cuda' else FP8MuonCPUTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
