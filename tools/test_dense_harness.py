"""Dense NTP comparator integration tests; CPU-only, tiny fixtures."""
import copy
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import accounting
import experiment
from data_fixture import Document, pack
from optimizers import build_optimizer, build_semantic_partition
from splot import DenseConfig, DenseLM


ROOT = Path(__file__).resolve().parents[1]
DENSE_CONFIGS = (
    'dense-ntp-512x6-q8kv4-ffn2432.json',
    'dense-ntp-512x6-q8kv2-ffn2176.json',
)


def tiny_config(vocab=264):
    return DenseConfig(vocab_size=vocab, width=64, layers=2, q_heads=4,
                       kv_heads=2, head_dim=16, intermediate=128)


def tiny_blocks(length=12, count=3, vocab=264):
    rows = []
    for doc in range(count):
        ids = [1] + [8 + ((doc * 11 + i) % (vocab - 8)) for i in range(length - 2)] + [2]
        labels = {1: [-100] * length, 2: [-100] * length}
        for i in range(length - 1):
            labels[1][i] = ids[i + 1]
        for i in range(length - 2):
            labels[2][i] = ids[i + 2]
        rows.append({'input_ids': ids, 'document_ids': [doc] * length,
                     'position_ids': list(range(length)), 'valid': [True] * length,
                     'labels': labels})
    return rows


def batch(block, device='cpu'):
    x = tuple(torch.tensor([block[k]], device=device) for k in
              ('input_ids', 'document_ids', 'position_ids', 'valid'))
    y = {k: torch.tensor([block['labels'][k]], device=device) for k in (1, 2)}
    return x, y


def config(name):
    return json.loads((ROOT / 'configs/experiments' / name).read_text())


class DenseAccountingTests(unittest.TestCase):
    def test_candidate_counts_and_splot_v16384_a0(self):
        for name, params, prefill512, prefill2048 in (
            ('dense-ntp-512x6-q8kv4-ffn2432.json', 35527168, 18993381376, 85637201920),
            ('dense-ntp-512x6-q8kv2-ffn2176.json', 32381440, 17382768640, 79194750976),
        ):
            cfg = config(name)
            shape = accounting.DenseShape(
                vocab=cfg['dense']['vocab_size'], width=cfg['dense']['width'],
                layers=cfg['dense']['layers'], q_heads=cfg['dense']['q_heads'],
                kv_heads=cfg['dense']['kv_heads'], head=cfg['dense']['head_dim'],
                dense=cfg['dense']['intermediate'])
            self.assertEqual(accounting.dense_account(shape, 512)['physical'], params)
            self.assertEqual(accounting.dense_account(shape, 512)['prefill_all_logits_macs'], prefill512)
            self.assertEqual(accounting.dense_account(shape, 2048)['prefill_all_logits_macs'], prefill2048)
            self.assertEqual(cfg['training']['mode'], 'A0')
            self.assertEqual(cfg['training']['loops'], 1)
            self.assertNotIn('recurrence', cfg)
            self.assertNotIn('expert_intermediate', cfg)
        splot = config('splot-nano-v16384-a0.json')
        shape = accounting.Shape(
            vocab=splot['dense']['vocab_size'], d=splot['dense']['width'],
            q_heads=splot['dense']['q_heads'], kv_heads=splot['dense']['kv_heads'],
            gdn_heads=splot['dense']['q_heads'], head=splot['dense']['head_dim'],
            dense=splot['dense']['intermediate'], expert=splot['expert_intermediate'])
        result = accounting.account(shape, loops=2, context=512, readouts=1, offsets=0)
        self.assertEqual(result['physical'], 36763620)
        self.assertEqual(result['base'], 36615780)
        self.assertEqual(result['auxiliary'], 147840)
        self.assertEqual(result['logical_blocks'], 12)
        self.assertEqual(result['prefill_all_logits_macs'], 17381326848)
        self.assertEqual(splot['training']['mode'], 'A0')
        self.assertEqual(splot['training']['balance_weight'], 0.01)

    def test_candidate_instantiation_and_partition(self):
        for name, expected in (
            ('dense-ntp-512x6-q8kv4-ffn2432.json', 35527168),
            ('dense-ntp-512x6-q8kv2-ffn2176.json', 32381440),
        ):
            cfg = config(name)
            model = DenseLM(DenseConfig(**cfg['dense']))
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            partition = build_semantic_partition(model)
            self.assertEqual(partition.metadata['inventory_numel'], expected)
            self.assertEqual(partition.metadata['groups']['muon']['numel'] +
                             partition.metadata['groups']['adamw']['numel'], expected)
            self.assertEqual(len({id(p) for p in partition.muon + partition.adamw}),
                             len(partition.muon) + len(partition.adamw))
            names = {e['name']: e['group'] for g in ('muon', 'adamw')
                     for e in partition.metadata['groups'][g]['entries']}
            self.assertEqual(names['embedding.weight'], 'adamw')
            self.assertEqual(names['blocks.0.attn.q.weight'], 'muon')
            self.assertEqual(names['blocks.0.attn.k.weight'], 'muon')
            self.assertEqual(names['blocks.0.gate.weight'], 'muon')
            self.assertEqual(names['blocks.0.n1.weight'], 'adamw')
            self.assertEqual(names['norm.weight'], 'adamw')
            self.assertEqual(sum(1 for key in names if key == 'embedding.weight'), 1)
            del model, partition
            gc.collect()


class DenseFamilyTests(unittest.TestCase):
    def test_family_defaults_and_fail_closed(self):
        self.assertEqual(experiment.resolve_model_family({}), 'splot')
        self.assertEqual(experiment.resolve_model_family({'model_family': 'dense-ntp'}), 'dense-ntp')
        with self.assertRaises(ValueError):
            experiment.resolve_model_family({'model_family': 'dense-a2'})
        cfg = config('dense-ntp-512x6-q8kv2-ffn2176.json')
        experiment.validate_dense_config(cfg, cfg['training'])
        mutations = (
            ('recurrence', {}),
            ('expert_intermediate', 512),
        )
        for key, value in mutations:
            bad = copy.deepcopy(cfg); bad[key] = value
            with self.assertRaises(ValueError):
                experiment.validate_dense_config(bad, bad['training'])
        for key, value in (('mode', 'A2'), ('loops', 2), ('precision', 'fp8'),
                           ('fp8_weight_reuse', True), ('balance_weight', 0.01),
                           ('depth_weight', 0.25), ('offset_weight', 0.25)):
            bad = copy.deepcopy(cfg); bad['training'][key] = value
            with self.assertRaises(ValueError):
                experiment.validate_dense_config(bad, bad['training'])

    def test_dense_objective_matches_manual_ntp_and_labels(self):
        model = DenseLM(tiny_config())
        block = tiny_blocks()[0]
        x, y = batch(block)
        objective = experiment.dense_ntp_objective(model, x, y)
        logits, _ = model(*x)
        manual = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y[1].reshape(-1),
            ignore_index=-100, reduction='sum') / y[1].ne(-100).sum()
        torch.testing.assert_close(objective.final_ntp, manual, rtol=0, atol=0)
        torch.testing.assert_close(objective.total, objective.final_ntp, rtol=0, atol=0)
        for value in (objective.early_ntp, objective.offset2, objective.balance):
            self.assertEqual(float(value), 0.0)
        self.assertEqual(objective.offset2_count, 0)
        objective.total.backward()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.grad is not None),
                         sum(p.numel() for p in model.parameters()))


class DenseTrainingTests(unittest.TestCase):
    def test_continuous_four_equals_two_plus_two(self):
        blocks = tiny_blocks()
        cfg = {'lr': .01, 'weight_decay': 0}
        identity = {'schema': 1, 'model_family': 'dense-ntp', 'fixture': 'tiny'}

        def run(path, steps, resume=False):
            experiment.seed_all(99 if resume else 42)
            model = DenseLM(tiny_config())
            opt, _ = build_optimizer(model, cfg)
            history = experiment.train(
                model, opt, blocks, experiment.dense_ntp_objective, identity, path,
                steps=steps, batch_size=1, seed=42, device='cpu', resume=resume,
                checkpoint_interval=1)
            val = experiment.evaluate(model, blocks, lambda m, x: m(*x)[0],
                                      batch_size=1, device='cpu')
            return model, opt, history, val, torch.get_rng_state(), random.getstate()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            full_model, full_opt, full_history, full_val, full_rng, full_py = run(out / 'full', 4)
            run(out / 'split', 2)
            split_model, split_opt, split_history, split_val, split_rng, split_py = run(out / 'split', 4, True)
            for left, right in zip(full_model.parameters(), split_model.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            for left, right in zip(full_opt.state.values(), split_opt.state.values()):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertEqual(full_val, split_val)
            clean = lambda rows: [{k: v for k, v in row.items() if k != 'step_seconds'}
                                  for row in rows]
            self.assertEqual(clean(full_history), clean(split_history))
            self.assertTrue(torch.equal(full_rng, split_rng))
            self.assertEqual(full_py, split_py)

    def test_cross_family_cli_resume_rejected(self):
        dense = {
            'model_family': 'dense-ntp',
            'dense': {'vocab_size': 264, 'width': 64, 'layers': 2,
                      'q_heads': 4, 'kv_heads': 2, 'head_dim': 16, 'intermediate': 128},
            'training': {'data': 'synthetic-v1', 'precision': 'fp32', 'mode': 'A0',
                         'loops': 1, 'seed': 42, 'sequence_length': 96, 'batch_size': 1,
                         'lr': .01, 'weight_decay': 0, 'depth_weight': 0,
                         'offset_weight': 0, 'balance_weight': 0, 'checkpoint_interval': 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dense_path = tmp / 'dense.json'
            dense_path.write_text(json.dumps(dense))
            splot_path = ROOT / 'configs/experiments/splot-smoke-synthetic.json'
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(
                str(ROOT / p) for p in ('src', 'tools', 'tools/rep')))
            first = subprocess.run(
                [sys.executable, 'tools/experiment.py', '--config', str(dense_path),
                 '--out', str(tmp / 'run'), '--steps', '1', '--device', 'cpu'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(
                [sys.executable, 'tools/experiment.py', '--config', str(splot_path),
                 '--out', str(tmp / 'run'), '--steps', '2', '--device', 'cpu',
                 '--resume'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('run identity mismatch', second.stderr)


    def test_legacy_splot_cli_identity_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(
                str(ROOT / p) for p in ('src', 'tools', 'tools/rep')))
            result = subprocess.run(
                [sys.executable, 'tools/experiment.py', '--config',
                 str(ROOT / 'configs/experiments/splot-smoke-synthetic.json'),
                 '--out', str(tmp / 'splot'), '--steps', '1', '--device', 'cpu'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((tmp / 'splot' / 'run.json').read_text())
            self.assertEqual(report['parameters'], 179952)
            self.assertNotIn('model_family', report['identity'])
            self.assertNotIn('model_family', report)
            self.assertNotIn('dense_shape', report)


class DenseIngestionTests(unittest.TestCase):
    def test_shared_synthetic_v2_and_pilot_loading(self):
        import pilot_ingestion as pilot
        import test_pilot_ingestion as pilot_fixture
        import test_tokenized_shards as shard_fixture

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pilot_root = tmp / 'pilot'
            pilot_sha, _ = pilot_fixture.write_fixture(pilot_root)
            loaded = pilot.load_pilot_dataset(pilot_root, pilot_sha,
                                               expected_model_vocab=264)
            self.assertEqual(set(loaded['blocks']), {'train', 'validation'})
            self.assertGreaterEqual(len(loaded['blocks']['train']), 32)

            v2_root = tmp / 'v2'
            shard_fixture.materialize(v2_root)
            manifest_sha = hashlib.sha256((v2_root / 'manifest.json').read_bytes()).hexdigest()
            vocab = shard_fixture.DEMO_BUNDLE['artifact']['actual_vocab_size']
            blocks, summary = experiment.load_shard_dataset(
                v2_root, manifest_sha, expected_vocab=vocab, expected_length=96)
            self.assertEqual(set(blocks), {'train', 'validation'})
            self.assertEqual(summary['training_splits'], ['train', 'validation'])

    def test_dense_bf16_capability_if_supported(self):
        try:
            policy = experiment.resolve_precision('cpu', 'bf16')
        except RuntimeError as exc:
            print('dense BF16 unsupported by this CPU runner:', exc)
            return
        self.assertEqual(policy['autocast_dtype'], torch.bfloat16)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main(verbosity=2)
