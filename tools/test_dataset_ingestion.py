"""Focused tests for verified v2 ingestion and checkpoint cadence."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiment import load_shard_dataset, seed_all, train
from tokenized_shards import (PackConfig, build_demo_bundle, manifest_sha256,
                              materialize_dataset)


IGNORE = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}


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


def compare_cadence_dirs(a, b):
    """Compare interval1/interval2 runs without comparing their identity."""
    dirs = [Path(a), Path(b)]
    runs = [json.loads((path / 'run.json').read_text()) for path in dirs]
    checkpoints = [torch.load(path / 'checkpoint.pt', map_location='cpu',
                              weights_only=True) for path in dirs]
    checks = {
        'parameters_equal': runs[0]['parameters'] == runs[1]['parameters'],
        'validation_equal': runs[0]['validation'] == runs[1]['validation'],
        'non_timing_history_equal': (
            [{k: v for k, v in row.items() if k not in IGNORE} for row in runs[0]['history']]
            == [{k: v for k, v in row.items() if k not in IGNORE} for row in runs[1]['history']]),
        'model_equal': True,
        'optimizer_equal': True,
        'rng_equal': True,
    }
    for key, target in (('model', 'model_equal'), ('optimizer', 'optimizer_equal'),
                        ('rng', 'rng_equal')):
        try:
            tree_equal(checkpoints[0][key], checkpoints[1][key], key)
        except AssertionError:
            checks[target] = False
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'parameters': runs[0]['parameters'], 'steps': len(runs[0]['history']),
            'validation': runs[0]['validation'],
            'ignored_history_fields': sorted(IGNORE)}


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


def tiny_run(path, steps, *, interval, resume=False, identity=None, corrupt=False):
    seed_all(42 if not resume else 99)
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    if corrupt:
        original = opt.step

        def step(closure=None):
            result = original(closure)
            with torch.no_grad():
                model[1].weight[0, 0] = float('nan')
            return result

        opt.step = step
    stats = {}
    history = train(model, opt, tiny_blocks(), tiny_objective,
                    identity or {'id': 1, 'checkpoint_interval': interval},
                    path, steps=steps, batch_size=1, seed=42, device='cpu',
                    resume=resume, post_update_check=True,
                    checkpoint_interval=interval, checkpoint_stats=stats)
    return model, opt, history, stats


def write_run_dir(path, model, opt, history):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    torch.save({'identity': {'tag': path.name}, 'model': model.state_dict(),
                'optimizer': opt.state_dict(), 'rng': {'python': None, 'torch': None,
                                                       'cuda': None}, 'history': history},
               path / 'checkpoint.pt')
    (path / 'run.json').write_text(json.dumps({
        'identity': {'tag': path.name}, 'parameters': 33,
        'history': history, 'validation': {'ntp_labels': 1, 'nll_sum': .5,
                                           'mean_nll': .5, 'perplexity': 1.6487212707},
    }, indent=2, sort_keys=True) + '\n')


class DatasetIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='splot-t0014-')
        cls.root = Path(cls.tmp.name) / 'demo'
        bundle = build_demo_bundle()
        cls.manifest = materialize_dataset(
            cls.root, bundle['documents'], bundle['artifact'],
            PackConfig(length=96, max_blocks_per_shard=8), bundle['source_identity'])
        cls.sha = manifest_sha256(cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_verified_v2_blocks_and_counts(self):
        blocks, summary = load_shard_dataset(self.root, self.sha,
                                             expected_vocab=320, expected_length=96)
        self.assertEqual(set(blocks), {'train', 'validation'})
        self.assertEqual(len(blocks['train']), 2)
        self.assertEqual(len(blocks['validation']), 1)
        self.assertEqual(sum(len(block['input_ids']) for block in blocks['train']), 192)
        train_valid = sum(sum(block['valid']) for block in blocks['train'])
        train_ntp = sum(sum(value != -100 for value in block['labels'][1])
                        for block in blocks['train'])
        train_offset = sum(sum(value != -100 for value in block['labels'][2])
                           for block in blocks['train'])
        validation_ntp = sum(sum(value != -100 for value in block['labels'][1])
                             for block in blocks['validation'])
        self.assertEqual((train_valid, train_ntp, train_offset), (142, 134, 126))
        self.assertEqual(validation_ntp, 14)
        for block in blocks['train'] + blocks['validation']:
            self.assertEqual(set(block['labels']), {1, 2})
            for offset in (1, 2):
                for index, value in enumerate(block['labels'][offset]):
                    if value != -100:
                        self.assertEqual(value, block['input_ids'][index + offset])
        self.assertEqual(summary['format'], 'splot-tokenized-shards-v2')
        self.assertEqual(summary['version'], 2)
        self.assertEqual(summary['manifest_sha256'], self.sha)
        self.assertEqual(summary['tokenizer']['actual_vocab_size'], 320)
        self.assertEqual(summary['training_splits'], ['train', 'validation'])

    def test_manifest_and_dimension_rejection(self):
        with self.assertRaises(ValueError):
            load_shard_dataset(self.root, '0' * 64, expected_vocab=320, expected_length=96)
        with self.assertRaises(ValueError):
            load_shard_dataset(self.root, self.sha, expected_vocab=321, expected_length=96)
        with self.assertRaises(ValueError):
            load_shard_dataset(self.root, self.sha, expected_vocab=320, expected_length=128)
        with self.assertRaises(ValueError):
            load_shard_dataset(self.root, 'not-a-hash', expected_vocab=320, expected_length=96)
        with self.assertRaises((OSError, ValueError)):
            load_shard_dataset(self.root / 'missing', self.sha,
                               expected_vocab=320, expected_length=96)
        with self.assertRaises(ValueError):
            load_shard_dataset(self.root, self.sha.upper(), expected_vocab=320,
                               expected_length=96)

    def test_tampered_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0014-tamper-') as tmp:
            copy_root = Path(tmp) / 'copy'
            shutil.copytree(self.root, copy_root)
            manifest = copy_root / 'manifest.json'
            manifest.write_bytes(manifest.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                load_shard_dataset(copy_root, self.sha, expected_vocab=320,
                                   expected_length=96)

    def test_smoke_shard_config(self):
        base = json.loads(Path('configs/experiments/splot-smoke-synthetic-fp8-muon.json').read_text())
        cfg = json.loads(Path('configs/experiments/splot-smoke-shards-fp8-muon.json').read_text())
        self.assertEqual(cfg['dense']['vocab_size'], 320)
        self.assertEqual(cfg['training']['data'], 'synthetic-shards-v2')
        self.assertEqual(cfg['training']['sequence_length'], 96)
        self.assertEqual(cfg['training']['optimizer'], base['training']['optimizer'])
        self.assertEqual(cfg['training']['mode'], 'A2')
        self.assertEqual(cfg['training']['loops'], 2)
        self.assertEqual(cfg['training']['batch_size'], 1)


class CheckpointCadenceTests(unittest.TestCase):
    def test_interval_one_legacy_writes_every_step(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0014-int1-') as tmp:
            _, _, history, stats = tiny_run(Path(tmp) / 'run', 4, interval=1)
            self.assertEqual(len(history), 4)
            self.assertEqual(stats['writes'], 4)
            self.assertEqual(stats['write_steps'], [1, 2, 3, 4])
            self.assertEqual(stats['initial_history_steps'], 0)
            self.assertEqual(stats['final_history_steps'], 4)

    def test_interval_two_saves_final_off_boundary(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0014-int2-') as tmp:
            _, _, history, stats = tiny_run(Path(tmp) / 'run', 3, interval=2)
            self.assertEqual(len(history), 3)
            self.assertEqual(stats['writes'], 2)
            self.assertEqual(stats['write_steps'], [2, 3])
            checkpoint = torch.load(Path(tmp) / 'run' / 'checkpoint.pt',
                                    map_location='cpu', weights_only=True)
            self.assertEqual(len(checkpoint['history']), 3)

    def test_interval_two_split_resume_equivalence(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0014-resume-') as tmp:
            root = Path(tmp)
            a, ao, ah, astats = tiny_run(root / 'continuous', 4, interval=2)
            tiny_run(root / 'split', 2, interval=2)
            b, bo, bh, bstats = tiny_run(root / 'split', 4, interval=2, resume=True)
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            tree_equal(ao.state_dict(), bo.state_dict())
            self.assertEqual([{k: v for k, v in row.items() if k not in IGNORE} for row in ah],
                             [{k: v for k, v in row.items() if k not in IGNORE} for row in bh])
            self.assertEqual(astats['writes'], 2)
            self.assertEqual(bstats['writes'], 1)
            self.assertEqual(bstats['initial_history_steps'], 2)
            self.assertEqual(bstats['final_history_steps'], 4)

    def test_failed_guard_preserves_prior_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0014-guard-') as tmp:
            root = Path(tmp)
            tiny_run(root / 'run', 2, interval=2)
            checkpoint = root / 'run' / 'checkpoint.pt'
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with self.assertRaises(FloatingPointError):
                tiny_run(root / 'run', 3, interval=2, resume=True, corrupt=True)
            after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(len(torch.load(checkpoint, map_location='cpu',
                                            weights_only=True)['history']), 2)

    def test_interval_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0014-mismatch-') as tmp:
            root = Path(tmp)
            tiny_run(root / 'run', 1, interval=2)
            with self.assertRaises(ValueError):
                tiny_run(root / 'run', 2, interval=1, resume=True)

    def test_interval_one_vs_two_comparator(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0014-compare-') as tmp:
            root = Path(tmp)
            m1, o1, h1, _ = tiny_run(root / 'one', 3, interval=1)
            m2, o2, h2, _ = tiny_run(root / 'two', 3, interval=2)
            write_run_dir(root / 'one-run', m1, o1, h1)
            write_run_dir(root / 'two-run', m2, o2, h2)
            result = compare_cadence_dirs(root / 'one-run', root / 'two-run')
            self.assertEqual(result['status'], 'passed', result['checks'])
            self.assertTrue(result['checks']['non_timing_history_equal'])
            self.assertTrue(result['checks']['model_equal'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DatasetIngestionTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointCadenceTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
