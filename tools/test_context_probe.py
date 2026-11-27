"""Focused CPU tests for the T0012 contiguous-context probe."""
import argparse
import json
import math
import sys
import unittest
from pathlib import Path

import torch

import context_probe as probe
from experiment import seed_all
from splot.fp8 import convert_selected_linears


class ContiguousCaseTests(unittest.TestCase):
    def test_boundaries_labels_and_counts(self):
        for length in (128, 512, 2048, 4096):
            inputs, labels, metadata = probe.build_contiguous_case(length, 42)
            input_ids, document_ids, position_ids, valid = inputs
            self.assertEqual(tuple(input_ids.shape), (1, length))
            self.assertEqual(int(input_ids[0, 0]), 1)
            self.assertEqual(int(input_ids[0, -1]), 2)
            self.assertTrue(bool(((input_ids[0, 1:-1] >= probe.CONTENT_MIN)
                                  & (input_ids[0, 1:-1] <= probe.CONTENT_MAX)).all()))
            self.assertTrue(bool((document_ids == 0).all()))
            self.assertTrue(torch.equal(position_ids[0], torch.arange(length)))
            self.assertTrue(bool(valid.all()))
            self.assertEqual(metadata['valid_tokens'], length)
            self.assertEqual(metadata['ntp_labels'], length - 1)
            self.assertEqual(metadata['offset2_labels'], length - 2)
            self.assertTrue(metadata['labels_exact'])
            self.assertTrue(metadata['bos_ok'] and metadata['eos_ok'])
            self.assertTrue(metadata['positions_ok'] and metadata['documents_ok'])
            for t in range(length - 1):
                self.assertEqual(int(labels[1][0, t]), int(input_ids[0, t + 1]))
            for t in range(length - 2):
                self.assertEqual(int(labels[2][0, t]), int(input_ids[0, t + 2]))
            self.assertEqual(int(labels[1][0, -1]), -100)
            self.assertEqual(int(labels[2][0, -1]), -100)
            self.assertEqual(int(labels[2][0, -2]), -100)

    def test_fixed_seed_and_invalid_inputs(self):
        a = probe.build_contiguous_case(128, 42)
        b = probe.build_contiguous_case(128, 42)
        c = probe.build_contiguous_case(128, 43)
        self.assertEqual(a[2]['input_sha256'], b[2]['input_sha256'])
        self.assertNotEqual(a[2]['input_sha256'], c[2]['input_sha256'])
        self.assertTrue(torch.equal(a[0][0], b[0][0]))
        self.assertFalse(torch.equal(a[0][0], c[0][0]))
        for length in (0, 1, 2, -1, 3.5, True):
            with self.assertRaises(ValueError):
                probe.build_contiguous_case(length)
        with self.assertRaises(ValueError):
            probe.build_contiguous_case(128, 1.5)
        with self.assertRaises(RuntimeError):
            probe.run_case('cpu', 'bf16', 128)
        with self.assertRaises(ValueError):
            probe.load_config('fp16')

    def test_phase_and_rate_math(self):
        phases = {
            'prepare': .1, 'transfer': .2, 'forward': 1., 'backward': 2.,
            'checks': .3, 'optimizer_step': .4, 'post_update_checks': .2,
            'end_to_end': 4.5,
        }
        summary = probe.phase_summary(phases, 512)
        self.assertAlmostEqual(summary['phase_sum_seconds'], 4.2)
        self.assertAlmostEqual(summary['unaccounted_seconds'], .3)
        self.assertGreater(summary['tokens_per_second'], 0)
        self.assertAlmostEqual(probe.tokens_per_second(512, 4.5), 512 / 4.5)
        for tokens, seconds in ((0, 1), (-1, 1), (1, 0), (1, -1), (1, float('nan'))):
            with self.assertRaises(ValueError):
                probe.tokens_per_second(tokens, seconds)
        bad = dict(phases, end_to_end=1.)
        with self.assertRaises(ValueError):
            probe.phase_summary(bad, 512)
        with self.assertRaises(ValueError):
            probe.phase_summary({'prepare': 1.}, 512)


class ModelAndConfigTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_paired_configs_and_initial_model_identity(self):
        path_bf16, cfg_bf16 = probe.load_config('bf16')
        path_fp8, cfg_fp8 = probe.load_config('fp8')
        self.assertEqual(path_bf16.name, 'splot-nano-synthetic-bf16-muon.json')
        self.assertEqual(path_fp8.name, 'splot-nano-synthetic-fp8-muon.json')
        left = json.loads(path_bf16.read_text())
        right = json.loads(path_fp8.read_text())
        left['training'].pop('precision')
        right['training'].pop('precision')
        self.assertEqual(left, right)
        seed_all(42)
        a = probe.model_from(cfg_bf16)
        digest_a = probe.initial_state_digest(a)
        seed_all(42)
        b = probe.model_from(cfg_bf16)
        digest_b = probe.initial_state_digest(b)
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(sum(value.numel() for value in a.parameters()),
                         probe.MODEL_PARAMETERS)
        seed_all(42)
        c = probe.model_from(cfg_fp8)
        self.assertEqual(probe.initial_state_digest(c), digest_a)
        policy = convert_selected_linears(c)
        self.assertEqual(policy['selected_count'], 156)
        self.assertEqual(policy['excluded_count'], 11)
        self.assertEqual(policy['optimizer_scope'], 'adamw-or-muon-hybrid')
        self.assertEqual(probe.initial_state_digest(c), digest_a)

    def test_invalid_probe_arguments(self):
        for precision, length in (('bf16', 2), ('fp8', 0), ('bf16', -1)):
            with self.assertRaises(ValueError):
                probe.run_case('cuda', precision, length)
        with self.assertRaises(ValueError):
            probe.run_case('cuda', 'fp16', 128)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContiguousCaseTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ModelAndConfigTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
