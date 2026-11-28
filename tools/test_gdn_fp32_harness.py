"""CLI tests for FP32 scan/chunk64 identity, resume and fail-closed behavior."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs/experiments'
PYTHONPATH = os.pathsep.join(str(ROOT / p) for p in ('src', 'tools', 'tools/rep'))


def load_config(name):
    return json.loads((CFG / name).read_text())


def write_config(directory, name, cfg):
    path = Path(directory) / name
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    return path


def run_cli(config, out, steps, resume=False, expect=0):
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUTF8='1')
    command = [sys.executable, str(ROOT / 'tools/experiment.py'),
               '--config', str(config), '--out', str(out), '--steps', str(steps),
               '--device', 'cpu']
    if resume:
        command.append('--resume')
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=120)
    if expect == 0:
        if proc.returncode != 0:
            raise AssertionError(f'CLI failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}')
    else:
        if proc.returncode == 0:
            raise AssertionError(f'CLI unexpectedly passed:\n{proc.stdout}\n{proc.stderr}')
    return proc


def checkpoint(path):
    return torch.load(Path(path) / 'checkpoint.pt', map_location='cpu', weights_only=True)


def run_json(path):
    return json.loads((Path(path) / 'run.json').read_text())


def assert_state_equal(test, a, b):
    if torch.is_tensor(a):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            assert_state_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_state_equal(test, x, y)
    else:
        test.assertEqual(a, b)


def non_timing_history(rows):
    ignore = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}
    return [{key: value for key, value in row.items() if key not in ignore} for row in rows]


class FP32HarnessTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_cli_a0_a2_scan_chunk_resume_exact(self):
        base = load_config('splot-smoke-synthetic-fp32-scan.json')
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('A0', 'A2'):
                for algorithm in ('scan', 'chunk64'):
                    cfg = copy.deepcopy(base)
                    cfg['training']['mode'] = mode
                    cfg['recurrence']['gdn_algorithm'] = algorithm
                    if mode == 'A0':
                        cfg['training']['depth_weight'] = 0
                        cfg['training']['offset_weight'] = 0
                    tag = f'{mode}-{algorithm}'
                    config_path = write_config(tmp, f'{tag}.json', cfg)
                    continuous = Path(tmp) / f'{tag}-continuous'
                    split = Path(tmp) / f'{tag}-split'
                    run_cli(config_path, continuous, 4)
                    run_cli(config_path, split, 2)
                    checkpoint_hash = hashlib.sha256((split / 'checkpoint.pt').read_bytes()).hexdigest()
                    run_cli(config_path, split, 4, resume=True)
                    a = checkpoint(continuous)
                    b = checkpoint(split)
                    for key in ('model', 'optimizer', 'rng'):
                        assert_state_equal(self, a[key], b[key])
                    self.assertEqual(a['identity'], b['identity'])
                    ja, jb = run_json(continuous), run_json(split)
                    self.assertEqual(ja['identity'], jb['identity'])
                    self.assertEqual(non_timing_history(ja['history']), non_timing_history(jb['history']))
                    self.assertEqual(ja['validation'], jb['validation'])
                    self.assertNotEqual(checkpoint_hash,
                                        hashlib.sha256((split / 'checkpoint.pt').read_bytes()).hexdigest())

    def test_cli_cross_mode_and_precision_fail_closed(self):
        base = load_config('splot-smoke-synthetic-fp32-scan.json')
        with tempfile.TemporaryDirectory() as tmp:
            chunk = copy.deepcopy(base)
            chunk['recurrence']['gdn_algorithm'] = 'chunk64'
            chunk_path = write_config(tmp, 'chunk64.json', chunk)
            scan = copy.deepcopy(base)
            scan['recurrence']['gdn_algorithm'] = 'scan'
            scan_path = write_config(tmp, 'scan.json', scan)
            run = Path(tmp) / 'cross'
            run_cli(chunk_path, run, 2)
            before = hashlib.sha256((run / 'checkpoint.pt').read_bytes()).hexdigest()
            proc = run_cli(scan_path, run, 4, resume=True, expect=2)
            self.assertIn('identity mismatch', proc.stderr + proc.stdout)
            after = hashlib.sha256((run / 'checkpoint.pt').read_bytes()).hexdigest()
            self.assertEqual(before, after)

            for precision in ('bf16', 'fp8'):
                bad = copy.deepcopy(chunk)
                bad['training']['precision'] = precision
                bad_path = write_config(tmp, f'chunk64-{precision}.json', bad)
                out = Path(tmp) / f'chunk64-{precision}'
                proc = run_cli(bad_path, out, 1, expect=2)
                self.assertIn('requires precision fp32', proc.stderr + proc.stdout)
                self.assertFalse((out / 'checkpoint.pt').exists())
                self.assertFalse((out / 'run.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
