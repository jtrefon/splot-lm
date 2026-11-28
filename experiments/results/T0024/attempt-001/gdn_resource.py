"""T0024 resource measurement for one BF16/FP32 GDN scan or chunked method case."""
import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch

from splot.gdn import delta_chunk, delta_scan

H, D = 6, 64
NAMES = ('q', 'k', 'v', 'g', 'beta', 'state')
CHUNKS = {'chunk16': 16, 'chunk32': 32, 'chunk64': 64}
MIN_GPU_BYTES = 20480 * 1024 * 1024
MIN_HOST_RAM_BYTES = 8 * 1024 ** 3
MIN_SCRATCH_BYTES = 4 * 1024 ** 3


def sha_digest(tensors, names=NAMES):
    h = hashlib.sha256()
    for name, tensor in zip(names, tensors):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def make_inputs(length, seed=42):
    generator = torch.Generator(device='cpu').manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    q = randn((length, H, D))
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12) / 8
    k = randn((length, H, D))
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = 0.1 * randn((length, H, D))
    g = torch.empty((length, H), dtype=torch.float32).uniform_(-0.1, -0.001, generator=generator)
    beta = torch.empty((length, H), dtype=torch.float32).uniform_(0.1, 0.9, generator=generator)
    state = 0.01 * randn((H, D, D))
    return q, k, v, g, beta, state


def clone_leaves(base):
    return tuple(value.detach().clone().requires_grad_(True) for value in base)


def forward_method(method, leaves):
    q, k, v, g, beta, state = leaves
    if method == 'scan':
        out, final = delta_scan(q, k, v, g, beta, state)
    else:
        chunk = CHUNKS[method]
        outputs = []
        current = state
        for start in range(0, q.shape[0], chunk):
            end = min(start + chunk, q.shape[0])
            part, current = delta_chunk(q[start:end], k[start:end], v[start:end],
                                        g[start:end], beta[start:end], current)
            outputs.append(part)
        out = torch.cat(outputs, dim=0)
        final = current
    loss = out.square().mean() + final.square().mean()
    return out, final, loss


def run_forward_backward(method, base):
    leaves = clone_leaves(base)
    out, final, loss = forward_method(method, leaves)
    loss.backward()
    grads = tuple(value.grad.detach() for value in leaves)
    return {
        'out': out.detach(),
        'state': final.detach(),
        'loss': loss.detach(),
        'grads': grads,
    }


def memory_state():
    free, total = torch.cuda.mem_get_info()
    return {
        'free_bytes': free,
        'total_bytes': total,
        'allocated_bytes': torch.cuda.memory_allocated(),
        'reserved_bytes': torch.cuda.memory_reserved(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    }


def host_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def compare_tensor(candidate, reference, kind):
    delta = candidate.double() - reference.double()
    rel_l2 = delta.norm().item() / max(reference.double().norm().item(), 1e-12)
    max_abs = delta.abs().max().item()
    ref_max = reference.double().abs().max().item()
    finite = bool(torch.isfinite(candidate).all().item())
    if kind == 'grad':
        rel_ok = rel_l2 <= 5e-4
        max_ok = max_abs <= 1e-5 * max(1.0, ref_max)
    else:
        rel_ok = rel_l2 <= 1e-4
        max_ok = max_abs <= 1e-4 * max(1.0, ref_max)
    return {
        'rel_l2': rel_l2, 'max_abs': max_abs, 'reference_max_abs': ref_max,
        'relative_ok': rel_ok, 'max_abs_ok': max_ok, 'finite': finite,
        'status': 'passed' if rel_ok and max_ok and finite else 'failed',
    }


def compare_result(candidate, reference):
    components = {
        'output': compare_tensor(candidate['out'], reference['out'], 'output'),
        'state': compare_tensor(candidate['state'], reference['state'], 'state'),
        'loss': compare_tensor(candidate['loss'], reference['loss'], 'loss'),
    }
    for name, value, ref in zip(NAMES, candidate['grads'], reference['grads']):
        components[f'grad_{name}'] = compare_tensor(value, ref, 'grad')
    return {
        'components': components,
        'status': 'passed' if all(item['status'] == 'passed' for item in components.values()) else 'failed',
    }


def to_cpu(result):
    return {
        'out': result['out'].cpu(),
        'state': result['state'].cpu(),
        'loss': result['loss'].cpu(),
        'grads': tuple(value.cpu() for value in result['grads']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--length', type=int, choices=(512, 2048), required=True)
    parser.add_argument('--method', choices=('scan', 'chunk16', 'chunk32', 'chunk64'), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    report = {
        'status': 'blocked',
        'stage': 'resources',
        'task': 'T0024',
        'attempt': 1,
        'length': args.length,
        'method': args.method,
        'seed': args.seed,
        'device': 'cuda',
        'precision': 'float32',
        'source_commit': '6e8911662ac56df665147cb79e412f0383cf71d8',
        'gdn_sha256': '46eaade0fc574353a68654b6dfa52f1c93af271a97b1632aa43551ac20e85f7b',
        'resources': {},
        'gate': {},
        'repeats': [],
    }
    scratch = args.out.parent
    scratch.mkdir(parents=True, exist_ok=True)
    if args.out.resolve().is_relative_to(args.source_root.resolve()):
        raise SystemExit('output must be outside source checkout')
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    resources = {
        'gpu_free_bytes': gpu_free, 'gpu_total_bytes': gpu_total,
        'host_ram_available_bytes': host_available(),
        'scratch_free_bytes': shutil.disk_usage(scratch).free,
        'minimum_gpu_bytes': MIN_GPU_BYTES,
        'minimum_host_ram_bytes': MIN_HOST_RAM_BYTES,
        'minimum_scratch_bytes': MIN_SCRATCH_BYTES,
    }
    report['resources'].update(resources)
    if (gpu_free < MIN_GPU_BYTES or resources['host_ram_available_bytes'] < MIN_HOST_RAM_BYTES
            or resources['scratch_free_bytes'] < MIN_SCRATCH_BYTES):
        report['error'] = 'resource precondition failed'
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        return 2

    stage = 'input'
    try:
        cpu_inputs = make_inputs(args.length, args.seed)
        report['input'] = {
            'sha256': sha_digest(cpu_inputs),
            'length': args.length, 'heads': H, 'dim': D,
            'chunk': None if args.method == 'scan' else CHUNKS[args.method],
            'dtype': 'float32',
        }
        base = tuple(value.to(device) for value in cpu_inputs)
        report['resources']['input_baseline'] = memory_state()

        stage = 'reference_gate'
        reference = to_cpu(run_forward_backward('scan', base))
        candidate = to_cpu(run_forward_backward(args.method, base))
        gate = compare_result(candidate, reference)
        gate['reference_digest'] = sha_digest((reference['out'], reference['state'], *reference['grads']),
                                              ('out', 'state', *NAMES))
        gate['candidate_digest'] = sha_digest((candidate['out'], candidate['state'], *candidate['grads']),
                                              ('out', 'state', *NAMES))
        report['gate'] = gate
        if gate['status'] != 'passed':
            report['status'] = 'failed'
            report['stage'] = 'reference_gate'
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
            return 1
        del reference, candidate
        gc.collect()
        torch.cuda.empty_cache()

        stage = 'warmup'
        warm = run_forward_backward(args.method, base)
        report['warmup'] = {
            'loss': float(warm['loss'].item()),
            'finite': all(bool(torch.isfinite(value).all().item())
                          for value in (warm['out'], warm['state'], *warm['grads'])),
        }
        if not report['warmup']['finite']:
            report['status'] = 'failed'
            report['stage'] = 'warmup'
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
            return 1
        del warm
        gc.collect()
        torch.cuda.empty_cache()

        stage = 'timing'
        reference = to_cpu(run_forward_backward('scan', base))
        for repeat in range(1, 4):
            leaves = clone_leaves(base)
            torch.cuda.synchronize()
            baseline = memory_state()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            out, final, loss = forward_method(args.method, leaves)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - start
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - start
            peak = memory_state()
            grads = tuple(value.grad.detach() for value in leaves)
            candidate = to_cpu({'out': out, 'state': final, 'loss': loss, 'grads': grads})
            comparison = compare_result(candidate, reference)
            digest = sha_digest((candidate['out'], candidate['state'], *candidate['grads']),
                                ('out', 'state', *NAMES))
            row = {
                'repeat': repeat,
                'forward_seconds': forward_seconds,
                'backward_seconds': backward_seconds,
                'total_seconds': forward_seconds + backward_seconds,
                'baseline_allocated_bytes': baseline['allocated_bytes'],
                'baseline_reserved_bytes': baseline['reserved_bytes'],
                'peak_allocated_bytes': peak['peak_allocated_bytes'],
                'peak_reserved_bytes': peak['peak_reserved_bytes'],
                'increment_allocated_bytes': peak['peak_allocated_bytes'] - baseline['allocated_bytes'],
                'increment_reserved_bytes': peak['peak_reserved_bytes'] - baseline['reserved_bytes'],
                'comparison': comparison,
                'candidate_digest': digest,
                'digest_matches_gate': digest == gate['candidate_digest'],
                'finite': all(bool(torch.isfinite(value).all().item())
                              for value in (out, final, *grads)),
            }
            report['repeats'].append(row)
            del leaves, out, final, loss, grads, candidate, baseline, peak
            gc.collect()
            torch.cuda.empty_cache()
            if comparison['status'] != 'passed' or not row['finite'] or not row['digest_matches_gate']:
                report['status'] = 'failed'
                report['stage'] = f'timing_repeat_{repeat}'
                break
        if report['status'] not in ('failed',):
            report['status'] = 'passed'
            report['stage'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['stage'] = stage
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['resources']['free_after_cleanup'] = memory_state()
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
