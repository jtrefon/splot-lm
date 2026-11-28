"""T0024 numerical qualification: delta_scan reference vs chunked delta_chunk compositions."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from splot.gdn import delta_chunk, delta_scan

H, D = 6, 64
CHUNKS = (16, 32, 64)
NAMES = ('q', 'k', 'v', 'g', 'beta', 'state')


def sha_digest(inputs):
    h = hashlib.sha256()
    for name, tensor in zip(NAMES, inputs):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def make_inputs(length, seed, dtype, decay):
    generator = torch.Generator(device='cpu').manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=dtype)

    q = randn((length, H, D))
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12) / 8
    k = randn((length, H, D))
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = 0.1 * randn((length, H, D))
    low, high = (-0.1, -0.001) if decay == 'ordinary' else (-0.005, -0.0001)
    g = torch.empty((length, H), dtype=dtype).uniform_(low, high, generator=generator)
    beta = torch.empty((length, H), dtype=dtype).uniform_(0.1, 0.9, generator=generator)
    state = 0.01 * randn((H, D, D))
    return q, k, v, g, beta, state


def leaves(inputs):
    return tuple(value.detach().clone().requires_grad_(True) for value in inputs)


def loss_and_grads(out, final, values):
    loss = out.square().mean() + final.square().mean()
    grads = torch.autograd.grad(loss, values)
    return loss, grads


def run_scan(inputs):
    q, k, v, g, beta, state = leaves(inputs)
    out, final = delta_scan(q, k, v, g, beta, state)
    loss, grads = loss_and_grads(out, final, (q, k, v, g, beta, state))
    return {'out': out.detach(), 'state': final.detach(), 'loss': loss.detach(),
            'grads': tuple(value.detach() for value in grads)}


def run_chunk(inputs, chunk):
    q, k, v, g, beta, state = leaves(inputs)
    length = q.shape[0]
    outputs = []
    current = state
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        out, current = delta_chunk(q[start:end], k[start:end], v[start:end],
                                   g[start:end], beta[start:end], current)
        outputs.append(out)
    out = torch.cat(outputs, dim=0)
    loss, grads = loss_and_grads(out, current, (q, k, v, g, beta, state))
    return {'out': out.detach(), 'state': current.detach(), 'loss': loss.detach(),
            'grads': tuple(value.detach() for value in grads)}


def compare(candidate, reference, mode, kind):
    delta = candidate.double() - reference.double()
    ref_norm = reference.double().norm().item()
    rel_l2 = delta.norm().item() / max(ref_norm, 1e-12)
    max_abs = delta.abs().max().item()
    ref_max_abs = reference.double().abs().max().item()
    finite = bool(torch.isfinite(candidate).all().item())
    if mode == 'fp64':
        rel_ok = rel_l2 <= 1e-9
        max_ok = max_abs <= 1e-9
    elif kind == 'grad':
        rel_ok = rel_l2 <= 5e-4
        max_ok = max_abs <= 1e-5 * max(1.0, ref_max_abs)
    else:
        rel_ok = rel_l2 <= 1e-4
        max_ok = max_abs <= 1e-4 * max(1.0, ref_max_abs)
    return {
        'rel_l2': rel_l2,
        'max_abs': max_abs,
        'reference_max_abs': ref_max_abs,
        'relative_ok': rel_ok,
        'max_abs_ok': max_ok,
        'finite': finite,
        'status': 'passed' if rel_ok and max_ok and finite else 'failed',
    }


def case_result(mode, seed, decay, length):
    inputs = make_inputs(length, seed, torch.float64 if mode == 'fp64' else torch.float32, decay)
    reference = run_scan(inputs)
    case = {
        'mode': mode,
        'dtype': 'float64' if mode == 'fp64' else 'float32',
        'seed': seed,
        'decay': decay,
        'length': length,
        'input_sha256': sha_digest(inputs),
        'reference': {
            'loss': float(reference['loss']),
            'output_norm': float(reference['out'].double().norm()),
            'state_norm': float(reference['state'].double().norm()),
            'gradient_norms': {name: float(value.double().norm())
                               for name, value in zip(NAMES, reference['grads'])},
        },
        'chunks': {},
    }
    failed = set()
    for chunk in CHUNKS:
        if chunk in failed:
            case['chunks'][str(chunk)] = {
                'status': 'skipped',
                'reason': 'dependent higher length skipped after prior numerical failure for this chunk',
            }
            continue
        candidate = run_chunk(inputs, chunk)
        components = {
            'output': compare(candidate['out'], reference['out'], mode, 'output'),
            'state': compare(candidate['state'], reference['state'], mode, 'state'),
            'loss': compare(candidate['loss'], reference['loss'], mode, 'loss'),
        }
        for name, value, ref in zip(NAMES, candidate['grads'], reference['grads']):
            components[f'grad_{name}'] = compare(value, ref, mode, 'grad')
        status = 'passed' if all(item['status'] == 'passed' for item in components.values()) else 'failed'
        if status == 'failed':
            failed.add(chunk)
        case['chunks'][str(chunk)] = {
            'status': status,
            'components': components,
        }
    case['status'] = 'passed' if any(item['status'] == 'passed'
                                     for item in case['chunks'].values()) else 'failed'
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    cases = []
    # FP64 oracle, seed 42.
    for length in (33, 129):
        cases.append(case_result('fp64', 42, 'ordinary', length))
    # FP32 ordinary seeds 42/43: T129, T512, T2048.
    for seed in (42, 43):
        for length in (129, 512, 2048):
            cases.append(case_result('fp32', seed, 'ordinary', length))
    # FP32 slow-decay stress at T129, seeds 42/43.
    for seed in (42, 43):
        cases.append(case_result('fp32', seed, 'slow', 129))

    eligibility = {}
    for length in (512, 2048):
        eligibility[str(length)] = {}
        for chunk in CHUNKS:
            rows = [case for case in cases
                    if case['mode'] == 'fp32' and case['decay'] == 'ordinary'
                    and case['length'] == length and case['seed'] in (42, 43)]
            eligibility[str(length)][str(chunk)] = bool(
                len(rows) == 2 and all(row['chunks'][str(chunk)]['status'] == 'passed' for row in rows))
        eligibility[str(length)]['scan'] = True

    passed_chunks = sum(1 for case in cases for item in case['chunks'].values()
                        if item['status'] == 'passed')
    failed_chunks = sum(1 for case in cases for item in case['chunks'].values()
                        if item['status'] == 'failed')
    skipped_chunks = sum(1 for case in cases for item in case['chunks'].values()
                         if item['status'] == 'skipped')
    report = {
        'status': 'completed',
        'task': 'T0024',
        'attempt': 1,
        'device': 'cpu',
        'source_commit': '6e8911662ac56df665147cb79e412f0383cf71d8',
        'gdn_sha256': '46eaade0fc574353a68654b6dfa52f1c93af271a97b1632aa43551ac20e85f7b',
        'chunks': list(CHUNKS),
        'thresholds': {
            'fp64': {'relative_l2': 1e-9, 'max_abs': 1e-9},
            'fp32_output_state': {'relative_l2': 1e-4, 'max_abs_scale': 1e-4},
            'fp32_gradient': {'relative_l2': 5e-4, 'max_abs_scale': 1e-5},
            'near_zero_denominator': 1e-12,
        },
        'oracle_limitations': [
            'delta_chunk is an independent dense triangular solve intended as a small finite-log-decay oracle, not a production kernel.',
            'It has no built-in chunking; T0024 composes it over caller-selected consecutive slices with carried FP32 state.',
            'Existing GDN numerical tests are small float64/NumPy cases; they do not cover FP32 projected-tensor gradients at T512/T2048 or CUDA resource behavior.',
            'The full-slice chunk solve is O(T^2) in the slice length; composition changes the cost profile but is not a custom kernel.',
            'This is a component feasibility test only; it does not show that GDN dominates the full model or that a production kernel is faster.',
        ],
        'cases': cases,
        'eligibility': eligibility,
        'summary': {
            'cases': len(cases),
            'chunk_outcomes': len(cases) * len(CHUNKS),
            'passed_chunks': passed_chunks,
            'failed_chunks': failed_chunks,
            'skipped_chunks': skipped_chunks,
            'elapsed_seconds': time.perf_counter() - started,
        },
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'status': report['status'],
        'cases': len(cases),
        'passed_chunks': passed_chunks,
        'failed_chunks': failed_chunks,
        'skipped_chunks': skipped_chunks,
        'eligibility': eligibility,
        'elapsed_seconds': report['summary']['elapsed_seconds'],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
