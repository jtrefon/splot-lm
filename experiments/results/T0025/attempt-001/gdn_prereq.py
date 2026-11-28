"""T0025 prerequisite: FP64 distinct-document/reset and carried split/tail gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from splot.gdn import delta_chunk, delta_scan

H, D = 6, 64
CHUNKS = (16, 32, 64)
NAMES = ('q', 'k', 'v', 'g', 'beta', 'state')


def digest(inputs):
    h = hashlib.sha256()
    for name, tensor in zip(NAMES, inputs):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def make_inputs(length, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    dtype = torch.float64

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=dtype)

    q = randn((length, H, D))
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12) / 8
    k = randn((length, H, D))
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = 0.1 * randn((length, H, D))
    g = torch.empty((length, H), dtype=dtype).uniform_(-0.1, -0.001, generator=generator)
    beta = torch.empty((length, H), dtype=dtype).uniform_(0.1, 0.9, generator=generator)
    state = 0.01 * randn((H, D, D))
    return q, k, v, g, beta, state


def clone(inputs):
    return tuple(value.detach().clone().requires_grad_(True) for value in inputs)


def loss_grads(out, final, values):
    loss = out.square().mean() + final.square().mean()
    return loss.detach(), tuple(value.detach() for value in torch.autograd.grad(loss, values))


def scan(inputs):
    q, k, v, g, beta, state = clone(inputs)
    out, final = delta_scan(q, k, v, g, beta, state)
    loss, grads = loss_grads(out, final, (q, k, v, g, beta, state))
    return {'out': out.detach(), 'state': final.detach(), 'loss': loss, 'grads': grads}


def compose(leaves, chunks):
    q, k, v, g, beta, state = leaves
    outputs = []
    current = state
    for lo, hi, chunk in chunks:
        for start in range(lo, hi, chunk):
            end = min(start + chunk, hi)
            part, current = delta_chunk(q[start:end], k[start:end], v[start:end],
                                        g[start:end], beta[start:end], current)
            outputs.append(part)
    out = torch.cat(outputs, dim=0)
    loss, grads = loss_grads(out, current, leaves)
    return {'out': out.detach(), 'state': current.detach(), 'loss': loss, 'grads': grads}


def chunked_full(inputs, chunk):
    return compose(clone(inputs), [(0, inputs[0].shape[0], chunk)])


def chunked_split(inputs, chunk, split):
    return compose(clone(inputs), [(0, split, chunk), (split, inputs[0].shape[0], chunk)])


def compare(candidate, reference, kind):
    delta = candidate.double() - reference.double()
    rel = delta.norm().item() / max(reference.double().norm().item(), 1e-12)
    max_abs = delta.abs().max().item()
    finite = bool(torch.isfinite(candidate).all().item())
    return {'rel_l2': rel, 'max_abs': max_abs, 'finite': finite,
            'status': 'passed' if rel <= 1e-9 and max_abs <= 1e-9 and finite else 'failed'}


def compare_result(candidate, reference):
    components = {
        'output': compare(candidate['out'], reference['out'], 'output'),
        'state': compare(candidate['state'], reference['state'], 'state'),
        'loss': compare(candidate['loss'], reference['loss'], 'loss'),
    }
    for name, value, ref in zip(NAMES, candidate['grads'], reference['grads']):
        components[f'grad_{name}'] = compare(value, ref, 'grad')
    return {'components': components,
            'status': 'passed' if all(item['status'] == 'passed'
                                       for item in components.values()) else 'failed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--t0024-numerical', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0024 = json.loads(args.t0024_numerical.read_text())
    reference_input = next(case['input_sha256'] for case in t0024['cases']
                           if case['mode'] == 'fp64' and case['seed'] == 42
                           and case['decay'] == 'ordinary' and case['length'] == 129)
    doc_a = make_inputs(129, 42)
    doc_b = make_inputs(129, 43)
    digest_a = digest(doc_a)
    digest_b = digest(doc_b)
    if digest_a != reference_input:
        raise ValueError('carry document does not match T0024 FP64 T129 seed42 input')
    report = {
        'status': 'pending',
        'task': 'T0025',
        'attempt': 1,
        'gate': 'prerequisite_reset',
        'dtype': 'float64',
        'seed_a': 42,
        'seed_b': 43,
        'length': 129,
        'split': 37,
        'tail': 92,
        'chunks': list(CHUNKS),
        't0024_input_match': True,
        'input_digest_a': digest_a,
        'input_digest_b': digest_b,
        'thresholds': {'relative_l2': 1e-9, 'max_abs': 1e-9,
                       'denominator_floor': 1e-12},
        'independent_documents': {},
        'carried_split': {},
    }
    a_ref = scan(doc_a)
    b_ref = scan(doc_b)
    independent_ok = True
    for chunk in CHUNKS:
        a = chunked_full(doc_a, chunk)
        b = chunked_full(doc_b, chunk)
        # Run A again after B in the same process to expose any cross-document state.
        a_again = chunked_full(doc_a, chunk)
        result = {
            'doc_a_vs_a_reference': compare_result(a, a_ref),
            'doc_b_vs_b_reference': compare_result(b, b_ref),
            'doc_a_rerun_after_b_vs_a_reference': compare_result(a_again, a_ref),
            'doc_a_rerun_matches_first_a': compare_result(a_again, a),
            'cross_document_a_b_outputs_identical': bool(torch.equal(a['out'], b['out'])),
        }
        result['status'] = 'passed' if all(
            result[key]['status'] == 'passed' for key in
            ('doc_a_vs_a_reference', 'doc_b_vs_b_reference',
             'doc_a_rerun_after_b_vs_a_reference', 'doc_a_rerun_matches_first_a')) else 'failed'
        independent_ok = independent_ok and result['status'] == 'passed'
        report['independent_documents'][str(chunk)] = result

    carry_ok = True
    for chunk in CHUNKS:
        candidate = chunked_split(doc_a, chunk, 37)
        result = compare_result(candidate, a_ref)
        result['split'] = 37
        result['tail'] = 92
        carry_ok = carry_ok and result['status'] == 'passed'
        report['carried_split'][str(chunk)] = result
    report['status'] = 'passed' if independent_ok and carry_ok else 'failed'
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'status': report['status'],
        'independent': {k: v['status'] for k, v in report['independent_documents'].items()},
        'carried_split': {k: v['status'] for k, v in report['carried_split'].items()},
        'input_digest_a': digest_a,
    }, indent=2, sort_keys=True))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
