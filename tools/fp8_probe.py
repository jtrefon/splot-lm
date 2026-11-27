"""Bounded native FP8 scaled-GEMM feasibility probe for Ada RTX 4090.

This is an exploratory operator qualification, not model training. The only
native FP8 path exercised is torch._scaled_mm on actual float8 tensors.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

FP8_DTYPES = {
    'e4m3fn': torch.float8_e4m3fn,
    'e5m2': torch.float8_e5m2,
}
FP32_THRESHOLD = .15
DEQUANT_THRESHOLD = .025
LARGE_SHAPES = ((16, 384, 512), (64, 384, 512), (512, 384, 512),
                (64, 384, 384), (64, 512, 384), (64, 384, 1152))
SMALL_SHAPES = ((1, 384, 512), (3, 384, 512), (7, 384, 512), (17, 384, 512))


def dtype_of(dtype):
    if isinstance(dtype, str):
        try:
            return FP8_DTYPES[dtype]
        except KeyError as exc:
            raise ValueError(f'unsupported FP8 dtype: {dtype}') from exc
    if dtype not in FP8_DTYPES.values():
        raise ValueError(f'unsupported FP8 dtype: {dtype}')
    return dtype


def quantize_per_tensor(x, dtype):
    """Current per-tensor scaling; returns FP8 data, scale and audit counts."""
    dtype = dtype_of(dtype)
    if not torch.is_tensor(x) or x.dtype != torch.float32:
        raise ValueError('quantization input must be a float32 tensor')
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite input cannot be quantized')
    maximum = float(torch.finfo(dtype).max)
    maximum_abs = float(x.abs().max().item()) if x.numel() else 0.
    scale = maximum_abs / maximum if maximum_abs > 0 else 1.
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid quantization scale')
    scaled = x / scale
    saturation = int((scaled.abs() > maximum).sum().item())
    q = torch.clamp(scaled, -maximum, maximum).to(dtype)
    q_float = q.float()
    nonzero = int((x != 0).sum().item())
    q_zero = int((q_float == 0).sum().item())
    q_zero_nonzero = int(((q_float == 0) & (x != 0)).sum().item())
    return {
        'q': q,
        'scale': scale,
        'dtype': str(dtype).replace('torch.', ''),
        'saturation_count': saturation,
        'quantized_zero_count': q_zero,
        'quantized_zero_fraction': q_zero / x.numel() if x.numel() else 0.,
        'underflow_count': q_zero_nonzero,
        'underflow_fraction': q_zero_nonzero / nonzero if nonzero else 0.,
        'nonzero_input_count': nonzero,
    }


def dequantize(q, scale):
    if not torch.is_tensor(q) or q.dtype not in FP8_DTYPES.values():
        raise ValueError('dequantization requires an FP8 tensor')
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid dequantization scale')
    return q.float() * scale


def relative_frobenius(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f'shape mismatch: {tuple(actual.shape)} != {tuple(reference.shape)}')
    numerator = torch.linalg.vector_norm((actual.float() - reference.float()).flatten())
    denominator = torch.linalg.vector_norm(reference.float().flatten())
    return float(numerator / (denominator + 1e-12))


def padding_plan(real_m, multiple=16):
    if type(real_m) is not int or real_m < 1 or type(multiple) is not int or multiple < 1:
        raise ValueError('invalid padding dimensions')
    padded = ((real_m + multiple - 1) // multiple) * multiple
    return {
        'real_M': real_m, 'padded_M': padded, 'padding_rows': padded - real_m,
        'padding_fraction': (padded - real_m) / padded,
        'overhead_ratio': padded / real_m,
        'multiple': multiple, 'padding_value': 0,
    }


def pad_rows(x, padded_m):
    if x.ndim != 2 or padded_m < x.shape[0]:
        raise ValueError('invalid row padding')
    if padded_m == x.shape[0]:
        return x, False
    out = torch.zeros((padded_m, x.shape[1]), dtype=x.dtype, device=x.device)
    out[:x.shape[0]] = x
    return out, True


def as_row_major(x):
    if x.ndim != 2:
        raise ValueError('expected a matrix')
    if x.stride() == (x.shape[1], 1):
        return x, False
    return x.contiguous(), True


def as_column_major(x):
    if x.ndim != 2:
        raise ValueError('expected a matrix')
    if x.stride() == (1, x.shape[0]):
        return x, False
    return x.t().contiguous().t(), True


def scale_tensor(scale, device):
    return torch.tensor(float(scale), dtype=torch.float32, device=device)


def scaled_mm(a, b, scale_a, scale_b):
    return torch._scaled_mm(a, b, scale_tensor(scale_a, a.device), scale_tensor(scale_b, b.device),
                            out_dtype=torch.bfloat16)


def _q(x, dtype):
    return quantize_per_tensor(x, dtype)


def prepare_forward(x, w, device):
    qx, qw = _q(x, 'e4m3fn'), _q(w, 'e4m3fn')
    b, copied = as_column_major(qw['q'].t())
    return {'a': qx['q'], 'b': b, 'qx': qx, 'qw': qw, 'copy_rhs': copied,
            'input_shapes': [list(qx['q'].shape), list(b.shape)],
            'input_strides': [list(qx['q'].stride()), list(b.stride())],
            'input_dtypes': [str(qx['q'].dtype).replace('torch.', ''), str(b.dtype).replace('torch.', '')],
            'scales': [qx['scale'], qw['scale']]}


def prepare_dgrad(dy, w, device):
    qdy, qw = _q(dy, 'e5m2'), _q(w, 'e4m3fn')
    b, copied = as_column_major(qw['q'])
    return {'a': qdy['q'], 'b': b, 'qdy': qdy, 'qw': qw, 'copy_rhs': copied,
            'input_shapes': [list(qdy['q'].shape), list(b.shape)],
            'input_strides': [list(qdy['q'].stride()), list(b.stride())],
            'input_dtypes': [str(qdy['q'].dtype).replace('torch.', ''), str(b.dtype).replace('torch.', '')],
            'scales': [qdy['scale'], qw['scale']]}


def prepare_wgrad(dy, x, device):
    qdy, qx = _q(dy, 'e5m2'), _q(x, 'e4m3fn')
    a, copy_lhs = as_row_major(qdy['q'].t())
    b, copy_rhs = as_column_major(qx['q'])
    return {'a': a, 'b': b, 'qdy': qdy, 'qx': qx, 'copy_lhs': copy_lhs, 'copy_rhs': copy_rhs,
            'input_shapes': [list(a.shape), list(b.shape)],
            'input_strides': [list(a.stride()), list(b.stride())],
            'input_dtypes': [str(a.dtype).replace('torch.', ''), str(b.dtype).replace('torch.', '')],
            'scales': [qdy['scale'], qx['scale']]}


def call_prepared(prepared):
    return scaled_mm(prepared['a'], prepared['b'], prepared['scales'][0], prepared['scales'][1])


def _dispatch(prepared, output):
    return {
        'operator': 'torch._scaled_mm',
        'input_shapes': prepared['input_shapes'],
        'input_strides': prepared['input_strides'],
        'input_dtypes': prepared['input_dtypes'],
        'scales': prepared['scales'],
        'scale_dtype': 'float32 scalar',
        'out_dtype': str(output.dtype).replace('torch.', ''),
        'output_shape': list(output.shape),
        'output_stride': list(output.stride()),
        'layout_copies': {k: prepared[k] for k in ('copy_lhs', 'copy_rhs') if k in prepared},
        'native_fp8_inputs': all('float8' in dtype for dtype in prepared['input_dtypes']),
    }


def _timing(fn, device, warmups=5, iterations=20):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    times.sort()
    return {
        'method': 'host perf_counter around synchronized CUDA call',
        'warmups': warmups, 'iterations': iterations,
        'median_seconds': statistics.median(times),
        'p95_seconds': times[min(len(times) - 1, math.ceil(.95 * len(times)) - 1)],
        'min_seconds': times[0], 'max_seconds': times[-1],
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
    }


def _bf16_forward(x, w):
    return x.bfloat16() @ w.bfloat16().t()


def _bf16_dgrad(dy, w):
    return dy.bfloat16() @ w.bfloat16()


def _bf16_wgrad(dy, x):
    return dy.bfloat16().t() @ x.bfloat16()


def _forward_quant_inclusive(x, w):
    qx, qw = _q(x, 'e4m3fn'), _q(w, 'e4m3fn')
    b, _ = as_column_major(qw['q'].t())
    return scaled_mm(qx['q'], b, qx['scale'], qw['scale'])


def _dgrad_quant_inclusive(dy, w):
    qdy, qw = _q(dy, 'e5m2'), _q(w, 'e4m3fn')
    b, _ = as_column_major(qw['q'])
    return scaled_mm(qdy['q'], b, qdy['scale'], qw['scale'])


def _wgrad_quant_inclusive(dy, x):
    qdy, qx = _q(dy, 'e5m2'), _q(x, 'e4m3fn')
    a, _ = as_row_major(qdy['q'].t())
    b, _ = as_column_major(qx['q'])
    return scaled_mm(a, b, qdy['scale'], qx['scale'])


def _case_inputs(m, k, n, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, generator=generator, device=device) * 1.
    w = torch.randn(n, k, generator=generator, device=device) * .02
    dy = torch.randn(m, n, generator=generator, device=device) * .1
    return x, w, dy


def _refs(real_x, real_w, real_dy, qx, qw, qdy, pad_m, device):
    xq, wq, dyq = dequantize(qx['q'], qx['scale']), dequantize(qw['q'], qw['scale']), dequantize(qdy['q'], qdy['scale'])
    refs = {
        'fp32': {'forward': real_x @ real_w.t(), 'dgrad': real_dy @ real_w, 'wgrad': real_dy.t() @ real_x},
        'dequant': {'forward': xq @ wq.t(), 'dgrad': dyq @ wq, 'wgrad': dyq.t() @ xq},
    }
    if pad_m != real_x.shape[0]:
        refs['dequant']['forward'] = refs['dequant']['forward'][:real_x.shape[0]]
        refs['dequant']['dgrad'] = refs['dequant']['dgrad'][:real_x.shape[0]]
    return refs


def _check_op(name, output, reference, dequant_reference, trim, thresholds, all_zero=False):
    got = output[:trim] if trim else output
    if all_zero:
        exact = int(torch.count_nonzero(got).item()) == 0
        entry = {'status': 'passed' if exact else 'failed', 'finite': bool(torch.isfinite(got).all()),
                 'zero_exact': exact, 'error_vs_fp32': 0. if exact else relative_frobenius(got, reference),
                 'error_vs_dequant': 0. if exact else relative_frobenius(got, dequant_reference),
                 'thresholds': thresholds}
        return entry, got
    err_fp32 = relative_frobenius(got, reference)
    err_dequant = relative_frobenius(got, dequant_reference)
    passed = (bool(torch.isfinite(got).all()) and err_fp32 <= thresholds['fp32']
              and err_dequant <= thresholds['dequant'])
    return {'status': 'passed' if passed else 'failed', 'finite': bool(torch.isfinite(got).all()),
            'error_vs_fp32': err_fp32, 'error_vs_dequant': err_dequant,
            'thresholds': thresholds, 'output_shape': list(got.shape)}, got


def _bench_op(name, prepared, prepare_fn, quant_fn, bf_fn, ref, trim, device):
    if name == 'forward':
        pre_fn = lambda: call_prepared(prepared)
    elif name == 'dgrad':
        pre_fn = lambda: call_prepared(prepared)
    else:
        pre_fn = lambda: call_prepared(prepared)
    try:
        return {'prequantized': _timing(pre_fn, device),
                'quantization_inclusive': _timing(quant_fn, device),
                'bf16_baseline': _timing(bf_fn, device),
                'shape': prepared.get('bench_shape', {})}
    except Exception as exc:
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}


def _process_case(case_id, m, k, n, seed, device, all_zero=False):
    plan = padding_plan(m)
    padded_m = plan['padded_M']
    if all_zero:
        x = torch.zeros(m, k, device=device)
        w = torch.zeros(n, k, device=device)
        dy = torch.zeros(m, n, device=device)
    else:
        x, w, dy = _case_inputs(m, k, n, seed, device)
    x_pad, x_padded = pad_rows(x, padded_m)
    dy_pad, dy_padded = pad_rows(dy, padded_m)
    forward = prepare_forward(x_pad, w, device)
    dgrad = prepare_dgrad(dy_pad, w, device)
    wgrad = prepare_wgrad(dy_pad, x_pad, device)
    forward['padded_m'] = padded_m
    dgrad['padded_m'] = padded_m
    wgrad['padded_m'] = padded_m
    refs = _refs(x, w, dy, forward['qx'], forward['qw'], dgrad['qdy'], padded_m, device)
    thresholds = {'fp32': FP32_THRESHOLD, 'dequant': DEQUANT_THRESHOLD}
    def stats(entry):
        return {key: value for key, value in entry.items() if key != 'q'}
    bench_shape = {'M': m, 'K': k, 'N': n, 'padded_M': padded_m}
    forward['bench_shape'] = dgrad['bench_shape'] = wgrad['bench_shape'] = bench_shape
    case = {'case_id': case_id, 'shape': {'M': m, 'K': k, 'N': n},
            'padding': plan, 'all_zero': all_zero,
            'quantization': {'x': stats(forward['qx']), 'w': stats(forward['qw']),
                             'dy': stats(dgrad['qdy'])},
            'formats': {'forward': 'e4m3fn x e4m3fn',
                        'dgrad': 'e5m2 x e4m3fn', 'wgrad': 'e5m2 x e4m3fn'},
            'thresholds': thresholds, 'ops': {}, 'benchmarks': {}}
    try:
        out_f = call_prepared(forward)
        entry, got = _check_op('forward', out_f, refs['fp32']['forward'], refs['dequant']['forward'],
                               m, thresholds, all_zero)
        entry['dispatch'] = _dispatch(forward, out_f)
        case['ops']['forward'] = entry
        out_dx = call_prepared(dgrad)
        entry, got_dx = _check_op('dgrad', out_dx, refs['fp32']['dgrad'], refs['dequant']['dgrad'],
                                  m, thresholds, all_zero)
        entry['dispatch'] = _dispatch(dgrad, out_dx)
        case['ops']['dgrad'] = entry
        out_dw = call_prepared(wgrad)
        entry, got_dw = _check_op('wgrad', out_dw, refs['fp32']['wgrad'], refs['dequant']['wgrad'],
                                  None, thresholds, all_zero)
        entry['dispatch'] = _dispatch(wgrad, out_dw)
        case['ops']['wgrad'] = entry
        case['status'] = 'passed' if all(v['status'] == 'passed' for v in case['ops'].values()) else 'failed'
        if case['status'] == 'passed' and not all_zero:
            case['benchmarks']['forward'] = _bench_op(
                'forward', forward, None, lambda: _forward_quant_inclusive(x_pad, w),
                lambda: _bf16_forward(x, w), refs['fp32']['forward'], m, device)
            case['benchmarks']['dgrad'] = _bench_op(
                'dgrad', dgrad, None, lambda: _dgrad_quant_inclusive(dy_pad, w),
                lambda: _bf16_dgrad(dy, w), refs['fp32']['dgrad'], m, device)
            case['benchmarks']['wgrad'] = _bench_op(
                'wgrad', wgrad, None, lambda: _wgrad_quant_inclusive(dy_pad, x_pad),
                lambda: _bf16_wgrad(dy, x), refs['fp32']['wgrad'], None, device)
            if any(v.get('status') == 'failed' for v in case['benchmarks'].values()):
                case['status'] = 'failed'
    except Exception as exc:
        case['status'] = 'failed'
        case['first_failure'] = f'{type(exc).__name__}: {exc}'
    return case


def _profile(prepared, device):
    try:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True) as prof:
            call_prepared(prepared)
        torch.cuda.synchronize(device)
        keys = []
        for item in prof.key_averages():
            if item.device_time_total > 0:
                keys.append({'key': item.key, 'device_time_us': item.device_time_total})
        return {'status': 'passed', 'operator_keys': keys[:20],
                'table_excerpt': prof.key_averages().table(
                    sort_by='self_device_time_total', row_limit=20)[:4000]}
    except Exception as exc:
        return {'status': 'not_run', 'error': f'{type(exc).__name__}: {exc}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda',), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(root):
        parser.error('output must be outside the repository')
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if not hasattr(torch, '_scaled_mm'):
        parser.error('torch._scaled_mm unavailable')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    report = {
        'schema_version': 1, 'task': 'T0007', 'status': 'failed',
        'device': args.device, 'seed': args.seed,
        'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'capability': list(torch.cuda.get_device_capability(0)),
        'scaled_mm_schema': str(torch.ops.aten._scaled_mm.default._schema),
        'float8_dtypes': sorted(name for name in dir(torch) if 'float8' in name),
        'profiler_available': hasattr(torch, 'profiler'),
        'thresholds': {'fp32_relative_frobenius': FP32_THRESHOLD,
                       'dequantized_operand_relative_frobenius': DEQUANT_THRESHOLD},
        'benchmark_method': 'prequantized vs quantization-inclusive vs BF16; 5 warmups, 20 synchronized measured iterations',
        'cases': [], 'first_failure': None,
        'limitations': [
            'Operator feasibility only; not model training, convergence or language quality.',
            'FP8 output is BF16; no FP8-only accumulation claim.',
            'No Tensor Core instruction claim without profiler kernel verification.',
            'Synthetic independent normal inputs; no production shapes or corpus.',
            'Quantization is current per-tensor scaling only; no delayed scaling, MXFP8, NVFP4 or block scaling.',
            'No package, CUDA driver/toolkit or model/harness change.',
        ],
    }
    try:
        try:
            quantize_per_tensor(torch.tensor([float('inf')]), torch.float8_e4m3fn)
            report['nonfinite_rejection'] = {'status': 'failed', 'error': 'nonfinite input accepted'}
        except ValueError as exc:
            report['nonfinite_rejection'] = {'status': 'passed', 'exception': f'ValueError: {exc}'}
        index = 0
        for m, k, n in LARGE_SHAPES:
            index += 1
            report['cases'].append(_process_case(f'M{m}_K{k}_N{n}', m, k, n,
                                                 args.seed + index, device))
        for m, k, n in SMALL_SHAPES:
            index += 1
            report['cases'].append(_process_case(f'M{m}_K{k}_N{n}_padded', m, k, n,
                                                 args.seed + index, device))
        report['cases'].append(_process_case('all_zero_M16_K384_N512', 16, 384, 512,
                                             args.seed + 100, device, all_zero=True))
        failures = [c for c in report['cases'] if c['status'] != 'passed']
        if failures:
            report['first_failure'] = failures[0].get('first_failure') or failures[0]['case_id']
        report['status'] = 'passed' if not failures and report['nonfinite_rejection']['status'] == 'passed' else 'failed'
        if report['status'] == 'passed':
            largest = next((c for c in report['cases'] if c['shape']['M'] == 512), report['cases'][0])
            # Recreate the representative prepared operands for profiling.
            x, w, _ = _case_inputs(largest['shape']['M'], largest['shape']['K'], largest['shape']['N'], 999, device)
            report['profiler'] = _profile(prepare_forward(x, w, device), device)
    except Exception as exc:
        report['status'] = 'failed'
        report['first_failure'] = f'{type(exc).__name__}: {exc}'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'first_failure': report['first_failure'],
                      'cases': len(report['cases']),
                      'passed_cases': sum(c['status'] == 'passed' for c in report['cases'])}, indent=2))
    sys.exit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
