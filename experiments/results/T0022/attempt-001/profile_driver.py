"""T0022 bounded BF16 backward-attribution driver.

Reuses the pinned T0020/T0012 context_probe model/config/warmup/input/objective
helpers. Adds control-versus-profiled digest capture and a bounded torch.profiler
summary. It does not modify the reference source, config, audit or warmup.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from context_probe import (_assert_pre_update, _move_inputs, _partition_summary,
                           _precheck, _setup, _warmup, build_contiguous_case,
                           load_config, tensor_digest)
from experiment import assert_post_update_finite
from optimizers import optimizer_state_inventory
from splot import training_objective

PARAMETERS = 43055076
MIN_GPU_BYTES = 20480 * 1024 * 1024
MIN_HOST_RAM_BYTES = 12 * 1024 ** 3
MIN_SCRATCH_BYTES = 8 * 1024 ** 3


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def meminfo_available_bytes():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        fields[key] = int(value.strip().split()[0]) * 1024
    return fields.get('MemAvailable', 0), fields.get('MemTotal', 0)


def memory_state(device):
    free, total = torch.cuda.mem_get_info(device)
    return {
        'free_bytes': free,
        'total_bytes': total,
        'allocated_bytes': torch.cuda.memory_allocated(device),
        'reserved_bytes': torch.cuda.memory_reserved(device),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
    }


def all_tensors_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_tensors_finite(item) for item in value)
    return True


def canonical_optimizer(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return {'kind': 'tensor', 'dtype': str(tensor.dtype),
                'shape': list(tensor.shape),
                'sha256': sha(tensor.numpy().tobytes())}
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: f'{type(item[0]).__name__}:{item[0]}')
        return {'kind': 'dict', 'items': [[f'{type(key).__name__}:{key}', canonical_optimizer(item)]
                                          for key, item in items]}
    if isinstance(value, (list, tuple)):
        return {'kind': 'list', 'items': [canonical_optimizer(item) for item in value]}
    if isinstance(value, bool):
        return {'kind': 'bool', 'value': value}
    if isinstance(value, int):
        return {'kind': 'int', 'value': value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('nonfinite scalar in optimizer state')
        return {'kind': 'float', 'value': repr(value)}
    if isinstance(value, str):
        return {'kind': 'str', 'value': value}
    if value is None:
        return {'kind': 'none'}
    raise TypeError(f'unsupported optimizer value: {type(value).__name__}')


def optimizer_digest(optimizer):
    canonical = canonical_optimizer(optimizer.state_dict())
    payload = json.dumps(canonical, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return sha(payload)


def gradient_inventory(model):
    named = []
    none_gradients = []
    dtypes = set()
    elements = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            none_gradients.append(name)
            continue
        if parameter.grad.dtype != torch.float32:
            raise TypeError(f'non-FP32 gradient for {name}: {parameter.grad.dtype}')
        named.append((name, parameter.grad))
        dtypes.add(str(parameter.grad.dtype))
        elements += parameter.grad.numel()
    return {
        'gradient_count': len(named),
        'gradient_elements': elements,
        'gradient_dtypes': sorted(dtypes),
        'none_gradients': none_gradients,
        'gradient_digest': tensor_digest(named),
    }


def profile_summary(prof):
    events = list(prof.events())
    averages = list(prof.key_averages())

    def record(item):
        return {
            'key': item.key,
            'count': int(item.count),
            'self_cpu_us': float(getattr(item, 'self_cpu_time_total', 0.0)),
            'cpu_total_us': float(getattr(item, 'cpu_time_total', 0.0)),
            'self_device_us': float(getattr(item, 'self_device_time_total', 0.0)),
            'device_total_us': float(getattr(item, 'device_time_total', 0.0)),
        }

    top_cpu = sorted(averages, key=lambda item: getattr(item, 'self_cpu_time_total', 0.0),
                     reverse=True)[:30]
    top_device = sorted(averages, key=lambda item: getattr(item, 'self_device_time_total', 0.0),
                        reverse=True)[:30]
    return {
        'activities': ['cpu', 'cuda'],
        'record_shapes': False,
        'profile_memory': False,
        'with_stack': False,
        'units': 'microseconds',
        'total_events': len(events),
        'key_average_count': len(averages),
        'top_self_cpu': [record(item) for item in top_cpu],
        'top_self_device': [record(item) for item in top_device],
        'module_attribution': 'unassigned',
        'module_attribution_note': 'Top keys are generic profiler/ATen labels; no reliable GDN, attention, expert or optimizer module mapping was inferred.',
    }


def source_record(source_root):
    root = Path(source_root)
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    probe = root / 'tools/context_probe.py'
    config = root / 'configs/experiments/splot-nano-synthetic-bf16-muon.json'
    return {
        'source_commit': head,
        'source_clean': not bool(dirty),
        'probe_sha256': file_sha(probe),
        'config': 'configs/experiments/splot-nano-synthetic-bf16-muon.json',
        'config_sha256': file_sha(config),
    }


def run_case(source_root, mode, length, seed, out):
    if mode not in ('control', 'profile'):
        raise ValueError('mode must be control or profile')
    if length not in (512, 2048):
        raise ValueError('T0022 requires length 512 or 2048')
    if seed != 42:
        raise ValueError('T0022 requires seed 42')
    source = Path(source_root)
    out = Path(out)
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError('output must be outside the source checkout')
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    host_available, host_total = meminfo_available_bytes()
    scratch_free = shutil.disk_usage(out.parent).free
    gpu_free, gpu_total = torch.cuda.mem_get_info(device)
    resources = {
        'host_ram_available_bytes': host_available,
        'host_ram_total_bytes': host_total,
        'scratch_free_bytes': scratch_free,
        'gpu_free_bytes': gpu_free,
        'gpu_total_bytes': gpu_total,
        'minimum_host_ram_bytes': MIN_HOST_RAM_BYTES,
        'minimum_scratch_bytes': MIN_SCRATCH_BYTES,
        'minimum_gpu_bytes': MIN_GPU_BYTES,
    }
    report = {
        'status': 'blocked',
        'stage': 'resources',
        'mode': mode,
        'length': length,
        'seed': seed,
        'precision': 'bf16',
        'device': 'cuda',
        'source': source_record(source),
        'resources': resources,
    }
    if host_available < MIN_HOST_RAM_BYTES or scratch_free < MIN_SCRATCH_BYTES or gpu_free < MIN_GPU_BYTES:
        report['error'] = 'resource precondition failed'
        out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        return report

    stage = 'warmup'
    try:
        config_path, cfg = load_config('bf16')
        report['config'] = str(config_path)
        report['resources']['free_before_case'] = memory_state(device)
        warmup = _warmup(cfg, 'bf16', device, length, seed)
        report['warmup'] = warmup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report['resources']['free_after_warmup'] = memory_state(device)

        stage = 'measured_setup'
        setup = _setup(cfg, 'bf16', device)
        inputs, labels, case = build_contiguous_case(length, seed)
        report['input'] = case
        report['identity'] = {
            'initialization_sha256': setup['initialization_sha256'],
            'input_sha256': case['input_sha256'],
            'partition_plan_sha256': _partition_summary(setup['optimizer_policy'])['plan_sha256'],
            'parameters': setup['parameters'],
        }
        report['checks'] = {
            'parameters_exact': setup['parameters'] == PARAMETERS,
            'warmup_status': warmup['status'] == 'passed',
            'warmup_init_equal': warmup['initialization_sha256'] == setup['initialization_sha256'],
            'warmup_input_equal': warmup['input_sha256'] == case['input_sha256'],
            'warmup_partition_equal': warmup['partition_plan_sha256'] ==
                                      _partition_summary(setup['optimizer_policy'])['plan_sha256'],
            'labels_exact': case['labels_exact'] is True,
        }

        stage = 'precheck'
        precheck = _precheck(setup['model'], inputs, labels, device, False)
        report['precheck'] = precheck
        report['checks']['precheck_finite'] = bool(precheck['logits_finite'] and
                                                   precheck['hidden_finite'] and
                                                   precheck['offset2_finite'])
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report['resources']['free_before_update'] = memory_state(device)

        model = setup['model']
        optimizer = setup['optimizer']
        stage = 'measured_update'
        phases = {}
        measured_start = time.perf_counter()
        phase = time.perf_counter()
        moved_inputs, moved_labels = _move_inputs(inputs, labels, device)
        torch.cuda.synchronize(device)
        phases['move_inputs'] = time.perf_counter() - phase

        phase = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        phases['zero_grad'] = time.perf_counter() - phase

        phase = time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            objective = training_objective(model, moved_inputs, moved_labels, mode='A2', loops=2)
        torch.cuda.synchronize(device)
        phases['forward'] = time.perf_counter() - phase

        phase = time.perf_counter()
        if mode == 'profile':
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=False, profile_memory=False, with_stack=False) as prof:
                objective.total.backward()
                torch.cuda.synchronize(device)
            report['profile'] = profile_summary(prof)
        else:
            objective.total.backward()
            torch.cuda.synchronize(device)
        phases['backward'] = time.perf_counter() - phase

        phase = time.perf_counter()
        _assert_pre_update(model, objective, False)
        phases['pre_update_checks'] = time.perf_counter() - phase

        phase = time.perf_counter()
        gradients = gradient_inventory(model)
        phases['gradient_digest'] = time.perf_counter() - phase

        phase = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        phases['optimizer'] = time.perf_counter() - phase

        phase = time.perf_counter()
        assert_post_update_finite(model, optimizer, 'measured post-update')
        phases['post_update_checks'] = time.perf_counter() - phase
        measured_seconds = time.perf_counter() - measured_start

        phase = time.perf_counter()
        losses = {name: float(getattr(objective, name).detach())
                  for name in ('total', 'final_ntp', 'early_ntp', 'offset2', 'balance')}
        model_digest = tensor_digest(
            [('param:' + name, value) for name, value in model.named_parameters()] +
            [('buffer:' + name, value) for name, value in model.named_buffers()])
        optimizer_digest_value = optimizer_digest(optimizer)
        state = optimizer_state_inventory(optimizer)
        digest_seconds = time.perf_counter() - phase
        phases['digests'] = digest_seconds
        phases['measured_update'] = measured_seconds

        model_finite = all(bool(torch.isfinite(value).all())
                           for value in list(model.parameters()) + list(model.buffers()))
        optimizer_finite = bool(state['total']['finite']) and all_tensors_finite(optimizer.state_dict())
        gradients_finite = all(bool(torch.isfinite(value).all())
                               for value in (parameter.grad for parameter in model.parameters()
                                             if parameter.grad is not None))
        report.update({
            'losses': losses,
            'gradients': gradients,
            'model': {
                'digest': model_digest,
                'parameter_count': sum(1 for _ in model.parameters()),
                'elements': sum(value.numel() for value in model.parameters()),
                'finite': model_finite,
            },
            'optimizer': {
                'digest': optimizer_digest_value,
                'state': state,
                'finite': optimizer_finite,
            },
            'phases_seconds': phases,
            'memory': {
                'after_update': memory_state(device),
            },
        })
        gc.collect()
        torch.cuda.empty_cache()
        report['memory']['free_after_cleanup'] = memory_state(device)
        report['checks'].update({
            'losses_finite': all(math.isfinite(value) for value in losses.values()),
            'gradients_finite': gradients_finite,
            'model_finite': model_finite,
            'optimizer_finite': optimizer_finite,
        })
        required = ('parameters_exact', 'warmup_status', 'warmup_init_equal',
                    'warmup_input_equal', 'warmup_partition_equal', 'labels_exact',
                    'precheck_finite', 'losses_finite', 'gradients_finite',
                    'model_finite', 'optimizer_finite')
        report['status'] = 'passed' if all(report['checks'][key] for key in required) else 'failed'
        report['stage'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['stage'] = stage
        report['error'] = f'{type(exc).__name__}: {exc}'
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('control', 'profile'), required=True)
    parser.add_argument('--length', type=int, choices=(512, 2048), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--instruction-commit', default=None)
    args = parser.parse_args()
    report = run_case(args.source_root, args.mode, args.length, args.seed, args.out)
    if args.instruction_commit:
        data = json.loads(args.out.read_text())
        data['instruction_commit'] = args.instruction_commit
        args.out.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'status': report.get('status'),
        'mode': report.get('mode'),
        'length': report.get('length'),
        'losses': report.get('losses'),
        'gradient_digest': (report.get('gradients') or {}).get('gradient_digest'),
        'model_digest': (report.get('model') or {}).get('digest'),
        'optimizer_digest': (report.get('optimizer') or {}).get('digest'),
        'profile_events': (report.get('profile') or {}).get('total_events'),
    }, sort_keys=True))
    return 0 if report.get('status') == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
