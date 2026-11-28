"""Build bounded T0022 attribution aggregates from the four sanitized cases."""
import argparse
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def top_names(items, limit=10):
    return [item['key'] for item in items[:limit]]


def find_key(items, key):
    for item in items:
        if item['key'] == key:
            return item
    return None


def profile_summary(case, equality):
    profile = case['profile']
    phases = case['phases_seconds']
    control = equality['control_sha256']
    record = {
        'length': case['length'],
        'control_case_sha256': equality['control_sha256'],
        'profile_case_sha256': equality['profile_sha256'],
        'control_backward_seconds': equality.get('control_backward_seconds'),
        'profile_backward_seconds': phases['backward'],
        'control_measured_update_seconds': equality.get('control_measured_update_seconds'),
        'profile_measured_update_seconds': phases.get('measured_update'),
        'profile_to_control_backward_ratio': (phases['backward'] /
                                                equality['control_backward_seconds']),
        'total_events': profile['total_events'],
        'key_average_count': profile['key_average_count'],
        'units': profile['units'],
        'top_self_cpu': profile['top_self_cpu'][:10],
        'top_self_device': profile['top_self_device'][:10],
        'top_self_cpu_names': top_names(profile['top_self_cpu'], 10),
        'top_self_device_names': top_names(profile['top_self_device'], 10),
        'sum_top30_self_cpu_us': sum(item['self_cpu_us'] for item in profile['top_self_cpu']),
        'sum_top30_self_device_us': sum(item['self_device_us']
                                        for item in profile['top_self_device']),
        'sum_top30_self_cpu_seconds': sum(item['self_cpu_us']
                                          for item in profile['top_self_cpu']) / 1e6,
        'sum_top30_self_device_seconds': sum(item['self_device_us']
                                             for item in profile['top_self_device']) / 1e6,
        'module_attribution': 'unassigned',
        'module_attribution_note': ('Profiler keys are generic ATen/profiler labels. '
                                    'No reliable GDN, attention, expert or optimizer '
                                    'module mapping was inferred.'),
    }
    launch = find_key(profile['top_self_cpu'], 'cudaLaunchKernel')
    record['cuda_launch_kernel'] = None if launch is None else {
        'count': launch['count'],
        'self_cpu_us': launch['self_cpu_us'],
        'cpu_total_us': launch['cpu_total_us'],
        'self_device_us': launch['self_device_us'],
    }
    record['top30_self_device_seconds_over_profiled_backward'] = (
        record['sum_top30_self_device_seconds'] / phases['backward'])
    record['top30_self_cpu_seconds_over_profiled_backward'] = (
        record['sum_top30_self_cpu_seconds'] / phases['backward'])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    d = args.cases_dir
    cases = {}
    for length in (512, 2048):
        for mode in ('control', 'profile'):
            path = d / f'case-{length}-{mode}.json'
            cases[(length, mode)] = load(path)
    equalities = {length: load(d / f'equality-{length}.json') for length in (512, 2048)}
    for length in (512, 2048):
        require(cases[(length, 'control')]['status'] == 'passed', f'{length} control')
        require(cases[(length, 'profile')]['status'] == 'passed', f'{length} profile')
        require(equalities[length]['status'] == 'passed', f'{length} equality')
        equalities[length]['control_backward_seconds'] = (
            cases[(length, 'control')]['phases_seconds']['backward'])
        equalities[length]['control_measured_update_seconds'] = (
            cases[(length, 'control')]['phases_seconds']['measured_update'])
    profiles = {str(length): profile_summary(cases[(length, 'profile')], equalities[length])
                for length in (512, 2048)}
    common_cpu = sorted(set(profiles['512']['top_self_cpu_names']) &
                        set(profiles['2048']['top_self_cpu_names']))
    common_device = sorted(set(profiles['512']['top_self_device_names']) &
                           set(profiles['2048']['top_self_device_names']))
    report = {
        'status': 'passed',
        'task': 'T0022',
        'attempt': 1,
        'precision': 'bf16',
        'lengths': [512, 2048],
        'exactness': {
            '512': equalities[512],
            '2048': equalities[2048],
            'all_passed': equalities[512]['status'] == 'passed' and
                          equalities[2048]['status'] == 'passed',
            'control_profile_exact': True,
            't0020_reference_exact': True,
        },
        'profiles': profiles,
        'operator_attribution': {
            'level': 'generic ATen/profiler keys only',
            'common_top_self_cpu_names': common_cpu,
            'common_top_self_device_names': common_device,
            'module_attribution': 'unassigned',
            'module_attribution_note': ('No reliable attribution to GDN scan, attention/logits, '
                                        'expert dispatch, optimizer or quantization was possible '
                                        'from generic keys. No percentages by module are claimed.'),
            'confidence': 'low for module attribution; bounded operator-level only',
        },
        'profiler_perturbation': {
            'note': 'Profiled backward wall time includes profiler instrumentation and is much larger than the unprofiled control; it is not a performance measurement.',
            'ratios': {length: profiles[length]['profile_to_control_backward_ratio']
                       for length in ('512', '2048')},
            'top30_device_fraction_of_profiled_backward': {
                length: profiles[length]['top30_self_device_seconds_over_profiled_backward']
                for length in ('512', '2048')},
            'top30_cpu_fraction_of_profiled_backward': {
                length: profiles[length]['top30_self_cpu_seconds_over_profiled_backward']
                for length in ('512', '2048')},
        },
        'next_question': ('Scoped instrumentation IMP: tag backward operators by module/path '
                          '(GDN recurrent scan vs attention/logits vs expert dispatch) and rerun '
                          'this exact four-case equality/profiling harness. Exactness guard: full '
                          'gradient, model and optimizer digests must equal the current control at '
                          'both lengths; no equation, tolerance, precision or optimizer change.'),
        'limitations': [
            'Four single-update cases; no steady-state, convergence or production-throughput claim.',
            'torch.profiler adds large perturbation; instrumented latency is not a performance result.',
            'Only bounded top-30 key averages are published; full traces were not exported.',
            'Self CPU/device times overlap and are not additive; no module attribution is inferred from generic ATen names.',
            'No real data, checkpoint, Mini, kernel, package or reference-source change.',
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'status': report['status'],
        'exactness': report['exactness']['all_passed'],
        'profiles': {length: {
            'total_events': profiles[length]['total_events'],
            'control_backward_seconds': profiles[length]['control_backward_seconds'],
            'profile_backward_seconds': profiles[length]['profile_backward_seconds'],
            'top_cpu': profiles[length]['top_self_cpu_names'][:5],
            'top_device': profiles[length]['top_self_device_names'][:5],
            'launch': profiles[length]['cuda_launch_kernel'],
        } for length in ('512', '2048')},
        'next_question': report['next_question'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
