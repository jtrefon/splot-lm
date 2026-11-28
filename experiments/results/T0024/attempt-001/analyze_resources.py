"""Aggregate T0024 numerical eligibility and CUDA resource cases."""
import argparse
import json
import statistics
from pathlib import Path

METHODS = ('scan', 'chunk16', 'chunk32', 'chunk64')
LENGTHS = (512, 2048)


def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    d = args.dir
    numerical = json.loads((d / 'numerical.json').read_text())
    cases = {(length, method): json.loads((d / f'resource-{length}-{method}.json').read_text())
             for length in LENGTHS for method in METHODS}

    numerical_inputs = {}
    for case in numerical['cases']:
        if case['mode'] == 'fp32' and case['seed'] == 42 and case['decay'] == 'ordinary':
            numerical_inputs[case['length']] = case['input_sha256']
    eligibility = numerical['eligibility']

    cases_out = {}
    for length in LENGTHS:
        cases_out[str(length)] = {}
        for method in METHODS:
            case = cases[(length, method)]
            if method != 'scan':
                if not eligibility[str(length)][method.replace('chunk', '')]:
                    raise ValueError(f'{length}/{method} was not numerically eligible')
            if case['status'] != 'passed' or case['gate']['status'] != 'passed':
                raise ValueError(f'{length}/{method} resource gate failed')
            if case['input']['sha256'] != numerical_inputs[length]:
                raise ValueError(f'{length}/{method} input digest differs from numerical case')
            if len(case['repeats']) != 3:
                raise ValueError(f'{length}/{method} lacks three repeats')
            for repeat in case['repeats']:
                if (repeat['comparison']['status'] != 'passed' or not repeat['finite']
                        or not repeat['digest_matches_gate']):
                    raise ValueError(f'{length}/{method} repeat gate failed')
            base = case['resources']['input_baseline']
            repeated = case['repeats']
            row = {
                'method': method,
                'input_sha256': case['input']['sha256'],
                'input_baseline_allocated_bytes': base['allocated_bytes'],
                'input_baseline_reserved_bytes': base['reserved_bytes'],
                'gate_status': case['gate']['status'],
                'gate_worst': {
                    name: {'rel_l2': item['rel_l2'], 'max_abs': item['max_abs'],
                           'status': item['status']}
                    for name, item in case['gate']['components'].items()
                },
                'warmup': case['warmup'],
                'repeats': repeated,
                'forward_seconds': stats([x['forward_seconds'] for x in repeated]),
                'backward_seconds': stats([x['backward_seconds'] for x in repeated]),
                'total_seconds': stats([x['total_seconds'] for x in repeated]),
                'peak_allocated_bytes': stats([x['peak_allocated_bytes'] for x in repeated]),
                'peak_reserved_bytes': stats([x['peak_reserved_bytes'] for x in repeated]),
                'increment_over_input_allocated_bytes': stats(
                    [x['peak_allocated_bytes'] - base['allocated_bytes'] for x in repeated]),
                'increment_over_input_reserved_bytes': stats(
                    [x['peak_reserved_bytes'] - base['reserved_bytes'] for x in repeated]),
                'baseline_allocated_bytes': stats([x['baseline_allocated_bytes'] for x in repeated]),
                'baseline_reserved_bytes': stats([x['baseline_reserved_bytes'] for x in repeated]),
                'reserved_first_repeat_bytes': repeated[0]['peak_reserved_bytes'],
                'reserved_repeats_2_3_bytes': [x['peak_reserved_bytes'] for x in repeated[1:]],
            }
            cases_out[str(length)][method] = row

    paired = {}
    for length in LENGTHS:
        scan_total = cases_out[str(length)]['scan']['total_seconds']['median']
        scan_forward = cases_out[str(length)]['scan']['forward_seconds']['median']
        scan_backward = cases_out[str(length)]['scan']['backward_seconds']['median']
        paired[str(length)] = {}
        for method in ('chunk16', 'chunk32', 'chunk64'):
            row = cases_out[str(length)][method]
            paired[str(length)][method] = {
                'scan_over_chunk_total': scan_total / row['total_seconds']['median'],
                'chunk_over_scan_total': row['total_seconds']['median'] / scan_total,
                'scan_over_chunk_forward': scan_forward / row['forward_seconds']['median'],
                'scan_over_chunk_backward': scan_backward / row['backward_seconds']['median'],
                'scan_peak_allocated_over_chunk': (
                    cases_out[str(length)]['scan']['peak_allocated_bytes']['median'] /
                    row['peak_allocated_bytes']['median']),
            }

    fastest = {}
    for length in LENGTHS:
        fastest[str(length)] = min(
            METHODS,
            key=lambda method: cases_out[str(length)][method]['total_seconds']['median'])

    numerical_worst = {'fp64_rel_l2': 0.0, 'fp64_max_abs': 0.0,
                       'fp32_rel_l2': 0.0, 'fp32_max_abs': 0.0}
    for case in numerical['cases']:
        for chunk, result in case['chunks'].items():
            if result['status'] == 'skipped':
                continue
            for component in result['components'].values():
                if case['mode'] == 'fp64':
                    numerical_worst['fp64_rel_l2'] = max(numerical_worst['fp64_rel_l2'],
                                                         component['rel_l2'])
                    numerical_worst['fp64_max_abs'] = max(numerical_worst['fp64_max_abs'],
                                                          component['max_abs'])
                else:
                    numerical_worst['fp32_rel_l2'] = max(numerical_worst['fp32_rel_l2'],
                                                         component['rel_l2'])
                    numerical_worst['fp32_max_abs'] = max(numerical_worst['fp32_max_abs'],
                                                          component['max_abs'])

    report = {
        'status': 'passed',
        'task': 'T0024',
        'attempt': 1,
        'lengths': list(LENGTHS),
        'methods': list(METHODS),
        'numerical': {
            'cases': numerical['summary']['cases'],
            'chunk_outcomes': numerical['summary']['chunk_outcomes'],
            'passed': numerical['summary']['passed_chunks'],
            'failed': numerical['summary']['failed_chunks'],
            'skipped': numerical['summary']['skipped_chunks'],
            'eligibility': eligibility,
            'worst': numerical_worst,
            'oracle_limitations': numerical['oracle_limitations'],
        },
        'cases': cases_out,
        'paired': paired,
        'fastest_component_method': fastest,
        'allocator_state_note': (
            'A pre-timing reference scan can leave allocator cache reserved. '
            'Absolute peak allocated is stable across repeats; the first repeat peak reserved '
            'may include that retained cache, while repeats 2-3 show the method-specific '
            'reserved peak. Both absolute peaks and increments over the recorded input baseline '
            'are reported; no retry was performed.'),
        'conclusion': (
            'All three chunk sizes preserved outputs, carried state and all six input gradients '
            'within the preregistered FP32 thresholds at both lengths. In this isolated component '
            'measurement, chunk64 had the lowest median synchronized forward+backward time at both '
            'lengths; the ordering was chunk64 < chunk32 < chunk16 < scan. This is component-only '
            'evidence, not a full-model or production-kernel speed claim.'),
        'next_question': (
            'Integrate the numerically qualified chunk composition behind an opt-in GDN algorithm '
            'flag in a separate reviewed IMP and compare full Nano forward/backward at T512/T2048 '
            'against scan with exact full-model loss/gradient gates and full-model memory/timing; '
            'no production kernel claim from this component result.'),
        'limitations': [
            'Two lengths, one seed and three synchronized repeats; descriptive component timing only.',
            'The scan reference is a Python recurrence; the chunk path is multiple triangular solves. Differences include launch/Python overhead and are not kernel-level attribution.',
            'Peak memory is allocator data; first-repeat reserved memory can include retained cache from the pre-timing reference scan.',
            'The existing delta_chunk is a small finite-log-decay oracle, not a production kernel.',
            'No full-model, sustained-throughput, quality, training or architecture-superiority claim.',
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'status': report['status'],
        'numerical_passed': report['numerical']['passed'],
        'fastest': report['fastest_component_method'],
        'paired': report['paired'],
        'allocator_note': report['allocator_state_note'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
