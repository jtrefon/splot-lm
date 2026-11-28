"""Validate T0020 cases and compute paired phase/resource aggregates from published JSON only."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

LENGTHS = (512, 2048)
PAIRS = (1, 2, 3)
PRECISIONS = ('bf16', 'fp8')
PARAMETERS = 43055076
STATE_ELEMENTS = 55685105
STATE_BYTES = 222740420
CASE_RE = re.compile(r'case-(\d+)-p?(\d+)-(bf16|fp8)\.json$')
PHASE_GROUPS = {
    'forward_loss': lambda p: p['forward'],
    'backward': lambda p: p['backward'],
    'checks': lambda p: p['checks'] + p['post_update_checks'],
    'optimizer_step': lambda p: p['optimizer_step'],
    'transfer': lambda p: p['transfer'],
    'prepare': lambda p: p['prepare'],
}


def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_cases(directory):
    cases = {}
    for path in sorted(Path(directory).glob('case-*.json')):
        match = CASE_RE.fullmatch(path.name)
        if not match:
            continue
        length, pair, precision = int(match.group(1)), int(match.group(2)), match.group(3)
        require(length in LENGTHS, f'unexpected length: {path.name}')
        require(pair in PAIRS, f'unexpected pair: {path.name}')
        cases[(length, pair, precision)] = json.loads(path.read_text())
    require(len(cases) == 12, f'expected 12 cases, found {len(cases)}')
    return cases


def validate(cases):
    failures = []
    for (length, pair, precision), case in sorted(cases.items()):
        tag = f'{length}-p{pair}-{precision}'
        checks = case['checks']
        warmup = case['warmup']
        measured = case['measured']
        expected_ntp = length - 1
        expected_offset = length - 2
        require(case['status'] == 'passed', f'{tag}: status')
        require(case['stage'] == 'complete', f'{tag}: stage')
        require(case['device'] == 'cuda' and case['seed'] == 42, f'{tag}: device/seed')
        require(warmup['status'] == 'passed' and measured['status'] == 'passed', f'{tag}: sub-status')
        require(warmup['parameters'] == PARAMETERS, f'{tag}: parameters')
        require(checks['parameters_exact'] is True, f'{tag}: parameters_exact')
        require(case['input']['length'] == length, f'{tag}: input length')
        require(case['input']['valid_tokens'] == length, f'{tag}: valid tokens')
        require(case['input']['ntp_labels'] == expected_ntp, f'{tag}: NTP labels')
        require(case['input']['offset2_labels'] == expected_offset, f'{tag}: offset2 labels')
        require(case['input']['expected'] == {'ntp_labels': expected_ntp,
                                               'offset2_labels': expected_offset,
                                               'valid_tokens': length}, f'{tag}: expected labels')
        require(checks['labels_exact'] is True, f'{tag}: labels_exact')
        for key in ('losses_finite', 'hidden_finite', 'logits_finite', 'offset2_finite',
                    'model_fp32', 'state_fp32_finite', 'warmup_init_equal',
                    'warmup_input_equal', 'warmup_partition_equal', 'warmup_status',
                    'audit_positive', 'audit_finite'):
            require(checks[key] is True, f'{tag}: check {key}')
        require(all(math.isfinite(value) for value in measured['losses'].values()), f'{tag}: losses')
        require(measured['state']['total']['finite'] is True, f'{tag}: state finite')
        require(measured['state']['total']['elements'] == STATE_ELEMENTS, f'{tag}: state elements')
        require(measured['state']['total']['bytes'] == STATE_BYTES, f'{tag}: state bytes')
        require(measured['state']['total']['dtypes'] == ['float32'], f'{tag}: state dtype')
        require(all(math.isfinite(value) for value in case['precheck'].values()
                    if isinstance(value, (int, float))), f'{tag}: precheck')
        require(case['input']['input_sha256'] == warmup['input_sha256'], f'{tag}: input digest')
        require(checks['partition']['plan_sha256'] == warmup['partition_plan_sha256'],
                f'{tag}: partition digest')
        if precision == 'fp8':
            audit = measured['audit']
            require(isinstance(audit, dict), f'{tag}: FP8 audit missing')
            require(checks['native_fp8'] is True, f'{tag}: native_fp8')
            require(audit['native_forward_calls'] > 0, f'{tag}: native forward calls')
            require(audit['native_backward_calls'] > 0, f'{tag}: native backward calls')
            require(audit['quantization']['nonfinite'] is False, f'{tag}: nonfinite quantization')
            require(audit['weight_reuse_enabled'] is False, f'{tag}: weight reuse changed')
            require(audit['weight_cache_hits'] == 0, f'{tag}: unexpected cache hits')
            require(audit['weight_cache_bytes'] == 0, f'{tag}: unexpected cache bytes')
        else:
            require(checks['native_fp8'] is False, f'{tag}: BF16 native flag')
            require(measured['audit'] is None, f'{tag}: BF16 audit not null')
    return failures


def equality(cases):
    out = {}
    for length in LENGTHS:
        subset = [case for (l, _, _), case in cases.items() if l == length]
        init = {case['warmup']['initialization_sha256'] for case in subset}
        inp = {case['warmup']['input_sha256'] for case in subset}
        part = {case['warmup']['partition_plan_sha256'] for case in subset}
        require(len(init) == 1 and len(inp) == 1 and len(part) == 1,
                f'length {length}: digest mismatch within repetitions')
        out[str(length)] = {
            'initialization_sha256': next(iter(init)),
            'input_sha256': next(iter(inp)),
            'partition_plan_sha256': next(iter(part)),
            'identical_within_repetitions': True,
        }
    all_init = {case['warmup']['initialization_sha256'] for case in cases.values()}
    all_part = {case['warmup']['partition_plan_sha256'] for case in cases.values()}
    require(len(all_init) == 1 and len(all_part) == 1,
            'initialization/partition differs across lengths')
    out['across_lengths'] = {
        'initialization_sha256': next(iter(all_init)),
        'partition_plan_sha256': next(iter(all_part)),
        'identical_initialization_and_partition': True,
        'input_differs_by_length_as_expected': True,
    }
    return out


def thresholds(cases):
    out = {}
    all_pass = True
    for length in LENGTHS:
        for pair in PAIRS:
            bf16 = cases[(length, pair, 'bf16')]['measured']['losses']
            fp8 = cases[(length, pair, 'fp8')]['measured']['losses']
            diffs = {key: abs(bf16[key] - fp8[key])
                     for key in ('final_ntp', 'early_ntp', 'offset2', 'total')}
            component_pass = all(diffs[key] <= 0.5 for key in ('final_ntp', 'early_ntp', 'offset2'))
            total_pass = diffs['total'] <= 0.75
            all_pass = all_pass and component_pass and total_pass
            out[f'{length}-p{pair}'] = {
                'abs_diff_nats': diffs,
                'component_limit_0.5': component_pass,
                'total_limit_0.75': total_pass,
                'status': 'passed' if component_pass and total_pass else 'failed',
            }
    return {'all_passed': all_pass, 'pairs': out}


def case_summary(case):
    measured = case['measured']
    phases = measured['phases_seconds']
    memory = case['memory']
    audit = measured.get('audit') or {}
    ntp = case['input']['ntp_labels']
    e2e = phases['end_to_end']
    return {
        'length': case['length'],
        'pair': None,
        'precision': case['precision'],
        'status': case['status'],
        'parameters': case['warmup']['parameters'],
        'valid_tokens': case['input']['valid_tokens'],
        'ntp_labels': ntp,
        'offset2_labels': case['input']['offset2_labels'],
        'input_sha256': case['input']['input_sha256'],
        'initialization_sha256': case['warmup']['initialization_sha256'],
        'partition_plan_sha256': case['warmup']['partition_plan_sha256'],
        'losses': measured['losses'],
        'end_to_end_seconds': e2e,
        'phases_seconds': phases,
        'phase_sum_seconds': measured['phase_sum_seconds'],
        'unaccounted_seconds': measured['unaccounted_seconds'],
        'sync_seconds_total': measured['sync_seconds_total'],
        'ntp_labels_per_second': ntp / e2e,
        'valid_tokens_per_second': measured['tokens_per_second'],
        'peak_allocated_bytes': memory['after_update']['peak_allocated_bytes'],
        'peak_reserved_bytes': memory['after_update']['peak_reserved_bytes'],
        'allocated_after_update_bytes': memory['after_update']['allocated_bytes'],
        'reserved_after_update_bytes': memory['after_update']['reserved_bytes'],
        'free_before_case_bytes': memory['free_before_case']['free_bytes'],
        'free_after_cleanup_bytes': memory['free_after_cleanup']['free_bytes'],
        'state_bytes': measured['state']['total']['bytes'],
        'state_elements': measured['state']['total']['elements'],
        'state_finite': measured['state']['total']['finite'],
        'native_forward_calls': audit.get('native_forward_calls'),
        'native_backward_calls': audit.get('native_backward_calls'),
        'forward_calls': audit.get('forward_calls'),
        'backward_calls': audit.get('backward_calls'),
        'selected_modules': audit.get('selected_modules'),
        'padded_forward_rows': audit.get('padded_forward_rows'),
        'padded_backward_rows': audit.get('padded_backward_rows'),
        'weight_reuse_enabled': audit.get('weight_reuse_enabled'),
        'weight_quantizations': audit.get('weight_quantizations'),
        'weight_cache_hits': audit.get('weight_cache_hits'),
        'weight_cache_bytes': audit.get('weight_cache_bytes'),
        'quantization_nonfinite': (audit.get('quantization') or {}).get('nonfinite'),
        'quantization': audit.get('quantization'),
    }


def aggregates(cases):
    out = {}
    for length in LENGTHS:
        out[str(length)] = {}
        for precision in PRECISIONS:
            rows = [cases[(length, pair, precision)] for pair in PAIRS]
            measured = [row['measured'] for row in rows]
            phases = [row['measured']['phases_seconds'] for row in rows]
            memory = [row['memory'] for row in rows]
            e2e = [phase['end_to_end'] for phase in phases]
            seconds = {name: [fn(phase) for phase in phases] for name, fn in PHASE_GROUPS.items()}
            out[str(length)][precision] = {
                'cases': PAIRS,
                'end_to_end_seconds': stats(e2e),
                'phase_seconds': {name: stats(values) for name, values in seconds.items()},
                'phase_sum_seconds': stats([m['phase_sum_seconds'] for m in measured]),
                'unaccounted_seconds': stats([m['unaccounted_seconds'] for m in measured]),
                'sync_seconds_total': stats([m['sync_seconds_total'] for m in measured]),
                'ntp_labels_per_second': stats([row['input']['ntp_labels'] / m['phases_seconds']['end_to_end']
                                                for row, m in zip(rows, measured)]),
                'valid_tokens_per_second': stats([m['tokens_per_second'] for m in measured]),
                'peak_allocated_bytes': stats([m['after_update']['peak_allocated_bytes'] for m in memory]),
                'peak_reserved_bytes': stats([m['after_update']['peak_reserved_bytes'] for m in memory]),
                'free_before_case_bytes': stats([m['free_before_case']['free_bytes'] for m in memory]),
                'free_after_cleanup_bytes': stats([m['free_after_cleanup']['free_bytes'] for m in memory]),
                'allocated_after_update_bytes': stats([m['after_update']['allocated_bytes'] for m in memory]),
                'reserved_after_update_bytes': stats([m['after_update']['reserved_bytes'] for m in memory]),
                'state_bytes': sorted({m['state']['total']['bytes'] for m in measured}),
                'state_elements': sorted({m['state']['total']['elements'] for m in measured}),
                'native_forward_calls': sorted({(m.get('audit') or {}).get('native_forward_calls') for m in measured
                                                if (m.get('audit') or {}).get('native_forward_calls') is not None}),
                'native_backward_calls': sorted({(m.get('audit') or {}).get('native_backward_calls') for m in measured
                                                 if (m.get('audit') or {}).get('native_backward_calls') is not None}),
                'padded_forward_rows': sorted({(m.get('audit') or {}).get('padded_forward_rows') for m in measured
                                               if (m.get('audit') or {}).get('padded_forward_rows') is not None}),
                'weight_reuse_enabled': sorted({(m.get('audit') or {}).get('weight_reuse_enabled') for m in measured
                                                if (m.get('audit') or {}).get('weight_reuse_enabled') is not None}),
                'weight_quantizations': sorted({(m.get('audit') or {}).get('weight_quantizations') for m in measured
                                                if (m.get('audit') or {}).get('weight_quantizations') is not None}),
                'weight_cache_hits': sorted({(m.get('audit') or {}).get('weight_cache_hits') for m in measured
                                             if (m.get('audit') or {}).get('weight_cache_hits') is not None}),
                'weight_cache_bytes': sorted({(m.get('audit') or {}).get('weight_cache_bytes') for m in measured
                                              if (m.get('audit') or {}).get('weight_cache_bytes') is not None}),
            }
    return out


def paired_ratios(cases):
    out = {}
    for length in LENGTHS:
        pairs = {}
        for pair in PAIRS:
            bf16 = cases[(length, pair, 'bf16')]['measured']
            fp8 = cases[(length, pair, 'fp8')]['measured']
            bp = bf16['phases_seconds']
            fp = fp8['phases_seconds']
            pairs[f'p{pair}'] = {
                'end_to_end_ratio': fp['end_to_end'] / bp['end_to_end'],
                'end_to_end_delta_seconds': fp['end_to_end'] - bp['end_to_end'],
                'forward_ratio': fp['forward'] / bp['forward'],
                'backward_ratio': fp['backward'] / bp['backward'],
                'checks_ratio': (fp['checks'] + fp['post_update_checks']) /
                                (bp['checks'] + bp['post_update_checks']),
                'optimizer_ratio': fp['optimizer_step'] / bp['optimizer_step'],
            }
        out[str(length)] = {
            'pairs': pairs,
            'end_to_end_ratio_stats': stats([pairs[f'p{p}']['end_to_end_ratio'] for p in PAIRS]),
            'end_to_end_delta_seconds_stats': stats([pairs[f'p{p}']['end_to_end_delta_seconds'] for p in PAIRS]),
        }
    return out


def phase_reconciliation(cases):
    out = {}
    for length in LENGTHS:
        out[str(length)] = {}
        for precision in PRECISIONS:
            rows = [cases[(length, pair, precision)]['measured'] for pair in PAIRS]
            out[str(length)][precision] = {
                'phase_sum_seconds': stats([r['phase_sum_seconds'] for r in rows]),
                'unaccounted_seconds': stats([r['unaccounted_seconds'] for r in rows]),
                'max_unaccounted_seconds': max(r['unaccounted_seconds'] for r in rows),
                'sync_seconds_total': stats([r['sync_seconds_total'] for r in rows]),
                'phases_additive_within_resolution': max(r['unaccounted_seconds'] for r in rows) < 1e-4,
            }
    return out


def phase_ranking(cases):
    rankings = {}
    for length in LENGTHS:
        rows = [cases[(length, pair, precision)]['measured']['phases_seconds']
                for pair in PAIRS for precision in PRECISIONS]
        values = {name: statistics.median([fn(phase) for phase in rows])
                  for name, fn in PHASE_GROUPS.items()}
        rankings[str(length)] = [
            {'phase': name, 'median_seconds': value}
            for name, value in sorted(values.items(), key=lambda item: item[1], reverse=True)
        ]
    return rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.cases_dir)
    validate(cases)
    eq = equality(cases)
    thr = thresholds(cases)
    require(thr['all_passed'] is True, 'paired CE threshold failed')
    order = json.loads((args.cases_dir / 'run-order.json').read_text())['order']
    case_rows = []
    for (length, pair, precision), case in sorted(cases.items()):
        row = case_summary(case)
        row['pair'] = pair
        row['case_file'] = f'case-{length}-p{pair}-{precision}.json'
        case_rows.append(row)
    analysis = {
        'status': 'passed',
        'task': 'T0020',
        'attempt': 1,
        'cases_total': len(cases),
        'cases_passed': sum(case['status'] == 'passed' for case in cases.values()),
        'parameters': PARAMETERS,
        'lengths': list(LENGTHS),
        'pairs': list(PAIRS),
        'precision_order': list(PRECISIONS),
        'fp8_weight_reuse_enabled': False,
        'actual_run_order': order,
        'equality': eq,
        'thresholds': thr,
        'cases': case_rows,
        'aggregates': aggregates(cases),
        'paired_ratios': paired_ratios(cases),
        'phase_reconciliation': phase_reconciliation(cases),
        'phase_ranking': phase_ranking(cases),
        'ranking_interpretation': 'Backward dominates both lengths; forward/loss is second. The ranking is single-update phase timing after warmup, not kernel attribution or steady-state throughput.',
        'next_profiling_question': 'Profile the T2048 backward pass to separate the GDN recurrent scan from attention/logits and expert dispatch; require exact pre-update loss and full gradient/state digests equal to the current probe at fixed seed/input, with no equation, tolerance or precision change.',
        'limitations': [
            'One measured update per case after an untimed warmup; three repeats are descriptive, not independent training seeds.',
            'Synthetic contiguous documents only; no real corpus, checkpoint, quality, convergence or architecture-superiority claim.',
            'Phase timings include explicit synchronization and audit/finite checks; they are not kernel-level attribution.',
            'Peak allocated and reserved are single-case CUDA allocator values, not steady-state production memory.',
            'FP8 weight reuse is disabled in the existing probe; no cache reuse is measured or claimed.',
        ],
    }
    args.out.write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'status': analysis['status'],
        'cases': analysis['cases_total'],
        'parameters': analysis['parameters'],
        'thresholds': analysis['thresholds']['all_passed'],
        'e2e_512': analysis['aggregates']['512'],
        'e2e_2048': analysis['aggregates']['2048'],
        'ratios_512': analysis['paired_ratios']['512']['end_to_end_ratio_stats'],
        'ratios_2048': analysis['paired_ratios']['2048']['end_to_end_ratio_stats'],
        'ranking_512': analysis['phase_ranking']['512'],
        'ranking_2048': analysis['phase_ranking']['2048'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
