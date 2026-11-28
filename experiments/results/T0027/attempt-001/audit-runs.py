#!/usr/bin/env python3
"""T0027 per-run structural/numerical audit. Read-only; writes only its audit JSON."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

MANIFEST = 'dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55'
COMMIT = '003cf8db6126f0aaefdd599ac159f2503457f709'
SOURCE = 'c84bd8a6e94c5d0f4d429e44a43c7bffd9a6dfb9a263b9a3440126e23b4aa087'
TOKENIZER = '104d3927bb55cdd2a911dbe21c8e755264756cedb5663bd39365c3d4cba3864b'
LOGICAL = '70ba8d35d2ae334a8601f6389dc39208f3636937cb9dbfda639ead7ab30c052c'
ARMS = {
    'S': {'parameters': 36763620, 'gradient': 36615780, 'dormant': 147840,
          'family': 'splot', 'loops': 2},
    'P': {'parameters': 35527168, 'gradient': 35527168, 'dormant': 0,
          'family': 'dense-ntp', 'loops': 1},
    'C': {'parameters': 32381440, 'gradient': 32381440, 'dormant': 0,
          'family': 'dense-ntp', 'loops': 1},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--meta', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--arm', choices=tuple(ARMS), required=True)
    p.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    expected = ARMS[a.arm]
    run = json.loads((a.run_dir / 'run.json').read_text())
    cfg = json.loads(a.config.read_text())
    meta = json.loads(a.meta.read_text())
    matrix = json.loads(a.matrix.read_text())
    order = matrix['data_order'][str(a.seed)]['order']
    order_sha = matrix['data_order'][str(a.seed)]['sha256']
    checks = {}

    def check(name, condition, detail):
        checks[name] = {'status': 'passed' if bool(condition) else 'failed',
                        'detail': detail}

    hist = run.get('history', [])
    flat = [int(i) for row in hist for i in row.get('batch_indices', [])]
    actual_sha = digest(flat)
    check('exit_and_resources',
          meta.get('status') == 'completed' and meta.get('exit_code') == 0
          and meta.get('preflight', {}).get('gpu_free_mib', 0) >= 20480
          and meta.get('preflight', {}).get('host_ram_available_bytes', 0) >= 12 * 1024 ** 3
          and meta.get('preflight', {}).get('scratch_free_bytes', 0) >= 4 * 1024 ** 3
          and meta.get('wall_seconds', 0) > 0,
          f"exit={meta.get('exit_code')} gpu={meta.get('preflight', {}).get('gpu_free_mib')} MiB "
          f"ram={meta.get('preflight', {}).get('host_ram_available_bytes')} B "
          f"scratch={meta.get('preflight', {}).get('scratch_free_bytes')} B "
          f"wall={meta.get('wall_seconds')} s")
    check('source_identity', run.get('identity', {}).get('commit') == COMMIT,
          f"identity.commit={run.get('identity', {}).get('commit')}")
    check('config_identity',
          run.get('config') == cfg and run.get('identity', {}).get('settings') == cfg['training']
          and run.get('identity', {}).get('config_sha256') == digest(cfg),
          f"config_sha256={run.get('identity', {}).get('config_sha256')} derived={digest(cfg)}")
    check('dataset_identity',
          run.get('identity', {}).get('data_sha256') == digest(run.get('data_manifest', {}))
          and run.get('data_manifest', {}).get('manifest_sha256') == MANIFEST
          and run.get('data_manifest', {}).get('source_snapshot_sha256') == SOURCE
          and run.get('data_manifest', {}).get('tokenizer_sha256') == TOKENIZER
          and run.get('data_manifest', {}).get('logical_sha256') == LOGICAL
          and run.get('data_policy', {}).get('manifest_sha256') == MANIFEST
          and run.get('data_policy', {}).get('source_snapshot_sha256') == SOURCE
          and run.get('data_policy', {}).get('tokenizer_sha256') == TOKENIZER
          and run.get('data_policy', {}).get('logical_digest') == LOGICAL
          and run.get('identity', {}).get('environment', {}).get('dataset_manifest_sha256') == MANIFEST
          and run.get('identity', {}).get('environment', {}).get('dataset_source_snapshot_sha256') == SOURCE
          and run.get('identity', {}).get('environment', {}).get('dataset_tokenizer_sha256') == TOKENIZER,
          'manifest/source/tokenizer/logical pins and data digest')
    check('vocab_policy',
          run.get('data_manifest', {}).get('vocab_size') == 16384
          and run.get('data_manifest', {}).get('model_vocab_size') == 16384
          and run.get('data_policy', {}).get('model_vocab_size') == 16384
          and run.get('data_policy', {}).get('tokenizer_actual_vocab_size') == 16384
          and run.get('data_policy', {}).get('unused_token_id_count') == 0
          and run.get('data_policy', {}).get('unused_token_id_ranges') == []
          and run.get('data_policy', {}).get('test_used_for_training') is False
          and run.get('data_policy', {}).get('test_validated_only') is True,
          'actual V16384, zero unused IDs, test integrity-only')
    splits = run.get('data_manifest', {}).get('splits', {})
    check('split_counts',
          splits.get('train', {}).get('blocks') == 128
          and splits.get('train', {}).get('valid_tokens') == 46204
          and splits.get('train', {}).get('ntp_labels') == 46076
          and splits.get('train', {}).get('offset2_labels') == 45948
          and splits.get('validation', {}).get('blocks') == 16
          and splits.get('validation', {}).get('ntp_labels') == 5250
          and splits.get('test', {}).get('blocks') == 16,
          json.dumps(splits, sort_keys=True))
    check('data_order',
          len(hist) == 128 and flat == order and actual_sha == order_sha
          and sorted(flat) == list(range(128)) and len(set(flat)) == 128,
          f"steps={len(hist)} unique={len(set(flat))} sha256={actual_sha}")
    sum_valid = sum(int(row.get('valid_tokens', 0)) for row in hist)
    sum_ntp = sum(int(row.get('ntp_labels', 0)) for row in hist)
    sum_off2 = sum(int(row.get('offset2_labels_available', 0)) for row in hist)
    check('training_counts',
          sum_valid == 46204 and sum_ntp == 46076 and sum_off2 == 45948
          and run.get('total_valid_tokens') == sum_valid
          and run.get('total_ntp_labels') == sum_ntp,
          f"valid={sum_valid} ntp={sum_ntp} offset2_available={sum_off2}")
    finite_keys = ('objective', 'final_ntp', 'early_ntp', 'offset2', 'balance', 'step_seconds')
    finite_rows = all(finite(row.get(k)) for row in hist for k in finite_keys)
    positive_steps = all(row.get('step_seconds', 0) > 0 for row in hist)
    check('finite_training',
          finite_rows and positive_steps
          and all(finite(row.get('peak_cuda_allocated_bytes')) for row in hist)
          and all(finite(row.get('peak_cuda_reserved_bytes')) for row in hist),
          f"rows={len(hist)} finite_losses={finite_rows} positive_steps={positive_steps}")
    val = run.get('validation', {})
    n = val.get('ntp_labels')
    mean_ok = finite(val.get('mean_nll')) and finite(val.get('nll_sum')) and n == 5250 \
        and math.isclose(val['mean_nll'], val['nll_sum'] / n, rel_tol=1e-12, abs_tol=1e-12)
    ppl_ok = finite(val.get('perplexity')) and math.isclose(
        val['perplexity'], math.exp(val['mean_nll']), rel_tol=1e-12, abs_tol=1e-12)
    check('validation_recompute',
          n == 5250 and mean_ok and ppl_ok and finite(run.get('validation_seconds'))
          and run.get('validation_seconds', 0) > 0 and 'test' not in val,
          f"labels={n} nll_sum={val.get('nll_sum')} mean={val.get('mean_nll')} "
          f"ppl={val.get('perplexity')} seconds={run.get('validation_seconds')}")
    inv = run.get('parameter_inventory', {})
    gradient_ok = (inv.get('gradient_present_parameter_numel') == expected['gradient']
                   if a.arm != 'S' else
                   0 < inv.get('gradient_present_parameter_numel', 0) <= expected['gradient'])
    check('parameter_inventory',
          run.get('parameters') == expected['parameters']
          and inv.get('physical_parameter_numel') == expected['parameters']
          and gradient_ok
          and inv.get('mtp_parameter_numel') == expected['dormant']
          and inv.get('dormant_mtp_parameter_numel') == expected['dormant'],
          f"total={run.get('parameters')} inventory={json.dumps(inv, sort_keys=True)}")
    pol = run.get('precision_policy', {})
    check('precision_policy',
          pol.get('requested') == 'bf16' and pol.get('policy') == 'bf16-autocast-fp32-master'
          and pol.get('autocast_dtype') == 'bfloat16'
          and run.get('identity', {}).get('environment', {}).get('master_dtype') == 'float32'
          and pol.get('bf16_supported') is True
          and run.get('identity', {}).get('environment', {}).get('tf32_enabled') is False
          and run.get('identity', {}).get('environment', {}).get('deterministic_algorithms') is True
          and run.get('fp8_policy') is None,
          json.dumps(pol, sort_keys=True))
    op = run.get('optimizer_policy', {})
    top = cfg.get('training', {}).get('optimizer', {})
    check('optimizer_policy',
          op.get('name') == 'muon' and op.get('muon') == top.get('muon')
          and op.get('adamw') == top.get('adamw')
          and run.get('identity', {}).get('environment', {}).get('optimizer') == 'muon',
          f"name={op.get('name')} implementation={op.get('implementation')}")
    cs = run.get('checkpoint_stats', {})
    check('checkpoint_schedule',
          run.get('identity', {}).get('checkpoint_interval') == 16
          and run.get('identity', {}).get('environment', {}).get('checkpoint_interval') == 16
          and run.get('checkpoint_policy', {}).get('interval') == 16
          and cs.get('interval') == 16 and cs.get('writes') == 8
          and cs.get('write_steps') == list(range(16, 129, 16))
          and cs.get('final_history_steps') == 128
          and cs.get('bytes_last', 0) > 0 and cs.get('bytes_total', 0) > 0
          and finite(cs.get('write_seconds')),
          f"writes={cs.get('writes')} steps={cs.get('write_steps')} bytes_last={cs.get('bytes_last')}")
    check('checkpoint_file',
          meta.get('checkpoint', {}).get('bytes') == cs.get('bytes_last')
          and isinstance(meta.get('checkpoint', {}).get('sha256'), str)
          and len(meta['checkpoint']['sha256']) == 64,
          f"bytes={meta.get('checkpoint', {}).get('bytes')} sha256={meta.get('checkpoint', {}).get('sha256')}")
    invoc = run.get('invocation', {})
    check('fresh_invocation',
          invoc.get('mode') == 'fresh' and invoc.get('initial_history_steps') == 0
          and invoc.get('final_history_steps') == 128
          and invoc.get('invocation_history_steps') == 128,
          json.dumps(invoc, sort_keys=True))
    check('config_policy',
          cfg.get('model_family') == expected['family']
          and cfg['training'].get('seed') == a.seed
          and cfg['training'].get('precision') == 'bf16'
          and cfg['training'].get('checkpoint_interval') == 1
          and cfg['training'].get('offset_weight') == 0
          and cfg['training'].get('depth_weight') == 0
          and cfg['training'].get('loops') == expected['loops'],
          f"seed={cfg['training'].get('seed')} precision={cfg['training'].get('precision')}")
    check('mac_reporting',
          finite(run.get('analytical_forward_macs')) and run.get('analytical_forward_macs', 0) > 0
          and run.get('measured_flops') is None and run.get('energy_joules') is None,
          f"analytical_forward_macs={run.get('analytical_forward_macs')}")

    errors = [name for name, row in checks.items() if row['status'] != 'passed']
    weighted_nll = sum(float(row['final_ntp']) * int(row['ntp_labels']) for row in hist)
    step_seconds = sum(float(row['step_seconds']) for row in hist)
    peaks_alloc = [row['peak_cuda_allocated_bytes'] for row in hist]
    peaks_res = [row['peak_cuda_reserved_bytes'] for row in hist]
    audit = {
        'status': 'passed' if not errors else 'failed',
        'run_id': meta.get('run_id'), 'arm': a.arm, 'seed': a.seed,
        'checks': checks, 'errors': errors,
        'metrics': {
            'validation_ntp_labels': n,
            'validation_nll_sum': val.get('nll_sum'),
            'validation_mean_nll': val.get('mean_nll'),
            'validation_perplexity': val.get('perplexity'),
            'validation_seconds': run.get('validation_seconds'),
            'weighted_pre_update_train_nll_sum': weighted_nll,
            'total_valid_tokens': run.get('total_valid_tokens'),
            'total_ntp_labels': run.get('total_ntp_labels'),
            'step_seconds_sum': step_seconds,
            'valid_tokens_per_step_second': (run.get('total_valid_tokens') / step_seconds
                                             if step_seconds else None),
            'wall_seconds': meta.get('wall_seconds'),
            'peak_cuda_allocated_bytes': max(peaks_alloc) if peaks_alloc else None,
            'peak_cuda_reserved_bytes': max(peaks_res) if peaks_res else None,
            'checkpoint_sha256': meta.get('checkpoint', {}).get('sha256'),
            'checkpoint_bytes': meta.get('checkpoint', {}).get('bytes'),
            'checkpoint_writes': cs.get('writes'),
            'checkpoint_write_seconds': cs.get('write_seconds'),
            'analytical_forward_macs': run.get('analytical_forward_macs'),
            'parameters': run.get('parameters'),
            'parameter_inventory': inv,
            'packed_sha256': run.get('identity', {}).get('packed_sha256'),
        },
    }
    a.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': audit['status'], 'run_id': audit['run_id'],
                      'errors': errors, 'metrics': audit['metrics']}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
