"""Compare T0022 control/profile cases and the accepted T0020 BF16 reference."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, choices=(512, 2048), required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    profiled = json.loads(args.profile.read_text())
    reference = json.loads(args.reference.read_text())
    require(control['status'] == 'passed' and profiled['status'] == 'passed',
            'control/profile case did not pass')
    require(control['mode'] == 'control' and profiled['mode'] == 'profile',
            'unexpected mode')
    require(control['length'] == profiled['length'] == args.length,
            'length mismatch')

    same = {
        'identity_equal': control['identity'] == profiled['identity'],
        'losses_equal': control['losses'] == profiled['losses'],
        'gradient_inventory_equal': control['gradients'] == profiled['gradients'],
        'model_digest_equal': control['model'] == profiled['model'],
        'optimizer_digest_equal': control['optimizer'] == profiled['optimizer'],
    }
    required_checks = ('parameters_exact', 'warmup_status', 'warmup_init_equal',
                       'warmup_input_equal', 'warmup_partition_equal', 'labels_exact',
                       'precheck_finite', 'losses_finite', 'gradients_finite',
                       'model_finite', 'optimizer_finite')
    same['all_control_checks_passed'] = all(control['checks'][key] for key in required_checks)
    same['all_profile_checks_passed'] = all(profiled['checks'][key] for key in required_checks)
    same['all_passed'] = all(same.values())

    reference_identity = {
        'input_sha256': reference['input']['input_sha256'],
        'initialization_sha256': reference['warmup']['initialization_sha256'],
        'partition_plan_sha256': reference['warmup']['partition_plan_sha256'],
        'parameters': reference['warmup']['parameters'],
    }
    reference_result = {
        'identity_match': control['identity'] == reference_identity,
        'losses_match': control['losses'] == reference['measured']['losses'],
        'profile_matches_reference': profiled['losses'] == reference['measured']['losses'],
        'parameters_match': control['identity']['parameters'] == 43055076,
        'all_passed': False,
    }
    reference_result['all_passed'] = all(value for key, value in reference_result.items()
                                         if key != 'all_passed')
    report = {
        'status': 'passed' if same['all_passed'] and reference_result['all_passed'] else 'failed',
        'length': args.length,
        'control_sha256': sha(args.control),
        'profile_sha256': sha(args.profile),
        'reference_sha256': sha(args.reference),
        'reference_label': f'experiments/results/T0020/attempt-001/case-{args.length}-1-bf16.json',
        'same_mode_exact': same,
        't0020_reference': reference_result,
        'control_gradient_digest': control['gradients']['gradient_digest'],
        'profile_gradient_digest': profiled['gradients']['gradient_digest'],
        'control_model_digest': control['model']['digest'],
        'profile_model_digest': profiled['model']['digest'],
        'control_optimizer_digest': control['optimizer']['digest'],
        'profile_optimizer_digest': profiled['optimizer']['digest'],
        'control_losses': control['losses'],
        'profile_losses': profiled['losses'],
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
