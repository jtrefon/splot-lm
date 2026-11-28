"""Static T0023 preflight matrix; not an executed training protocol."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = {
    'splot-v16384-a0': 'splot-nano-v16384-a0.json',
    'dense-v16384-parameter': 'dense-ntp-512x6-q8kv4-ffn2432.json',
    'dense-v16384-forward-mac': 'dense-ntp-512x6-q8kv2-ffn2176.json',
}
SEEDS = (42, 43, 44)
UPDATES = 128
POSITIONS = 128 * 512
VALID = 46204
NTP = 46076


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def main():
    configs = {}
    for arm, name in CONFIGS.items():
        path = ROOT / 'configs/experiments' / name
        payload = json.loads(path.read_text(encoding='utf-8'))
        configs[arm] = {
            'file': name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'canonical_digest': canonical_digest(payload),
            'optimizer': payload['training'].get('optimizer'),
            'model_family': payload.get('model_family', 'splot'),
        }
    assert len({json.dumps(v['optimizer'], sort_keys=True) for v in configs.values()}) == 1
    orders = {}
    for seed in SEEDS:
        order = list(range(128))
        random.Random(seed).shuffle(order)
        orders[str(seed)] = {
            'order': order,
            'sha256': canonical_digest(order),
            'is_permutation': sorted(order) == list(range(128)),
        }
    runs = []
    for arm in CONFIGS:
        for seed in SEEDS:
            runs.append({
                'run_id': f'{arm}-seed{seed}',
                'arm': arm,
                'config': configs[arm]['file'],
                'config_sha256': configs[arm]['sha256'],
                'seed': seed,
                'updates': UPDATES,
                'positions': POSITIONS,
                'valid_tokens': VALID,
                'ntp_labels': NTP,
                'offset2_labels': 0,
                'data_order_sha256': orders[str(seed)]['sha256'],
                'one_pass': True,
            })
    payload = {
        'schema': 'splot-t0023-preflight-matrix-v1',
        'instruction_commit': 'a87adbd1e7186926fa1766adc4aa8470b71d8579',
        'tested_commit': '57629a9ea2ab956ec24e1d05fb153f79b5bd2a64',
        'status': 'proposed only; not executed; attempt-002 hosted required CI failed',
        'configs': configs,
        'seeds': list(SEEDS),
        'data_order': orders,
        'runs': runs,
        'counts': {'positions_per_pass': POSITIONS, 'valid_tokens_per_pass': VALID,
                   'ntp_labels_per_pass': NTP, 'offset2_labels_ntp_only': 0},
        'notes': ['No quality margins, GPU allocation, actual corpus or training are included.',
                  'Attempt-001 failed on identity.dataset_manifest_sha256; attempt-002 corrected that path but read data_policy.dataset_manifest_sha256, while the field is data_policy.manifest_sha256. No further correction or rerun was performed.'],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
