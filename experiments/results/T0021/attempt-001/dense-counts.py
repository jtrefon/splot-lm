"""Reproducible T0021 analytical counts for the dense comparator configs."""
import hashlib
import json
from pathlib import Path

import accounting

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = (
    'dense-ntp-512x6-q8kv4-ffn2432.json',
    'dense-ntp-512x6-q8kv2-ffn2176.json',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_row(name):
    path = ROOT / 'configs/experiments' / name
    cfg = json.loads(path.read_text(encoding='utf-8'))
    d = cfg['dense']
    shape = accounting.DenseShape(vocab=d['vocab_size'], width=d['width'], layers=d['layers'],
                                  q_heads=d['q_heads'], kv_heads=d['kv_heads'],
                                  head=d['head_dim'], dense=d['intermediate'])
    counts = {}
    for context in (512, 2048):
        result = accounting.dense_account(shape, context)
        counts[str(context)] = {
            'physical': result['physical'],
            'decode_macs': result['decode_macs'],
            'prefill_all_logits_macs': result['prefill_all_logits_macs'],
        }
    return {
        'config': name,
        'config_sha256': digest(path),
        'shape': {'vocab': d['vocab_size'], 'width': d['width'], 'layers': d['layers'],
                  'q_heads': d['q_heads'], 'kv_heads': d['kv_heads'],
                  'head': d['head_dim'], 'intermediate': d['intermediate']},
        'counts': counts,
    }


def main():
    dense = [dense_row(name) for name in CONFIGS]
    splot_path = ROOT / 'configs/experiments/splot-nano-v16384-a0.json'
    splot_cfg = json.loads(splot_path.read_text(encoding='utf-8'))
    d = splot_cfg['dense']
    shape = accounting.Shape(vocab=d['vocab_size'], d=d['width'], q_heads=d['q_heads'],
                             kv_heads=d['kv_heads'], gdn_heads=d['q_heads'],
                             head=d['head_dim'], dense=d['intermediate'],
                             expert=splot_cfg['expert_intermediate'])
    splot = {}
    for context in (512, 2048):
        result = accounting.account(shape, loops=2, context=context, readouts=1, offsets=0)
        splot[str(context)] = {k: result[k] for k in
                               ('physical', 'base', 'auxiliary', 'logical_blocks',
                                'decode_macs', 'prefill_all_logits_macs', 'cache_bytes')}
    payload = {
        'schema': 'splot-t0021-dense-counts-v1',
        'instruction_commit': '2babd431a8c0cb61a0f1129b4aae8c2383d7dbf8',
        'source_commit': '75fced45d0cff3888e3606a46a529fb3714e93df',
        'status': 'analytical only; no training or measured hardware result',
        'formulas': {
            'dense_params': 'V*d + L*(d*(2q+2kv)+3*d*m+2*d)+d',
            'dense_decode': 'L*(d*(2q+2kv)+3*d*m)+L*2*q*T+d*V',
            'dense_prefill': 'T*(L*(d*(2q+2kv)+3*d*m)+d*V)+L*q*T*(T+1)',
        },
        'dense': dense,
        'splot_v16384_a0': {
            'config': splot_path.name,
            'config_sha256': digest(splot_path),
            'counts': splot,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
