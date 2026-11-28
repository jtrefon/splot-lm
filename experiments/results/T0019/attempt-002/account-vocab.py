"""Load the pinned accounting module and print exact SPLOT shapes for one vocab.

Usage:
    python account-vocab.py --accounting /path/to/pinned/tools/accounting.py --vocab 16384
"""
import argparse
import importlib.util
import json
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location('splot_accounting_pinned', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounting', required=True)
    parser.add_argument('--vocab', required=True, type=int)
    args = parser.parse_args()
    module = load(Path(args.accounting))
    shape = module.Shape(vocab=args.vocab, d=384, q_heads=6, kv_heads=2,
                         gdn_heads=6, dense=1024, expert=512)
    result = {'vocab': args.vocab, 'shape': 'Nano d384 q6/kv2 gdn6 dense1024 expert512'}
    for name, options in (
        ('ntp_512', {'context': 512, 'readouts': 1, 'offsets': 0}),
        ('ntp_2048', {'context': 2048, 'readouts': 1, 'offsets': 0}),
        ('a2_512', {'context': 512, 'readouts': 2, 'offsets': 2}),
        ('a2_2048', {'context': 2048, 'readouts': 2, 'offsets': 2}),
    ):
        value = module.account(shape, loops=2, **options)
        result[name] = {key: value[key] for key in (
            'physical', 'base', 'auxiliary', 'logical_blocks', 'decode_macs',
            'prefill_all_logits_macs', 'cache_bytes', 'gdn_snapshot_bytes')}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
