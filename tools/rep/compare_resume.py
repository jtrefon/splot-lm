"""Compare trusted, self-produced REP checkpoint/run pairs on the same device."""
import argparse
import json
import math
from pathlib import Path
import torch


IGNORE = {'step_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}


def equal(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError('tensor mismatch: ' + path)
    elif type(a) is not type(b):
        raise ValueError('type mismatch: ' + path)
    elif isinstance(a, dict):
        if a.keys() != b.keys(): raise ValueError('keys mismatch: ' + path)
        for k in a: equal(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b): raise ValueError('length mismatch: ' + path)
        for i, (x, y) in enumerate(zip(a, b)): equal(x, y, f'{path}.{i}')
    elif a != b:
        raise ValueError('value mismatch: ' + path)


def compare(a, b, steps, parameters):
    runs = [json.loads((p/'run.json').read_text()) for p in (a, b)]
    cps = [torch.load(p/'checkpoint.pt', map_location='cpu', weights_only=True) for p in (a, b)]
    for run, cp in zip(runs, cps):
        if run['parameters'] != parameters or len(run['history']) != steps or len(cp['history']) != steps:
            raise ValueError('unexpected inventory or step count')
        equal(run['identity'], cp['identity'], 'run/checkpoint identity')
        equal(run['history'], cp['history'], 'run/checkpoint history')
        if run['validation']['ntp_labels'] <= 0 or not all(math.isfinite(run['validation'][k]) for k in ('mean_nll', 'perplexity')):
            raise ValueError('invalid validation')
        for row in run['history']:
            if not all(math.isfinite(row[k]) for k in ('objective','final_ntp','early_ntp','offset2','balance')):
                raise ValueError('nonfinite history')
    for k in ('identity','model','optimizer','rng'): equal(cps[0][k], cps[1][k], k)
    for k in ('identity','config','data_manifest','parameters','validation','total_valid_tokens','total_ntp_labels','analytical_forward_macs'):
        equal(runs[0][k], runs[1][k], k)
    history = lambda r: [{k:v for k,v in row.items() if k not in IGNORE} for row in r['history']]
    equal(history(runs[0]), history(runs[1]), 'non-timing history')
    return {'status':'passed','steps':steps,'parameters':parameters,'model_optimizer_rng_equal':True,
            'non_timing_history_equal':True,'validation_equal':True,'ignored_history_fields':sorted(IGNORE)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('continuous',type=Path); p.add_argument('resumed',type=Path)
    p.add_argument('--steps',type=int,required=True); p.add_argument('--parameters',type=int,required=True)
    a=p.parse_args()
    try: print(json.dumps(compare(a.continuous,a.resumed,a.steps,a.parameters),indent=2))
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc: p.exit(1,f'FAILED: {exc}\n')


if __name__=='__main__': main()
