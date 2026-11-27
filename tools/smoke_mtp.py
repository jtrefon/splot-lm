"""Fixed synthetic A0/A1/A2 training smoke; not a language-quality experiment."""
import argparse
import json
import torch
from splot import RecurrentConfig, MTPLM, HybridLM, SPLOTLM, training_objective
from data_fixture import Document, pack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--loops', type=int, default=2)
    p.add_argument('--mode', choices=('A0','A1','A2'), default='A2')
    models=p.add_mutually_exclusive_group()
    models.add_argument('--hybrid', action='store_true')
    models.add_argument('--sparse', action='store_true')
    a=p.parse_args()
    if a.steps<1 or not 1<=a.loops<=4: p.error('positive steps and 1..4 loops required')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    torch.set_num_threads(1); torch.manual_seed(42)
    blocks=pack([Document('en','en','en','education','A prism bends light.'),
                 Document('pl','pl','pl','education','Żółty pryzmat załamuje światło.')],64)
    x=tuple(torch.tensor([b[k] for b in blocks],device=a.device) for k in
            ('input_ids','document_ids','position_ids','valid'))
    labels={k:torch.tensor([b['labels'][k] for b in blocks],device=a.device) for k in (1,2)}
    m=(SPLOTLM if a.sparse else HybridLM if a.hybrid else MTPLM)(RecurrentConfig()).to(a.device)
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0)
    history=[]
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True)
        o=training_objective(m,x,labels,mode=a.mode,loops=a.loops)
        if not o.total.isfinite(): raise RuntimeError('nonfinite loss')
        o.total.backward()
        for name,param in m.named_parameters():
            inactive=name.startswith(('offset_projection.','offset_norm.')) and a.mode!='A2'
            if inactive:
                if param.grad is not None: raise RuntimeError('inactive head received gradient')
            elif a.sparse and '.moe.experts.' in name and param.grad is None:
                continue  # unselected experts have no gradient
            elif param.grad is None or not param.grad.isfinite().all():
                raise RuntimeError('missing or nonfinite gradient: '+name)
        opt.step()
        history.append({k:getattr(o,k).item() for k in ('total','final_ntp','early_ntp','offset2','balance')})
    if a.steps>=40 and history[-1]['total']>=.6*history[0]['total']:
        raise RuntimeError('synthetic overfit smoke gate failed')
    print(json.dumps(dict(torch=torch.__version__,device=a.device,seed=42,steps=a.steps,
                          loops=a.loops,mode=a.mode,hybrid=a.hybrid,sparse=a.sparse,parameters=m.parameter_count(),
                          ntp_count=o.ntp_count,offset2_count=o.offset2_count,
                          initial=history[0],final=history[-1]),indent=2))


if __name__=='__main__': main()
