"""Synthetic recurrent NTP/all-depth overfit and cache smoke; not a benchmark."""
import argparse
import json
import torch
from splot import RecurrentConfig, RecurrentLM, ntp_loss
from data_fixture import Document, pack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--loops', type=int, default=2)
    a = p.parse_args()
    if a.steps < 1 or not 1 <= a.loops <= 4:
        p.error('positive steps and 1..4 loops required')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    torch.set_num_threads(1); torch.manual_seed(42)
    blocks = pack([Document('en','en','en','education','A prism bends light.'),
                   Document('pl','pl','pl','education','Żółty pryzmat załamuje światło.')],64)
    x = tuple(torch.tensor([b[k] for b in blocks],device=a.device) for k in
              ('input_ids','document_ids','position_ids','valid'))
    labels = torch.tensor([b['labels'][1] for b in blocks],device=a.device)
    model = RecurrentLM(RecurrentConfig()).to(a.device)
    opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
    readouts = tuple(range(1,a.loops+1))
    losses = []
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True)
        out = model(*x,loops=a.loops,readouts=readouts)
        loss = ntp_loss(out.logits[a.loops],labels)
        if a.loops > 1:
            loss = loss+.25*sum(ntp_loss(out.logits[r],labels) for r in readouts[:-1])/(a.loops-1)
        if not loss.isfinite(): raise RuntimeError('nonfinite loss')
        loss.backward()
        if not all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()):
            raise RuntimeError('missing or nonfinite gradient')
        opt.step(); losses.append(loss.item())
    if a.steps >= 40 and losses[-1] >= .5*losses[0]: raise RuntimeError('overfit smoke gate failed')
    model.eval()
    with torch.no_grad():
        full = model(*x,loops=a.loops,readouts=readouts)
        outputs = {r:[] for r in readouts}; cache = None
        for i in range(x[0].shape[1]):
            out = model(*(v[:,i:i+1] for v in x),loops=a.loops,readouts=readouts,cache=cache,use_cache=True)
            cache = out.cache
            for r in readouts: outputs[r].append(out.logits[r])
        errors = {}
        for r in readouts:
            y = torch.cat(outputs[r],dim=1)
            torch.testing.assert_close(y,full.logits[r],atol=1e-5,rtol=1e-4)
            errors[r] = (y-full.logits[r]).abs().max().item()
    print(json.dumps(dict(torch=torch.__version__,device=a.device,seed=42,steps=a.steps,
                          loops=a.loops,parameters=sum(p.numel() for p in model.parameters()),
                          initial_loss=losses[0],final_loss=losses[-1],cache_max_abs_error=errors),indent=2))


if __name__ == '__main__': main()
