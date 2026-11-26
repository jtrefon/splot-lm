"""Overfit fixed original synthetic token sequence; no external data or quality claim."""
import argparse
import json
import math
import torch
from splot import DenseConfig, DenseLM, ntp_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--config', help='JSON DenseConfig fields')
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--bf16', action='store_true')
    a = p.parse_args()
    if a.steps < 1 or (a.bf16 and a.device != 'cuda'):
        p.error('positive steps; BF16 smoke requires CUDA')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device(a.device)
    if a.config:
        with open(a.config, encoding='utf-8') as f: config = DenseConfig(**json.load(f))
    else:
        config = DenseConfig()
    if config.vocab_size < 264: p.error('fixture requires vocabulary >=264')
    model = DenseLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
    text = ['A prism bends light.', 'Żółty pryzmat załamuje światło.']
    tokens = [[1, *[8+b for b in t.encode('utf-8')], 2] for t in text]
    size = max(map(len, tokens))
    x = torch.tensor([t+[0]*(size-len(t)) for t in tokens], device=device)
    valid = x.ne(0)
    doc = torch.where(valid, 0, -1).long()
    pos = torch.where(valid, torch.arange(size, device=device)[None, :], 0).long()
    labels = torch.full_like(x, -100)
    labels[:, :-1] = torch.where(valid[:, 1:], x[:, 1:], -100)
    losses = []
    for _ in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=a.device, dtype=torch.bfloat16, enabled=a.bf16):
            logits, _ = model(x, doc, pos, valid)
            loss = ntp_loss(logits, labels)
        if not math.isfinite(loss.item()): raise RuntimeError('nonfinite loss')
        loss.backward()
        if not all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()):
            raise RuntimeError('nonfinite or missing gradient')
        optimizer.step(); losses.append(loss.item())
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=a.device, dtype=torch.bfloat16, enabled=a.bf16):
        y, _ = model(x, doc, pos, valid)
        cached, cache = [], None
        for i in range(size):
            z, cache = model(x[:, i:i+1], doc[:, i:i+1], pos[:, i:i+1], valid[:, i:i+1], cache=cache, use_cache=True)
            cached.append(z)
        error = (y-torch.cat(cached, 1)).abs().max().item()
        torch.testing.assert_close(y, torch.cat(cached, 1), atol=.05 if a.bf16 else 1e-5, rtol=.05 if a.bf16 else 1e-4)
    if a.steps >= 40 and not losses[-1] < losses[0]*.5: raise RuntimeError('overfit smoke gate failed')
    print(json.dumps(dict(torch=torch.__version__, device=a.device, bf16=a.bf16, steps=a.steps,
                          seed=42, parameters=sum(p.numel() for p in model.parameters()),
                          initial_loss=losses[0], final_loss=losses[-1], cache_max_abs_error=error), indent=2))


if __name__ == '__main__': main()
