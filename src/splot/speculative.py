"""Slow exact-speculation reference; CPU float64 sampling, single document/row."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Prefix:
    length: int
    cache: object
    next_logits: torch.Tensor
    offset2_logits: torch.Tensor | None = None


class ModelRunner:
    """Fixed-depth MTPLM runner. Returned caches are read-only by convention."""
    def __init__(self, model, loops, *, offset2=False):
        if model.training:
            raise ValueError('runner requires eval model')
        self.model, self.loops, self.offset2 = model, loops, offset2

    @torch.no_grad()
    def advance(self, prefix, tokens):
        if not tokens:
            raise ValueError('empty advance')
        start = 0 if prefix is None else prefix.length
        device = next(self.model.parameters()).device
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        doc = torch.zeros_like(x)
        pos = torch.arange(start, start+len(tokens), device=device)[None,:]
        out = self.model(x, doc, pos, torch.ones_like(x,dtype=torch.bool),
                         loops=self.loops, readouts=(self.loops,),
                         offset2_depths=(self.loops,) if self.offset2 else (),
                         cache=None if prefix is None else prefix.cache, use_cache=True)
        logits = out.logits[self.loops][0].detach().cpu().double()
        aux = out.offset2[self.loops][0,-1].detach().cpu().double() if self.offset2 else None
        return Prefix(start+len(tokens),out.cache,logits[-1],aux), logits


def distribution(logits, temperature=1.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    x = logits.detach().cpu().double()
    if x.ndim != 1 or not x.numel() or torch.isnan(x).any() or torch.isposinf(x).any() or not torch.isfinite(x).any():
        raise ValueError('invalid logits')
    # Subtract max first, avoiding positive overflow for small temperatures.
    return ((x-x.max())/temperature).softmax(-1)


def draw(p, rng):
    return int(torch.multinomial(p,1,generator=rng).item())


def correction(p,q):
    r=(p-q).clamp_min(0)
    mass=r.sum()
    if mass <= 0 or not torch.isfinite(mass):
        raise ArithmeticError('rejection with zero residual mass')
    return r/mass


def verify_token(p,q,y,rng):
    if p.shape != q.shape or q[y] <= 0:
        raise ValueError('proposal outside support or incompatible vocabulary')
    u=torch.rand((),dtype=torch.float64,generator=rng).item()
    if u*q[y].item() < p[y].item():
        return y,True
    return draw(correction(p,q),rng),False


@dataclass(frozen=True)
class Result:
    tokens: tuple
    target: Prefix
    shallow: Prefix
    proposed: int
    accepted: int
    rejected: int
    rounds: int


def generate(target, shallow, prompt, max_new_tokens, *, strategy='D1', draft_length=2,
             eos=None, temperature=1., greedy=False, rng=None):
    """Separate shallow/target prefill. Verify chunk once; replay on rejection.

    Greedy mode uses equality with target argmax, not stochastic acceptance.
    D2 draws independent q1/q2 from the same committed shallow prefix.
    Target and shallow must share token IDs, and must not mutate old states.
    """
    if not prompt or type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError('nonempty prompt and nonnegative integer budget required')
    if strategy not in ('D1','D2') or type(draft_length) is not int or draft_length < 1 or (strategy=='D2' and draft_length!=2):
        raise ValueError('invalid drafting strategy/length')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('invalid temperature')
    if rng is None: rng=torch.Generator().manual_seed(42)
    t,_=target.advance(None,list(prompt)); s,_=shallow.advance(None,list(prompt))
    output=[]; proposed=accepted=rejected=rounds=0
    while len(output)<max_new_tokens:
        rounds+=1
        remaining=max_new_tokens-len(output)
        count=min(draft_length,remaining)
        drafts=[]; qs=[]; scratch=s
        for i in range(count):
            logits=(s.next_logits if i==0 else s.offset2_logits) if strategy=='D2' else scratch.next_logits
            if logits is None: raise ValueError('D2 requires offset-2 logits')
            q=distribution(logits,temperature)
            y=int(q.argmax()) if greedy else draw(q,rng)
            drafts.append(y); qs.append(q)
            if y==eos: break
            if strategy=='D1' and i+1<count:
                scratch,_=shallow.advance(scratch,[y])
        proposed+=len(drafts)
        verified, after=target.advance(t,drafts)
        emitted=[]; failed=False
        for i,(y,q) in enumerate(zip(drafts,qs)):
            p=distribution(t.next_logits if i==0 else after[i-1],temperature)
            if p.shape!=q.shape: raise ValueError('target and proposal vocab mismatch')
            if greedy:
                z=int(p.argmax()); ok=z==y
            else:
                z,ok=verify_token(p,q,y,rng)
            emitted.append(z)
            if ok: accepted+=1
            else:
                rejected+=1; failed=True; break
            if z==eos: break
        if failed:
            # Discard verified suffix, replay accepted prefix plus correction once.
            t,_=target.advance(t,emitted)
        else:
            t=verified
            if emitted[-1]!=eos and len(emitted)<remaining:
                p=distribution(t.next_logits,temperature)
                bonus=int(p.argmax()) if greedy else draw(p,rng)
                emitted.append(bonus)
                t,_=target.advance(t,[bonus])
        # Reference replay: always synchronize shallow from committed prefix.
        s,_=shallow.advance(s,emitted)
        output.extend(emitted)
        if output[-1]==eos: break
    return Result(tuple(output),t,s,proposed,accepted,rejected,rounds)
