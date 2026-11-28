"""Slow exact-speculation reference; CPU float64 sampling, single document/row."""
from dataclasses import dataclass
import math
import time
import torch

TRACE_SCHEMA = 'splot.mtp.trace.v2'


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
             eos=None, temperature=1., greedy=False, rng=None, observer=None, clock=None):
    """Separate shallow/target prefill. Verify chunk once; replay on rejection.

    Greedy mode uses equality with target argmax, not stochastic acceptance.
    D2 draws independent q1/q2 from the same committed shallow prefix.
    Target and shallow must share token IDs, and must not mutate old states.

    ``observer`` is an optional callable for bounded structured trace events.
    Default ``None`` adds no work and preserves tokens, RNG draws, model calls,
    returned caches and the exact API behavior. ``clock`` may inject integer
    monotonic nanoseconds for deterministic tests; it is unused without an
    observer. Events never contain tokens, text, logits or caches.
    """
    if not prompt or type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError('nonempty prompt and nonnegative integer budget required')
    if strategy not in ('D1','D2') or type(draft_length) is not int or draft_length < 1 or (strategy=='D2' and draft_length!=2):
        raise ValueError('invalid drafting strategy/length')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('invalid temperature')
    if rng is None: rng=torch.Generator().manual_seed(42)
    emit=None; seq=0
    if observer is not None:
        if not callable(observer): raise TypeError('observer must be callable')
        now=time.monotonic_ns if clock is None else clock
        def emit(kind,**fields):
            nonlocal seq
            stamp=now()
            if type(stamp) is not int: raise TypeError('clock must return integer nanoseconds')
            event={'schema':TRACE_SCHEMA,'seq':seq,'kind':kind,'timestamp_ns':stamp}
            event.update(fields)
            observer(event)
            seq+=1
        emit('run_start',strategy=strategy,draft_length=draft_length,
             max_new_tokens=max_new_tokens,prompt_length=len(prompt),
             greedy=bool(greedy),temperature=float(temperature),eos=eos is not None)
        emit('phase_start',role='target',phase='target_prefill',round=0,before_length=0)
    t,_=target.advance(None,list(prompt))
    if emit:
        emit('phase_end',role='target',phase='target_prefill',round=0,
             before_length=0,after_length=t.length,requested_tokens=len(prompt),
             actual_tokens=len(prompt),model_calls=1)
        emit('phase_start',role='shallow',phase='shallow_prefill',round=0,before_length=0)
    s,_=shallow.advance(None,list(prompt))
    if emit:
        emit('phase_end',role='shallow',phase='shallow_prefill',round=0,
             before_length=0,after_length=s.length,requested_tokens=len(prompt),
             actual_tokens=len(prompt),model_calls=1)
    output=[]; proposed=accepted=rejected=rounds=0
    while len(output)<max_new_tokens:
        rounds+=1
        remaining=max_new_tokens-len(output)
        count=min(draft_length,remaining)
        t_before=t.length; s_before=s.length; out_before=len(output)
        if emit:
            emit('phase_start',role='shallow',phase='shallow_draft',round=rounds,
                 before_length=s_before)
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
        if emit:
            emit('phase_end',role='shallow',phase='shallow_draft',round=rounds,
                 before_length=s_before,after_length=scratch.length,
                 requested_tokens=0 if strategy=='D2' else max(0,count-1),
                 actual_tokens=scratch.length-s_before,
                 model_calls=scratch.length-s_before)
        proposed+=len(drafts)
        if emit:
            emit('phase_start',role='target',phase='target_verify',round=rounds,
                 before_length=t_before)
        verified, after=target.advance(t,drafts)
        if emit:
            emit('phase_end',role='target',phase='target_verify',round=rounds,
                 before_length=t_before,after_length=verified.length,
                 requested_tokens=len(drafts),actual_tokens=len(drafts),
                 model_calls=1)
        emitted=[]; failed=False; statuses=['unexamined']*len(drafts)
        round_accepted=0
        for i,(y,q) in enumerate(zip(drafts,qs)):
            p=distribution(t.next_logits if i==0 else after[i-1],temperature)
            if p.shape!=q.shape: raise ValueError('target and proposal vocab mismatch')
            if greedy:
                z=int(p.argmax()); ok=z==y
            else:
                z,ok=verify_token(p,q,y,rng)
            emitted.append(z)
            if ok:
                accepted+=1; round_accepted+=1; statuses[i]='accepted'
            else:
                rejected+=1; statuses[i]='rejected'; failed=True; break
            if z==eos: break
        if emit:
            for i,status in enumerate(statuses,1):
                emit('draft',round=rounds,offset=i,status=status,proposed=True,
                     verified=status!='unexamined')
        bonus=0
        if failed:
            # Discard verified suffix, replay accepted prefix plus correction once.
            if emit:
                emit('phase_start',role='target',phase='target_rejection_replay',
                     round=rounds,before_length=t.length)
            replay_before=t.length
            t,_=target.advance(t,emitted)
            if emit:
                emit('phase_end',role='target',phase='target_rejection_replay',
                     round=rounds,before_length=replay_before,after_length=t.length,
                     requested_tokens=len(emitted),actual_tokens=len(emitted),
                     model_calls=1)
        else:
            t=verified
            if emitted[-1]!=eos and len(emitted)<remaining:
                if emit:
                    emit('phase_start',role='target',phase='target_bonus_advance',
                         round=rounds,before_length=t.length)
                bonus_before=t.length
                p=distribution(t.next_logits,temperature)
                bonus_token=int(p.argmax()) if greedy else draw(p,rng)
                emitted.append(bonus_token); bonus=1
                t,_=target.advance(t,[bonus_token])
                if emit:
                    emit('phase_end',role='target',phase='target_bonus_advance',
                         round=rounds,before_length=bonus_before,
                         after_length=t.length,requested_tokens=1,actual_tokens=1,
                         model_calls=1)
        # Reference replay: always synchronize shallow from committed prefix.
        if emit:
            emit('phase_start',role='shallow',phase='shallow_committed_replay',
                 round=rounds,before_length=s_before)
        s,_=shallow.advance(s,emitted)
        if emit:
            emit('phase_end',role='shallow',phase='shallow_committed_replay',
                 round=rounds,before_length=s_before,after_length=s.length,
                 requested_tokens=len(emitted),actual_tokens=len(emitted),
                 model_calls=1)
        output.extend(emitted)
        eos_committed=bool(eos is not None and output[-1]==eos)
        if emit:
            round_rejected=1 if failed else 0
            emit('round_commit',round=rounds,strategy=strategy,
                 proposed=len(drafts),verified=round_accepted+round_rejected,
                 accepted=round_accepted,rejected=round_rejected,
                 unexamined=len(drafts)-round_accepted-round_rejected,
                 correction=1 if failed else 0,bonus=bonus,committed=len(emitted),
                 target_before=t_before,target_after=t.length,
                 shallow_before=s_before,shallow_after=s.length,
                 output_before=out_before,output_after=len(output),
                 eos_committed=eos_committed)
        if output[-1]==eos: break
    if emit:
        emit('run_end',rounds=rounds,proposed=proposed,accepted=accepted,
             rejected=rejected,output_length=len(output),target_length=t.length,
             shallow_length=s.length)
    return Result(tuple(output),t,s,proposed,accepted,rejected,rounds)
