"""Analytical reference architecture accounting, not measured FLOPs/VRAM."""
from dataclasses import dataclass, asdict
import argparse
import json


@dataclass(frozen=True)
class Shape:
    d: int = 768
    vocab: int = 32768
    q_heads: int = 12
    kv_heads: int = 4
    head: int = 64
    gdn_heads: int = 12
    conv: int = 4
    dense: int = 2048
    expert: int = 1024
    experts: int = 8
    topk: int = 2

    def validate(self):
        if min(asdict(self).values()) <= 0:
            raise ValueError('dimensions must be positive')
        if self.q_heads % self.kv_heads or self.topk > self.experts or self.head % 2:
            raise ValueError('invalid grouping, routing or RoPE dimension')


def account(s=Shape(), loops=2, context=2048, readouts=1, mtp=True, offsets=0):
    s.validate()
    if not 1 <= loops <= 4 or not 1 <= readouts <= loops or context < 1:
        raise ValueError('invalid depth, context or readouts')
    if not 0 <= offsets <= min(2, loops) or (offsets and not mtp):
        raise ValueError('invalid number of offset-2 readouts')
    d, q, kv, g = s.d, s.q_heads*s.head, s.kv_heads*s.head, s.gdn_heads*s.head
    gqa = d*(2*q+2*kv)
    gated = gqa+d*q
    gdn_linear = d*(5*g+2*s.gdn_heads)
    conv = s.conv*3*g
    gdn = gdn_linear+conv+2*s.gdn_heads+s.head
    expert = 3*d*s.expert
    dense = 3*d*s.dense
    router = d*s.experts
    base = d*s.vocab+4*(gqa+dense+2*d)+3*gdn+gated+4*((s.experts+1)*expert+router+2*d)+d
    aux = d*d+d if mtp else 0
    # Final readout only: each shared physical core block activates top-k plus shared expert.
    active_min = base-4*(s.experts-s.topk)*expert
    active_max = base-4*(s.experts-min(s.experts, loops*s.topk))*expert
    # Count contraction MACs only; embedding lookup, norms, gates, scalar ops excluded.
    attention = 2*q*context
    delta = 3*s.gdn_heads*s.head*s.head
    dense_block = gqa+dense+attention
    core = 3*(gdn_linear+conv+delta)+gated+attention+4*((s.topk+1)*expert+router)
    decode = (2+2*readouts)*dense_block+loops*core+readouts*d*s.vocab+offsets*(d*d+d*s.vocab)
    # For all T outputs in one unpadded causal prefill, sum attention lengths 1..T.
    prefill = context*(decode-((2+2*readouts)+loops)*attention)+((2+2*readouts)+loops)*q*context*(context+1)
    gdn_state = 4*s.gdn_heads*s.head*s.head+2*(s.conv-1)*3*g
    kv_state = 4*kv*context
    return dict(physical=base+aux, base=base, auxiliary=aux, gdn=gdn, gqa=gqa,
                gated_gqa=gated, target_unique_active_min=active_min,
                target_unique_active_max=active_max, logical_blocks=2+4*loops+2*readouts,
                decode_macs=decode, prefill_all_logits_macs=prefill,
                cache_bytes=3*loops*gdn_state+(2+2*readouts+loops)*kv_state,
                gdn_snapshot_bytes=3*loops*gdn_state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loops', type=int, default=2)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--readouts', type=int, default=1)
    p.add_argument('--offsets', type=int, default=0)
    p.add_argument('--no-mtp', action='store_true')
    p.add_argument('--nano', action='store_true')
    a = p.parse_args()
    shape = Shape(d=384, q_heads=6, kv_heads=2, gdn_heads=6, dense=1024, expert=512) if a.nano else Shape()
    print(json.dumps(account(shape, a.loops, a.context, a.readouts, not a.no_mtp, a.offsets), indent=2))
