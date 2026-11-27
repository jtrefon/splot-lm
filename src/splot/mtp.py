"""Native offset-2 prediction and controlled A0/A1/A2 training objectives."""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .dense import RMSNorm, ntp_loss
from .recurrent import RecurrentLM, RecurrentOutput


@dataclass(frozen=True)
class MTPOutput(RecurrentOutput):
    offset2: dict


class MTPLM(RecurrentLM):
    def __init__(self, config):
        super().__init__(config)
        d = config.dense.width
        self.offset_projection = nn.Linear(d, d, bias=False)
        self.offset_norm = RMSNorm(d, config.dense.eps)
        nn.init.normal_(self.offset_projection.weight, std=config.dense.init_std)

    def parameter_count(self):
        d = self.config.dense.width
        return self.config.parameter_count() + d*d+d

    def forward(self, input_ids, document_ids, position_ids, valid, *, loops=None,
                readouts=None, offset2_depths=(), cache=None, use_cache=False):
        loops = self.config.default_loops if loops is None else loops
        readouts = (loops,) if readouts is None else tuple(readouts)
        depths = tuple(offset2_depths)
        if any(type(r) is not int or r not in readouts for r in depths):
            raise ValueError('offset-2 depths must be selected NTP readouts')
        out = super().forward(input_ids, document_ids, position_ids, valid,
                              loops=loops, readouts=readouts, cache=cache, use_cache=use_cache)
        aux = {r: F.linear(self.offset_norm(self.offset_projection(out.hidden[r])),
                           self.embedding.weight).masked_fill(~valid[..., None], 0)
               for r in sorted(set(depths))}
        return MTPOutput(out.logits, out.hidden, out.cache, aux)


@dataclass(frozen=True)
class Objective:
    total: torch.Tensor
    final_ntp: torch.Tensor
    early_ntp: torch.Tensor
    offset2: torch.Tensor
    ntp_count: int
    offset2_count: int


def training_objective(model, inputs, labels, *, mode='A2', loops=None,
                       depth_weight=.25, offset_weight=.25):
    """Pre-shifted, packer-masked labels; no labels passed to model.forward."""
    loops = model.config.default_loops if loops is None else loops
    if type(loops) is not int or not 1 <= loops <= model.config.max_loops:
        raise ValueError('invalid depth')
    if mode not in ('A0', 'A1', 'A2'):
        raise ValueError('invalid objective mode')
    if not all(math.isfinite(w) and w >= 0 for w in (depth_weight, offset_weight)):
        raise ValueError('invalid objective weights')
    if 1 not in labels or (mode == 'A2' and 2 not in labels):
        raise ValueError('missing offset labels')
    readouts = (loops,) if mode == 'A0' else tuple(range(1, loops+1))
    depths = tuple(sorted({1, loops})) if mode == 'A2' else ()
    out = model(*inputs, loops=loops, readouts=readouts, offset2_depths=depths)
    final = ntp_loss(out.logits[loops], labels[1])
    zero = final.new_zeros(())
    early = (torch.stack([ntp_loss(out.logits[r], labels[1]) for r in readouts[:-1]]).mean()
             if len(readouts) > 1 else zero)
    aux = (torch.stack([ntp_loss(out.offset2[r], labels[2]) for r in depths]).mean()
           if depths else zero)
    return Objective(final+depth_weight*early+offset_weight*aux, final, early, aux,
                     int(labels[1].ne(-100).sum()), int(labels[2].ne(-100).sum()) if depths else 0)
