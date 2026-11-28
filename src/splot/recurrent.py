"""Fixed-depth all-GQA sandwich control; shared weights, independent logical KV."""
from dataclasses import dataclass, field
import math
import torch
from torch import nn
import torch.nn.functional as F
from .dense import DenseConfig, Block, RMSNorm, segments


@dataclass(frozen=True)
class RecurrentConfig:
    dense: DenseConfig = field(default_factory=lambda: DenseConfig(layers=8))
    prelude: int = 2
    core: int = 4
    coda: int = 2
    max_loops: int = 4
    default_loops: int = 2
    injection: str = 'none'
    injection_strength: float = .1
    residual_scale: str = 'unscaled'
    depth_embedding: bool = False
    gdn_algorithm: str = 'scan'

    def __post_init__(self):
        if min(self.prelude, self.core, self.coda) < 1 or self.dense.layers != self.prelude+self.core+self.coda:
            raise ValueError('invalid physical topology')
        if not 1 <= self.default_loops <= self.max_loops <= 4:
            raise ValueError('invalid depth cap/default')
        if self.injection not in ('none', 'additive', 'convex') or not 0 <= self.injection_strength <= 1:
            raise ValueError('invalid injection')
        if self.residual_scale not in ('unscaled', 'sqrt_cap', 'linear_cap'):
            raise ValueError('invalid residual scale')
        if self.gdn_algorithm not in ('scan', 'chunk64'):
            raise ValueError('gdn_algorithm must be scan or chunk64')

    def parameter_count(self):
        return self.dense.parameter_count() + (self.max_loops*self.dense.width if self.depth_embedding else 0)

    def gamma(self):
        if self.residual_scale == 'sqrt_cap':
            return 1/math.sqrt(2*self.core*self.max_loops)
        if self.residual_scale == 'linear_cap':
            return .5/(self.max_loops*math.sqrt(self.core))
        return 1.


@dataclass(frozen=True)
class RecurrentCache:
    owner: object
    rows: tuple
    loops: int
    readouts: tuple
    states: tuple  # ordered ((stage, depth, block), per-row KV) pairs


@dataclass(frozen=True)
class RecurrentOutput:
    logits: dict  # requested depth -> [B,T,V]
    hidden: dict  # requested depth -> pre-coda [B,T,d]
    cache: RecurrentCache | None
    routing: tuple = field(default=(), kw_only=True)


class RecurrentLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.dense
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        self.prelude = nn.ModuleList(Block(c) for _ in range(config.prelude))
        self.core = nn.ModuleList(Block(c) for _ in range(config.core))
        self.coda = nn.ModuleList(Block(c) for _ in range(config.coda))
        self.norm = RMSNorm(c.width, c.eps)
        self.depth = nn.Parameter(torch.zeros(config.max_loops, c.width)) if config.depth_embedding else None
        self._cache_owner = object()
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=c.init_std)

    def _apply_block(self, block, x, positions, plans, previous, gamma, valid, key):
        x,state=block(x,positions,plans,previous,residual_scale=gamma)
        return x,state,None

    def forward(self, input_ids, document_ids, position_ids, valid, *, loops=None,
                readouts=None, cache=None, use_cache=False):
        c = self.config
        loops = c.default_loops if loops is None else loops
        if not isinstance(loops, int) or isinstance(loops, bool) or not 1 <= loops <= c.max_loops:
            raise ValueError('invalid requested depth')
        requested = (loops,) if readouts is None else tuple(readouts)
        if not requested or any(not isinstance(r, int) or isinstance(r, bool) or not 1 <= r <= loops for r in requested):
            raise ValueError('invalid readout depths')
        requested = tuple(sorted(set(requested)))
        if input_ids.ndim != 2 or not input_ids.numel() or any(t.shape != input_ids.shape for t in (document_ids, position_ids, valid)):
            raise ValueError('expected nonempty matching [B,T] tensors')
        if any(t.dtype != torch.long for t in (input_ids, document_ids, position_ids)) or valid.dtype != torch.bool:
            raise ValueError('expected int64 IDs and bool mask')
        if any(t.device != input_ids.device for t in (document_ids, position_ids, valid)):
            raise ValueError('metadata device mismatch')
        if cache is not None and not use_cache:
            raise ValueError('cache requires use_cache')
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError('cache requires eval and no_grad/inference_mode')
        b = input_ids.shape[0]
        keys = ([('P', 0, i) for i in range(c.prelude)] +
                [('C', r, i) for r in range(1, loops+1) for i in range(c.core)] +
                [('Q', r, i) for r in requested for i in range(c.coda)])
        previous = {} if cache is None else dict(cache.states)
        if cache is not None and (cache.owner is not self._cache_owner or len(cache.rows) != b or
                                  cache.loops != loops or cache.readouts != requested or
                                  len(cache.states) != len(keys) or set(previous) != set(keys)):
            raise ValueError('cache ownership, depth, readout or layout mismatch')
        plans, rows = segments(document_ids, position_ids, valid, cache.rows if cache else (None,)*b)
        states = {}
        routing = []

        def run(stack, x, stage, r, gamma=1.):
            for i, block in enumerate(stack):
                key = (stage, r, i)
                x,state,route=self._apply_block(block,x,position_ids,plans,previous.get(key,(None,)*b),gamma,valid,key)
                if route is not None: routing.append((key,route))
                if use_cache:
                    states[key] = state
            return x

        z = run(self.prelude, self.embedding(input_ids), 'P', 0)
        h = z
        logits, hidden = {}, {}
        for r in range(1, loops+1):
            if r > 1:
                a = c.injection_strength/r
                if c.injection == 'additive':
                    h = h+a*z
                elif c.injection == 'convex':
                    h = (1-a)*h+a*z
            if self.depth is not None:
                h = h+self.depth[r-1]
            h = run(self.core, h, 'C', r, c.gamma())
            if r in requested:
                hidden[r] = h.masked_fill(~valid[..., None], 0)
                y = run(self.coda, h, 'Q', r)
                logits[r] = F.linear(self.norm(y), self.embedding.weight).masked_fill(~valid[..., None], 0)
        next_cache = RecurrentCache(self._cache_owner, rows, loops, requested,
                                    tuple((k, states[k]) for k in keys)) if use_cache else None
        return RecurrentOutput(logits, hidden, next_cache, routing=tuple(routing))
