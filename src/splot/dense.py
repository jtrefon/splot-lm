"""Slow, explicit dense GQA reference. Inference cache retains last document per row."""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DenseConfig:
    vocab_size: int = 264
    width: int = 32
    layers: int = 2
    q_heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    intermediate: int = 64
    rope_theta: float = 1e6
    eps: float = 1e-5
    init_std: float = .02

    def __post_init__(self):
        if min(self.vocab_size, self.width, self.layers, self.q_heads,
               self.kv_heads, self.head_dim, self.intermediate) <= 0:
            raise ValueError('nonpositive dimension')
        if self.q_heads % self.kv_heads or self.head_dim % 2:
            raise ValueError('invalid GQA groups or RoPE dimensions')
        if not all(math.isfinite(x) and x > 0 for x in (self.rope_theta, self.eps, self.init_std)):
            raise ValueError('invalid numerical configuration')

    def parameter_count(self):
        d, q, k = self.width, self.q_heads*self.head_dim, self.kv_heads*self.head_dim
        return self.vocab_size*d + self.layers*(d*(2*q+2*k)+3*d*self.intermediate+2*d)+d


@dataclass(frozen=True)
class RowState:
    document: int
    length: int


@dataclass(frozen=True)
class DenseCache:
    owner: object
    rows: tuple
    layers: tuple  # layer -> batch row -> (rotated K, V), or None


def _accum(x):
    return x if x.dtype == torch.float64 else x.float()


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = _accum(x)
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True)+self.eps) * self.weight).to(x.dtype)


def rope(x, positions, theta):
    # x: [heads, time, head_dim], interleaved pair convention.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv = theta ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=dtype)/x.shape[-1])
    ang = positions.to(dtype)[:, None] * inv
    a, b = _accum(x[..., 0::2]), _accum(x[..., 1::2])
    return torch.stack((a*ang.cos()-b*ang.sin(), a*ang.sin()+b*ang.cos()), -1).flatten(-2).to(x.dtype)


def segments(document_ids, position_ids, valid, old_rows):
    """Build contiguous logical document runs; invalid slots are skipped entirely."""
    plans, rows = [], []
    for docs, pos, mask, old in zip(document_ids.tolist(), position_ids.tolist(), valid.tolist(), old_rows):
        runs = []
        doc, length = (old.document, old.length) if old is not None else (None, 0)
        for i, (di, pi, vi) in enumerate(zip(docs, pos, mask)):
            if not vi:
                if di != -1 or pi != 0:
                    raise ValueError('padding requires document -1 and position 0')
                continue
            if di < 0:
                raise ValueError('valid token needs nonnegative document ID')
            if di != doc:
                doc, length = di, 0
                runs.append(([], False))
            elif not runs:
                runs.append(([], old is not None))
            if pi != length:
                raise ValueError('positions must advance within a document and reset on change')
            runs[-1][0].append(i)
            length += 1
        plans.append(runs)
        rows.append(None if doc is None else RowState(doc, length))
    return plans, tuple(rows)


class GQA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.width, c.q_heads*c.head_dim, bias=False)
        self.k = nn.Linear(c.width, c.kv_heads*c.head_dim, bias=False)
        self.v = nn.Linear(c.width, c.kv_heads*c.head_dim, bias=False)
        self.o = nn.Linear(c.q_heads*c.head_dim, c.width, bias=False)

    def forward(self, x, positions, plans, previous):
        c = self.c
        out = torch.zeros_like(x)
        next_rows = []
        for b, runs in enumerate(plans):
            final = previous[b]
            for indices, continuing in runs:
                ix = torch.tensor(indices, device=x.device)
                u = x[b, ix]
                n = len(indices)
                q = rope(self.q(u).view(n, c.q_heads, c.head_dim).transpose(0, 1), positions[b, ix], c.rope_theta)
                k = rope(self.k(u).view(n, c.kv_heads, c.head_dim).transpose(0, 1), positions[b, ix], c.rope_theta)
                v = self.v(u).view(n, c.kv_heads, c.head_dim).transpose(0, 1)
                if continuing:
                    if final is None:
                        raise ValueError('missing cache for continuing document')
                    k = torch.cat((final[0], k), dim=1)
                    v = torch.cat((final[1], v), dim=1)
                past = k.shape[1]-n
                q = q.reshape(c.kv_heads, c.q_heads//c.kv_heads, n, c.head_dim)
                score = torch.einsum('hgtd,hsd->hgts', _accum(q), _accum(k))/math.sqrt(c.head_dim)
                allowed = torch.arange(k.shape[1], device=x.device)[None, :] <= past+torch.arange(n, device=x.device)[:, None]
                prob = score.masked_fill(~allowed, float('-inf')).softmax(-1)
                y = torch.einsum('hgts,hsd->hgtd', prob, _accum(v)).to(x.dtype)
                out[b, ix] = self.o(y.reshape(c.q_heads, n, c.head_dim).transpose(0, 1).reshape(n, -1)).to(out.dtype)
                final = (k, v)
            next_rows.append(final)
        return out, tuple(next_rows)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.width, c.eps), RMSNorm(c.width, c.eps)
        self.attn = GQA(c)
        self.gate = nn.Linear(c.width, c.intermediate, bias=False)
        self.up = nn.Linear(c.width, c.intermediate, bias=False)
        self.down = nn.Linear(c.intermediate, c.width, bias=False)

    def forward(self, x, positions, plans, previous):
        y, state = self.attn(self.n1(x), positions, plans, previous)
        x = x+y
        u = self.n2(x)
        return x+self.down(F.silu(self.gate(u))*self.up(u)), state


class DenseLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width, config.eps)
        self._cache_owner = object()
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=config.init_std)

    def forward(self, input_ids, document_ids, position_ids, valid, *, cache=None, use_cache=False):
        if input_ids.ndim != 2 or not input_ids.numel() or any(t.shape != input_ids.shape for t in (document_ids, position_ids, valid)):
            raise ValueError('expected nonempty matching [B,T] tensors')
        if any(t.dtype != torch.long for t in (input_ids, document_ids, position_ids)) or valid.dtype != torch.bool:
            raise ValueError('expected int64 IDs and bool valid mask')
        if any(t.device != input_ids.device for t in (document_ids, position_ids, valid)):
            raise ValueError('metadata device mismatch')
        if cache is not None and not use_cache:
            raise ValueError('cache requires use_cache')
        if use_cache and (self.training or torch.is_grad_enabled()):
            raise ValueError('cache requires eval and no_grad/inference_mode')
        b = input_ids.shape[0]
        if cache is not None and (cache.owner is not self._cache_owner or len(cache.rows) != b or len(cache.layers) != len(self.blocks)):
            raise ValueError('cache ownership or shape mismatch')
        plans, rows = segments(document_ids, position_ids, valid, cache.rows if cache else (None,)*b)
        x = self.embedding(input_ids)
        states = []
        for i, block in enumerate(self.blocks):
            x, state = block(x, position_ids, plans, cache.layers[i] if cache else (None,)*b)
            states.append(state)
        logits = F.linear(self.norm(x), self.embedding.weight).masked_fill(~valid[..., None], 0)
        return logits, DenseCache(self._cache_owner, rows, tuple(states)) if use_cache else None


def ntp_loss(logits, labels):
    """Labels are already shifted by the packer; do not shift a second time."""
    if labels.shape != logits.shape[:-1] or labels.dtype != torch.long:
        raise ValueError('invalid labels')
    return F.cross_entropy(_accum(logits).reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction='sum') / labels.ne(-100).sum().clamp_min(1)
