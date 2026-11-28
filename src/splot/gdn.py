"""Learned Gated DeltaNet v1, slow differentiable reference (RFC 0003)."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .dense import RMSNorm, _accum


def delta_scan(q, k, v, g, beta, state):
    """[T,H,D] projected inputs, [T,H] gates, [H,D,D] key-by-value state."""
    outputs = []
    for qt, kt, vt, gt, bt in zip(q, k, v, g, beta):
        decayed = gt.exp()[:, None, None]*state
        error = vt-torch.einsum('hk,hkv->hv', kt, decayed)
        state = decayed+torch.einsum('hk,hv->hkv', kt, bt[:, None]*error)
        outputs.append(torch.einsum('hk,hkv->hv', qt, state))
    return torch.stack(outputs), state


def delta_chunk(q, k, v, g, beta, state):
    """Independent dense triangular solve; small finite-log-decay oracle only."""
    q, k, v, g, beta = (x.transpose(0, 1) for x in (q, k, v, g, beta))
    cumulative = g.cumsum(-1)
    decay = (cumulative[:, :, None]-cumulative[:, None, :]).clamp_max(0).exp()
    gram = k@k.transpose(-1, -2)
    lower = (beta[:, :, None]*decay*gram).tril(-1)
    eye = torch.eye(k.shape[1], dtype=k.dtype, device=k.device)
    rhs = beta[:, :, None]*(v-cumulative.exp()[:, :, None]*(k@state))
    u = torch.linalg.solve_triangular(lower+eye, rhs, upper=False, unitriangular=True)
    out = cumulative.exp()[:, :, None]*(q@state)+((q@k.transpose(-1, -2))*decay).tril()@u
    final = cumulative[:, -1].exp()[:, None, None]*state+k.transpose(-1, -2)@(decay[:, -1, :, None]*u)
    return out.transpose(0, 1), final


def delta_chunk64(q, k, v, g, beta, state):
    """Compose the existing delta_chunk oracle over consecutive <=64-token slices.

    State is carried across slices with no detach; the final short slice is
    solved by the same oracle. This is an experimental reference path, not a
    production kernel, and changes no GDN equations.
    """
    if q.shape[0] < 1:
        raise ValueError('delta_chunk64 requires a nonempty sequence')
    outputs = []
    current = state
    for start in range(0, q.shape[0], 64):
        end = min(start+64, q.shape[0])
        out, current = delta_chunk(q[start:end], k[start:end], v[start:end],
                                   g[start:end], beta[start:end], current)
        outputs.append(out)
    return torch.cat(outputs, dim=0), current


class GatedDeltaNet(nn.Module):
    """Equal Q/K/V head widths; raw QKV history and FP32 matrix state."""
    ALGORITHMS = ('scan', 'chunk', 'chunk64')

    def __init__(self, c, *, algorithm='scan'):
        super().__init__()
        if algorithm not in self.ALGORITHMS:
            raise ValueError('unknown delta algorithm')
        self.algorithm = algorithm
        self.c = c
        self.heads, self.dim = c.q_heads, c.head_dim
        self.width = self.heads*self.dim
        d, h, w = c.width, self.heads, self.width
        self.q = nn.Linear(d,w,bias=False)
        self.k = nn.Linear(d,w,bias=False)
        self.v = nn.Linear(d,w,bias=False)
        self.z = nn.Linear(d,w,bias=False)
        self.a = nn.Linear(d,h,bias=False)
        self.b = nn.Linear(d,h,bias=False)
        self.o = nn.Linear(w,d,bias=False)
        self.conv = nn.Parameter(torch.empty(4,3*w))  # oldest to newest
        self.log_A = nn.Parameter(torch.empty(h))
        self.dt_bias = nn.Parameter(torch.empty(h))
        self.norm = RMSNorm(self.dim,c.eps)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=c.init_std)
        nn.init.uniform_(self.conv,-.5,.5)
        with torch.no_grad():
            self.log_A.uniform_(1.,16.).log_()
            dt = torch.empty_like(self.dt_bias).uniform_(math.log(.001),math.log(.1)).exp()
            self.dt_bias.copy_(dt+torch.log(-torch.expm1(-dt)))

    def run(self, x, previous, *, algorithm=None):
        """One contiguous valid document run; history stores raw projected QKV."""
        algorithm = self.algorithm if algorithm is None else algorithm
        if algorithm not in self.ALGORITHMS:
            raise ValueError('unknown delta algorithm')
        raw = torch.cat((self.q(x),self.k(x),self.v(x)),dim=-1)
        if previous is None:
            state = _accum(raw).new_zeros(self.heads,self.dim,self.dim)
            history = raw.new_zeros(3,3*self.width)
        else:
            state, history = previous
        window = torch.cat((history,raw),dim=0)
        conv = F.silu((window.unfold(0,4,1)*self.conv.T).sum(-1)).to(raw.dtype)
        q,k,v = (_accum(y.reshape(-1,self.heads,self.dim)) for y in conv.split(self.width,dim=-1))
        q = q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)/math.sqrt(self.dim)
        k = k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
        g = -_accum(self.log_A).exp()*F.softplus(_accum(self.a(x))+_accum(self.dt_bias))
        beta = _accum(self.b(x)).sigmoid()
        # Explicit contractions must remain FP32 under autocast (FP64 oracle preserved).
        with torch.autocast(device_type=x.device.type,enabled=False):
            if algorithm == 'scan': out,state = delta_scan(q,k,v,g,beta,state)
            elif algorithm == 'chunk': out,state = delta_chunk(q,k,v,g,beta,state)
            elif algorithm == 'chunk64': out,state = delta_chunk64(q,k,v,g,beta,state)
            else: raise ValueError('unknown delta algorithm')
        z = F.silu(self.z(x).reshape(-1,self.heads,self.dim))
        y = self.o((self.norm(out)*z).reshape(-1,self.width).to(x.dtype))
        return y,(state,window[-3:].clone())

    def forward(self,x,positions,plans,previous):
        out = torch.zeros_like(x); states=[]
        for b,runs in enumerate(plans):
            final = previous[b]
            for indices,continuing in runs:
                ix = torch.tensor(indices,device=x.device)
                if continuing and final is None: raise ValueError('missing GDN prefix state')
                y,final = self.run(x[b,ix],final if continuing else None)
                out[b,ix] = y.to(out.dtype)
            states.append(final)
        return out,tuple(states)
