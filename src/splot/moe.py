"""Deterministic all-resident sparse expert reference, RFC 0004."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .dense import _accum


@dataclass(frozen=True)
class Route:
    indices: torch.Tensor  # valid flattened [B,T] positions
    probabilities: torch.Tensor  # [N,E], FP32 or FP64 oracle
    selected: torch.Tensor  # [N,K]
    weights: torch.Tensor  # [N,K]


class Expert(nn.Module):
    def __init__(self,d,m,std):
        super().__init__()
        self.gate=nn.Linear(d,m,bias=False)
        self.up=nn.Linear(d,m,bias=False)
        self.down=nn.Linear(m,d,bias=False)
        for p in self.parameters(): nn.init.normal_(p,std=std)

    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))


class SparseFFN(nn.Module):
    def __init__(self,d,m,experts=8,topk=2,std=.02):
        super().__init__()
        if min(d,m,experts,topk)<1 or topk>experts: raise ValueError('invalid expert dimensions')
        self.topk=topk
        self.router=nn.Linear(d,experts,bias=False)
        nn.init.normal_(self.router.weight,std=std)
        self.experts=nn.ModuleList(Expert(d,m,std) for _ in range(experts))
        self.shared=Expert(d,m,std)

    def forward(self,x,valid):
        if x.ndim!=3 or valid.shape!=x.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('expected [B,T,d] and bool valid [B,T]')
        flat=x.reshape(-1,x.shape[-1]); ix=valid.reshape(-1).nonzero().flatten()
        h=flat.index_select(0,ix)
        # FP32 router computation even under autocast; preserve FP64 for oracle.
        with torch.autocast(device_type=x.device.type,enabled=False):
            logits=F.linear(_accum(h),_accum(self.router.weight))
            p=logits.softmax(-1)
            chosen=p.argsort(dim=-1,descending=True,stable=True)[:,:self.topk]
            w=p.gather(1,chosen); w=w/w.sum(-1,keepdim=True)
        route=Route(ix,p,chosen,w)
        if not ix.numel(): return torch.zeros_like(x),route
        y=_accum(self.shared(h))
        for e,expert in enumerate(self.experts):
            rows,slots=(chosen==e).nonzero(as_tuple=True)
            if rows.numel():
                contribution=_accum(expert(h.index_select(0,rows)))*w[rows,slots,None]
                y=y.index_add(0,rows,contribution)
        return torch.zeros_like(flat).index_copy(0,ix,y.to(flat.dtype)).reshape_as(x),route


def balance_loss(records):
    """Mean over physical pools; aggregate valid assignments across all passes."""
    pools={}
    for (_,_,block),route in records: pools.setdefault(block,[]).append(route)
    losses=[]
    for routes in pools.values():
        p=torch.cat([r.probabilities for r in routes]); ids=torch.cat([r.selected for r in routes])
        e=p.shape[-1]
        if not p.shape[0]: losses.append(p.sum()*0); continue
        f=torch.bincount(ids.flatten(),minlength=e).to(p.dtype)/ids.numel()
        losses.append(e*(f*p.mean(0)).sum())
    if not losses: raise ValueError('no physical routing pools')
    return torch.stack(losses).mean()


@torch.no_grad()
def routing_summary(records,document_ids,position_ids):
    """Per-pass loads, entropy and same-document temporal/depth overlap (no raw text)."""
    result={}; previous={}
    for key,r in records:
        _,depth,block=key; n,e=r.probabilities.shape; k=r.selected.shape[-1]
        ids=torch.full((*document_ids.shape,k),-1,device=document_ids.device,dtype=torch.long)
        ids.reshape(-1,k)[r.indices]=r.selected
        loads=torch.bincount(r.selected.flatten(),minlength=e)
        # Flatten matched pairs after mask; use explicit indexing for [B,T,K].
        def pair(a,b,mask):
            aa,bb=a[mask],b[mask]
            return (aa[:,:,None]==bb[:,None,:]).any(-1).float().mean().item() if aa.numel() else None
        valid=ids[...,0]>=0
        temporal=valid[:,1:]&valid[:,:-1]&document_ids[:,1:].eq(document_ids[:,:-1])&position_ids[:,1:].eq(position_ids[:,:-1]+1)
        old=previous.get(block)
        result[key]={'tokens':n,'assignments':loads.tolist(),'dead':int(loads.eq(0).sum()),
                     'entropy':(-torch.special.xlogy(r.probabilities,r.probabilities).sum(-1).mean()).item() if n else None,
                     'time_overlap':pair(ids[:,1:],ids[:,:-1],temporal),
                     'depth_overlap':pair(ids,old,valid & old[...,0].ge(0)) if old is not None else None}
        previous[block]=ids
    return result
