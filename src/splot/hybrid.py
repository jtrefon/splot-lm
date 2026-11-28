"""Shared 3 GDN + 1 gated GQA core, dense FFNs, native MTP."""
from torch import nn
from .dense import GQA
from .gdn import GatedDeltaNet
from .mtp import MTPLM


class HybridLM(MTPLM):
    def __init__(self,config):
        if (config.prelude,config.core,config.coda)!=(2,4,2):
            raise ValueError('hybrid reference requires 2/4/2 topology')
        super().__init__(config)
        c=config.dense
        for block in self.core[:3]:
            block.attn=GatedDeltaNet(c,algorithm=config.gdn_algorithm)
        self.core[3].attn=GQA(c,gated=True)
        for m in self.core[3].attn.modules():
            if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=c.init_std)

    def parameter_count(self):
        c=self.config.dense
        d,h,D=c.width,c.q_heads,c.head_dim
        q,kv=h*D,c.kv_heads*D
        gqa=d*(2*q+2*kv)
        gdn=d*(5*q+2*h)+12*q+2*h+D
        return super().parameter_count()+3*(gdn-gqa)+d*q
