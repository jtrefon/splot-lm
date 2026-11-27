"""Full SPLOT all-resident reference: hybrid mixing, recurrence, MoE and MTP."""
from dataclasses import dataclass
from .hybrid import HybridLM
from .mtp import MTPOutput
from .moe import SparseFFN, balance_loss


@dataclass(frozen=True)
class SparseOutput(MTPOutput):
    balance_loss: object


class SPLOTLM(HybridLM):
    def __init__(self,config,*,expert_intermediate=None):
        super().__init__(config)
        c=config.dense
        self.expert_intermediate=c.intermediate//2 if expert_intermediate is None else expert_intermediate
        if type(self.expert_intermediate) is not int or self.expert_intermediate<1:
            raise ValueError('invalid expert width')
        for block in self.core:
            del block.gate,block.up,block.down
            block.moe=SparseFFN(c.width,self.expert_intermediate,std=c.init_std)

    def parameter_count(self):
        c=self.config.dense
        return super().parameter_count()+4*(9*3*c.width*self.expert_intermediate+8*c.width-3*c.width*c.intermediate)

    def _apply_block(self,block,x,positions,plans,previous,gamma,valid,key):
        if key[0]!='C': return super()._apply_block(block,x,positions,plans,previous,gamma,valid,key)
        y,state=block.attn(block.n1(x),positions,plans,previous)
        x=x+gamma*y
        y,route=block.moe(block.n2(x),valid)
        return x+gamma*y,state,route

    def forward(self,*args,**kwargs):
        o=super().forward(*args,**kwargs)
        return SparseOutput(o.logits,o.hidden,o.cache,o.offset2,balance_loss(o.routing),routing=o.routing)
