import unittest
import torch
from splot import (DenseConfig, RecurrentConfig, SPLOTLM, SparseFFN, balance_loss,
                   routing_summary, training_objective, ModelRunner, generate)
from test_mtp import fixture


class MoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x,self.labels=fixture()
        self.m=SPLOTLM(RecurrentConfig()).double().eval()

    def equal(self,a,b):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_dispatch_and_gradient_oracle(self):
        m=SparseFFN(8,12).double(); x=torch.randn(2,4,8,dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([[True,True,False,True],[True,False,True,True]])
        y,r=m(x,mask)
        rows=[]
        for h in x[mask]:
            p=m.router(h).softmax(-1)
            ids=sorted(range(8),key=lambda e:(-p[e].item(),e))[:2]
            w=p[ids]/p[ids].sum()
            rows.append(m.shared(h)+sum(w[i]*m.experts[e](h) for i,e in enumerate(ids)))
        oracle=torch.zeros_like(x); oracle[mask]=torch.stack(rows)
        self.equal(y,oracle); self.equal(r.weights.sum(-1),torch.ones(6,dtype=torch.float64))
        self.assertEqual(r.selected.numel(),12)
        params=(x,*m.parameters())
        a=torch.autograd.grad(y.square().sum(),params,allow_unused=True)
        b=torch.autograd.grad(oracle.square().sum(),params,allow_unused=True)
        for u,v in zip(a,b):
            if u is None: self.assertIsNone(v)
            else: self.equal(u,v)

    def test_ties_padding_and_balance(self):
        m=SparseFFN(8,12).double()
        with torch.no_grad(): m.router.weight.zero_()
        x=torch.randn(2,3,8,dtype=torch.float64); valid=torch.tensor([[True,False,True],[False,False,True]])
        y,r=m(x,valid)
        self.assertEqual(r.selected.tolist(),[[0,1]]*3)
        self.equal(y[~valid],torch.zeros_like(y[~valid]))
        loss=balance_loss([(('C',1,0),r)]); self.equal(loss,torch.tensor(1.,dtype=torch.float64))
        loss.backward(); self.assertGreater(m.router.weight.grad.norm().item(),0)
        z,empty=m(x,torch.zeros_like(valid)); self.equal(z,torch.zeros_like(z))
        self.assertEqual(empty.selected.shape,(0,2)); self.assertEqual(balance_loss([(('C',1,0),empty)]).item(),0)
        self.assertRaises(ValueError,SparseFFN,8,12,8,9)

    def test_exact_candidate_counts(self):
        self.assertEqual(sum(p.numel() for p in self.m.parameters()),179952)
        for d,h,kv,m,expected in ((384,6,2,1024,43055076),(768,12,4,2048,146987784)):
            c=RecurrentConfig(dense=DenseConfig(vocab_size=32768,width=d,layers=8,q_heads=h,kv_heads=kv,head_dim=64,intermediate=m))
            with torch.device('meta'): model=SPLOTLM(c)
            self.assertEqual(sum(p.numel() for p in model.parameters()),expected)
            self.assertEqual(model.parameter_count(),expected)

    def test_depth_document_cache_and_routes(self):
        with torch.no_grad():
            for r in (1,2,4):
                kw=dict(loops=r,readouts=tuple(range(1,r+1)),offset2_depths=(1,r),use_cache=True)
                full=self.m(*self.x,**kw); cache=None; chunks=[]
                for lo,hi in ((0,3),(3,9),(9,16)):
                    o=self.m(*(v[:,lo:hi] for v in self.x),cache=cache,**kw);cache=o.cache;chunks.append(o)
                for k in full.logits: self.equal(full.logits[k],torch.cat([o.logits[k] for o in chunks],1))
                for k in full.offset2: self.equal(full.offset2[k],torch.cat([o.offset2[k] for o in chunks],1))
                for (ka,sa),(kb,sb) in zip(full.cache.states,cache.states):
                    self.assertEqual(ka,kb)
                    for a,b in zip(sa,sb):
                        for u,v in zip(a,b): self.equal(u,v)
                self.assertEqual(len(full.routing),4*r)
                for key,route in full.routing:
                    self.assertEqual(route.selected.shape,(int(self.x[3].sum()),2))
                    for lo,hi,o in zip((0,3,9),(3,9,16),chunks):
                        table=torch.full((*self.x[0].shape,2),-1,dtype=torch.long)
                        table.reshape(-1,2)[route.indices]=route.selected
                        part=dict(o.routing)[key]
                        self.assertTrue(torch.equal(table[:,lo:hi][self.x[3][:,lo:hi]],part.selected))
                stats=routing_summary(full.routing,self.x[1],self.x[2]);self.assertEqual(len(stats),4*r)
                for val in stats.values(): self.assertEqual(sum(val['assignments']),2*int(self.x[3].sum()))
            shallow=self.m(*self.x,loops=2,readouts=(1,2)); deep=self.m(*self.x,loops=4,readouts=(1,2,4))
            for r in (1,2): self.equal(shallow.logits[r],deep.logits[r])
            x,d,p,v=self.x; alt=x.clone();alt[0,2:5]=70
            z=self.m(alt,d,p,v,loops=4,readouts=(1,2,4))
            for r in (1,2,4):
                self.equal(deep.logits[r][0,:2],z.logits[r][0,:2]);self.equal(deep.logits[r][0,5:],z.logits[r][0,5:]);self.equal(deep.logits[r][1],z.logits[r][1])

    def test_speculative_replay(self):
        t=ModelRunner(self.m,4);s=ModelRunner(self.m,1,offset2=True);prompt=[1,80,81,82]
        state,_=t.advance(None,prompt); expected=[]
        for _ in range(6):
            y=int(state.next_logits.argmax());expected.append(y);state,_=t.advance(state,[y])
        for strategy in ('D1','D2'):
            for greedy in (True,False):
                z=generate(t,s,prompt,6,strategy=strategy,greedy=greedy,rng=torch.Generator().manual_seed(9))
                if greedy:self.assertEqual(z.tokens,tuple(expected))
                for runner,actual in ((t,z.target),(s,z.shallow)):
                    fresh,_=runner.advance(None,prompt+list(z.tokens));self.equal(fresh.next_logits,actual.next_logits)
                    for (ka,a),(kb,b) in zip(fresh.cache.states,actual.cache.states):
                        self.assertEqual(ka,kb)
                        for u,v in zip(a[0],b[0]):self.equal(u,v)
                print(strategy,greedy,z.accepted,z.rejected)

    def test_sparse_objective_and_precision(self):
        m=self.m.float().train(); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0); losses=[]
        for _ in range(12):
            opt.zero_grad(set_to_none=True); o=training_objective(m,self.x,self.labels,loops=2)
            torch.testing.assert_close(o.total,o.final_ntp+.25*o.early_ntp+.25*o.offset2+.01*o.balance)
            o.total.backward()
            self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in m.parameters()))
            for b in m.core:self.assertGreater(b.moe.router.weight.grad.norm().item(),0)
            opt.step();losses.append(o.total.item())
        self.assertLess(losses[-1],losses[0]);print('12-step loss',losses[0],losses[-1])
        m.eval()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            o=m(*self.x,loops=2)
            self.assertTrue(all(r.probabilities.dtype==torch.float32 for _,r in o.routing))
            self.assertTrue(o.logits[2].isfinite().all())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
