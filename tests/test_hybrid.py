import unittest
import torch
from splot import DenseConfig, RecurrentConfig, HybridLM, GatedDeltaNet, ModelRunner, generate, training_objective
from test_mtp import fixture


class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.c = RecurrentConfig()
        self.m = HybridLM(self.c).double().eval()
        self.x,self.labels=fixture()

    def equal(self,a,b):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_learned_chunk_scan_and_gradients(self):
        m=GatedDeltaNet(self.c.dense).double()
        x=torch.randn(7,32,dtype=torch.float64,requires_grad=True)
        state=(torch.randn(4,8,8,dtype=torch.float64),torch.randn(3,96,dtype=torch.float64))
        y,s=m.run(x,state)
        a,t=m.run(x,state,algorithm='chunk')
        self.equal(y,a)
        for u,v in zip(s,t): self.equal(u,v)
        params=(x,*m.parameters())
        g1=torch.autograd.grad(y.square().sum()+s[0].square().sum(),params)
        g2=torch.autograd.grad(a.square().sum()+t[0].square().sum(),params)
        for u,v in zip(g1,g2): self.equal(u,v)
        y1,s1=m.run(x[:3],state); y2,s2=m.run(x[3:],s1)
        self.equal(y,torch.cat((y1,y2)))
        for u,v in zip(s,s2): self.equal(u,v)

    def test_counts(self):
        self.assertEqual(sum(p.numel() for p in self.m.parameters()),self.m.parameter_count())
        for d,h,kv,m in ((384,6,2,1024),(768,12,4,2048)):
            c=RecurrentConfig(dense=DenseConfig(vocab_size=32768,width=d,layers=8,q_heads=h,kv_heads=kv,head_dim=64,intermediate=m))
            with torch.device('meta'): model=HybridLM(c)
            self.assertEqual(sum(p.numel() for p in model.parameters()),model.parameter_count())

    def test_full_chunk_and_token_cache(self):
        with torch.no_grad():
            for r in (1,2,4):
                kw=dict(loops=r,readouts=tuple(range(1,r+1)),offset2_depths=(1,r),use_cache=True)
                full=self.m(*self.x,**kw)
                for cuts in (range(17),(0,3,9,16)):
                    cache=None; parts=[]
                    for lo,hi in zip(cuts,list(cuts)[1:]):
                        o=self.m(*(v[:,lo:hi] for v in self.x),cache=cache,**kw)
                        cache=o.cache; parts.append(o)
                    for key in full.logits: self.equal(full.logits[key],torch.cat([p.logits[key] for p in parts],1))
                    for key in full.offset2: self.equal(full.offset2[key],torch.cat([p.offset2[key] for p in parts],1))
                    for (ka,sa),(kb,sb) in zip(full.cache.states,cache.states):
                        self.assertEqual(ka,kb)
                        for a,b in zip(sa,sb):
                            if a is None: self.assertIsNone(b)
                            else:
                                for u,v in zip(a,b): self.equal(u,v)
                gdn=[s[0] for k,s in full.cache.states if k[0]=='C' and k[2]<3]
                self.assertEqual(len(gdn),3*r)
                self.assertEqual(len({s[0].data_ptr() for s in gdn}),3*r)

    def test_depth_and_document_isolation(self):
        a=self.m(*self.x,loops=2,readouts=(1,2),offset2_depths=(1,2))
        b=self.m(*self.x,loops=4,readouts=(1,2,4),offset2_depths=(1,2))
        for r in (1,2): self.equal(a.logits[r],b.logits[r]); self.equal(a.offset2[r],b.offset2[r])
        x,d,p,v=self.x; alt=x.clone(); alt[0,2:5]=70
        z=self.m(alt,d,p,v,loops=4,readouts=(1,2,4),offset2_depths=(1,2))
        for r in (1,2,4):
            self.equal(b.logits[r][0,:2],z.logits[r][0,:2]); self.equal(b.logits[r][0,5:],z.logits[r][0,5:]); self.equal(b.logits[r][1],z.logits[r][1])
            self.equal(b.logits[r][~v],torch.zeros_like(b.logits[r][~v]))

    def test_replay_and_speculation(self):
        t=ModelRunner(self.m,4); s=ModelRunner(self.m,1,offset2=True); prompt=[1,80,81,82]
        prefix,_=t.advance(None,prompt)
        saved=[(key,tuple(v.clone() for v in row[0])) for key,row in prefix.cache.states]
        t.advance(prefix,[40,41]); t.advance(prefix,[42])
        for (ka,a),(kb,b) in zip(saved,prefix.cache.states):
            self.assertEqual(ka,kb)
            for u,v in zip(a,b[0]): self.equal(u,v)
        expected=[]; state=prefix
        for _ in range(6):
            y=int(state.next_logits.argmax()); expected.append(y); state,_=t.advance(state,[y])
        for strategy in ('D1','D2'):
            for greedy in (True,False):
                z=generate(t,s,prompt,6,strategy=strategy,greedy=greedy,rng=torch.Generator().manual_seed(9))
                if greedy: self.assertEqual(z.tokens,tuple(expected))
                for runner,actual in ((t,z.target),(s,z.shallow)):
                    fresh,_=runner.advance(None,prompt+list(z.tokens))
                    self.equal(fresh.next_logits,actual.next_logits)
                    for (ka,a),(kb,b) in zip(fresh.cache.states,actual.cache.states):
                        self.assertEqual(ka,kb)
                        for u,v in zip(a[0],b[0]): self.equal(u,v)
                    if runner.offset2: self.equal(fresh.offset2_logits,actual.offset2_logits)
                print(strategy,greedy,'accepted',z.accepted,'rejected',z.rejected)

    def test_fp32_training_and_state_dtype(self):
        m=self.m.float().train(); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0)
        losses=[]
        for _ in range(12):
            opt.zero_grad(set_to_none=True); obj=training_objective(m,self.x,self.labels,loops=2)
            obj.total.backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))
            opt.step(); losses.append(obj.total.item())
        self.assertLess(losses[-1],losses[0]); print('hybrid A2 12 steps',losses[0],losses[-1])
        m.eval()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            out=m(*self.x,loops=2,use_cache=True)
            for k,rows in out.cache.states:
                if k[0]=='C' and k[2]<3:
                    for matrix,history in rows:
                        self.assertEqual(matrix.dtype,torch.float32); self.assertEqual(history.dtype,torch.bfloat16)
            self.assertTrue(out.logits[2].isfinite().all())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
