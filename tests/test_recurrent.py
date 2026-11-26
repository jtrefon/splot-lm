import copy
import unittest
from dataclasses import replace
import torch
import torch.nn.functional as F
from splot import DenseConfig, DenseLM, RecurrentConfig, RecurrentLM, ntp_loss
from splot.dense import segments
from data_fixture import Document, pack


def fixture():
    blocks = pack([Document('a','a','en','text','abc'), Document('b','b','pl','text','żół'),
                   Document('c','c','en','text','test')], 16)
    return tuple(torch.tensor([b[k] for b in blocks]) for k in
                 ('input_ids','document_ids','position_ids','valid')), torch.tensor([b['labels'][1] for b in blocks])


class RecurrentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.c = RecurrentConfig()
        self.m = RecurrentLM(self.c).double().eval()
        self.data, self.labels = fixture()

    def equal(self, a, b):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_physical_count_and_r1_dense_equivalence(self):
        m = self.m
        self.assertEqual(sum(p.numel() for p in m.parameters()), self.c.parameter_count())
        dense = DenseLM(self.c.dense).double().eval()
        dense.embedding.load_state_dict(m.embedding.state_dict()); dense.norm.load_state_dict(m.norm.state_dict())
        for dst, src in zip(dense.blocks, [*m.prelude, *m.core, *m.coda]):
            dst.load_state_dict(src.state_dict())
        expected, _ = dense(*self.data)
        self.equal(expected, m(*self.data, loops=1).logits[1])
        self.assertEqual(len(list(m.parameters())), len({id(p) for p in m.parameters()}))

    def test_depth_prefix_invariance_variants(self):
        for injection in ('none','additive','convex'):
            for scale in ('unscaled','sqrt_cap','linear_cap'):
                m = RecurrentLM(replace(self.c, injection=injection, residual_scale=scale, depth_embedding=True)).double().eval()
                with torch.no_grad():
                    m.depth.normal_(std=.02)
                    a = m(*self.data, loops=2, readouts=(1,2))
                    b = m(*self.data, loops=4, readouts=(1,2,4))
                for r in (1,2):
                    self.equal(a.hidden[r], b.hidden[r]); self.equal(a.logits[r], b.logits[r])

    def test_cache_equivalence_and_distinct_logical_states(self):
        with torch.no_grad():
            for depth in (1,2,4):
                readouts = tuple(range(1,depth+1))
                full = self.m(*self.data, loops=depth, readouts=readouts)
                for cuts in (range(17), (0,3,9,16)):
                    cache, outputs = None, {r:[] for r in readouts}
                    for a,b in zip(cuts,list(cuts)[1:]):
                        z = self.m(*(x[:,a:b] for x in self.data), loops=depth, readouts=readouts, cache=cache, use_cache=True)
                        cache = z.cache
                        for r in readouts: outputs[r].append(z.logits[r])
                    for r in readouts: self.equal(torch.cat(outputs[r],1), full.logits[r])
                    states = dict(cache.states)
                    self.assertEqual(len(states),2+4*depth+2*depth)
                    self.assertEqual(len({s[0][0].data_ptr() for s in states.values()}),len(states))
                    for s in states.values():
                        self.assertEqual(s[0][0].shape[1],cache.rows[0].length)

    def test_causal_document_isolation_and_padding(self):
        x,doc,pos,valid = self.data
        original = self.m(*self.data, loops=4, readouts=(1,4))
        alt = x.clone(); alt[0,2:5] = 70
        changed = self.m(alt,doc,pos,valid,loops=4,readouts=(1,4))
        for r in (1,4):
            self.equal(original.logits[r][0,:2], changed.logits[r][0,:2])
            self.equal(original.logits[r][0,5:], changed.logits[r][0,5:])
            self.equal(original.logits[r][1], changed.logits[r][1])
            self.equal(original.logits[r][~valid], torch.zeros_like(original.logits[r][~valid]))

    def test_cache_branch_replay_and_contract(self):
        with torch.no_grad():
            kw = dict(loops=4,readouts=(1,4),use_cache=True)
            prefix = self.m(*(x[:,:3] for x in self.data), **kw).cache
            saved = [(k, s[0][0].clone()) for k,s in prefix.states]
            alt = self.data[0][:,3:4].clone(); alt[0,0] = 80
            self.m(alt,*(x[:,3:4] for x in self.data[1:]),cache=prefix,**kw)
            for k,t in saved: self.equal(t,dict(prefix.states)[k][0][0])
            replay = self.m(*(x[:,3:4] for x in self.data),cache=prefix,**kw)
            full = self.m(*(x[:,:4] for x in self.data),**kw)
            for r in (1,4): self.equal(replay.logits[r],full.logits[r][:,-1:])
            with self.assertRaises(ValueError): self.m(*(x[:,3:4] for x in self.data),loops=2,cache=prefix,use_cache=True)
            with self.assertRaises(ValueError): self.m(*(x[:,3:4] for x in self.data),loops=4,cache=prefix,use_cache=True)
        with self.assertRaises(ValueError): self.m(*self.data,loops=5)

    def test_shared_gradients_equal_untied_sum(self):
        m = self.m
        untied_core = [copy.deepcopy(m.core) for _ in range(2)]
        untied_coda = [copy.deepcopy(m.coda) for _ in range(2)]
        out = m(*self.data,loops=2,readouts=(1,2))
        sum(ntp_loss(y,self.labels) for y in out.logits.values()).backward()
        shared = [[p.grad.clone() for p in stack.parameters()] for stack in (m.core,m.coda)]
        m.zero_grad(set_to_none=True)
        x,doc,pos,valid = self.data
        plans,_ = segments(doc,pos,valid,(None,)*x.shape[0])
        def run(stack,h):
            for block in stack: h,_ = block(h,pos,plans,(None,)*x.shape[0])
            return h
        h = run(m.prelude,m.embedding(x)); losses=[]
        for core,coda in zip(untied_core,untied_coda):
            h = run(core,h)
            y = F.linear(m.norm(run(coda,h)),m.embedding.weight).masked_fill(~valid[...,None],0)
            losses.append(ntp_loss(y,self.labels))
        sum(losses).backward()
        for expected, copies in zip(shared,(untied_core,untied_coda)):
            for a,ps in zip(expected,zip(*(list(c.parameters()) for c in copies))):
                self.equal(a,sum(p.grad for p in ps))

    def test_fixture_training_and_finite_gradients(self):
        m = self.m.float().train(); opt = torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0)
        losses=[]
        for _ in range(12):
            opt.zero_grad(set_to_none=True)
            out=m(*self.data,loops=2,readouts=(1,2))
            loss=ntp_loss(out.logits[2],self.labels)+.25*ntp_loss(out.logits[1],self.labels)
            loss.backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters()))
            opt.step(); losses.append(loss.item())
        self.assertLess(losses[-1],losses[0])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
