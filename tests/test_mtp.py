import unittest
import torch
from splot import RecurrentConfig, RecurrentLM, MTPLM, ntp_loss, training_objective
from data_fixture import Document, pack


def fixture():
    blocks = pack([Document('a','a','en','text','abc'), Document('b','b','pl','text','żół'),
                   Document('c','c','en','text','test')], 16)
    return (tuple(torch.tensor([b[k] for b in blocks]) for k in
                  ('input_ids','document_ids','position_ids','valid')),
            {k: torch.tensor([b['labels'][k] for b in blocks]) for k in (1,2)})


class MTPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.m = MTPLM(RecurrentConfig()).double().eval()
        self.x, self.labels = fixture()

    def equal(self, a, b):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_inventory_and_unchanged_target(self):
        torch.manual_seed(42)
        base = RecurrentLM(self.m.config).double().eval()
        self.assertEqual(sum(p.numel() for p in self.m.parameters()), 83776)
        self.assertEqual(self.m.parameter_count()-base.config.parameter_count(), 1056)
        with torch.no_grad():
            a = base(*self.x,loops=2,readouts=(1,2))
            b = self.m(*self.x,loops=2,readouts=(1,2),offset2_depths=(1,2))
            for r in (1,2): self.equal(a.logits[r],b.logits[r])
        self.assertEqual(sum(p is self.m.embedding.weight for p in self.m.parameters()),1)
        self.assertEqual(self.m(*self.x).offset2,{})

    def test_exact_objective_and_depth_one_dedup(self):
        for r in (1,2,4):
            depths = tuple(sorted({1,r}))
            out = self.m(*self.x,loops=r,readouts=range(1,r+1),offset2_depths=depths)
            final = ntp_loss(out.logits[r],self.labels[1])
            early = sum(ntp_loss(out.logits[i],self.labels[1]) for i in range(1,r))/max(1,r-1)
            aux = sum(ntp_loss(out.offset2[i],self.labels[2]) for i in depths)/len(depths)
            for mode in ('A0','A1','A2'):
                obj = training_objective(self.m,self.x,self.labels,mode=mode,loops=r)
                expected = final + (.25*early if mode != 'A0' else 0) + (.25*aux if mode == 'A2' else 0)
                self.equal(obj.total,expected)
            self.assertEqual(set(out.offset2),set(depths))
        out = self.m(*self.x,loops=1,offset2_depths=(1,1))
        self.assertEqual(list(out.offset2),[1])

    def test_causal_offset_and_document_isolation(self):
        x,d,p,v = self.x
        a = self.m(*self.x,loops=4,readouts=(1,4),offset2_depths=(1,4))
        alt=x.clone(); alt[0,2:5]=70
        b = self.m(alt,d,p,v,loops=4,readouts=(1,4),offset2_depths=(1,4))
        for r in (1,4):
            self.equal(a.offset2[r][0,:2],b.offset2[r][0,:2])
            self.equal(a.offset2[r][0,5:],b.offset2[r][0,5:])
            self.equal(a.offset2[r][1],b.offset2[r][1])
            self.equal(a.offset2[r][~v],torch.zeros_like(a.offset2[r][~v]))
        for k in (1,2):
            for row in range(x.shape[0]):
                for t in range(x.shape[1]):
                    expected = x[row,t+k].item() if t+k<x.shape[1] and v[row,t] and d[row,t]==d[row,t+k] else -100
                    self.assertEqual(self.labels[k][row,t].item(),expected)

    def test_auxiliary_gradient_and_inactive_controls(self):
        for mode in ('A0','A1','A2'):
            self.m.zero_grad(set_to_none=True)
            obj=training_objective(self.m,self.x,self.labels,mode=mode)
            obj.total.backward()
            if mode=='A2':
                for p in self.m.offset_projection.parameters(): self.assertGreater(p.grad.norm().item(),0)
                self.assertGreater(self.m.offset_norm.weight.grad.norm().item(),0)
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in self.m.parameters()))
            else:
                self.assertIsNone(self.m.offset_projection.weight.grad)
                self.assertIsNone(self.m.offset_norm.weight.grad)
        self.m.zero_grad(set_to_none=True)
        out=self.m(*self.x,loops=2,readouts=(1,2),offset2_depths=(1,2))
        ntp_loss(out.offset2[2],self.labels[2]).backward()
        self.assertGreater(self.m.embedding.weight.grad.norm().item(),0)
        self.assertGreater(self.m.core[0].attn.q.weight.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in self.m.coda.parameters()))

    def test_all_ignored_labels(self):
        labels={k:torch.full_like(v,-100) for k,v in self.labels.items()}
        obj=training_objective(self.m,self.x,labels)
        obj.total.backward()
        self.assertEqual(obj.total.item(),0)
        self.assertEqual((obj.ntp_count,obj.offset2_count),(0,0))
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero().item()==0 for p in self.m.parameters()))

    def test_cached_offset_equivalence_and_no_extra_state(self):
        with torch.no_grad():
            kw=dict(loops=4,readouts=(1,4),offset2_depths=(1,4),use_cache=True)
            full=self.m(*self.x,**kw)
            plain=self.m(*self.x,loops=4,readouts=(1,4),use_cache=True)
            self.assertEqual(len(full.cache.states),len(plain.cache.states))
            cache=None; pieces={1:[],4:[]}
            for lo,hi in ((0,3),(3,9),(9,16)):
                out=self.m(*(x[:,lo:hi] for x in self.x),cache=cache,**kw)
                cache=out.cache
                for r in pieces: pieces[r].append(out.offset2[r])
            for r in pieces: self.equal(full.offset2[r],torch.cat(pieces[r],1))

    def test_invalid_objective_inputs(self):
        for kw in ({'mode':'bad'},{'loops':0},{'offset_weight':-1},{'depth_weight':float('nan')}):
            with self.assertRaises(ValueError): training_objective(self.m,self.x,self.labels,**kw)
        with self.assertRaises(ValueError): training_objective(self.m,self.x,{1:self.labels[1]})
        with self.assertRaises(ValueError): self.m(*self.x,loops=2,offset2_depths=(1,))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
