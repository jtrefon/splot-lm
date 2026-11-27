import unittest
from collections import Counter
import torch
from splot import MTPLM, RecurrentConfig, ModelRunner, Prefix, generate
from splot.speculative import distribution, correction, verify_token


class Toy:
    """Two-token vocabulary; target next-token probabilities depend on last token."""
    def __init__(self, target=False, forced=None):
        self.target=target; self.forced=forced
        self.calls=[]

    def logits(self,tokens):
        if self.forced is not None:
            return torch.tensor(self.forced,dtype=torch.float64).log()
        p=([.8,.2] if tokens[-1]==0 else [.35,.65]) if self.target else ([.25,.75] if tokens[-1]==0 else [.9,.1])
        return torch.tensor(p,dtype=torch.float64).log()

    def advance(self,prefix,tokens):
        old=() if prefix is None else prefix.cache
        self.calls.append((old,tuple(tokens)))
        seq=old; out=[]
        for t in tokens:
            seq=seq+(t,); out.append(self.logits(seq))
        return Prefix(len(seq),seq,out[-1],torch.tensor([.6,.4],dtype=torch.float64).log()),torch.stack(out)


class SpeculativeTests(unittest.TestCase):
    def test_single_token_probability_identity(self):
        for p,q in (([.2,.3,.5],[.6,.3,.1]),([1.,0.,0.],[0.,1.,0.]),([.1,.2,.7],[.1,.2,.7])):
            p,q=torch.tensor(p,dtype=torch.float64),torch.tensor(q,dtype=torch.float64)
            accepted=torch.minimum(p,q)
            reject=1-accepted.sum()
            recovered=accepted+reject*correction(p,q) if reject>1e-14 else accepted
            torch.testing.assert_close(recovered,p,atol=1e-14,rtol=1e-14)
        rng=torch.Generator().manual_seed(0)
        self.assertEqual(verify_token(torch.tensor([1.,0.]),torch.tensor([0.,1.]),1,rng),(0,False))
        self.assertEqual(verify_token(torch.tensor([1.,0.]),torch.tensor([1.,0.]),0,rng),(0,True))

    def test_joint_sampling_both_proposals(self):
        # Target joint P(y1,y2) for prefix [0], not merely marginal frequencies.
        expected={(0,0):.64,(0,1):.16,(1,0):.07,(1,1):.13}
        for strategy in ('D1','D2'):
            rng=torch.Generator().manual_seed(123)
            counts=Counter(); n=3000
            for _ in range(n):
                z=generate(Toy(True),Toy(),[0],2,strategy=strategy,rng=rng)
                counts[z.tokens]+=1
                self.assertEqual(z.target.cache,(0,)+z.tokens)
                self.assertEqual(z.shallow.cache,z.target.cache)
            errors={k:abs(counts[k]/n-v) for k,v in expected.items()}
            print(strategy,'joint counts',dict(sorted(counts.items())),'max error',max(errors.values()))
            self.assertLess(max(errors.values()),.04)

    def test_forced_rejection_replay_at_each_position(self):
        # D2: q1 deterministic 0, q2 deterministic 0; target first accepts 0, next rejects 0.
        class Target(Toy):
            def logits(self,tokens):
                return torch.tensor([1.,0.] if len(tokens)==1 else [0.,1.],dtype=torch.float64).log()
        class Shallow(Toy):
            def advance(self,prefix,tokens):
                s,l=super().advance(prefix,tokens)
                return Prefix(s.length,s.cache,s.next_logits,torch.tensor([0.,float('-inf')])),l
        for target,expected in ((Toy(forced=[0.,1.]),(1,1)),(Target(),(0,1))):
            shallow=Shallow(forced=[1.,0.])
            out=generate(target,shallow,[0],2,strategy='D2',rng=torch.Generator().manual_seed(4))
            self.assertEqual(out.tokens,expected); self.assertEqual(out.target.cache,(0,)+expected)
            self.assertEqual(out.shallow.cache,out.target.cache)
            self.assertGreaterEqual(out.rejected,1)
            self.assertEqual(out.accepted+out.rejected,2)
            self.assertEqual(target.calls[1],((0,),(0,0)))
            self.assertEqual(target.calls[2],((0,),expected if expected[0]==0 else (1,)))

    def test_eos_budget_and_bonus(self):
        for strategy in ('D1','D2'):
            t,s=Toy(forced=[0.,1.]),Toy(forced=[0.,1.])
            z=generate(t,s,[0],5,strategy=strategy,eos=1,greedy=True)
            self.assertEqual(z.tokens,(1,)); self.assertEqual(z.target.cache,(0,1))
            z=generate(t,s,[0],0,strategy=strategy,greedy=True)
            self.assertEqual(z.tokens,()); self.assertEqual(z.rounds,0)
        # D1 identical distributions, all accepted plus bonus, no overshoot.
        z=generate(Toy(forced=[1.,0.]),Toy(forced=[1.,0.]),[1],3,greedy=True)
        self.assertEqual(z.tokens,(0,0,0)); self.assertEqual((z.accepted,z.proposed,z.rounds),(2,2,1))
        z=generate(Toy(forced=[0.,1.]),Toy(forced=[1.,0.]),[0],5,eos=1,greedy=True)
        self.assertEqual(z.tokens,(1,)); self.assertEqual(z.rejected,1)
        z=generate(Toy(forced=[1.,0.]),Toy(forced=[0.,1.]),[0],3,eos=1,greedy=True)
        self.assertEqual(z.tokens,(0,0,0))  # rejected draft EOS does not stop

    def test_actual_model_greedy_and_cache_replay(self):
        torch.manual_seed(42)
        m=MTPLM(RecurrentConfig()).double().eval()
        target=ModelRunner(m,4); shallow=ModelRunner(m,1,offset2=True)
        prompt=[1,80,81,82]; n=6
        state,_=target.advance(None,prompt); expected=[]
        for _ in range(n):
            token=int(state.next_logits.argmax()); expected.append(token)
            state,_=target.advance(state,[token])
        for strategy in ('D1','D2'):
            for greedy in (True,False):
                z=generate(target,shallow,prompt,n,strategy=strategy,greedy=greedy,rng=torch.Generator().manual_seed(9))
                if greedy: self.assertEqual(z.tokens,tuple(expected))
                self.assertEqual(z.target.length,len(prompt)+n)
                for runner,actual in ((target,z.target),(shallow,z.shallow)):
                    fresh,_=runner.advance(None,prompt+list(z.tokens))
                    torch.testing.assert_close(fresh.next_logits,actual.next_logits,atol=1e-10,rtol=1e-10)
                    for (ka,sa),(kb,sb) in zip(fresh.cache.states,actual.cache.states):
                        self.assertEqual(ka,kb)
                        for a,b in zip(sa[0],sb[0]): torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
                    if runner.offset2: torch.testing.assert_close(fresh.offset2_logits,actual.offset2_logits,atol=1e-10,rtol=1e-10)
                print('model',strategy,'greedy',greedy,'tokens',z.tokens,'proposed',z.proposed,'accepted',z.accepted,'rejected',z.rejected)

    def test_validation(self):
        for kw in ({'strategy':'D3'},{'draft_length':0},{'temperature':0},{'temperature':float('nan')},{'strategy':'D2','draft_length':3}):
            with self.assertRaises(ValueError): generate(Toy(True),Toy(),[0],2,**kw)
        with self.assertRaises(ValueError): generate(Toy(True),Toy(),[],2)
        with self.assertRaises(ValueError): distribution(torch.tensor([float('-inf'),float('-inf')]))
        with self.assertRaises(ValueError): verify_token(torch.tensor([1.,0.]),torch.tensor([1.,0.]),1,torch.Generator())
        torch.testing.assert_close(distribution(torch.tensor([1000.,-1000.]),1e-100),torch.tensor([1.,0.],dtype=torch.float64))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
