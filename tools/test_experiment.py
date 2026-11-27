import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from experiment import batch_indices, digest, train, evaluate, seed_all, logical_macs


def blocks():
    return [dict(input_ids=[i,1,0],document_ids=[0,0,-1],position_ids=[0,1,0],valid=[True,True,False],labels={1:[1,-100,-100],2:[-100]*3}) for i in range(5)]


def model():
    return torch.nn.Sequential(torch.nn.Embedding(5,4),torch.nn.Dropout(.2),torch.nn.Linear(4,5))


def objective(m,x,y):
    loss=torch.nn.functional.cross_entropy(m(x[0]).flatten(0,1),y[1].flatten(),ignore_index=-100)
    zero=loss.new_zeros(())
    return SimpleNamespace(total=loss,final_ntp=loss,early_ntp=zero,offset2=zero,balance=zero)


class HarnessTests(unittest.TestCase):
    def test_schedule(self):
        for epoch in range(3):
            rows=[i for step in range(epoch*3,epoch*3+3) for i in batch_indices(5,2,step,42)]
            self.assertEqual(sorted(rows),list(range(5)))
        self.assertEqual(len(batch_indices(5,2,2,42)),1)
        self.assertEqual(batch_indices(5,2,4,42),batch_indices(5,2,4,42))

    def test_resume_exact_and_reject_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            def run(path,steps,resume=False,identity=None):
                seed_all(42 if not resume else 99); m=model(); opt=torch.optim.AdamW(m.parameters(),lr=.01)
                h=train(m,opt,blocks(),objective,identity or {'id':1},Path(tmp)/path,steps=steps,batch_size=2,seed=42,device='cpu',resume=resume)
                return m,opt,h
            a,ao,ah=run('a',6); run('b',2); b,bo,bh=run('b',6,True)
            for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
            for x,y in zip(ao.state.values(),bo.state.values()):
                for key in x: torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
            for x,y in zip(ah,bh): self.assertEqual({k:v for k,v in x.items() if k!='step_seconds'},{k:v for k,v in y.items() if k!='step_seconds'})
            with self.assertRaises(ValueError): run('b',7,True,{'id':2})
            with self.assertRaises(FileExistsError): run('a',1)
            with self.assertRaises(ValueError): run('b',1,True)

    def test_token_weighted_eval(self):
        seed_all(42); m=model(); predict=lambda m,x:m(x[0])
        a=evaluate(m,blocks(),predict,batch_size=2,device='cpu'); b=evaluate(m,blocks(),predict,batch_size=5,device='cpu')
        self.assertEqual(a['ntp_labels'],5); self.assertAlmostEqual(a['mean_nll'],b['mean_nll'],places=6)

    def test_logical_macs(self):
        self.assertEqual(logical_macs(blocks(),[0,1],lambda n:n*n),8)

    def test_hash(self):
        self.assertEqual(digest({'b':2,'a':1}),digest({'a':1,'b':2}))
        self.assertNotEqual(digest({'a':1}),digest({'a':2}))


if __name__=='__main__':
    torch.set_num_threads(1); unittest.main(verbosity=2)
