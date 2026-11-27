"""Full SPLOT integration test; run from repository checkout."""
import tempfile
import unittest
from pathlib import Path
import torch
from experiment import train, evaluate, seed_all
from data_fixture import synthetic_documents, split_documents, pack
from splot import RecurrentConfig, SPLOTLM, training_objective


class Integration(unittest.TestCase):
    def test_sparse_resume_and_validation(self):
        splits=split_documents(synthetic_documents())
        blocks=pack(splits['train'],96)
        val=pack(splits['validation'],96)
        with tempfile.TemporaryDirectory() as tmp:
            def run(name,steps,resume=False):
                seed_all(99 if resume else 42)
                m=SPLOTLM(RecurrentConfig())
                o=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0)
                h=train(m,o,blocks,lambda m,x,y:training_objective(m,x,y,mode='A2',loops=2),
                        {'fixture':'integration-v1'},Path(tmp)/name,steps=steps,
                        batch_size=1,seed=42,device='cpu',resume=resume)
                v=evaluate(m,val,lambda m,x:m(*x,loops=2).logits[2],batch_size=1,device='cpu')
                return m,h,v
            a,ah,av=run('a',2)
            run('b',1)
            b,bh,bv=run('b',2,True)
            self.assertEqual(sum(p.numel() for p in a.parameters()),179952)
            for x,y in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(x,y,atol=0,rtol=0)
            self.assertEqual([r['objective'] for r in ah],[r['objective'] for r in bh])
            self.assertEqual(av,bv)
            self.assertGreater(av['ntp_labels'],0)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main(verbosity=2)
