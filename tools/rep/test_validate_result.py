import json
from pathlib import Path
import tempfile
import unittest
from validate_result import validate, sha


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)/'T0001'/'attempt-001'; self.d.mkdir(parents=True)
        for n, s in {'summary.md':'# test\n','environment.json':'{}','metrics.jsonl':'{"loss":1}\n','execution.log':'test\n'}.items():
            (self.d/n).write_text(s)
        self.r = dict(schema_version=1, task_id='T0001', attempt=1, task_commit='a'*40, source_commit='b'*40, tested_commit='b'*40,
                      status='passed', checks={'cpu':{'status':'passed','detail':'exit 0','evidence':['execution.log']}},
                      files={p.name:sha(p) for p in self.d.iterdir()}, deviations=[], limitations=[], external_artifacts=[])

    def check(self):
        (self.d/'result.json').write_text(json.dumps(self.r))
        return validate(self.d,'T0001','b'*40,['cpu'])

    def test_pass(self): self.assertEqual(self.check(),'passed')
    def test_failure(self):
        self.r['status']='failed'; self.r['checks']['cpu']['status']='failed'; self.assertEqual(self.check(),'failed')
    def test_blocked(self):
        self.r['status']='blocked'; self.r['tested_commit']=None; self.r['checks']['cpu']['status']='blocked'; self.assertEqual(self.check(),'blocked')
    def test_status_mismatch(self):
        self.r['checks']['cpu']['status']='failed'
        with self.assertRaises(ValueError): self.check()
    def test_hash_mismatch(self):
        (self.d/'execution.log').write_text('changed')
        with self.assertRaises(ValueError): self.check()
    def test_pin_mismatch(self):
        self.r['tested_commit']='c'*40
        with self.assertRaises(ValueError): self.check()
    def test_missing_check(self):
        self.r['checks']={}
        with self.assertRaises(ValueError): self.check()
    def test_unlisted_file(self):
        (self.d/'extra.txt').write_text('unexpected')
        with self.assertRaises(ValueError): self.check()
    def test_deviation_not_pass(self):
        self.r['deviations']=['changed tolerance']
        with self.assertRaises(ValueError): self.check()


if __name__=='__main__': unittest.main(verbosity=2)
