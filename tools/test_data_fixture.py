import unittest
from data_fixture import (Document, encode, decode, digest, split_documents,
                          pack, synthetic_documents, manifest, BOS, EOS, PAD, IGNORE)


class DataContract(unittest.TestCase):
    def test_utf8_roundtrip(self):
        for s in ('Zażółć gęślą jaźń', 'a\t\r\n  b', 'e\u0301', '🧪 <|eos|>'):
            self.assertEqual(decode(encode(s)), s)
            self.assertTrue(all(8 <= i < 264 for i in encode(s)))

    def test_split_determinism_and_disjointness(self):
        docs = synthetic_documents()
        a = split_documents(docs)
        self.assertEqual(a, split_documents(reversed(docs)))
        self.assertTrue(all(a.values()))
        for i, part in enumerate(a.values()):
            for other in list(a.values())[i+1:]:
                self.assertFalse({digest(d.text) for d in part} & {digest(d.text) for d in other})
                self.assertFalse({d.group for d in part} & {d.group for d in other})
        self.assertEqual({d.language for d in a['train']}, {'en', 'pl'})

    def test_transitive_group_dedup(self):
        docs = [Document('a', 'g1', 'pl', 'text', 'ó'),
                Document('b', 'g2', 'pl', 'text', 'o\u0301'),
                Document('c', 'g2', 'en', 'text', 'other'),
                Document('d', 'g1', 'en', 'text', 'more')]
        split = split_documents(docs)
        self.assertEqual(sorted(len(p) for p in split.values()), [0, 0, 3])
        self.assertEqual(digest(docs[0].text), digest(docs[1].text))

    def test_packing_and_offset_masks(self):
        docs = [Document('a', 'a', 'en', 'text', 'ab'), Document('b', 'b', 'en', 'text', 'c')]
        b, = pack(docs, 9)
        self.assertEqual(b['input_ids'], [BOS, *encode('ab'), EOS, BOS, *encode('c'), EOS, PAD, PAD])
        self.assertEqual(b['position_ids'], [0, 1, 2, 3, 0, 1, 2, 0, 0])
        self.assertEqual(b['valid'], [True]*7 + [False]*2)
        self.assertEqual(b['labels'][1], [*encode('ab'), EOS, IGNORE, *encode('c'), EOS, IGNORE, IGNORE, IGNORE])
        self.assertEqual(b['labels'][2], [encode('b')[0], EOS, IGNORE, IGNORE, EOS, IGNORE, IGNORE, IGNORE, IGNORE])

    def test_no_silent_truncation_and_no_empty_batch(self):
        with self.assertRaises(ValueError):
            pack(synthetic_documents()[:1], 4)
        self.assertEqual(pack([], 8), [])
        with self.assertRaises(ValueError):
            split_documents([synthetic_documents()[0]] * 2)

    def test_manifest(self):
        m = manifest()
        self.assertEqual(sum(p['documents'] for p in m.values()), 80)
        self.assertEqual(m, manifest())
        self.assertTrue(all(len(p['sha256']) == 64 for p in m.values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
