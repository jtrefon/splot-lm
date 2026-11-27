"""Focused tests for the bounded splot-pilot-blocks-v1 loader."""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import pilot_ingestion as pilot


def canonical(value):
    return pilot._canonical(value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(family, split, language):
    digest = sha(f'{family}:{split}:{language}'.encode())
    return {
        'family_id': family, 'language': language, 'split': split,
        'content_sha256': digest, 'raw_sha256': digest,
        'revision': 'synthetic-rev-1',
        'rights_evidence_url': 'https://example.invalid/rights',
        'source_url': 'https://example.invalid/source',
    }


def block(doc_id, vocab_size=264, seed=0, bad_label=False):
    rng = random.Random(seed)
    content = [8 + rng.randrange(vocab_size - 8) for _ in range(510)]
    input_ids = [1] + content + [2]
    document_ids = [doc_id] * 512
    position_ids = list(range(512))
    valid = [True] * 512
    labels = {}
    for offset in (1, 2):
        values = []
        for index in range(512):
            if index + offset < 512:
                values.append(input_ids[index + offset])
            else:
                values.append(-100)
        labels[str(offset)] = values
    if bad_label:
        labels['1'][0] = vocab_size + 1
    return {'input_ids': input_ids, 'document_ids': document_ids,
            'position_ids': position_ids, 'valid': valid, 'labels': labels}


def shard_bytes(blocks):
    return b''.join(canonical(value) + b'\n' for value in blocks)


def write_fixture(root, *, duplicate_doc=False, bad_label=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    sources = [source('train-en', 'train', 'en'), source('train-pl', 'train', 'pl'),
               source('val-en', 'validation', 'en'), source('test-pl', 'test', 'pl')]
    train = [block(index, seed=index, bad_label=bad_label and index == 0)
             for index in range(32)]
    validation = [block(100 + index, seed=100 + index) for index in range(2)]
    test = [block(200, seed=200)]
    if duplicate_doc:
        validation[0]['document_ids'] = train[0]['document_ids']
    parts = {'train': shard_bytes(train), 'validation': shard_bytes(validation),
             'test': shard_bytes(test)}
    shards = []
    for split in ('train', 'validation', 'test'):
        blocks = {'train': train, 'validation': validation, 'test': test}[split]
        valid = sum(sum(value['valid']) for value in blocks)
        ntp = sum(sum(value != -100 for value in value['labels']['1']) for value in blocks)
        offset = sum(sum(value != -100 for value in value['labels']['2']) for value in blocks)
        (root / f'{split}.jsonl').write_bytes(parts[split])
        shards.append({'split': split, 'path': f'{split}.jsonl',
                       'sha256': sha(parts[split]), 'bytes': len(parts[split]),
                       'blocks': len(blocks), 'valid_tokens': valid,
                       'ntp_labels': ntp, 'offset2_labels': offset})
    tokenizer = canonical({'algorithm': 'synthetic-test', 'actual_vocab_size': 264})
    (root / 'tokenizer.json').write_bytes(tokenizer)
    logical = sha(b''.join(parts[split] for split in ('train', 'validation', 'test')))
    manifest = {
        'format': pilot.FORMAT, 'version': 1, 'sequence_length': 512,
        'vocab_size': 264, 'tokenizer_sha256': sha(tokenizer),
        'source_snapshot_sha256': pilot._source_snapshot(sources),
        'sources': sources, 'shards': shards, 'logical_sha256': logical,
    }
    data = canonical(manifest)
    (root / 'manifest.json').write_bytes(data)
    return sha(data), sources


class PilotLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='splot-t0017-pilot-')
        self.root = Path(self.tmp.name) / 'dataset'
        self.sha, self.sources = write_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_train_validation_and_counts(self):
        loaded = pilot.load_pilot_dataset(self.root, self.sha)
        self.assertEqual(set(loaded['blocks']), {'train', 'validation'})
        self.assertEqual(len(loaded['blocks']['train']), 32)
        self.assertEqual(len(loaded['blocks']['validation']), 2)
        summary = loaded['manifest']
        self.assertEqual(summary['format'], pilot.FORMAT)
        self.assertEqual(summary['sequence_length'], 512)
        self.assertEqual(summary['vocab_size'], 264)
        self.assertEqual(summary['model_vocab_size'], 32768)
        self.assertEqual(summary['unused_token_id_count'], 32768 - 264)
        self.assertEqual(summary['splits']['train']['valid_tokens'], 32 * 512)
        self.assertEqual(summary['splits']['train']['ntp_labels'], 32 * 511)
        self.assertEqual(summary['splits']['train']['offset2_labels'], 32 * 510)
        self.assertTrue(summary['test_validated_only'])
        for split in ('train', 'validation'):
            for item in loaded['blocks'][split]:
                self.assertEqual(set(item['labels']), {1, 2})
                for offset in (1, 2):
                    values = item['labels'][offset]
                    for index, value in enumerate(values):
                        if value != -100:
                            self.assertEqual(value, item['input_ids'][index + offset])

    def test_manifest_hash_source_and_model_rejection(self):
        with self.assertRaises(pilot.PilotError):
            pilot.load_pilot_dataset(self.root, '0' * 64)
        with self.assertRaises(pilot.PilotError):
            pilot.load_pilot_dataset(self.root, self.sha, expected_model_vocab=32767)
        with self.assertRaises(pilot.PilotError):
            pilot.load_pilot_dataset(self.root, self.sha,
                                     expected_source_snapshot_sha256='0' * 64)

    def test_tamper_extra_file_and_duplicate_document(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0017-tamper-') as tmp:
            copy_root = Path(tmp) / 'copy'
            shutil.copytree(self.root, copy_root)
            (copy_root / 'manifest.json').write_bytes(
                (copy_root / 'manifest.json').read_bytes() + b' ')
            with self.assertRaises(pilot.PilotError):
                pilot.load_pilot_dataset(copy_root, self.sha)
            copy_root2 = Path(tmp) / 'copy2'
            shutil.copytree(self.root, copy_root2)
            (copy_root2 / 'extra.txt').write_text('x')
            with self.assertRaises(pilot.PilotError):
                pilot.load_pilot_dataset(copy_root2, self.sha)
            dup = Path(tmp) / 'duplicate'
            dup_sha, _ = write_fixture(dup, duplicate_doc=True)
            with self.assertRaises(pilot.PilotError):
                pilot.load_pilot_dataset(dup, dup_sha)
            bad = Path(tmp) / 'bad-label'
            bad_sha, _ = write_fixture(bad, bad_label=True)
            with self.assertRaises(pilot.PilotError):
                pilot.load_pilot_dataset(bad, bad_sha)

    def test_shard_hash_mismatch(self):
        with tempfile.TemporaryDirectory(prefix='splot-t0017-shard-') as tmp:
            copy_root = Path(tmp) / 'copy'
            shutil.copytree(self.root, copy_root)
            data = (copy_root / 'train.jsonl').read_bytes()
            (copy_root / 'train.jsonl').write_bytes(data + b'\n')
            with self.assertRaises(pilot.PilotError):
                pilot.load_pilot_dataset(copy_root, self.sha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PilotLoaderTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
