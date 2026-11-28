"""Stdlib tests for the bounded streaming corpus prototype."""
import copy
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import pilot_stream as stream
import pilot_stream_producer as producer

T = 12
V = 264
TOKEN_BYTES = tuple([b""] * 8 + [bytes([value]) for value in range(256)])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(source_id, family, split, language, content):
    return producer.SourceSpec(
        source_id=source_id, family_id=family, language=language, split=split,
        source_url=f"https://example.invalid/{source_id}",
        revision="synthetic-stream-v1",
        rights_evidence_url="https://example.invalid/rights",
        raw=b"RAW:" + content, content=content)


def segment(doc_id, source_id, start, end, index, count, content,
            seg_start=None, seg_end=None):
    return producer.SegmentSpec(
        doc_id=doc_id, segment_id=f"seg-{doc_id:03d}",
        source_id=source_id, source_byte_start=start, source_byte_end=end,
        segment_byte_start=start if seg_start is None else seg_start,
        segment_byte_end=end if seg_end is None else seg_end,
        segment_index=index, segment_count=count,
        token_ids=tuple(8 + value for value in content))


def make_spec():
    tail = b"ab"
    return producer.StreamSpec(
        sequence_length=T, vocab_size=V, token_bytes=TOKEN_BYTES,
        sources=(
            source("src-train-en", "fam-train-en", "train", "en", b"ab"),
            source("src-train-pl", "fam-train-pl", "train", "pl", "a\u017c".encode()),
            source("src-train-tail", "fam-train-tail", "train", "en", tail),
            source("src-val", "fam-val", "validation", "en", b"abc"),
            source("src-test", "fam-test", "test", "en", b"z"),
        ),
        segments=(
            segment(0, "src-train-en", 0, 2, 0, 1, b"ab"),
            segment(1, "src-train-pl", 0, 3, 0, 1, "a\u017c".encode()),
            segment(2, "src-train-tail", 0, 2, 0, 2, b"a", 0, 1),
            segment(3, "src-train-tail", 0, 2, 1, 2, b"b", 1, 2),
            segment(4, "src-val", 0, 3, 0, 1, b"abc"),
            segment(5, "src-test", 0, 1, 0, 1, b"z"),
        ),
        shards_per_split={"train": 2, "validation": 1, "test": 1},
    )


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="splot-t0032-")
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.summary = producer.write_dataset(make_spec(), self.dataset)
        self.manifest_sha = self.summary["manifest_sha256"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_positive_loader_counts_and_streaming(self):
        loaded = stream.load_stream_dataset(self.dataset, self.manifest_sha)
        self.assertEqual(loaded["format"], stream.FORMAT)
        self.assertEqual(loaded["sequence_length"], T)
        self.assertEqual(loaded["vocab_size"], V)
        self.assertEqual(loaded["counts"]["blocks"], 4)
        self.assertEqual(loaded["counts"]["valid_tokens"], 23)
        self.assertEqual(loaded["counts"]["content_tokens"], 11)
        self.assertEqual(loaded["counts"]["ntp_labels"], 17)
        self.assertEqual(loaded["counts"]["eos_labels"], 6)
        self.assertEqual(loaded["counts"]["content_target_bytes"], 11)
        self.assertEqual(loaded["counts"]["selected_bytes"], 11)
        self.assertEqual(loaded["splits"]["train"]["blocks"], 2)
        self.assertEqual(loaded["splits"]["train"]["content_tokens"], 7)
        self.assertFalse(loaded["test_blocks_exposed"])
        self.assertEqual(sum(1 for _ in stream.iter_blocks(self.dataset, self.manifest_sha, "train")), 2)
        self.assertEqual(sum(1 for _ in stream.iter_blocks(self.dataset, self.manifest_sha, "validation")), 1)
        for block in stream.iter_blocks(self.dataset, self.manifest_sha, "train"):
            self.assertEqual(set(block["labels"]), {1, 2})
            self.assertEqual(len(block["input_ids"]), T)

    def test_hand_computed_bpb_and_targets(self):
        targets = list(stream.iter_targets(self.dataset, self.manifest_sha, "validation"))
        self.assertEqual(len(targets), 4)
        self.assertEqual([target["kind"] for target in targets],
                         ["content", "content", "content", "eos"])
        for target, nll in zip(targets, [0.1, 0.2, 0.3, 0.4]):
            target["nll"] = nll
        result = stream.aggregate_bpb(targets)
        self.assertEqual(result["content_labels"], 3)
        self.assertEqual(result["content_bytes"], 3)
        self.assertEqual(result["eos_labels"], 1)
        self.assertAlmostEqual(result["content_nll_sum"], 0.6)
        self.assertAlmostEqual(result["eos_nll_sum"], 0.4)
        self.assertAlmostEqual(result["ordinary_ntp_nll_sum"], 1.0)
        self.assertAlmostEqual(result["content_bpb"], 0.6 / (math.log(2) * 3))
        for bad in (
            [],
            [{"nll": -0.1, "byte_length": 1, "kind": "content"}],
            [{"nll": float("nan"), "byte_length": 1, "kind": "content"}],
            [{"nll": 10 ** 400, "byte_length": 1, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 0, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 1, "kind": "eos"}],
        ):
            with self.assertRaises(stream.StreamError):
                stream.aggregate_bpb(bad)

    def test_forged_target_identity(self):
        targets = list(stream.iter_targets(self.dataset, self.manifest_sha, "validation"))
        self.assertEqual(len(stream.verify_targets(
            self.dataset, self.manifest_sha, "validation", targets)), 4)
        forged = copy.deepcopy(targets)
        forged[0]["byte_length"] = 9
        with self.assertRaises(stream.StreamError):
            stream.verify_targets(self.dataset, self.manifest_sha, "validation", forged)
        duplicate = copy.deepcopy(targets)
        duplicate[1]["block_index"] = duplicate[0]["block_index"]
        duplicate[1]["position"] = duplicate[0]["position"]
        with self.assertRaises(stream.StreamError):
            stream.verify_targets(self.dataset, self.manifest_sha, "validation", duplicate)

    def test_test_non_exposure(self):
        with self.assertRaises(stream.StreamError):
            list(stream.iter_blocks(self.dataset, self.manifest_sha, "test"))
        with self.assertRaises(stream.StreamError):
            list(stream.iter_targets(self.dataset, self.manifest_sha, "test"))

    def test_deterministic_and_existing_output(self):
        other = self.root / "other"
        summary = producer.write_dataset(make_spec(), other)
        self.assertEqual(summary["manifest_sha256"], self.manifest_sha)
        for name in ("manifest.json", "tokenizer-bytes.json", "sources.json",
                     "documents.jsonl", "train-000.jsonl", "train-001.jsonl",
                     "validation-000.jsonl", "test-000.jsonl"):
            self.assertEqual((self.dataset / name).read_bytes(),
                             (other / name).read_bytes())
        with self.assertRaises(producer.ProducerError):
            producer.write_dataset(make_spec(), self.dataset)

    def test_file_hash_and_path_rejections(self):
        cases = []
        extra = self.root / "extra"
        shutil.copytree(self.dataset, extra)
        (extra / "extra.txt").write_text("x")
        cases.append(extra)
        shard = self.root / "shard"
        shutil.copytree(self.dataset, shard)
        (shard / "train-000.jsonl").write_bytes(
            (shard / "train-000.jsonl").read_bytes() + b"\n")
        cases.append(shard)
        source = self.root / "source"
        shutil.copytree(self.dataset, source)
        (source / "sources/src-train-en.content").write_bytes(b"zz")
        cases.append(source)
        link = self.root / "link"
        shutil.copytree(self.dataset, link)
        target = link / "train-real.jsonl"
        (link / "train-000.jsonl").replace(target)
        (link / "train-000.jsonl").symlink_to(target)
        cases.append(link)
        for root in cases:
            with self.assertRaises(stream.StreamError):
                stream.load_stream_dataset(root, self.manifest_sha)

    def test_semantic_tokenizer_and_counts_rejections(self):
        tokenizer = self.root / "tokenizer"
        shutil.copytree(self.dataset, tokenizer)
        doc = json.loads((tokenizer / "tokenizer-bytes.json").read_text())
        doc["token_bytes"][263] = ""
        raw = stream._canonical(doc)
        (tokenizer / "tokenizer-bytes.json").write_bytes(raw)
        manifest = json.loads((tokenizer / "manifest.json").read_text())
        manifest["tokenizer_sha256"] = sha(raw)
        manifest_raw = stream._canonical(manifest)
        (tokenizer / "manifest.json").write_bytes(manifest_raw)
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(tokenizer, sha(manifest_raw))

        counts = self.root / "counts"
        shutil.copytree(self.dataset, counts)
        manifest = json.loads((counts / "manifest.json").read_text())
        manifest["counts"]["blocks"] = 99
        manifest_raw = stream._canonical(manifest)
        (counts / "manifest.json").write_bytes(manifest_raw)
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(counts, sha(manifest_raw))

    def test_complete_alphabet_requires_264_vocab(self):
        with self.assertRaises(producer.ProducerError):
            producer.write_dataset(producer.StreamSpec(
                sequence_length=T, vocab_size=263,
                token_bytes=tuple([b""] * 8 + [bytes([i]) for i in range(255)]),
                sources=make_spec().sources, segments=make_spec().segments,
                shards_per_split={"train": 2, "validation": 1, "test": 1}),
                self.root / "bad")


if __name__ == "__main__":
    unittest.main(verbosity=2)
