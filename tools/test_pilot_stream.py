"""Stdlib tests for the bounded streaming corpus prototype."""
import copy
import dataclasses
import hashlib
import inspect
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pilot_stream as stream
import pilot_stream_audit as audit
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


def rewrite_manifest(root, mutate):
    path = Path(root) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    raw = stream._canonical(manifest)
    path.write_bytes(raw)
    return sha(raw)


def rewrite_sources(root, mutate):
    path = Path(root) / "sources.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    raw = stream._canonical(value)
    path.write_bytes(raw)
    return rewrite_manifest(
        root, lambda manifest: manifest.__setitem__("source_registry_sha256", sha(raw)))


def rewrite_documents(root, mutate):
    path = Path(root) / "documents.jsonl"
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        mutate(doc)
        lines.append(stream._canonical(doc) + b"\n")
    data = b"".join(lines)
    path.write_bytes(data)
    return rewrite_manifest(
        root, lambda manifest: manifest.__setitem__("document_metadata_sha256", sha(data)))


def refresh_shards(root):
    manifest_path = Path(root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logical = hashlib.sha256()
    for shard in manifest["shards"]:
        data = (Path(root) / shard["path"]).read_bytes()
        shard["bytes"] = len(data)
        shard["sha256"] = sha(data)
        logical.update(data)
    manifest["logical_sha256"] = logical.hexdigest()
    raw = stream._canonical(manifest)
    manifest_path.write_bytes(raw)
    return sha(raw)


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="splot-t0034-")
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
        self.assertEqual(loaded["target_identity_fields"],
                         list(stream.TARGET_IDENTITY_FIELDS))
        self.assertLessEqual(loaded["total_file_bytes"], stream.MAX_TOTAL_BYTES)
        identity = loaded["identity"]
        self.assertIsInstance(identity, stream.StreamIdentity)
        self.assertEqual(identity.manifest_sha256, self.manifest_sha)
        self.assertEqual(identity.counts, loaded["counts"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.manifest_sha256 = "0" * 64
        self.assertEqual(sum(1 for _ in stream.iter_blocks(
            self.dataset, self.manifest_sha, "train")), 2)
        self.assertEqual(sum(1 for _ in stream.iter_blocks(
            self.dataset, self.manifest_sha, "validation")), 1)
        for block in stream.iter_blocks(self.dataset, self.manifest_sha, "train"):
            self.assertEqual(set(block["labels"]), {1, 2})
            self.assertEqual(len(block["input_ids"]), T)
        first = next(stream.iter_targets(self.dataset, self.manifest_sha, "validation"))
        self.assertEqual(first["manifest_sha256"], self.manifest_sha)

    def test_hand_computed_bpb_and_targets(self):
        targets = list(stream.iter_targets(self.dataset, self.manifest_sha, "validation"))
        self.assertEqual(len(targets), 4)
        self.assertEqual([target["kind"] for target in targets],
                         ["content", "content", "content", "eos"])
        for target, nll in zip(targets, [0.1, 0.2, 0.3, 0.4]):
            target["nll"] = nll
        result = stream.aggregate_bpb(iter(targets))
        self.assertEqual(result["content_labels"], 3)
        self.assertEqual(result["content_bytes"], 3)
        self.assertEqual(result["eos_labels"], 1)
        self.assertAlmostEqual(result["content_nll_sum"], 0.6)
        self.assertAlmostEqual(result["eos_nll_sum"], 0.4)
        self.assertAlmostEqual(result["ordinary_ntp_nll_sum"], 1.0)
        self.assertAlmostEqual(result["content_bpb"], 0.6 / (math.log(2) * 3))
        combined = stream.verify_and_aggregate_bpb(
            self.dataset, self.manifest_sha, "validation", iter(targets))
        self.assertEqual(combined["verified_targets"], 4)
        self.assertEqual(combined["content_labels"], 3)
        self.assertAlmostEqual(combined["content_bpb"], 0.6 / (math.log(2) * 3))
        for bad in (
            [],
            [{"nll": -0.1, "byte_length": 1, "kind": "content"}],
            [{"nll": float("nan"), "byte_length": 1, "kind": "content"}],
            [{"nll": 10 ** 400, "byte_length": 1, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 0, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 1, "kind": "eos"}],
            [{"nll": True, "byte_length": 1, "kind": "content"}],
            [{"nll": 0.1, "byte_length": True, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 1, "kind": "bogus"}],
            [{"nll": 0.1, "kind": "content"}],
        ):
            with self.assertRaises(stream.StreamError):
                stream.aggregate_bpb(bad)

    def test_forged_target_identity(self):
        targets = list(stream.iter_targets(self.dataset, self.manifest_sha, "validation"))
        summary = stream.verify_targets(
            self.dataset, self.manifest_sha, "validation", iter(targets))
        self.assertEqual(summary["verified_targets"], 4)
        self.assertEqual(summary["content"], 3)
        self.assertEqual(summary["eos"], 1)
        self.assertEqual(summary["manifest_sha256"], self.manifest_sha)
        for mutate in (
            lambda item: item.__setitem__("byte_length", 9),
            lambda item: item.__setitem__("manifest_sha256", "0" * 64),
            lambda item: item.__setitem__("target_token_id", True),
            lambda item: item.__setitem__("split", "train"),
            lambda item: item.__setitem__("family_id", "other"),
            lambda item: item.__setitem__("position", -1),
            lambda item: item.__setitem__("byte_length", 1 << 30),
            lambda item: item.pop("kind"),
            lambda item: item.__setitem__("extra", 1),
        ):
            forged = copy.deepcopy(targets)
            mutate(forged[0])
            with self.assertRaises(stream.StreamError):
                stream.verify_targets(self.dataset, self.manifest_sha,
                                      "validation", forged)
        duplicate = copy.deepcopy(targets)
        duplicate[1]["block_index"] = duplicate[0]["block_index"]
        duplicate[1]["position"] = duplicate[0]["position"]
        with self.assertRaises(stream.StreamError):
            stream.verify_targets(self.dataset, self.manifest_sha,
                                  "validation", duplicate)
        with self.assertRaises(stream.StreamError):
            stream.verify_targets(self.dataset, self.manifest_sha,
                                  "validation", targets[:-1])
        with self.assertRaises(stream.StreamError):
            stream.verify_targets(self.dataset, self.manifest_sha,
                                  "validation", targets + [targets[-1]])

    def test_test_non_exposure(self):
        with self.assertRaises(stream.StreamError):
            list(stream.iter_blocks(self.dataset, self.manifest_sha, "test"))
        with self.assertRaises(stream.StreamError):
            list(stream.iter_targets(self.dataset, self.manifest_sha, "test"))
        with self.assertRaises(stream.StreamError):
            stream.verify_targets(self.dataset, self.manifest_sha, "test", [])
        with self.assertRaises(stream.StreamError):
            stream.verify_and_aggregate_bpb(
                self.dataset, self.manifest_sha, "test", [])

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
        self.assertNotIn("verify", inspect.signature(producer.write_dataset).parameters)
        with self.assertRaises(TypeError):
            producer.write_dataset(make_spec(), self.root / "bypass", verify=False)
        preserved = self.root / "other.staging"
        preserved.mkdir()
        (preserved / "keep.txt").write_text("keep")
        producer.write_dataset(make_spec(), self.root / "third")
        self.assertEqual((preserved / "keep.txt").read_text(), "keep")

    def test_partial_staging_cleanup(self):
        failed = self.root / "failed"
        keep = self.root / ".failed.staging-keep"
        keep.mkdir()
        (keep / "keep.txt").write_text("keep")
        with mock.patch("pilot_stream_producer._block",
                        side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                producer.write_dataset(make_spec(), failed)
        self.assertFalse(failed.exists())
        leftovers = [path for path in self.root.glob(".failed.staging-*")
                     if path.name != keep.name]
        self.assertFalse(leftovers)
        self.assertEqual((keep / "keep.txt").read_text(), "keep")
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep.txt").write_text("keep")
        with self.assertRaises(producer.ProducerError):
            producer.write_dataset(make_spec(), existing)
        self.assertEqual((existing / "keep.txt").read_text(), "keep")

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
        source_root = self.root / "source"
        shutil.copytree(self.dataset, source_root)
        (source_root / "sources/src-train-en.content").write_bytes(b"zz")
        cases.append(source_root)
        link = self.root / "link"
        shutil.copytree(self.dataset, link)
        target = link / "train-real.jsonl"
        (link / "train-000.jsonl").replace(target)
        (link / "train-000.jsonl").symlink_to(target)
        cases.append(link)
        for root in cases:
            with self.assertRaises(stream.StreamError):
                stream.load_stream_dataset(root, self.manifest_sha)
        root_link = self.root / "root-link"
        root_link.symlink_to(self.dataset, target_is_directory=True)
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(root_link, self.manifest_sha)
        traversal = self.root / "traversal"
        shutil.copytree(self.dataset, traversal)
        traversal_sha = rewrite_manifest(
            traversal, lambda manifest: manifest["shards"][0].__setitem__(
                "path", "../train-000.jsonl"))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(traversal, traversal_sha)

    def test_semantic_nonbool_and_cap_rejections(self):
        tokenizer = self.root / "tokenizer"
        shutil.copytree(self.dataset, tokenizer)
        doc = json.loads((tokenizer / "tokenizer-bytes.json").read_text())
        doc["token_bytes"][263] = ""
        raw = stream._canonical(doc)
        (tokenizer / "tokenizer-bytes.json").write_bytes(raw)
        tokenizer_sha = rewrite_manifest(
            tokenizer, lambda manifest: manifest.__setitem__("tokenizer_sha256", sha(raw)))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(tokenizer, tokenizer_sha)

        counts = self.root / "counts"
        shutil.copytree(self.dataset, counts)
        counts_sha = rewrite_manifest(
            counts, lambda manifest: manifest["counts"].__setitem__("blocks", 99))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(counts, counts_sha)

        nonbool = self.root / "nonbool"
        shutil.copytree(self.dataset, nonbool)
        nonbool_sha = rewrite_sources(
            nonbool, lambda value: value["sources"][0].__setitem__("raw_bytes", True))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(nonbool, nonbool_sha)

        shard_bytes = self.root / "shard-bytes"
        shutil.copytree(self.dataset, shard_bytes)
        shard_sha = rewrite_manifest(
            shard_bytes, lambda manifest: manifest["shards"][0].__setitem__(
                "bytes", float(manifest["shards"][0]["bytes"])))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(shard_bytes, shard_sha)

        doc_selected = self.root / "doc-selected"
        shutil.copytree(self.dataset, doc_selected)
        doc_sha = rewrite_documents(
            doc_selected, lambda value: value.__setitem__("selected_bytes", 1.0))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(doc_selected, doc_sha)

        cap = self.root / "cap"
        shutil.copytree(self.dataset, cap)
        cap_sha = rewrite_manifest(
            cap, lambda manifest: manifest["shards"][0].__setitem__(
                "bytes", stream.MAX_SHARD_BYTES + 1))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(cap, cap_sha)

        long_doc = self.root / "long-doc"
        shutil.copytree(self.dataset, long_doc)
        long_data = b"x" * (stream.MAX_METADATA_LINE + 10) + b"\n"
        (long_doc / "documents.jsonl").write_bytes(long_data)
        long_sha = rewrite_manifest(
            long_doc, lambda manifest: manifest.__setitem__(
                "document_metadata_sha256", sha(long_data)))
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(long_doc, long_sha)

        long_shard = self.root / "long-shard"
        shutil.copytree(self.dataset, long_shard)
        with (long_shard / "train-000.jsonl").open("ab") as handle:
            handle.write(b"x" * (stream.MAX_SHARD_LINE + 10) + b"\n")
        long_shard_sha = refresh_shards(long_shard)
        with self.assertRaises(stream.StreamError):
            stream.load_stream_dataset(long_shard, long_shard_sha)

    def test_bounded_full_read_guard(self):
        original = Path.read_bytes

        def guarded(path):
            if path.name.startswith(("train-", "validation-", "test-")) \
                    or "sources" in path.parts:
                raise AssertionError(f"unbounded read: {path}")
            return original(path)

        targets = list(stream.iter_targets(self.dataset, self.manifest_sha, "validation"))
        with mock.patch.object(Path, "read_bytes", guarded):
            stream.load_stream_dataset(self.dataset, self.manifest_sha)
            list(stream.iter_targets(self.dataset, self.manifest_sha, "validation"))
            stream.verify_targets(self.dataset, self.manifest_sha, "validation", targets)
            audit.audit(self.dataset, self.manifest_sha)

    def test_auditor_matches_and_is_independent(self):
        loaded = stream.load_stream_dataset(self.dataset, self.manifest_sha)
        result = audit.audit(self.dataset, self.manifest_sha)
        self.assertEqual(result["counts"], loaded["counts"])
        self.assertFalse(result["test_blocks_exposed"])
        self.assertEqual(result["test_blocks_audited"],
                         loaded["splits"]["test"]["blocks"])
        self.assertEqual(result["bounded_io"]["mode"], "incremental")
        lines = Path(audit.__file__).read_text(encoding="utf-8").splitlines()
        self.assertFalse(any(line.startswith("import pilot_stream")
                             or line.startswith("from pilot_stream")
                             for line in lines))

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
