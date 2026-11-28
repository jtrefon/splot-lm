"""Stdlib tests for the standalone byte-auditable pilot v2 prototype."""
import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import pilot_v2 as v2

T = 32
V = 32
TOKEN_BYTES = [b""] * 8 + [
    b"\xc4", b"\x85", b"\xc5", b"\xbc", b"a", b"b", b" ", b"\xc4\x85",
    b"\xc5\xbc", b"\xc3\xb3", b"\xe2\x82\xac", b"\xf0\x9f\x98\x80", b"\n",
    b"!", b"c", b"d", b"e", b"f", b"g", b"h", b"i", b"j", b"k", b"l",
]


def canonical(value):
    return v2._canonical(value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tokenizer_doc():
    return {
        "format": v2.TOKENIZER_FORMAT,
        "version": v2.TOKENIZER_VERSION,
        "actual_vocab_size": V,
        "special_token_ids": copy.deepcopy(v2.SPECIAL_IDS),
        "token_bytes": [value.hex() for value in TOKEN_BYTES],
    }


SOURCE_DEFS = [
    ("src-train-en", "fam-train-en", "train", "en", b"ab"),
    ("src-train-pl", "fam-train-pl", "train", "pl", "a\u017c".encode()),
    ("src-train-en2", "fam-train-en2", "train", "en", b"ab a"),
    ("src-val-en", "fam-val-en", "validation", "en", b"a b!"),
    ("src-val-pl", "fam-val-pl", "validation", "pl", "\u0105a".encode()),
    ("src-test-en", "fam-test-en", "test", "en", b"b a"),
    ("src-test-pl", "fam-test-pl", "test", "pl", "\u017c".encode()),
]

DOC_DEFS = [
    (0, "seg-0", "src-train-en", 0, 2, 0, 2, 0, 1, b"ab", [12, 13]),
    (1, "seg-1", "src-train-pl", 0, 3, 0, 3, 0, 1, "a\u017c".encode(),
     [12, 10, 11]),
    (2, "seg-2", "src-train-en2", 0, 4, 0, 2, 0, 2, b"ab", [12, 13]),
    (3, "seg-3", "src-train-en2", 0, 4, 2, 4, 1, 2, b" a", [14, 12]),
    (4, "seg-4", "src-val-en", 0, 4, 0, 4, 0, 1, b"a b!", [12, 14, 13, 21]),
    (5, "seg-5", "src-val-pl", 0, 3, 0, 3, 0, 1, "\u0105a".encode(), [15, 12]),
    (6, "seg-6", "src-test-en", 0, 3, 0, 3, 0, 1, b"b a", [13, 14, 12]),
    (7, "seg-7", "src-test-pl", 0, 2, 0, 2, 0, 1, "\u017c".encode(), [10, 11]),
]

SEGMENTS = {
    "train": [(0, [12, 13]), (1, [12, 10, 11]), (2, [12, 13]), (3, [14, 12])],
    "validation": [(4, [12, 14, 13, 21]), (5, [15, 12])],
    "test": [(6, [13, 14, 12]), (7, [10, 11])],
}


def make_sources():
    selected_by_source = {}
    for (_, _, source_id, _, _, segment_start, segment_end,
         _, _, _, _) in DOC_DEFS:
        selected_by_source[source_id] = (
            selected_by_source.get(source_id, 0) + segment_end - segment_start)
    result = []
    for source_id, family_id, split, language, full in SOURCE_DEFS:
        raw = b"RAW:" + full
        selected = selected_by_source[source_id]
        result.append({
            "source_id": source_id,
            "family_id": family_id,
            "language": language,
            "split": split,
            "source_url": f"https://example.invalid/{source_id}",
            "revision": "synthetic-rev-2",
            "rights_evidence_url": "https://example.invalid/rights",
            "raw_sha256": sha(raw),
            "content_sha256": sha(full),
            "raw_bytes": len(raw),
            "content_bytes": len(full),
            "selected_bytes": selected,
        })
    return result


def make_documents():
    sources = {item["source_id"]: item for item in make_sources()}
    result = []
    for (doc_id, segment_id, source_id, source_start, source_end,
         segment_start, segment_end, segment_index, segment_count,
         segment, tokens) in DOC_DEFS:
        source = sources[source_id]
        full = next(item[4] for item in SOURCE_DEFS if item[0] == source_id)
        result.append({
            "doc_id": doc_id,
            "segment_id": segment_id,
            "source_id": source_id,
            "family_id": source["family_id"],
            "language": source["language"],
            "split": source["split"],
            "source_byte_start": source_start,
            "source_byte_end": source_end,
            "segment_byte_start": segment_start,
            "segment_byte_end": segment_end,
            "full_document_sha256": sha(full),
            "segment_sha256": sha(segment),
            "segment_index": segment_index,
            "segment_count": segment_count,
            "truncated": (segment_end - segment_start) < (source_end - source_start),
            "selected_bytes": segment_end - segment_start,
            "token_count": len(tokens),
        })
    return result


def make_block(segments):
    input_ids = []
    document_ids = []
    position_ids = []
    valid = []
    control = []
    byte_lengths = []
    byte_offsets = []
    labels = {"1": [], "2": []}
    for doc_id, tokens in segments:
        segment_input = [v2.BOS] + list(tokens) + [v2.EOS]
        length = len(segment_input)
        input_ids.extend(segment_input)
        document_ids.extend([doc_id] * length)
        position_ids.extend(range(length))
        valid.extend([True] * length)
        control.extend([True] + [False] * len(tokens) + [True])
        lengths = [0] + [len(TOKEN_BYTES[token]) for token in tokens] + [0]
        offsets = []
        cursor = 0
        for token in tokens:
            offsets.append(cursor)
            cursor += len(TOKEN_BYTES[token])
        byte_lengths.extend(lengths)
        byte_offsets.extend([0] + offsets + [cursor])
        for offset in (1, 2):
            for index in range(length):
                labels[str(offset)].append(
                    segment_input[index + offset] if index + offset < length else v2.IGNORE)
    if len(input_ids) > T:
        raise AssertionError("fixture block exceeds sequence length")
    pad = T - len(input_ids)
    input_ids.extend([v2.PAD] * pad)
    document_ids.extend([-1] * pad)
    position_ids.extend([0] * pad)
    valid.extend([False] * pad)
    control.extend([True] * pad)
    byte_lengths.extend([0] * pad)
    byte_offsets.extend([-1] * pad)
    for offset in (1, 2):
        labels[str(offset)].extend([v2.IGNORE] * pad)
    return {
        "input_ids": input_ids,
        "document_ids": document_ids,
        "position_ids": position_ids,
        "valid": valid,
        "control_mask": control,
        "token_byte_lengths": byte_lengths,
        "token_byte_offsets": byte_offsets,
        "labels": labels,
    }


def make_source_bytes():
    return {
        "format": v2.SOURCE_FORMAT,
        "version": v2.SOURCE_VERSION,
        "sources": {
            source_id: {"raw": (b"RAW:" + full).hex(), "content": full.hex()}
            for source_id, _, _, _, full in SOURCE_DEFS
        },
    }


def make_fixture():
    sources = make_sources()
    documents = make_documents()
    blocks = {split: [make_block(items)] for split, items in SEGMENTS.items()}
    return {
        "sequence_length": T,
        "vocab_size": V,
        "sources": sources,
        "documents": documents,
        "segments": SEGMENTS,
        "blocks": blocks,
        "tokenizer": tokenizer_doc(),
        "source_bytes": make_source_bytes(),
    }


def blank_entity():
    return {key: 0 for key in v2.COUNT_FIELDS}


def add_entity(target, source):
    for key in v2.COUNT_FIELDS:
        target[key] += source[key]


def entity_for_doc(doc, tokens):
    content = len(tokens)
    selected = sum(len(TOKEN_BYTES[token]) for token in tokens)
    return {
        "valid_tokens": content + 2,
        "content_tokens": content,
        "ntp_labels": content + 1,
        "content_labels": content,
        "eos_labels": 1,
        "offset2_labels": content,
        "content_target_bytes": selected,
        "selected_bytes": selected,
    }


def counts_for(fixture):
    docs = {item["doc_id"]: item for item in fixture["documents"]}
    total = {"blocks": 0, **blank_entity(), "by_split": {},
             "by_language": {}, "by_family": {}}
    for split in v2.SPLIT_ORDER:
        split_entity = {"blocks": len(fixture["blocks"][split]), **blank_entity()}
        for doc_id, tokens in fixture["segments"][split]:
            doc = docs[doc_id]
            entity = entity_for_doc(doc, tokens)
            add_entity(split_entity, entity)
            add_entity(total, entity)
            family = total["by_family"].setdefault(doc["family_id"], blank_entity())
            language = total["by_language"].setdefault(doc["language"], blank_entity())
            add_entity(family, entity)
            add_entity(language, entity)
        total["blocks"] += split_entity["blocks"]
        total["by_split"][split] = split_entity
    return total


def write_fixture(root, fixture):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    tokenizer_bytes = canonical(fixture["tokenizer"])
    (root / "tokenizer-bytes.json").write_bytes(tokenizer_bytes)
    source_bytes = canonical(fixture["source_bytes"])
    (root / "source-bytes.json").write_bytes(source_bytes)
    docs = {item["doc_id"]: item for item in fixture["documents"]}
    shard_records = []
    shard_payloads = {}
    counts = counts_for(fixture)
    for split in v2.SPLIT_ORDER:
        payload = b"".join(canonical(block) + b"\n"
                           for block in fixture["blocks"][split])
        shard_payloads[split] = payload
        (root / f"{split}.jsonl").write_bytes(payload)
        split_counts = counts["by_split"][split]
        family_tokens = {}
        family_bytes = {}
        language_tokens = {}
        language_bytes = {}
        for doc_id, tokens in fixture["segments"][split]:
            doc = docs[doc_id]
            content = len(tokens)
            selected = sum(len(TOKEN_BYTES[token]) for token in tokens)
            family_tokens[doc["family_id"]] = family_tokens.get(doc["family_id"], 0) + content
            family_bytes[doc["family_id"]] = family_bytes.get(doc["family_id"], 0) + selected
            language_tokens[doc["language"]] = language_tokens.get(doc["language"], 0) + content
            language_bytes[doc["language"]] = language_bytes.get(doc["language"], 0) + selected
        shard_records.append({
            "split": split,
            "path": f"{split}.jsonl",
            "sha256": sha(payload),
            "bytes": len(payload),
            "blocks": split_counts["blocks"],
            "valid_tokens": split_counts["valid_tokens"],
            "content_tokens": split_counts["content_tokens"],
            "ntp_labels": split_counts["ntp_labels"],
            "content_labels": split_counts["content_labels"],
            "eos_labels": split_counts["eos_labels"],
            "offset2_labels": split_counts["offset2_labels"],
            "content_target_bytes": split_counts["content_target_bytes"],
            "selected_bytes": split_counts["selected_bytes"],
            "family_content_tokens": family_tokens,
            "family_content_bytes": family_bytes,
            "language_content_tokens": language_tokens,
            "language_content_bytes": language_bytes,
        })
    manifest = {
        "format": v2.FORMAT,
        "version": v2.VERSION,
        "sequence_length": fixture["sequence_length"],
        "vocab_size": fixture["vocab_size"],
        "tokenizer_sha256": sha(tokenizer_bytes),
        "tokenizer_special_ids": copy.deepcopy(v2.SPECIAL_IDS),
        "source_snapshot_sha256": v2._source_snapshot(fixture["sources"]),
        "source_bytes_sha256": sha(source_bytes),
        "sources": fixture["sources"],
        "documents": fixture["documents"],
        "shards": shard_records,
        "counts": counts,
        "logical_sha256": sha(b"".join(shard_payloads[split]
                                       for split in v2.SPLIT_ORDER)),
        "byte_coverage_sha256": v2.byte_coverage_digest(fixture["documents"]),
    }
    data = canonical(manifest)
    (root / "manifest.json").write_bytes(data)
    return sha(data)


class PilotV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="splot-t0028-")
        self.root = Path(self.tmp.name) / "dataset"
        self.fixture = make_fixture()
        self.sha = write_fixture(self.root, self.fixture)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, root=None, sha=None):
        return v2.load_v2_dataset(root or self.root, sha or self.sha)

    def test_positive_dataset_and_summary(self):
        dataset = self.load()
        summary = dataset["manifest"]
        self.assertEqual(summary["format"], v2.FORMAT)
        self.assertEqual(summary["sequence_length"], T)
        self.assertEqual(summary["vocab_size"], V)
        self.assertEqual(set(dataset["blocks"]), {"train", "validation"})
        self.assertEqual(summary["counts"]["blocks"], 3)
        self.assertEqual(summary["counts"]["valid_tokens"], 36)
        self.assertEqual(summary["counts"]["content_tokens"], 20)
        self.assertEqual(summary["counts"]["ntp_labels"], 28)
        self.assertEqual(summary["counts"]["content_labels"], 20)
        self.assertEqual(summary["counts"]["eos_labels"], 8)
        self.assertEqual(summary["counts"]["offset2_labels"], 20)
        self.assertEqual(summary["counts"]["content_target_bytes"], 21)
        self.assertEqual(summary["counts"]["selected_bytes"], 21)
        self.assertEqual(summary["counts"]["by_language"]["en"]["content_target_bytes"], 13)
        self.assertEqual(summary["counts"]["by_language"]["pl"]["content_target_bytes"], 8)
        self.assertEqual(summary["counts"]["by_split"]["test"]["content_target_bytes"], 5)
        self.assertFalse(summary["test_blocks_exposed"])
        self.assertNotIn(6, summary["documents"])
        self.assertNotIn(7, summary["documents"])
        self.assertIn(4, summary["documents"])
        self.assertEqual(len(summary["documents"]), 6)
        for key in ("logical_sha256", "byte_coverage_sha256", "source_snapshot_sha256",
                    "source_bytes_sha256"):
            self.assertRegex(summary[key], r"^[0-9a-f]{64}$")
        docs = {item["doc_id"]: item for item in self.fixture["documents"]}
        self.assertTrue(docs[2]["truncated"])
        self.assertTrue(docs[3]["truncated"])
        self.assertEqual(docs[2]["full_document_sha256"], docs[3]["full_document_sha256"])
        self.assertEqual(self.fixture["sources"][2]["selected_bytes"], 4)

    def test_hand_computed_bpb(self):
        dataset = self.load()
        targets = v2.scoring_targets(dataset, "validation")
        self.assertEqual(len(targets), 8)
        nlls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for target, nll in zip(targets, nlls):
            target["nll"] = nll
        result = v2.aggregate_content_bpb(targets)
        self.assertEqual(result["content_labels"], 6)
        self.assertEqual(result["content_bytes"], 7)
        self.assertEqual(result["eos_labels"], 2)
        self.assertAlmostEqual(result["content_nll_sum"], 2.3)
        self.assertAlmostEqual(result["eos_nll_sum"], 1.3)
        self.assertAlmostEqual(result["ordinary_ntp_nll_sum"], 3.6)
        self.assertAlmostEqual(result["content_bpb"], 2.3 / (math.log(2) * 7))
        self.assertEqual(result["ordinary_ntp_labels"], 8)
        kinds = [target["kind"] for target in targets]
        self.assertEqual(kinds, ["content", "content", "content", "content", "eos",
                                 "content", "content", "eos"])
        self.assertEqual([target["byte_length"] for target in targets],
                         [1, 1, 1, 1, 0, 2, 1, 0])

    def test_bpb_rejections(self):
        good = [{"nll": 0.1, "byte_length": 1, "kind": "content"}]
        self.assertGreater(v2.aggregate_content_bpb(good)["content_bpb"], 0)
        for bad in (
            [],
            [{"nll": float("nan"), "byte_length": 1, "kind": "content"}],
            [{"nll": float("inf"), "byte_length": 1, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 0, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 1, "kind": "eos"}],
            [{"nll": 0.1, "byte_length": 1, "kind": "control"}],
            [{"nll": 0.1, "byte_length": -1, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 1, "kind": "other"}],
            [{"nll": True, "byte_length": 1, "kind": "content"}],
            [{"nll": 0.1, "byte_length": 1}],
            [{"nll": 0.1, "byte_length": 1, "kind": "eos"}],
        ):
            with self.assertRaises(v2.PilotV2Error):
                v2.aggregate_content_bpb(bad)

    def test_test_split_not_exposed(self):
        dataset = self.load()
        with self.assertRaises(v2.PilotV2Error):
            v2.scoring_targets(dataset, "test")

    def test_deterministic_identity(self):
        with tempfile.TemporaryDirectory(prefix="splot-t0028-repeat-") as tmp:
            root = Path(tmp) / "dataset"
            sha = write_fixture(root, make_fixture())
            self.assertEqual(sha, self.sha)
            first = self.load()
            second = self.load(root, sha)
            self.assertEqual(first["manifest"]["logical_sha256"],
                             second["manifest"]["logical_sha256"])
            self.assertEqual(first["manifest"]["byte_coverage_sha256"],
                             second["manifest"]["byte_coverage_sha256"])
            self.assertEqual(first["manifest"]["counts"], second["manifest"]["counts"])

    def test_explicit_caps(self):
        self.assertEqual(v2.MAX_TOTAL_BYTES, 4 * 1024 * 1024)
        self.assertEqual(v2.MAX_TOTAL_BLOCKS, 128)
        self.assertEqual(v2.MAX_TRAIN_BLOCKS, 96)
        self.assertEqual(v2.MAX_VALIDATION_BLOCKS, 16)
        self.assertEqual(v2.MAX_TEST_BLOCKS, 16)
        self.assertEqual(v2.MAX_VOCAB, 16384)
        self.assertEqual(v2.SEQUENCE_LENGTH_MAX, 512)

    def assert_tamper_rejected(self, mutate, *, root=None, sha=None):
        with tempfile.TemporaryDirectory(prefix="splot-t0028-tamper-") as tmp:
            target = Path(tmp) / "dataset"
            fixture = copy.deepcopy(self.fixture)
            mutate(fixture)
            new_sha = write_fixture(target, fixture)
            with self.assertRaises(v2.PilotV2Error):
                v2.load_v2_dataset(target, new_sha)

    def test_block_and_label_tamper(self):
        cases = [
            lambda f: f["blocks"]["train"][0]["token_byte_lengths"].__setitem__(1, 2),
            lambda f: f["blocks"]["train"][0]["token_byte_offsets"].__setitem__(2, 99),
            lambda f: f["blocks"]["train"][0]["labels"]["1"].__setitem__(0, 999),
            lambda f: f["blocks"]["train"][0]["document_ids"].__setitem__(0, 999),
            lambda f: f["blocks"]["train"][0]["valid"].__setitem__(0, 1),
            lambda f: f["blocks"]["train"][0]["control_mask"].__setitem__(1, True),
            lambda f: f["blocks"]["train"][0]["labels"]["1"].__setitem__(T - 1, 0),
            lambda f: f["blocks"]["train"][0]["position_ids"].__setitem__(1, 5),
            lambda f: f["blocks"]["train"][0]["document_ids"].__setitem__(-1, 0),
            lambda f: f["blocks"]["train"][0]["input_ids"].__setitem__(-1, 1),
            lambda f: f["blocks"]["train"][0]["token_byte_offsets"].__setitem__(-1, 0),
        ]
        for mutate in cases:
            self.assert_tamper_rejected(mutate)

    def test_document_and_source_tamper(self):
        cases = [
            lambda f: f["documents"][0].__setitem__("source_id", "missing"),
            lambda f: f["documents"][0].__setitem__("split", "validation"),
            lambda f: f["documents"].append({
                **copy.deepcopy(f["documents"][0]), "segment_id": "seg-duplicate"}),
            lambda f: f["documents"][3].__setitem__("segment_byte_start", 1),
            lambda f: f["documents"][0].__setitem__("segment_sha256", "0" * 64),
            lambda f: f["documents"][2].__setitem__("full_document_sha256", "0" * 64),
            lambda f: f["documents"][0].__setitem__("selected_bytes", 3),
            lambda f: f["sources"][0].__setitem__("selected_bytes", 3),
        ]
        for mutate in cases:
            self.assert_tamper_rejected(mutate)

    def test_tokenizer_and_caps_tamper(self):
        self.assert_tamper_rejected(
            lambda f: f["tokenizer"]["token_bytes"].__setitem__(12, "ff"))
        self.assert_tamper_rejected(
            lambda f: f["tokenizer"]["special_token_ids"]["reserved"].__setitem__(0, 9))
        self.assert_tamper_rejected(
            lambda f: f["source_bytes"]["sources"]["src-train-en"].__setitem__(
                "content", b"zz".hex()))
        self.assert_tamper_rejected(
            lambda f: f["source_bytes"]["sources"]["src-train-en"].__setitem__(
                "raw", b"zz".hex()))
        self.assert_tamper_rejected(lambda f: f.__setitem__("sequence_length", 513))
        self.assert_tamper_rejected(lambda f: f.__setitem__("vocab_size", 16385))

    def test_invalid_utf8_rejected(self):
        with tempfile.TemporaryDirectory(prefix="splot-t0028-utf8-") as tmp:
            root = Path(tmp) / "dataset"
            fixture = copy.deepcopy(self.fixture)
            fixture["tokenizer"]["token_bytes"][12] = "ff"
            content = b"\xffb"
            raw = b"RAW:" + content
            source = fixture["sources"][0]
            source["raw_sha256"] = sha(raw)
            source["content_sha256"] = sha(content)
            source["raw_bytes"] = len(raw)
            source["content_bytes"] = len(content)
            fixture["source_bytes"]["sources"][source["source_id"]] = {
                "raw": raw.hex(), "content": content.hex()}
            document = fixture["documents"][0]
            document["full_document_sha256"] = sha(content)
            document["segment_sha256"] = sha(content)
            new_sha = write_fixture(root, fixture)
            with self.assertRaisesRegex(v2.PilotV2Error, "valid UTF-8"):
                v2.load_v2_dataset(root, new_sha)

    def test_manifest_digest_tamper(self):
        for key in ("source_snapshot_sha256", "byte_coverage_sha256"):
            with tempfile.TemporaryDirectory(prefix="splot-t0028-digest-") as tmp:
                root = Path(tmp) / "dataset"
                write_fixture(root, copy.deepcopy(self.fixture))
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                manifest[key] = "0" * 64
                data = canonical(manifest)
                (root / "manifest.json").write_bytes(data)
                with self.assertRaises(v2.PilotV2Error):
                    v2.load_v2_dataset(root, sha(data))

    def test_manifest_and_file_tamper(self):
        # Noncanonical manifest with a valid hash.
        raw = (self.root / "manifest.json").read_bytes() + b" "
        (self.root / "manifest.json").write_bytes(raw)
        with self.assertRaises(v2.PilotV2Error):
            self.load(sha=sha(raw))

        # Tokenizer hash mismatch.
        with tempfile.TemporaryDirectory(prefix="splot-t0028-file-") as tmp:
            root = Path(tmp) / "dataset"
            write_fixture(root, copy.deepcopy(self.fixture))
            (root / "tokenizer-bytes.json").write_bytes(
                (root / "tokenizer-bytes.json").read_bytes() + b" ")
            with self.assertRaises(v2.PilotV2Error):
                self.load(root, self.sha)

        # Source-bytes artifact hash mismatch.
        with tempfile.TemporaryDirectory(prefix="splot-t0028-source-") as tmp:
            root = Path(tmp) / "dataset"
            write_fixture(root, copy.deepcopy(self.fixture))
            (root / "source-bytes.json").write_bytes(
                (root / "source-bytes.json").read_bytes() + b" ")
            with self.assertRaises(v2.PilotV2Error):
                self.load(root, self.sha)

        # Shard hash mismatch.
        with tempfile.TemporaryDirectory(prefix="splot-t0028-shard-") as tmp:
            root = Path(tmp) / "dataset"
            write_fixture(root, copy.deepcopy(self.fixture))
            (root / "train.jsonl").write_bytes((root / "train.jsonl").read_bytes() + b"\n")
            with self.assertRaises(v2.PilotV2Error):
                self.load(root, self.sha)

        # Extra file.
        with tempfile.TemporaryDirectory(prefix="splot-t0028-extra-") as tmp:
            root = Path(tmp) / "dataset"
            write_fixture(root, copy.deepcopy(self.fixture))
            (root / "extra.txt").write_text("x")
            with self.assertRaises(v2.PilotV2Error):
                self.load(root, self.sha)

        # Symlink shard.
        with tempfile.TemporaryDirectory(prefix="splot-t0028-link-") as tmp:
            root = Path(tmp) / "dataset"
            write_fixture(root, copy.deepcopy(self.fixture))
            target = root / "train-real.jsonl"
            (root / "train.jsonl").replace(target)
            os.symlink(target, root / "train.jsonl")
            with self.assertRaises(v2.PilotV2Error):
                self.load(root, self.sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
