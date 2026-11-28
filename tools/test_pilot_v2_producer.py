"""Stdlib tests for the deterministic synthetic v2 producer."""
import copy
import tempfile
import unittest
from pathlib import Path

import pilot_v2 as v2
import pilot_v2_producer as producer
import test_pilot_v2 as fixture


def make_spec():
    sources = tuple(
        producer.SourceSpec(
            source_id=source_id,
            family_id=family_id,
            language=language,
            split=split,
            source_url=f"https://example.invalid/{source_id}",
            revision="synthetic-rev-2",
            rights_evidence_url="https://example.invalid/rights",
            raw=b"RAW:" + full,
            content=full,
        )
        for source_id, family_id, split, language, full in fixture.SOURCE_DEFS
    )
    segments = tuple(
        producer.SegmentSpec(
            doc_id=doc_id,
            segment_id=segment_id,
            source_id=source_id,
            source_byte_start=source_start,
            source_byte_end=source_end,
            segment_byte_start=segment_start,
            segment_byte_end=segment_end,
            segment_index=segment_index,
            segment_count=segment_count,
            token_ids=tuple(tokens),
        )
        for (doc_id, segment_id, source_id, source_start, source_end,
             segment_start, segment_end, segment_index, segment_count,
             segment, tokens) in fixture.DOC_DEFS
    )
    return producer.ProducerSpec(
        sequence_length=fixture.T,
        vocab_size=fixture.V,
        token_bytes=tuple(fixture.TOKEN_BYTES),
        sources=sources,
        segments=segments,
    )


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="splot-t0030-producer-")
        self.root = Path(self.tmp.name)
        self.spec = make_spec()

    def tearDown(self):
        self.tmp.cleanup()

    def test_matches_t0028_fixture(self):
        expected = self.root / "expected"
        expected_sha = fixture.write_fixture(expected, fixture.make_fixture())
        produced = self.root / "produced"
        summary = producer.write_dataset(self.spec, produced)
        self.assertEqual(summary["manifest_sha256"], expected_sha)
        for name in ("manifest.json", "tokenizer-bytes.json", "source-bytes.json",
                     "train.jsonl", "validation.jsonl", "test.jsonl"):
            self.assertEqual((produced / name).read_bytes(),
                             (expected / name).read_bytes())
        loaded = v2.load_v2_dataset(produced, summary["manifest_sha256"])
        self.assertEqual(loaded["manifest"]["counts"], summary["counts"])
        self.assertEqual(loaded["manifest"]["counts"]["content_target_bytes"], 21)

    def test_deterministic_repeat(self):
        first = self.root / "first"
        second = self.root / "second"
        a = producer.write_dataset(self.spec, first)
        b = producer.write_dataset(self.spec, second)
        self.assertEqual(a, b)
        for name in ("manifest.json", "tokenizer-bytes.json", "source-bytes.json",
                     "train.jsonl", "validation.jsonl", "test.jsonl"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_expected_coverage_matches_loader(self):
        summary = producer.expected_coverage(self.spec)
        root = self.root / "dataset"
        written = producer.write_dataset(self.spec, root)
        self.assertEqual(summary["counts"], written["counts"])
        loaded = v2.load_v2_dataset(root, written["manifest_sha256"])
        self.assertEqual(loaded["manifest"]["counts"], summary["counts"])
        self.assertEqual(summary["counts"]["blocks"], 3)
        self.assertEqual(summary["counts"]["valid_tokens"], 36)
        self.assertEqual(summary["counts"]["content_tokens"], 20)
        self.assertEqual(summary["counts"]["ntp_labels"], 28)
        self.assertEqual(summary["counts"]["eos_labels"], 8)
        self.assertEqual(summary["counts"]["content_target_bytes"], 21)
        self.assertEqual(summary["counts"]["by_language"]["en"]["content_target_bytes"], 13)
        self.assertEqual(summary["counts"]["by_language"]["pl"]["content_target_bytes"], 8)
        selected = {source["source_id"]: source["selected_bytes"]
                    for source in summary["sources"]}
        self.assertEqual(selected["src-train-en2"], 4)
        self.assertEqual(selected["src-train-pl"], 3)

    def test_refuses_existing_output(self):
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(producer.ProducerError):
            producer.write_dataset(self.spec, existing)
        file_path = self.root / "file"
        file_path.write_text("x")
        with self.assertRaises(producer.ProducerError):
            producer.write_dataset(self.spec, file_path)
        link = self.root / "link"
        link.symlink_to(self.root / "missing")
        with self.assertRaises(producer.ProducerError):
            producer.write_dataset(self.spec, link)

    def test_input_validation(self):
        bad_tokens = list(self.spec.token_bytes)
        bad_tokens[13] = bad_tokens[12]
        with self.assertRaises(producer.ProducerError):
            producer.expected_coverage(producer.ProducerSpec(
                self.spec.sequence_length, self.spec.vocab_size, tuple(bad_tokens),
                self.spec.sources, self.spec.segments))

        bad_segments = list(self.spec.segments)
        bad_segments[0] = producer.SegmentSpec(
            **{**bad_segments[0].__dict__, "token_ids": (13, 12)})
        with self.assertRaises(producer.ProducerError):
            producer.expected_coverage(producer.ProducerSpec(
                self.spec.sequence_length, self.spec.vocab_size, self.spec.token_bytes,
                self.spec.sources, tuple(bad_segments)))

        with self.assertRaises(producer.ProducerError):
            producer.expected_coverage(producer.ProducerSpec(
                513, self.spec.vocab_size, self.spec.token_bytes,
                self.spec.sources, self.spec.segments))

        no_test = tuple(seg for seg in self.spec.segments
                        if seg.source_id not in ("src-test-en", "src-test-pl"))
        with self.assertRaises(producer.ProducerError):
            producer.expected_coverage(producer.ProducerSpec(
                self.spec.sequence_length, self.spec.vocab_size, self.spec.token_bytes,
                self.spec.sources, no_test))

    def test_multi_document_blocks_and_tail(self):
        summary = producer.expected_coverage(self.spec)
        self.assertEqual(summary["counts"]["by_split"]["train"]["blocks"], 1)
        self.assertEqual(summary["counts"]["by_split"]["train"]["valid_tokens"], 17)
        self.assertEqual(summary["counts"]["by_split"]["validation"]["valid_tokens"], 10)
        self.assertEqual(summary["counts"]["by_split"]["test"]["valid_tokens"], 9)
        # The train block packs four document segments and pads the tail.
        root = self.root / "dataset"
        producer.write_dataset(self.spec, root)
        loaded = v2.load_v2_dataset(root, summary["manifest_sha256"])
        block = loaded["blocks"]["train"][0]
        self.assertEqual(block["document_ids"][16], 3)
        self.assertEqual(block["valid"][17:], [False] * 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
