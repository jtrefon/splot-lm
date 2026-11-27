import contextlib
import copy
import hashlib
import io
import json
import unittest
from collections import Counter

from bpe_reference import (
    ALLOWED_VOCAB_SIZES,
    BASE_VOCAB_SIZE,
    BYTE_COUNT,
    BYTE_OFFSET,
    MAX_DOCUMENTS,
    MAX_DOCUMENT_BYTES,
    MAX_MERGES,
    BpeError,
    Document,
    artifact_sha256,
    best_pair,
    bytes_to_token_ids,
    canonical_artifact_bytes,
    corpus_digest,
    count_adjacent_pairs,
    decode,
    demo_documents,
    demo_report,
    document_content_sha256,
    encode,
    load_artifact,
    main,
    merge_sequence,
    serialize_artifact,
    token_ids_to_bytes,
    train_bpe,
    train_input_digest,
    validate_artifact,
    validate_documents,
)
from corpus_reference import demo_report as corpus_demo_report


def doc(
    doc_id,
    text,
    *,
    split="train",
    language="en",
    domain="prose",
    quarantined=False,
):
    return Document(doc_id, split, language, domain, text, quarantined)


def small_artifact():
    return train_bpe(
        (
            doc("t1", "abab abab abab", language="en"),
            doc("t2", "cdcd cdcd cdcd", language="pl"),
        ),
        requested_vocab_size=288,
    )


class DocumentValidationTests(unittest.TestCase):
    def test_document_validation_fail_closed(self):
        bad = (
            (doc("a", "x"), doc("a", "y")),
            (Document("a", "dev", "en", "prose", "x"),),
            (Document("a", "train", "de", "prose", "x"),),
            (Document("a", "train", ["en"], "prose", "x"),),
            (Document("a", "train", "en", "math", "x"),),
            (Document("a", "train", "en", ["prose"], "x"),),
            (Document("a", "train", "en", "prose", ""),),
            (Document("a", "train", "en", "prose", "x" * (MAX_DOCUMENT_BYTES + 1)),),
            (Document("a", "train", "en", "prose", "\ud800"),),
            (Document("a", "train", "en", "prose", None),),
            (Document("a", "train", "en", "prose", "x", quarantined=1),),
        )
        for docs in bad:
            with self.assertRaises(BpeError, msg=repr(docs)):
                validate_documents(docs)
        with self.assertRaises(BpeError):
            validate_documents(tuple(doc(f"d{index}", "x") for index in range(MAX_DOCUMENTS + 1)))
        over_budget = tuple(doc(f"t{index}", "a" * MAX_DOCUMENT_BYTES) for index in range(65))
        with self.assertRaises(BpeError):
            validate_documents(over_budget)

    def test_documents_are_sorted_and_preserved(self):
        text = "  Zażółć\r\nCafe\u0301\t🦊  "
        docs = validate_documents(
            (
                doc("z", text),
                doc("a", "validation", split="validation"),
                doc("m", "test", split="test"),
            )
        )
        self.assertEqual([item.doc_id for item in docs], ["a", "m", "z"])
        self.assertEqual(docs[2].text, text)
        self.assertEqual(
            len(validate_documents((doc("max", "a" * MAX_DOCUMENT_BYTES),))),
            1,
        )


class PairAndMergeTests(unittest.TestCase):
    def test_adjacent_pair_counts_are_independent(self):
        counts = count_adjacent_pairs([[8, 9, 8, 9, 10], [9, 8]])
        self.assertEqual(
            counts,
            Counter({(8, 9): 2, (9, 8): 2, (9, 10): 1}),
        )

    def test_best_pair_tie_break(self):
        self.assertEqual(best_pair({(9, 8): 2, (8, 9): 2, (8, 8): 1}), (8, 9))
        self.assertIsNone(best_pair({}))

    def test_merge_sequence_nonoverlapping_left_to_right(self):
        self.assertEqual(merge_sequence([8, 9, 8, 9], (8, 9), 264), [264, 264])
        self.assertEqual(merge_sequence([8, 8, 8], (8, 8), 264), [264, 8])
        self.assertEqual(merge_sequence([9, 8, 9, 8], (8, 9), 264), [9, 264, 8])
        self.assertEqual(merge_sequence([], (8, 9), 264), [])

    def test_no_document_boundary_merge(self):
        artifact = train_bpe(
            (doc("a", "A"), doc("b", "B")),
            requested_vocab_size=288,
        )
        self.assertEqual(artifact["merges"], [])
        self.assertEqual(artifact["actual_vocab_size"], BASE_VOCAB_SIZE)


class TrainingTests(unittest.TestCase):
    def test_train_only_artifact_equality_after_holdout_and_quarantine_changes(self):
        base = (
            doc("t2", "abab abab abab", language="en"),
            doc("t1", "cdcd cdcd cdcd", language="pl"),
            doc("v1", "validation one", split="validation"),
            doc("x1", "test one", split="test"),
            doc(
                "q1",
                "quarantine one",
                split="train",
                quarantined=True,
            ),
        )
        first = train_bpe(base, requested_vocab_size=288)
        changed_holdouts = (
            doc("t2", "abab abab abab", language="en"),
            doc("t1", "cdcd cdcd cdcd", language="pl"),
            doc("v1", "validation changed", split="validation"),
            doc("x1", "test changed", split="test"),
            doc("q1", "quarantine changed", split="train", quarantined=True),
        )
        second = train_bpe(tuple(reversed(changed_holdouts)), requested_vocab_size=288)
        self.assertEqual(first, second)
        self.assertEqual(first["training_digest"], second["training_digest"])
        changed_train = (
            doc("t2", "abab abab abab", language="en"),
            doc("t1", "efef efef efef", language="pl"),
            doc("v1", "validation one", split="validation"),
            doc("x1", "test one", split="test"),
            doc("q1", "quarantine one", split="train", quarantined=True),
        )
        third = train_bpe(changed_train, requested_vocab_size=288)
        self.assertNotEqual(first["training_digest"], third["training_digest"])

    def test_tiny_corpus_stops_before_capacity(self):
        empty = train_bpe((doc("a", "a"),), requested_vocab_size=320)
        self.assertEqual(empty["merges"], [])
        self.assertEqual(empty["actual_vocab_size"], BASE_VOCAB_SIZE)
        one_merge = train_bpe((doc("a", "aa"),), requested_vocab_size=320)
        self.assertEqual(len(one_merge["merges"]), 1)
        self.assertEqual(one_merge["actual_vocab_size"], BASE_VOCAB_SIZE + 1)
        self.assertLess(one_merge["actual_vocab_size"], 320)

    def test_vocab_targets_and_merge_limits(self):
        text = "abab abab abab abab abab"
        no_merges = train_bpe((doc("a", text),), requested_vocab_size=264)
        self.assertEqual(no_merges["merges"], [])
        for requested in (288, 320):
            artifact = train_bpe((doc("a", text),), requested_vocab_size=requested)
            self.assertIn(requested, ALLOWED_VOCAB_SIZES)
            self.assertLessEqual(len(artifact["merges"]), requested - BASE_VOCAB_SIZE)
            self.assertLessEqual(len(artifact["merges"]), MAX_MERGES)
            self.assertEqual(
                artifact["actual_vocab_size"],
                BASE_VOCAB_SIZE + len(artifact["merges"]),
            )
        capped = train_bpe(demo_documents(), requested_vocab_size=288)
        self.assertEqual(len(capped["merges"]), 24)
        self.assertEqual(capped["actual_vocab_size"], 288)
        for requested in (265, 512, True, 264.0, None):
            with self.assertRaises(BpeError):
                train_bpe((doc("a", text),), requested_vocab_size=requested)
        for max_merges in (-1, MAX_MERGES + 1, True, 1.0, None):
            with self.assertRaises(BpeError):
                train_bpe((doc("a", text),), max_merges=max_merges)
        self.assertEqual(
            train_bpe((doc("a", text),), max_merges=0)["merges"],
            [],
        )

    def test_train_input_digest_is_sorted_and_content_sensitive(self):
        first = doc("b", "one")
        second = doc("a", "two")
        digest = train_input_digest((first, second))
        self.assertEqual(digest, train_input_digest((second, first)))
        changed = train_input_digest((doc("a", "two"), doc("b", "changed")))
        self.assertNotEqual(digest, changed)
        self.assertNotEqual(
            document_content_sha256("é"),
            document_content_sha256("e\u0301"),
        )

    def test_holdout_only_corpus_has_base_vocabulary(self):
        artifact = train_bpe(
            (doc("v", "validation only", split="validation"),),
            requested_vocab_size=320,
        )
        self.assertEqual(artifact["merges"], [])
        self.assertEqual(artifact["actual_vocab_size"], BASE_VOCAB_SIZE)


class ArtifactTests(unittest.TestCase):
    def test_artifact_roundtrip_reload_and_digest(self):
        artifact = small_artifact()
        raw = canonical_artifact_bytes(artifact)
        digest = artifact_sha256(artifact)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        self.assertEqual(load_artifact(raw, digest), artifact)
        self.assertEqual(load_artifact(serialize_artifact(artifact), digest), artifact)
        self.assertEqual(canonical_artifact_bytes(load_artifact(raw, digest)), raw)
        self.assertNotIn("timestamp", raw.decode("utf-8"))

    def test_artifact_malformed_rejected(self):
        artifact = demo_report()["artifact"]

        def assert_bad(mutator):
            candidate = copy.deepcopy(artifact)
            mutator(candidate)
            with self.assertRaises(BpeError, msg=repr(candidate)[:200]):
                validate_artifact(candidate)

        assert_bad(lambda value: value.pop("version"))
        assert_bad(lambda value: value.__setitem__("extra", 1))
        assert_bad(lambda value: value.__setitem__("version", 2))
        assert_bad(lambda value: value.__setitem__("algorithm", "other"))
        assert_bad(lambda value: value["special_tokens"].__setitem__("0", "X"))
        assert_bad(lambda value: value["special_tokens"].pop("7"))
        assert_bad(lambda value: value.__setitem__("byte_offset", 9))
        assert_bad(lambda value: value.__setitem__("byte_count", 255))
        assert_bad(lambda value: value.__setitem__("requested_vocab_size", 265))
        assert_bad(lambda value: value.__setitem__("requested_vocab_size", 288))
        assert_bad(lambda value: value.__setitem__("training_digest", "0" * 63))
        assert_bad(lambda value: value.__setitem__("training_digest", "A" * 64))
        assert_bad(lambda value: value.__setitem__("merges", {}))
        assert_bad(lambda value: value.__setitem__("merges", [[8]]))
        assert_bad(lambda value: value["merges"].__setitem__(0, [0, 8]))
        assert_bad(lambda value: value["merges"].__setitem__(0, [BASE_VOCAB_SIZE, 8]))
        assert_bad(lambda value: value["merges"].__setitem__(1, value["merges"][0]))
        assert_bad(lambda value: value.__setitem__("actual_vocab_size", value["actual_vocab_size"] + 1))
        assert_bad(lambda value: value.__setitem__("merges", value["merges"] + [[8, 9]] * (MAX_MERGES + 1)))
        assert_bad(lambda value: value.__setitem__("merges", [[8, "9"]]))
        assert_bad(lambda value: value.__setitem__("merges", [[True, 8]]))

    def test_load_artifact_rejects_bad_input(self):
        artifact = small_artifact()
        raw = canonical_artifact_bytes(artifact)
        digest = artifact_sha256(artifact)
        with self.assertRaises(BpeError):
            load_artifact(raw, "0" * 64)
        with self.assertRaises(BpeError):
            load_artifact(raw, "A" * 64)
        with self.assertRaises(BpeError):
            load_artifact(raw, 123)
        with self.assertRaises(BpeError):
            load_artifact(b"{}", hashlib.sha256(b"{}").hexdigest())
        with self.assertRaises(BpeError):
            load_artifact(
                b'{"version":1,"version":1}',
                hashlib.sha256(b'{"version":1,"version":1}').hexdigest(),
            )
        with self.assertRaises(BpeError):
            load_artifact(
                b'{"version": NaN}',
                hashlib.sha256(b'{"version": NaN}').hexdigest(),
            )
        with self.assertRaises(BpeError):
            load_artifact("\ud800", "0" * 64)
        self.assertEqual(load_artifact(raw, digest), artifact)


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.artifact = small_artifact()

    def test_all_256_byte_alphabet_roundtrip(self):
        data = bytes(range(BYTE_COUNT))
        tokens = bytes_to_token_ids(data)
        self.assertEqual(tokens, list(range(BYTE_OFFSET, BASE_VOCAB_SIZE)))
        self.assertEqual(token_ids_to_bytes(tokens, self.artifact), data)

    def test_unicode_whitespace_and_control_roundtrip(self):
        texts = (
            "Zażółć gęślą jaźń.",
            "Cafe\u0301",
            "emoji 🦊🚀",
            "  indentation\n\tkept\r\n",
            "def f():\n    return 'ż'\n",
            "\x00raw-control-character\x01",
        )
        for text in texts:
            tokens = encode(text, self.artifact)
            self.assertTrue(all(token >= BYTE_OFFSET for token in tokens), text)
            self.assertEqual(decode(tokens, self.artifact), text)

    def test_no_normalization(self):
        precomposed = "é"
        decomposed = "e\u0301"
        self.assertNotEqual(
            bytes_to_token_ids(precomposed.encode("utf-8")),
            bytes_to_token_ids(decomposed.encode("utf-8")),
        )
        self.assertNotEqual(encode(precomposed, self.artifact), encode(decomposed, self.artifact))
        self.assertEqual(decode(encode(precomposed, self.artifact), self.artifact), precomposed)
        self.assertEqual(decode(encode(decomposed, self.artifact), self.artifact), decomposed)

    def test_raw_special_spellings_never_emit_control(self):
        for text in (
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>",
            "<RESERVED_4>",
            "<|endoftext|>",
            "<pad>",
        ):
            tokens = encode(text, self.artifact)
            self.assertTrue(all(token >= BYTE_OFFSET for token in tokens), text)
            self.assertEqual(decode(tokens, self.artifact), text)

    def test_decode_rejects_control_and_unknown(self):
        with self.assertRaises(BpeError):
            decode([0, 1, 2, 3], self.artifact)
        self.assertEqual(decode([0, 1, 2, 3], self.artifact, allow_control=True), "")
        with self.assertRaises(BpeError):
            decode([self.artifact["actual_vocab_size"]], self.artifact)
        with self.assertRaises(BpeError):
            decode([-1], self.artifact)
        with self.assertRaises(BpeError):
            decode([True], self.artifact)
        with self.assertRaises(BpeError):
            decode(["8"], self.artifact)
        with self.assertRaises(BpeError):
            decode([BYTE_OFFSET + 255], self.artifact)


class DemoTests(unittest.TestCase):
    def test_demo_is_deterministic_and_reports_heldout(self):
        first = demo_report()
        second = demo_report()
        self.assertEqual(first, second)
        artifact = first["artifact"]
        validate_artifact(artifact)
        self.assertEqual(first["artifact_sha256"], artifact_sha256(artifact))
        self.assertEqual(len(first["corpus_digest"]), 64)
        self.assertEqual(first["training"]["quarantined_documents"], 1)
        self.assertEqual(
            artifact["actual_vocab_size"],
            BASE_VOCAB_SIZE + len(artifact["merges"]),
        )
        self.assertGreater(len(artifact["merges"]), 0)
        self.assertLessEqual(len(artifact["merges"]), MAX_MERGES)
        self.assertLessEqual(artifact["actual_vocab_size"], 320)
        self.assertEqual(first["control_safety"]["control_ids_emitted"], 0)
        self.assertEqual(len(first["heldout"]), 8)
        for key, stats in first["heldout"].items():
            self.assertEqual(stats["documents"], 1, key)
            self.assertGreater(stats["bytes"], 0, key)
            self.assertGreater(stats["tokens"], 0, key)
            self.assertTrue(stats["roundtrip"], key)
        self.assertNotIn("timestamp", json.dumps(first))

    def test_cli_demo(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--demo"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), demo_report())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(main([]), 2)

    def test_corpus_digest_and_demo_documents(self):
        docs = demo_documents()
        self.assertEqual(corpus_digest(docs), demo_report()["corpus_digest"])
        ids = [item.doc_id for item in docs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(corpus_digest(docs), corpus_digest(tuple(reversed(docs))))


class OriginalCorpusDigestTests(unittest.TestCase):
    def test_original_corpus_demo_digest_stable(self):
        report = corpus_demo_report()
        self.assertEqual(
            report["digest"],
            "b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f",
        )
        self.assertEqual(report, corpus_demo_report())


if __name__ == "__main__":
    unittest.main(verbosity=2)
