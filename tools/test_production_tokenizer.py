import contextlib
import copy
import hashlib
import io
import json
import os
import unittest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import bpe_reference as bpe
import production_tokenizer as pt


BACKEND = pt.backend_info()
BACKEND_AVAILABLE = bool(BACKEND.get("available"))
TOKENIZERS_LOADED_BY_PROBE = "tokenizers" in __import__("sys").modules
DEMO_DOCS = pt.demo_documents()


def refresh_artifact(artifact):
    artifact = copy.deepcopy(artifact)
    artifact["logical_digest"] = hashlib.sha256(
        pt._canonical_json(
            {key: value for key, value in artifact.items() if key != "logical_digest"}
        )
    ).hexdigest()
    return artifact


class StdlibBoundaryTests(unittest.TestCase):
    def test_backend_status_is_explicit(self):
        self.assertIn("available", BACKEND)
        self.assertEqual(BACKEND["expected"], pt.TOKENIZERS_VERSION)
        self.assertFalse(
            TOKENIZERS_LOADED_BY_PROBE,
            "default adapter import/probe must not import tokenizers",
        )
        if not BACKEND_AVAILABLE:
            print(f"backend unavailable: {BACKEND.get('reason')}; stdlib tests only")

    def test_malformed_artifact_rejected_without_backend(self):
        data = b"{}"
        digest = hashlib.sha256(data).hexdigest()
        with self.assertRaises(pt.TokenizerError):
            pt.load_artifact(data, digest)
        with self.assertRaises(pt.TokenizerError):
            pt.validate_artifact({})
        with self.assertRaises(pt.TokenizerError):
            pt.load_artifact(data, "0" * 64)

    def test_document_validation_bounds(self):
        docs = (
            pt.TrainingDocument("a", "text"),
            pt.TrainingDocument("a", "other"),
        )
        with self.assertRaises(pt.TokenizerError):
            pt.validate_documents(docs)
        with self.assertRaises(pt.TokenizerError):
            pt.validate_documents((pt.TrainingDocument("a", ""),))
        with self.assertRaises(pt.TokenizerError):
            pt.validate_documents(
                (pt.TrainingDocument("a", "x" * (pt.MAX_DOCUMENT_BYTES + 1)),)
            )
        with self.assertRaises(pt.TokenizerError):
            pt.validate_documents((pt.TrainingDocument("a", "\ud800"),))

    def test_approval_packet_and_lock_metadata(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        packet = json.loads(
            (root / "research/data/pilot-approval-packet-v0.1.json").read_text()
        )
        lock = json.loads(
            (root / "configs/data/tokenizers-0.23.3-lock.json").read_text()
        )
        candidates = packet["candidates"]
        prose = [item for item in candidates if item["pilot_role"] != "excluded_nonprose"]
        self.assertEqual(len(prose), 8)
        self.assertEqual(
            {language: sum(1 for item in prose if item["language"] == language)
             for language in ("en", "pl")},
            {"en": 4, "pl": 4},
        )
        pan = next(item for item in candidates if item["candidate_id"] == "wl-pan-tadeusz")
        self.assertEqual(pan["work_genre"], "verse_epic")
        self.assertEqual(pan["pilot_role"], "excluded_nonprose")
        splits = packet["pilot"]["proposed_splits"]
        for language in ("en", "pl"):
            self.assertEqual(len(splits["train"][language]), 2)
            self.assertEqual(len(splits["validation"][language]), 1)
            self.assertEqual(len(splits["test"][language]), 1)
        self.assertLessEqual(packet["pilot"]["raw_utf8_cap_bytes"], 10 * 1024 * 1024)
        self.assertTrue(all(item["status"] == "pending" for item in candidates))
        self.assertNotIn('"text":', json.dumps(packet))
        self.assertEqual(lock["package"]["version"], pt.TOKENIZERS_VERSION)
        self.assertEqual(lock["wheel"]["bytes"], 3101381)
        self.assertFalse(lock["installation"]["system_changes"])


if BACKEND_AVAILABLE:

    class BackendTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.artifact_16 = pt.train_tokenizer(DEMO_DOCS, 16384)
            cls.artifact_32 = pt.train_tokenizer(DEMO_DOCS, 32768)

        def test_independently_pinned_demo_digests(self):
            # Physical hashes pin the qualified macOS arm64 wheel; logical
            # digests are the canonical cross-platform identity. A different
            # physical hash must be reported as build nondeterminism, not
            # silently accepted.
            self.assertEqual(
                pt.artifact_sha256(self.artifact_16),
                "2f148ac98771660e8479411f90fa8c1ff0e80907896a1d823bf0a322542e6fff",
            )
            self.assertEqual(
                pt.artifact_sha256(self.artifact_32),
                "8026f3058d154ce86105ebf4b66ee4ccab941a9d569d39267edd5d84c8827b40",
            )
            self.assertEqual(
                self.artifact_16["logical_digest"],
                "b6474d4cff55fdd526c4bfcba342dada259353864fce33e1863d614b02972cf0",
            )
            self.assertEqual(
                self.artifact_32["logical_digest"],
                "1a49d99f5ffaa2be78aa9722dfcc8adf78bf41af3ea7faa9910dedb8bcdf0c58",
            )

        def test_special_ids_byte_alphabet_and_target_bound(self):
            tokenizer = pt._tokenizer(self.artifact_16)
            vocab = tokenizer.get_vocab()
            self.assertIsNone(tokenizer.normalizer)
            self.assertEqual(tokenizer.get_vocab_size(), self.artifact_16["config"]["actual_vocab_size"])
            self.assertLessEqual(tokenizer.get_vocab_size(), 16384)
            for token_id, token in pt.SPECIAL_TOKENS.items():
                self.assertEqual(tokenizer.token_to_id(token), token_id)
            alphabet = pt._backend()[3].ByteLevel.alphabet()
            self.assertEqual(len(alphabet), 256)
            self.assertTrue(all(token in vocab for token in alphabet))
            self.assertGreater(tokenizer.get_vocab_size(), 8)

        def test_train_only_isolation_and_order(self):
            base = pt.train_tokenizer(DEMO_DOCS, 16384)
            changed = []
            for doc in DEMO_DOCS:
                if doc.split == "train":
                    changed.append(doc)
                else:
                    changed.append(
                        pt.TrainingDocument(
                            doc.doc_id,
                            doc.text + " changed holdout",
                            doc.split,
                            doc.language,
                            doc.domain,
                            doc.quarantined,
                        )
                    )
            changed_artifact = pt.train_tokenizer(changed, 16384)
            self.assertEqual(
                pt.canonical_artifact_bytes(base),
                pt.canonical_artifact_bytes(changed_artifact),
            )
            reordered = pt.train_tokenizer(tuple(reversed(DEMO_DOCS)), 16384)
            self.assertEqual(
                pt.canonical_artifact_bytes(base),
                pt.canonical_artifact_bytes(reordered),
            )
            train_changed = []
            for doc in DEMO_DOCS:
                if doc.split == "train" and not doc.quarantined:
                    train_changed.append(
                        pt.TrainingDocument(
                            doc.doc_id,
                            doc.text + " extra train token",
                            doc.split,
                            doc.language,
                            doc.domain,
                            doc.quarantined,
                        )
                    )
                else:
                    train_changed.append(doc)
            self.assertNotEqual(
                base["training_digest"],
                pt.train_tokenizer(train_changed, 16384)["training_digest"],
            )

        def test_unicode_whitespace_code_and_all_bytes(self):
            texts = (
                "Zażółć gęślą jaźń.",
                "Cafe\u0301",
                "emoji 🦊🚀",
                "  indentation\n\tkept\r\n",
                "def f():\n    return 'ż'\n",
            )
            for text in texts:
                ids = pt.encode(text, self.artifact_16)
                self.assertTrue(all(token >= 8 for token in ids), text)
                self.assertNotIn(3, ids)
                self.assertEqual(pt.decode(ids, self.artifact_16), text)
                reference_ids = bpe.encode(text, bpe.demo_report()["artifact"])
                self.assertEqual(
                    bpe.decode(reference_ids, bpe.demo_report()["artifact"]), text
                )

        def test_literal_special_strings_fail_closed(self):
            for text in pt.SPECIAL_STRINGS:
                with self.assertRaises(pt.TokenizerError):
                    pt.encode(text, self.artifact_16)
                with self.assertRaises(pt.TokenizerError):
                    pt.encode(f"prefix {text} suffix", self.artifact_16)

        def test_boundaries_are_explicit(self):
            text = "Café boundary"
            ids = pt.encode(text, self.artifact_16, add_boundaries=True)
            self.assertEqual(ids[0], 1)
            self.assertEqual(ids[-1], 2)
            self.assertEqual(pt.decode(ids, self.artifact_16, allow_control=True), text)
            with self.assertRaises(pt.TokenizerError):
                pt.decode(ids, self.artifact_16)

        def test_decode_token_iterable_is_bounded(self):
            import itertools

            too_many = itertools.repeat(8, pt.MAX_TOKEN_COUNT + 1)
            with self.assertRaises(pt.TokenizerError):
                pt.decode(too_many, self.artifact_16)
            with self.assertRaises(pt.TokenizerError):
                pt.decode(
                    (8 for _ in range(pt.MAX_TOKEN_COUNT + 1)),
                    self.artifact_16,
                )

        def test_artifact_roundtrip_reload_and_digest(self):
            raw = pt.canonical_artifact_bytes(self.artifact_16)
            digest = pt.artifact_sha256(self.artifact_16)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            self.assertEqual(pt.load_artifact(raw, digest), self.artifact_16)
            self.assertEqual(pt.canonical_artifact_bytes(pt.load_artifact(raw, digest)), raw)
            with self.assertRaises(pt.TokenizerError):
                pt.load_artifact(raw, "0" * 64)
            pretty = json.dumps(self.artifact_16, indent=2, sort_keys=True).encode("utf-8")
            with self.assertRaises(pt.TokenizerError):
                pt.load_artifact(pretty, hashlib.sha256(pretty).hexdigest())

        def test_artifact_malformed_rejected(self):
            def bad(mutator):
                candidate = copy.deepcopy(self.artifact_16)
                mutator(candidate)
                with self.assertRaises(pt.TokenizerError):
                    pt.validate_artifact(candidate)

            bad(lambda value: value.pop("version"))
            bad(lambda value: value.__setitem__("extra", 1))
            bad(lambda value: value.__setitem__("version", 2))
            bad(lambda value: value.__setitem__("format", "other"))
            bad(lambda value: value["package"].__setitem__("version", "0.0.0"))
            bad(lambda value: value["config"].__setitem__("target_vocab_size", 42))
            bad(lambda value: value["config"].__setitem__("actual_vocab_size", 999999))
            bad(lambda value: value["config"].__setitem__("use_regex", False))
            bad(lambda value: value["config"].__setitem__("add_prefix_space", True))
            bad(lambda value: value["config"]["special_tokens"].__setitem__("0", "<X>"))
            bad(lambda value: value.__setitem__("training_digest", "0" * 63))
            bad(lambda value: value.__setitem__("train_doc_ids", ["z", "a"]))
            bad(lambda value: value.__setitem__("tokenizer", {}))
            bad(lambda value: value.__setitem__("logical_digest", "0" * 64))

        def test_demo_and_cli(self):
            first = pt.demo_report()
            second = pt.demo_report()
            self.assertEqual(first, second)
            self.assertEqual(set(first["targets"]), {"16384", "32768"})
            for target in first["targets"].values():
                self.assertLessEqual(target["actual_vocab_size"], target["requested_vocab_size"])
                self.assertTrue(target["repeat_artifact_identical"])
                self.assertTrue(target["repeat_logical_equal"])
                self.assertGreater(target["heldout_bytes"], 0)
                self.assertGreater(target["heldout_tokens"], 0)
                self.assertTrue(all(row["roundtrip"] and row["control_or_unk"] == 0
                                    for row in target["heldout"].values()))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(pt.main(["--demo"]), 0)
            self.assertEqual(json.loads(output.getvalue()), first)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(pt.main([]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
