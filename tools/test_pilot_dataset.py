import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pilot_dataset as pd
import production_tokenizer as pt

BACKEND = pt.backend_info()
BACKEND_AVAILABLE = bool(BACKEND.get("available"))
REGISTRY = pd.load_registry("configs/data/pilot-sources-v0.2.json")
SPECIALS = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>", 4: "<RES4>", 5: "<RES5>", 6: "<RES6>", 7: "<RES7>"}
FAKE_VOCAB = 32
FAKE_TOKENIZER = {
    "version": "1.0",
    "model": {
        "type": "BPE",
        "vocab": {name: token_id for token_id, name in SPECIALS.items()}
        | {f"t{token_id}": token_id for token_id in range(8, FAKE_VOCAB)},
        "merges": [],
    },
    "added_tokens": [
        {"id": token_id, "content": name} for token_id, name in SPECIALS.items()
    ],
    "normalizer": None,
}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def segment(family, language, split, start, text):
    data = text.encode("utf-8")
    return pd.TextSegment(
        family,
        language,
        split,
        start,
        start + len(text),
        text,
        len(data),
        hashlib.sha256(data).hexdigest(),
    )


def extracted(family, language, split, texts):
    offset = 0
    segments = []
    for text in texts:
        segments.append(segment(family, language, split, offset, text))
        offset += len(text)
    return pd.ExtractedSource(
        family,
        family,
        language,
        split,
        "git:" + "0" * 40,
        "https://standardebooks.org/ebooks/x",
        "https://standardebooks.org/ebooks/x",
        "https://www.wikidata.org/wiki/Special:EntityData/Q1.json",
        "a" * 64,
        "b" * 64,
        0,
        sum(len(text.encode()) for text in texts),
        (),
        tuple(segments),
    )


def source_records():
    records = []
    for source in REGISTRY["sources"]:
        records.append(
            {
                "family_id": source["family_id"],
                "language": source["language"],
                "split": source["split"],
                "content_sha256": sha("content:" + source["family_id"]),
                "raw_sha256": sha("raw:" + source["family_id"]),
                "revision": source["revision"],
                "rights_evidence_url": source["rights_evidence_url"],
                "source_url": source["source_url"],
            }
        )
    return records


def one_block(doc_id, content_ids):
    length = len(content_ids) + 2
    pad = pd.SEQUENCE_LENGTH - length
    input_ids = [pd.BOS, *content_ids, pd.EOS] + [pd.PAD] * pad
    document_ids = [doc_id] * length + [-1] * pad
    position_ids = list(range(length)) + [0] * pad
    valid = [True] * length + [False] * pad
    block = {
        "input_ids": input_ids,
        "document_ids": document_ids,
        "position_ids": position_ids,
        "valid": valid,
        "labels": {
            "1": [
                input_ids[index + 1]
                if index + 1 < length and document_ids[index] >= 0
                else pd.IGNORE
                for index in range(pd.SEQUENCE_LENGTH)
            ],
            "2": [
                input_ids[index + 2]
                if index + 2 < length and document_ids[index] >= 0
                else pd.IGNORE
                for index in range(pd.SEQUENCE_LENGTH)
            ],
        },
    }
    return block


def fake_blocks():
    train = []
    for index in range(32):
        family = "se-pride-and-prejudice" if index % 2 == 0 else "wl-lalka"
        language = "en" if index % 2 == 0 else "pl"
        train.append(
            pd.BlockCandidate(
                family,
                language,
                "train",
                index * 10,
                index * 10 + 10,
                (8 + index % 8, 16 + index % 8, 24 + index % 8),
                10,
                hashlib.sha256(f"train-{index}".encode()).hexdigest(),
            )
        )
    candidates = {
        "train": train,
        "validation": [
            pd.BlockCandidate("se-alices-adventures", "en", "validation", 0, 10, (15, 16, 17), 10, "c" * 64),
            pd.BlockCandidate("wl-quo-vadis", "pl", "validation", 20, 30, (18, 19, 20), 10, "e" * 64),
        ],
        "test": [
            pd.BlockCandidate("se-moby-dick", "en", "test", 0, 10, (21, 22, 23), 10, "d" * 64),
            pd.BlockCandidate("wl-faraon", "pl", "test", 20, 30, (24, 25, 26), 10, "f" * 64),
        ],
    }
    selected = pd.select_block_candidates(candidates)
    return pd.pack_blocks(selected, vocab_size=FAKE_VOCAB)[0]


class ExtractionTests(unittest.TestCase):
    def test_xhtml_main_text_and_paratext(self):
        data = b"""<html><head><title>ignore</title></head><body>
        <section><h2>I</h2><p>Hello <abbr>Mr.</abbr>&nbsp;Brown.</p>
        <figure><figcaption>caption</figcaption></figure>
        <p>Line<br/>two.</p></section></body></html>"""
        text = pd.extract_xhtml(data)
        self.assertIn("Hello Mr.\u00a0Brown.", text)
        self.assertIn("Line\ntwo.", text)
        self.assertNotIn("caption", text)
        self.assertNotIn("ignore", text)

    def test_wolne_xml_excludes_editorial_and_annotation_layers(self):
        data = b"""<utwor><powiesc>
        <abstrakt><akap>SUMMARY MUST GO</akap></abstrakt>
        <nota_red><akap>EDITORIAL NOTE MUST GO</akap></nota_red>
        <naglowek_rozdzial>I</naglowek_rozdzial>
        <akap>Text <motyw>THEME</motyw>after<pr>FOOTNOTE</pr>.</akap>
        <uwaga>NOTE MUST GO</uwaga>
        <strofa>Line one
Line two</strofa>
        </powiesc></utwor>"""
        text = pd.extract_wolne_xml(data)
        self.assertEqual(text, "I\n\nText after.\n\nLine one\nLine two")

    def test_wolne_xml_rejects_dtd_and_missing_prose(self):
        with self.assertRaises(pd.PilotDataError):
            pd.extract_wolne_xml(b'<!DOCTYPE x><utwor><powiesc><akap>x</akap></powiesc></utwor>')
        with self.assertRaises(pd.PilotDataError):
            pd.extract_wolne_xml(b"<utwor><meta/></utwor>")


class SegmentationTests(unittest.TestCase):
    def test_segmentation_is_lossless_and_bounded(self):
        text = "short paragraph.\n\n" + "A" * 5000 + "\n\ntail"
        segments = pd.segment_text(text, max_bytes=4096)
        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(all(segment.bytes <= 4096 for segment in segments))
        self.assertEqual("".join(segment.text for segment in segments), text)
        self.assertEqual([segment.start for segment in segments], [0] + [s.end for s in segments[:-1]])

    def test_segmentation_rejects_nul_and_bad_limit(self):
        with self.assertRaises(pd.PilotDataError):
            pd.normalize_text("bad\x00text")
        with self.assertRaises(pd.PilotDataError):
            pd.segment_text("text", max_bytes=4097)


class FamilyAndSelectionTests(unittest.TestCase):
    def test_registry_has_eight_families_and_verse_exclusion(self):
        self.assertEqual(len(REGISTRY["sources"]), 8)
        self.assertEqual(
            {source["split"] for source in REGISTRY["sources"]},
            {"train", "validation", "test"},
        )
        self.assertTrue(
            any(
                item["family_id"] == "wl-pan-tadeusz"
                and item["genre"] == "verse_epic"
                for item in REGISTRY["excluded_candidates"]
            )
        )

    def test_registry_rejects_split_mutation_and_unsafe_path(self):
        bad = copy.deepcopy(REGISTRY)
        next(item for item in bad["sources"] if item["family_id"] == "se-alices-adventures")["split"] = "train"
        with self.assertRaises(pd.PilotDataError):
            pd.validate_registry(bad)
        bad = copy.deepcopy(REGISTRY)
        bad["sources"][0]["parts"][0]["path"] = "../escape.xhtml"
        with self.assertRaises(pd.PilotDataError):
            pd.validate_registry(bad)

    def test_tokenizer_selection_is_train_only_and_bounded(self):
        sources = [
            extracted("se-pride-and-prejudice", "en", "train", ["a" * 1000, "b" * 1000, "c" * 1000]),
            extracted("wl-lalka", "pl", "train", ["d" * 1000, "e" * 1000, "f" * 1000]),
            extracted("se-alices-adventures", "en", "validation", ["v" * 1000]),
        ]
        selected, sampled = pd.select_tokenizer_segments(sources, max_bytes=2500)
        self.assertTrue(sampled)
        self.assertLessEqual(sum(item.bytes for item in selected), 2500)
        self.assertEqual({item.language for item in selected}, {"en", "pl"})
        self.assertTrue(all(item.split == "train" for item in selected))
        all_train = [item for source in sources if source.split == "train" for item in source.segments]
        selected_all, sampled_all = pd.select_tokenizer_segments(
            [source for source in sources if source.split == "train"],
            max_bytes=10_000,
        )
        self.assertFalse(sampled_all)
        self.assertEqual(len(selected_all), len(all_train))


class BlockTests(unittest.TestCase):
    def test_pack_blocks_are_fixed_shape_and_masked(self):
        blocks = fake_blocks()
        self.assertEqual([len(blocks[split]) for split in pd.SPLITS], [32, 2, 2])
        for split_blocks in blocks.values():
            for block in split_blocks:
                length, ntp, offset2 = pd.validate_block(block, FAKE_VOCAB)
                self.assertGreaterEqual(length, 3)
                self.assertEqual(ntp, length - 1)
                self.assertEqual(offset2, length - 2)
                self.assertEqual(block["input_ids"][0], pd.BOS)
                self.assertEqual(block["input_ids"][length - 1], pd.EOS)
                self.assertEqual(block["document_ids"][length:], [-1] * (pd.SEQUENCE_LENGTH - length))

    def test_block_rejections(self):
        block = one_block(0, [8, 9, 10])
        bad = copy.deepcopy(block)
        bad["input_ids"][0] = 8
        with self.assertRaises(pd.PilotDataError):
            pd.validate_block(bad, FAKE_VOCAB)
        bad = copy.deepcopy(block)
        bad["input_ids"][2] = 1
        with self.assertRaises(pd.PilotDataError):
            pd.validate_block(bad, FAKE_VOCAB)
        bad = copy.deepcopy(block)
        bad["labels"]["1"][0] = pd.IGNORE
        with self.assertRaises(pd.PilotDataError):
            pd.validate_block(bad, FAKE_VOCAB)
        bad = copy.deepcopy(block)
        bad["position_ids"][1] = 7
        with self.assertRaises(pd.PilotDataError):
            pd.validate_block(bad, FAKE_VOCAB)
        bad = copy.deepcopy(block)
        bad["document_ids"][1] = 1
        with self.assertRaises(pd.PilotDataError):
            pd.validate_block(bad, FAKE_VOCAB)
        with self.assertRaises(pd.PilotDataError):
            pd.validate_block(one_block(0, []), FAKE_VOCAB)

    def test_long_segment_splits_to_block_limit(self):
        class CharSession:
            def encode(self, text):
                return [8 + (ord(character) % 20) for character in text]

        source = extracted("se-pride-and-prejudice", "en", "train", ["x" * 700])
        candidates = pd.build_block_candidates([source], CharSession())
        self.assertGreaterEqual(len(candidates["train"]), 2)
        self.assertTrue(all(len(item.content_ids) <= 510 for item in candidates["train"]))
        self.assertEqual(
            sum(item.text_bytes for item in candidates["train"]),
            700,
        )
        self.assertTrue(all(item.end > item.start for item in candidates["train"]))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="splot-t0016-")
        self.root = Path(self.tmp.name)
        self.blocks = fake_blocks()
        self.result = pd.write_dataset(
            self.root,
            name="pilot-test",
            blocks_by_split=self.blocks,
            sources=source_records(),
            tokenizer_json=pd._canonical_json(FAKE_TOKENIZER),
            vocab_size=FAKE_VOCAB,
        )
        self.dataset = self.root / "pilot-test"
        self.manifest_sha = self.result["manifest_sha256"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_and_loader(self):
        loaded = pd.load_pilot_dataset(self.dataset, self.manifest_sha, expected_vocab=FAKE_VOCAB)
        self.assertEqual(loaded["manifest_sha256"], self.manifest_sha)
        self.assertEqual(loaded["manifest"]["sequence_length"], 512)
        self.assertEqual(loaded["manifest"]["vocab_size"], FAKE_VOCAB)
        self.assertEqual(set(loaded["blocks"]), set(pd.SPLITS))
        self.assertEqual(loaded["summary"]["splits"]["train"]["blocks"], 32)
        self.assertEqual(
            set(loaded["blocks"]["train"][0]["labels"]),
            {1, 2},
        )
        self.assertEqual(
            _sha((self.dataset / "train.jsonl").read_bytes()),
            loaded["manifest"]["shards"][0]["sha256"],
        )

    def test_repeat_materialization_is_identical(self):
        second = pd.write_dataset(
            self.root,
            name="pilot-test-2",
            blocks_by_split=self.blocks,
            sources=source_records(),
            tokenizer_json=pd._canonical_json(FAKE_TOKENIZER),
            vocab_size=FAKE_VOCAB,
        )
        compared = pd.compare_datasets(self.dataset, self.root / "pilot-test-2")
        self.assertEqual(compared["status"], "passed")
        self.assertEqual(second["manifest_sha256"], self.manifest_sha)

    def test_tampered_manifest_is_rejected(self):
        path = self.dataset / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(pd.PilotDataError):
            pd.load_pilot_dataset(self.dataset, self.manifest_sha)

    def test_extra_file_and_symlink_are_rejected(self):
        (self.dataset / "extra.txt").write_text("x")
        with self.assertRaises(pd.PilotDataError):
            pd.load_pilot_dataset(self.dataset, self.manifest_sha)
        (self.dataset / "extra.txt").unlink()
        (self.dataset / "link.jsonl").symlink_to(self.dataset / "train.jsonl")
        with self.assertRaises(pd.PilotDataError):
            pd.load_pilot_dataset(self.dataset, self.manifest_sha)

    def test_duplicate_document_id_is_rejected_by_writer(self):
        blocks = copy.deepcopy(self.blocks)
        validation = blocks["validation"][0]
        content_end = validation["valid"].index(False)
        validation["document_ids"][:content_end] = [0] * content_end
        with self.assertRaises(pd.PilotDataError):
            pd.write_dataset(
                self.root,
                name="duplicate",
                blocks_by_split=blocks,
                sources=source_records(),
                tokenizer_json=pd._canonical_json(FAKE_TOKENIZER),
                vocab_size=FAKE_VOCAB,
            )

    def test_source_snapshot_and_shard_hash_tamper(self):
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        manifest["sources"][0]["content_sha256"] = "0" * 64
        (self.dataset / "manifest.json").write_bytes(pd._canonical_json(manifest))
        with self.assertRaises(pd.PilotDataError):
            pd.load_pilot_dataset(self.dataset, _sha((self.dataset / "manifest.json").read_bytes()))
        # Restore the original and tamper a shard instead.
        shutil.rmtree(self.dataset)
        pd.write_dataset(
            self.root,
            name="pilot-test",
            blocks_by_split=self.blocks,
            sources=source_records(),
            tokenizer_json=pd._canonical_json(FAKE_TOKENIZER),
            vocab_size=FAKE_VOCAB,
        )
        shard = self.dataset / "train.jsonl"
        shard.write_bytes(shard.read_bytes() + b"\n")
        with self.assertRaises(pd.PilotDataError):
            pd.load_pilot_dataset(self.dataset, self.manifest_sha)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


if BACKEND_AVAILABLE:

    class BackendTests(unittest.TestCase):
        def test_real_pilot_training_and_session(self):
            docs = [
                pt.TrainingDocument("en-1", "Café society meets Kraków.", "train", "en", "prose"),
                pt.TrainingDocument("pl-1", "Zażółć gęślą jaźń.", "train", "pl", "prose"),
                pt.TrainingDocument("pl-2", "def f():\n    return 'ż'\n", "train", "pl", "prose"),
            ]
            artifact = pt.train_real_pilot_tokenizer(docs)
            self.assertLessEqual(
                artifact["config"]["actual_vocab_size"], 16384
            )
            self.assertEqual(
                artifact["config"]["target_vocab_size"], 16384
            )
            session = pt.create_session(artifact)
            for text in (
                "Café society meets Kraków.",
                "Zażółć gęślą jaźń.",
                "  indentation\n\tkept\r\n",
                "emoji 🦊🚀",
            ):
                ids = session.encode(text)
                self.assertTrue(all(token >= 8 for token in ids))
                self.assertEqual(session.decode(ids), text)
            self.assertEqual(
                len(artifact["tokenizer"]["model"]["vocab"]),
                artifact["config"]["actual_vocab_size"],
            )

        def test_real_pilot_rejects_holdout_reserved_and_bounds(self):
            with self.assertRaises(pt.TokenizerError):
                pt.train_real_pilot_tokenizer(
                    [pt.TrainingDocument("v", "text", "validation", "en", "prose")]
                )
            with self.assertRaises(pt.TokenizerError):
                pt.train_real_pilot_tokenizer(
                    [pt.TrainingDocument("r", "<PAD> literal", "train", "en", "prose")]
                )
            with self.assertRaises(pt.TokenizerError):
                pt.train_real_pilot_tokenizer(
                    [pt.TrainingDocument("big", "x" * 4097, "train", "en", "prose")]
                )
            with self.assertRaises(pt.TokenizerError):
                pt.train_real_pilot_tokenizer(
                    [
                        pt.TrainingDocument(f"d{index}", "x", "train", "en", "prose")
                        for index in range(pt.REAL_PILOT_MAX_DOCUMENTS + 1)
                    ]
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
