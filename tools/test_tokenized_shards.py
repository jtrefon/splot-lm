import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import bpe_reference as bpe
import corpus_reference as corpus
import data_fixture
import tokenized_shards as ts


DEMO_BUNDLE = ts.build_demo_bundle()
DEMO_CONFIG = ts.PackConfig(length=96, max_blocks_per_shard=8)


def token_doc(doc_id, text, numeric_id, *, language="en", domain="prose", split="train", group=None):
    artifact = bpe.train_bpe(
        (bpe.Document("train", "train", language, domain, text),),
        requested_vocab_size=264,
    )
    return (
        ts.TokenDocument(
            doc_id,
            group or f"cluster-{doc_id}",
            split,
            language,
            domain,
            "synthetic-source",
            "rev-1",
            bpe.document_content_sha256(text),
            numeric_id,
            tuple(bpe.encode(text, artifact)),
        ),
        artifact,
    )


def materialize(root, config=DEMO_CONFIG, bundle=DEMO_BUNDLE):
    return ts.materialize_dataset(
        root,
        bundle["documents"],
        bundle["artifact"],
        config,
        bundle["source_identity"],
    )


def load_manifest(root):
    data = (Path(root) / "manifest.json").read_bytes()
    return json.loads(data), data


def refresh_manifest(root, manifest):
    manifest = copy.deepcopy(manifest)
    manifest["physical_manifest_digest"] = ts._digest_bytes(
        ts._canonical_json(
            {key: value for key, value in manifest.items() if key != "physical_manifest_digest"}
        )
    )
    data = ts.manifest_bytes(manifest)
    (Path(root) / "manifest.json").write_bytes(data)
    return manifest, hashlib.sha256(data).hexdigest()


def read_blocks(root):
    blocks = []
    for shard in load_manifest(root)[0]["shards"]:
        data = (Path(root) / shard["path"]).read_bytes()
        blocks.extend(json.loads(line) for line in data.decode("utf-8").splitlines())
    return blocks


def write_blocks(root, manifest, blocks):
    by_split = {}
    for block in blocks:
        by_split.setdefault(block["split"], []).append(block)
    shards = []
    index = 0
    for split in ts.SPLIT_ORDER:
        split_blocks = sorted(by_split.get(split, []), key=lambda item: item["block_index"])
        size = manifest["pack_config"]["max_blocks_per_shard"]
        for start in range(0, len(split_blocks), size):
            chunk = split_blocks[start:start + size]
            data = b"".join(ts._canonical_json(block) + b"\n" for block in chunk)
            path = f"shards/{split}-{index:05d}.jsonl"
            (Path(root) / path).write_bytes(data)
            shards.append(
                {
                    "path": path,
                    "split": split,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "blocks": len(chunk),
                    "first_block": chunk[0]["block_index"],
                    "last_block": chunk[-1]["block_index"],
                }
            )
            index += 1
    # Remove old shard files not referenced by the rewritten manifest.
    allowed = {shard["path"] for shard in shards}
    for path in (Path(root) / "shards").glob("*"):
        if path.relative_to(Path(root)).as_posix() not in allowed:
            path.unlink()
    manifest = copy.deepcopy(manifest)
    manifest["shards"] = shards
    manifest["shard_count"] = len(shards)
    manifest["block_count"] = len(blocks)
    return refresh_manifest(root, manifest)


class CorpusBridgeTests(unittest.TestCase):
    def test_train_only_bpe_and_cluster_split_identity(self):
        bundle = DEMO_BUNDLE
        report = bundle["report"]
        self.assertEqual(
            report["digest"],
            "b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f",
        )
        self.assertTrue(report["audit"]["ok"])
        train_ids = sorted(
            cluster["representative"]
            for cluster in report["clusters"]
            if cluster["split"] == "train"
        )
        self.assertEqual(bundle["source_identity"]["train_doc_ids"], train_ids)
        split_clusters = {
            split: set(report["splits"][split])
            for split in ts.SPLIT_ORDER
        }
        for left in ts.SPLIT_ORDER:
            for right in ts.SPLIT_ORDER:
                if left < right:
                    self.assertFalse(split_clusters[left] & split_clusters[right])
        doc_ids = [doc.doc_id for doc in bundle["documents"]]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))
        self.assertEqual(
            bundle["artifact"]["training_digest"],
            bundle["source_identity"]["bpe_training_digest"],
        )
        quarantined = {
            member
            for item in bundle["source_identity"]["quarantine"]
            for member in item["member_ids"]
        }
        self.assertFalse(quarantined & set(doc_ids))
        for doc in bundle["documents"]:
            self.assertTrue(all(token >= bpe.BYTE_OFFSET for token in doc.tokens))
            self.assertEqual(
                bpe.decode(doc.tokens, bundle["artifact"]),
                next(
                    item["text"]
                    for item in report["inputs"]
                    if item["record_id"] == doc.doc_id
                ),
            )

    def test_old_demo_digests_unchanged(self):
        self.assertEqual(
            corpus.demo_report()["digest"],
            "b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f",
        )
        self.assertEqual(
            bpe.demo_report()["artifact_sha256"],
            "f67ef4ab5c343e3c9f81570b0d7426a5668e578aa84806058265d0aaf633ce7e",
        )


class PackingTests(unittest.TestCase):
    def test_hand_calculated_tiny_sequence_and_fixture_semantics(self):
        doc, artifact = token_doc("doc-a", "AB", 0)
        blocks = ts.pack_documents((doc,), artifact, ts.PackConfig(length=96))
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block["input_ids"][:5], [1, 73, 74, 2, 0])
        self.assertEqual(block["document_ids"][:5], [0, 0, 0, 0, -1])
        self.assertEqual(block["position_ids"][:5], [0, 1, 2, 3, 0])
        self.assertEqual(block["valid"][:5], [True, True, True, True, False])
        self.assertEqual(block["labels"]["1"][:4], [73, 74, 2, -100])
        self.assertEqual(block["labels"]["2"][:3], [74, 2, -100])
        fixture = data_fixture.pack(
            [data_fixture.Document("doc-a", "cluster-doc-a", "en", "prose", "AB")],
            96,
        )[0]
        for key in ("input_ids", "document_ids", "position_ids", "valid"):
            self.assertEqual(block[key], fixture[key])
        self.assertEqual(block["labels"]["1"], fixture["labels"][1])
        self.assertEqual(block["labels"]["2"], fixture["labels"][2])

    def test_multi_document_boundary_masks(self):
        first, artifact = token_doc("doc-a", "AB", 0)
        second = ts.TokenDocument(
            "doc-b",
            "cluster-doc-b",
            "train",
            "en",
            "prose",
            "synthetic-source",
            "rev-1",
            bpe.document_content_sha256("CD"),
            1,
            tuple(bpe.encode("CD", artifact)),
        )
        blocks = ts.pack_documents((first, second), artifact, ts.PackConfig(length=96))
        block = blocks[0]
        self.assertEqual(block["input_ids"][:9], [1, 73, 74, 2, 1, 75, 76, 2, 0])
        self.assertEqual(block["document_ids"][:9], [0, 0, 0, 0, 1, 1, 1, 1, -1])
        self.assertEqual(block["position_ids"][:9], [0, 1, 2, 3, 0, 1, 2, 3, 0])
        self.assertEqual(block["labels"]["1"][3], -100)
        self.assertEqual(block["labels"]["1"][7], -100)
        self.assertEqual(block["labels"]["2"][2], -100)
        self.assertEqual(block["labels"]["2"][6], -100)
        fixture = data_fixture.pack(
            [
                data_fixture.Document("doc-a", "cluster-doc-a", "en", "prose", "AB"),
                data_fixture.Document("doc-b", "cluster-doc-b", "en", "prose", "CD"),
            ],
            96,
        )[0]
        self.assertEqual(block["input_ids"], fixture["input_ids"])
        self.assertEqual(block["document_ids"], fixture["document_ids"])
        self.assertEqual(block["labels"]["1"], fixture["labels"][1])
        self.assertEqual(block["labels"]["2"], fixture["labels"][2])

    def test_overlong_document_is_rejected(self):
        doc, artifact = token_doc("doc-a", "A" * 100, 0)
        with self.assertRaises(ts.ShardError):
            ts.pack_documents((doc,), artifact, ts.PackConfig(length=96))

    def test_invalid_pack_config(self):
        doc, artifact = token_doc("doc-a", "AB", 0)
        for length in (3, 95, 97, 512):
            with self.assertRaises(ts.ShardError):
                ts.pack_documents((doc,), artifact, ts.PackConfig(length=length))
        for size in (0, 9, -1):
            with self.assertRaises(ts.ShardError):
                ts.pack_documents((doc,), artifact, ts.PackConfig(max_blocks_per_shard=size))
        with self.assertRaises(ts.ShardError):
            ts.pack_documents((doc,), artifact, ts.PackConfig(offsets=(1,)))

    def test_exact_tiny_counts(self):
        doc, artifact = token_doc("doc-a", "AB", 0)
        blocks = ts.pack_documents((doc,), artifact, ts.PackConfig(length=96))
        counts = ts.compute_counts((doc,), blocks, ts.PackConfig(length=96))
        self.assertEqual(
            counts["overall"],
            {
                "blocks": 1,
                "documents": 1,
                "content_tokens": 2,
                "special_tokens": 2,
                "valid_tokens": 4,
                "pad_tokens": 92,
                "ntp_labels": 3,
                "offset2_labels": 2,
            },
        )
        self.assertEqual(blocks[0]["valid"].count(True), 4)
        self.assertEqual(blocks[0]["input_ids"][:4], [1, 73, 74, 2])


class MaterializationTests(unittest.TestCase):
    def test_demo_materialization_and_partition_invariance(self):
        report = ts.demo_report()
        self.assertTrue(report["independent_materializations_identical"])
        self.assertTrue(report["partition_invariance"]["logical_digest_equal"])
        self.assertTrue(report["partition_invariance"]["packed_logical_digest_equal"])
        self.assertTrue(report["partition_invariance"]["physical_manifest_digest_differs"])
        self.assertGreater(report["partition_invariance"]["shard_count"], report["shards"])
        self.assertEqual(report["loaded_blocks"], report["blocks"])
        self.assertEqual(
            report["manifest_sha256"], ts.manifest_sha256(report["manifest"])
        )
        self.assertEqual(
            report["counts"]["overall"]["documents"],
            len(report["manifest"]["documents"]),
        )

    def test_demo_is_deterministic_and_cli(self):
        first = ts.demo_report()
        second = ts.demo_report()
        self.assertEqual(first, second)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = ts.main(["--demo"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), first)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(ts.main([]), 2)

    def test_materialize_load_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            manifest = materialize(root)
            loaded = ts.load_dataset(root, ts.manifest_sha256(manifest))
            self.assertEqual(len(loaded["blocks"]), manifest["block_count"])
            before = (root / "manifest.json").read_bytes()
            with self.assertRaises(ts.ShardError):
                materialize(root)
            self.assertEqual((root / "manifest.json").read_bytes(), before)

    def test_stable_order_and_identical_materialization(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            first = materialize(base / "first")
            second = ts.materialize_dataset(
                base / "second",
                tuple(reversed(DEMO_BUNDLE["documents"])),
                DEMO_BUNDLE["artifact"],
                DEMO_CONFIG,
                DEMO_BUNDLE["source_identity"],
            )
            self.assertEqual(ts.manifest_bytes(first), ts.manifest_bytes(second))
            self.assertEqual(
                ts._snapshot(base / "first"),
                ts._snapshot(base / "second"),
            )

    def test_atomic_failure_leaves_no_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            with self.assertRaises(ts.ShardError):
                ts.materialize_dataset(
                    root,
                    (),
                    DEMO_BUNDLE["artifact"],
                    DEMO_CONFIG,
                    DEMO_BUNDLE["source_identity"],
                )
            self.assertFalse(root.exists())


class LoaderCorruptionTests(unittest.TestCase):
    def _materialized(self, temp):
        root = Path(temp) / "dataset"
        manifest = materialize(root)
        return root, manifest, ts.manifest_sha256(manifest)

    def test_corrupt_shard_bytes_and_unsafe_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, digest = self._materialized(temp)
            shard = root / manifest["shards"][0]["path"]
            data = shard.read_bytes()
            shard.write_bytes(data + b"\n")
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)
            shard.write_bytes(data)
            mutated = copy.deepcopy(manifest)
            mutated["shards"][0]["path"] = "../escape.jsonl"
            _, new_digest = refresh_manifest(root, mutated)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, new_digest)

    def test_corrupt_manifest_counts_and_logical_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, _ = self._materialized(temp)
            for mutator in (
                lambda value: value["counts"]["overall"].__setitem__(
                    "pad_tokens", value["counts"]["overall"]["pad_tokens"] + 1
                ),
                lambda value: value.__setitem__("logical_digest", "0" * 64),
                lambda value: value.__setitem__("packed_logical_digest", "0" * 64),
            ):
                candidate = copy.deepcopy(manifest)
                mutator(candidate)
                _, digest = refresh_manifest(root, candidate)
                with self.assertRaises(ts.ShardError):
                    ts.load_dataset(root, digest)

    def test_corrupt_block_labels_and_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, _ = self._materialized(temp)
            blocks = read_blocks(root)
            blocks[0]["labels"]["1"][0] = 0
            _, digest = write_blocks(root, manifest, blocks)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)
            blocks = read_blocks(root)
            blocks[0]["valid"][0] = False
            _, digest = write_blocks(root, manifest, blocks)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)

    def test_duplicate_manifest_document_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, _ = self._materialized(temp)
            mutated = copy.deepcopy(manifest)
            mutated["documents"].append(copy.deepcopy(mutated["documents"][0]))
            _, digest = refresh_manifest(root, mutated)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)

    def test_invalid_token_and_position_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, _ = self._materialized(temp)
            blocks = read_blocks(root)
            blocks[0]["input_ids"][0] = DEMO_BUNDLE["artifact"]["actual_vocab_size"]
            _, digest = write_blocks(root, manifest, blocks)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)
            blocks = read_blocks(root)
            blocks[0]["position_ids"][0] = 1
            _, digest = write_blocks(root, manifest, blocks)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)

    def test_duplicate_json_keys_and_oversized_manifest_rejected(self):
        with self.assertRaises(ts.ShardError):
            ts._parse_canonical_json(b'{"a":1,"a":2}', "fixture")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            root.mkdir()
            data = b" " * (ts.MAX_MANIFEST_BYTES + 1)
            (root / "manifest.json").write_bytes(data)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, hashlib.sha256(data).hexdigest())

    def test_extra_file_missing_shard_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, digest = self._materialized(temp)
            extra = root / "extra.txt"
            extra.write_text("extra", encoding="utf-8")
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)
            extra.unlink()
            shard = root / manifest["shards"][0]["path"]
            shard.unlink()
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)
            shard.write_bytes(b"restored\n")
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, digest = self._materialized(temp)
            target = root / "shards" / "link"
            os.symlink(root / manifest["shards"][0]["path"], target)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)

    def test_noncanonical_manifest_and_shard_utf8(self):
        with tempfile.TemporaryDirectory() as temp:
            root, manifest, _ = self._materialized(temp)
            pretty = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            (root / "manifest.json").write_bytes(pretty)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, hashlib.sha256(pretty).hexdigest())
            manifest = copy.deepcopy(manifest)
            shard = root / manifest["shards"][0]["path"]
            shard.write_bytes(b"\xff\n")
            manifest["shards"][0]["bytes"] = len(b"\xff\n")
            manifest["shards"][0]["sha256"] = hashlib.sha256(b"\xff\n").hexdigest()
            _, digest = refresh_manifest(root, manifest)
            with self.assertRaises(ts.ShardError):
                ts.load_dataset(root, digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
