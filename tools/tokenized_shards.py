"""Portable synthetic tokenized shards and document-safe packing (T0010).

This is a reference bridge between the accepted synthetic corpus oracle and the
byte BPE oracle. It is not a production sharding/streaming pipeline, tokenizer
or data acquisition tool. All demo data is original synthetic text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import bpe_reference as bpe
import corpus_reference as corpus

FORMAT = "splot-tokenized-shards-v1"
FORMAT_VERSION = 1
PAD = 0
BOS = 1
EOS = 2
IGNORE = -100
SPLIT_ORDER = ("train", "validation", "test")
ALLOWED_LANGUAGES = frozenset({"en", "pl"})
ALLOWED_DOMAINS = frozenset({"prose"})
ALLOWED_PACK_LENGTHS = (96, 128, 256)
MAX_BLOCKS_PER_SHARD = 8
MAX_TOTAL_BLOCKS = 1024
MAX_SHARD_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILES = 2048
MAX_DOCUMENTS = 1000
MAX_DOCUMENT_BYTES = 4096
MAX_TRAIN_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024
REQUIRED_METRICS = (
    "blocks",
    "documents",
    "content_tokens",
    "special_tokens",
    "valid_tokens",
    "pad_tokens",
    "ntp_labels",
    "offset2_labels",
)
MANIFEST_KEYS = frozenset(
    {
        "format",
        "version",
        "pack_config",
        "tokenizer",
        "source_identity",
        "documents",
        "shards",
        "counts",
        "logical_digest",
        "packed_logical_digest",
        "physical_manifest_digest",
        "block_count",
        "shard_count",
    }
)
PACK_KEYS = frozenset(
    {
        "length",
        "offsets",
        "max_blocks_per_shard",
        "max_blocks_total",
        "pad",
        "bos",
        "eos",
        "ignore",
    }
)
TOKENIZER_KEYS = frozenset(
    {"artifact_sha256", "actual_vocab_size", "requested_vocab_size", "algorithm", "version"}
)
SOURCE_KEYS = frozenset(
    {
        "corpus_report_digest",
        "seed",
        "split_percentages",
        "source_approvals",
        "train_doc_ids",
        "bpe_training_digest",
        "cluster_splits",
        "document_splits",
        "quarantine",
    }
)
DOC_KEYS = frozenset(
    {
        "doc_id",
        "cluster_id",
        "split",
        "language",
        "domain",
        "source_id",
        "source_revision",
        "content_sha256",
        "numeric_id",
        "tokens",
    }
)
SHARD_KEYS = frozenset(
    {"path", "split", "bytes", "sha256", "blocks", "first_block", "last_block"}
)
BLOCK_KEYS = frozenset(
    {
        "split",
        "block_index",
        "language",
        "domain",
        "input_ids",
        "document_ids",
        "position_ids",
        "valid",
        "labels",
    }
)


class ShardError(ValueError):
    """Invalid corpus, pack configuration, manifest, shard or path."""


@dataclass(frozen=True)
class TokenDocument:
    doc_id: str
    cluster_id: str
    split: str
    language: str
    domain: str
    source_id: str
    source_revision: str
    content_sha256: str
    numeric_id: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class PackConfig:
    length: int = 96
    max_blocks_per_shard: int = MAX_BLOCKS_PER_SHARD
    offsets: tuple[int, ...] = (1, 2)


def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShardError(f"{name} must be a nonempty string")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_pack_config(config: PackConfig) -> None:
    if type(config.length) is not int or config.length not in ALLOWED_PACK_LENGTHS:
        raise ShardError(f"pack length must be one of {ALLOWED_PACK_LENGTHS}")
    if (
        type(config.max_blocks_per_shard) is not int
        or not 1 <= config.max_blocks_per_shard <= MAX_BLOCKS_PER_SHARD
    ):
        raise ShardError(
            f"max_blocks_per_shard must be between 1 and {MAX_BLOCKS_PER_SHARD}"
        )
    if tuple(config.offsets) != (1, 2):
        raise ShardError("offsets must be exactly (1, 2)")


def _validate_tokens(tokens: Any, actual_vocab_size: int, name: str) -> tuple[int, ...]:
    if not isinstance(tokens, (list, tuple)):
        raise ShardError(f"{name} tokens must be a list")
    result = []
    for token in tokens:
        if type(token) is not int:
            raise ShardError(f"{name} token IDs must be integers")
        if token < bpe.BYTE_OFFSET or token >= actual_vocab_size:
            raise ShardError(f"{name} token ID out of content range: {token}")
        result.append(token)
    return tuple(result)


def _normalise_allowlist(
    allowlist: Iterable[corpus.SourceApproval] | Mapping[str, corpus.SourceApproval],
) -> dict[str, corpus.SourceApproval]:
    if isinstance(allowlist, Mapping):
        entries = tuple(allowlist.values())
    else:
        try:
            entries = tuple(allowlist)
        except TypeError as exc:
            raise ShardError("allowlist must be iterable or a mapping") from exc
    return corpus.build_allowlist(entries)


def build_token_documents(
    report: Mapping[str, Any],
    *,
    requested_vocab_size: int = 320,
    max_merges: int = bpe.MAX_MERGES,
) -> dict[str, Any]:
    """Build encoded eligible representatives from an accepted corpus report."""
    if not isinstance(report, Mapping):
        raise ShardError("corpus report must be an object")
    if not report.get("audit", {}).get("ok"):
        raise ShardError("corpus report audit is not ok")
    report_digest = report.get("digest")
    if not _is_hex64(report_digest):
        raise ShardError("corpus report digest is invalid")
    inputs = {
        item["record_id"]: item["text"]
        for item in report.get("inputs", [])
        if isinstance(item, Mapping) and "record_id" in item and "text" in item
    }
    clusters = report.get("clusters")
    if not isinstance(clusters, list):
        raise ShardError("corpus report clusters must be a list")
    eligible = [cluster for cluster in clusters if cluster.get("split") is not None]
    if not eligible:
        raise ShardError("corpus report has no eligible clusters")
    train_clusters = [cluster for cluster in eligible if cluster["split"] == "train"]
    if not train_clusters:
        raise ShardError("corpus report has no eligible train clusters")

    train_docs = []
    for cluster in train_clusters:
        representative = cluster["representative"]
        if representative not in inputs:
            raise ShardError(f"missing input text for representative {representative!r}")
        member = next(
            (item for item in cluster["members"] if item["record_id"] == representative),
            None,
        )
        if member is None:
            raise ShardError(f"missing member metadata for representative {representative!r}")
        text = inputs[representative]
        if member["domain"] != "prose":
            raise ShardError("only prose documents are supported in this reference")
        train_docs.append(
            bpe.Document(
                representative,
                "train",
                member["language"],
                member["domain"],
                text,
            )
        )
    train_docs.sort(key=lambda item: item.doc_id)
    artifact = bpe.train_bpe(
        train_docs,
        requested_vocab_size=requested_vocab_size,
        max_merges=max_merges,
    )
    actual_vocab_size = artifact["actual_vocab_size"]

    documents: list[TokenDocument] = []
    cluster_splits: dict[str, str] = {}
    document_splits: dict[str, str] = {}
    approvals: set[tuple[str, str]] = set()
    for cluster in eligible:
        representative = cluster["representative"]
        split = cluster["split"]
        if split not in SPLIT_ORDER:
            raise ShardError(f"invalid cluster split: {split!r}")
        member = next(
            (item for item in cluster["members"] if item["record_id"] == representative),
            None,
        )
        if member is None:
            raise ShardError(f"missing member metadata for representative {representative!r}")
        if representative not in inputs:
            raise ShardError(f"missing input text for representative {representative!r}")
        text = inputs[representative]
        if member["domain"] != "prose":
            raise ShardError("only prose documents are supported in this reference")
        content_sha = member["raw_sha256"]
        if content_sha != bpe.document_content_sha256(text):
            raise ShardError(f"content digest mismatch for {representative!r}")
        tokens = tuple(bpe.encode(text, artifact))
        if any(token < bpe.BYTE_OFFSET for token in tokens):
            raise ShardError(f"encoder emitted a control ID for {representative!r}")
        if any(token >= actual_vocab_size for token in tokens):
            raise ShardError(f"encoder emitted an out-of-vocabulary ID for {representative!r}")
        if bpe.decode(tokens, artifact) != text:
            raise ShardError(f"round-trip failed for {representative!r}")
        approvals.add((member["source_id"], member["source_revision"]))
        cluster_splits[cluster["cluster_id"]] = split
        document_splits[representative] = split
        documents.append(
            TokenDocument(
                representative,
                cluster["cluster_id"],
                split,
                member["language"],
                member["domain"],
                member["source_id"],
                member["source_revision"],
                content_sha,
                0,
                tokens,
            )
        )
    documents.sort(key=lambda item: (SPLIT_ORDER.index(item.split), item.doc_id))
    numbered = tuple(
        TokenDocument(
            doc.doc_id,
            doc.cluster_id,
            doc.split,
            doc.language,
            doc.domain,
            doc.source_id,
            doc.source_revision,
            doc.content_sha256,
            index,
            doc.tokens,
        )
        for index, doc in enumerate(documents)
    )
    quarantine = [
        {
            "cluster_id": cluster["cluster_id"],
            "reasons": list(cluster.get("quarantine_reasons", [])),
            "member_ids": sorted(member["record_id"] for member in cluster["members"]),
        }
        for cluster in clusters
        if cluster.get("split") is None
    ]
    quarantine.sort(key=lambda item: item["cluster_id"])
    source_identity = {
        "corpus_report_digest": report_digest,
        "seed": report["seed"],
        "split_percentages": dict(report["split_percentages"]),
        "source_approvals": [
            {"source_id": source_id, "revision": revision}
            for source_id, revision in sorted(approvals)
        ],
        "train_doc_ids": sorted(cluster["representative"] for cluster in train_clusters),
        "bpe_training_digest": artifact["training_digest"],
        "cluster_splits": {key: cluster_splits[key] for key in sorted(cluster_splits)},
        "document_splits": {key: document_splits[key] for key in sorted(document_splits)},
        "quarantine": quarantine,
    }
    return {
        "documents": numbered,
        "artifact": artifact,
        "artifact_sha256": bpe.artifact_sha256(artifact),
        "source_identity": source_identity,
        "report": report,
    }


def build_demo_bundle() -> dict[str, Any]:
    return build_token_documents(corpus.demo_report())


def _block_labels(
    input_ids: Sequence[int],
    document_ids: Sequence[int],
    length: int,
    offsets: tuple[int, ...],
) -> dict[str, list[int]]:
    labels: dict[str, list[int]] = {}
    for offset in offsets:
        labels[str(offset)] = [
            input_ids[index + offset]
            if index + offset < length
            and document_ids[index] >= 0
            and document_ids[index] == document_ids[index + offset]
            else IGNORE
            for index in range(length)
        ]
    return labels


def _pack_split(
    documents: Sequence[TokenDocument],
    artifact: Mapping[str, Any],
    config: PackConfig,
    split: str,
) -> list[dict[str, Any]]:
    docs = sorted(documents, key=lambda item: item.doc_id)
    blocks: list[dict[str, Any]] = []
    input_ids: list[int] = []
    document_ids: list[int] = []
    position_ids: list[int] = []
    language: str | None = None
    domain: str | None = None

    def flush() -> None:
        nonlocal language, domain
        if not input_ids:
            return
        if language is None or domain is None:
            raise ShardError("block has no language/domain")
        pad = config.length - len(input_ids)
        padded_ids = input_ids + [PAD] * pad
        padded_docs = document_ids + [-1] * pad
        padded_positions = position_ids + [0] * pad
        valid = [doc_id >= 0 for doc_id in padded_docs]
        blocks.append(
            {
                "split": split,
                "block_index": len(blocks),
                "language": language,
                "domain": domain,
                "input_ids": padded_ids,
                "document_ids": padded_docs,
                "position_ids": padded_positions,
                "valid": valid,
                "labels": _block_labels(
                    padded_ids, padded_docs, config.length, config.offsets
                ),
            }
        )
        input_ids.clear()
        document_ids.clear()
        position_ids.clear()
        language = None
        domain = None

    for doc in docs:
        if doc.split != split:
            raise ShardError(f"document {doc.doc_id!r} is not in split {split!r}")
        if doc.language not in ALLOWED_LANGUAGES:
            raise ShardError(f"unsupported language: {doc.language!r}")
        if doc.domain not in ALLOWED_DOMAINS:
            raise ShardError(f"unsupported domain: {doc.domain!r}")
        tokens = _validate_tokens(
            doc.tokens, artifact["actual_vocab_size"], f"document {doc.doc_id!r}"
        )
        full = (BOS, *tokens, EOS)
        if len(full) > config.length:
            raise ShardError(
                f"overlong document {doc.doc_id!r}; explicit segmentation required"
            )
        if input_ids and (
            len(input_ids) + len(full) > config.length
            or language != doc.language
            or domain != doc.domain
        ):
            flush()
        if not input_ids:
            language = doc.language
            domain = doc.domain
        input_ids.extend(full)
        document_ids.extend([doc.numeric_id] * len(full))
        position_ids.extend(range(len(full)))
    flush()
    return blocks


def pack_documents(
    documents: Sequence[TokenDocument],
    artifact: Mapping[str, Any],
    config: PackConfig,
) -> list[dict[str, Any]]:
    """Pack whole documents into deterministic, homogeneous blocks."""
    _validate_pack_config(config)
    bpe.validate_artifact(artifact)
    docs = tuple(documents)
    seen_ids: set[str] = set()
    seen_numeric: set[int] = set()
    for doc in docs:
        if not isinstance(doc, TokenDocument):
            raise ShardError("documents must be TokenDocument instances")
        if doc.doc_id in seen_ids:
            raise ShardError(f"duplicate document ID: {doc.doc_id!r}")
        if doc.numeric_id in seen_numeric:
            raise ShardError(f"duplicate numeric document ID: {doc.numeric_id}")
        if doc.split not in SPLIT_ORDER:
            raise ShardError(f"invalid split: {doc.split!r}")
        seen_ids.add(doc.doc_id)
        seen_numeric.add(doc.numeric_id)
    blocks: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        blocks.extend(
            _pack_split(
                [doc for doc in docs if doc.split == split],
                artifact,
                config,
                split,
            )
        )
    if len(blocks) > MAX_TOTAL_BLOCKS:
        raise ShardError(f"total blocks exceed {MAX_TOTAL_BLOCKS}")
    return blocks


def _empty_metrics() -> dict[str, int]:
    return {key: 0 for key in REQUIRED_METRICS}


def _bump(metrics: dict[str, int], key: str, amount: int = 1) -> None:
    metrics[key] += amount


def _ensure_metric_map(
    target: dict[str, dict[str, int]], key: str
) -> dict[str, int]:
    if key not in target:
        target[key] = _empty_metrics()
    return target[key]


def compute_counts(
    documents: Sequence[TokenDocument],
    blocks: Sequence[Mapping[str, Any]],
    config: PackConfig,
) -> dict[str, Any]:
    """Count content/special/valid/pad/label observations per grouping."""
    overall = _empty_metrics()
    by_split: dict[str, dict[str, int]] = {}
    by_language: dict[str, dict[str, int]] = {}
    by_domain: dict[str, dict[str, int]] = {}
    by_sld: dict[str, dict[str, int]] = {}
    numeric = {doc.numeric_id: doc for doc in documents}
    for doc in documents:
        if doc.split not in SPLIT_ORDER or doc.language not in ALLOWED_LANGUAGES:
            raise ShardError("invalid document grouping")
        content = len(doc.tokens)
        metrics = (
            overall,
            _ensure_metric_map(by_split, doc.split),
            _ensure_metric_map(by_language, doc.language),
            _ensure_metric_map(by_domain, doc.domain),
            _ensure_metric_map(by_sld, f"{doc.split}/{doc.language}/{doc.domain}"),
        )
        for metric in metrics:
            _bump(metric, "documents")
            _bump(metric, "content_tokens", content)
            _bump(metric, "special_tokens", 2)
            _bump(metric, "valid_tokens", content + 2)
    for block in blocks:
        split = block["split"]
        language = block["language"]
        domain = block["domain"]
        if split not in SPLIT_ORDER:
            raise ShardError("block split is invalid")
        if any(numeric[doc_id].language != language for doc_id in block["document_ids"] if doc_id >= 0):
            raise ShardError("block mixes languages")
        if any(numeric[doc_id].domain != domain for doc_id in block["document_ids"] if doc_id >= 0):
            raise ShardError("block mixes domains")
        pad = sum(1 for value in block["valid"] if not value)
        ntp = sum(1 for value in block["labels"]["1"] if value != IGNORE)
        offset2 = sum(1 for value in block["labels"]["2"] if value != IGNORE)
        metrics = (
            overall,
            _ensure_metric_map(by_split, split),
            _ensure_metric_map(by_language, language),
            _ensure_metric_map(by_domain, domain),
            _ensure_metric_map(by_sld, f"{split}/{language}/{domain}"),
        )
        for metric in metrics:
            _bump(metric, "blocks")
            _bump(metric, "pad_tokens", pad)
            _bump(metric, "ntp_labels", ntp)
            _bump(metric, "offset2_labels", offset2)
    return {
        "overall": overall,
        "by_split": {key: by_split[key] for key in sorted(by_split)},
        "by_language": {key: by_language[key] for key in sorted(by_language)},
        "by_domain": {key: by_domain[key] for key in sorted(by_domain)},
        "by_split_language_domain": {key: by_sld[key] for key in sorted(by_sld)},
    }


def _build_shards(
    blocks: Sequence[Mapping[str, Any]],
    config: PackConfig,
) -> list[dict[str, Any]]:
    if len(blocks) > MAX_TOTAL_BLOCKS:
        raise ShardError(f"total blocks exceed {MAX_TOTAL_BLOCKS}")
    shards: list[dict[str, Any]] = []
    total_bytes = 0
    for split in SPLIT_ORDER:
        split_blocks = sorted(
            (block for block in blocks if block["split"] == split),
            key=lambda block: block["block_index"],
        )
        for index in range(0, len(split_blocks), config.max_blocks_per_shard):
            chunk = split_blocks[index:index + config.max_blocks_per_shard]
            data = b"".join(_canonical_json(block) + b"\n" for block in chunk)
            if len(data) > MAX_SHARD_BYTES:
                raise ShardError(f"shard exceeds {MAX_SHARD_BYTES} bytes")
            total_bytes += len(data)
            if total_bytes > MAX_TOTAL_BYTES:
                raise ShardError(f"total shard bytes exceed {MAX_TOTAL_BYTES}")
            path = f"shards/{split}-{len(shards):05d}.jsonl"
            shards.append(
                {
                    "path": path,
                    "split": split,
                    "bytes": len(data),
                    "sha256": _digest_bytes(data),
                    "blocks": len(chunk),
                    "first_block": chunk[0]["block_index"],
                    "last_block": chunk[-1]["block_index"],
                }
            )
    return shards


def _document_records(documents: Sequence[TokenDocument]) -> list[dict[str, Any]]:
    ordered = sorted(
        documents,
        key=lambda item: (SPLIT_ORDER.index(item.split), item.doc_id),
    )
    return [
        {
            "doc_id": doc.doc_id,
            "cluster_id": doc.cluster_id,
            "split": doc.split,
            "language": doc.language,
            "domain": doc.domain,
            "source_id": doc.source_id,
            "source_revision": doc.source_revision,
            "content_sha256": doc.content_sha256,
            "numeric_id": doc.numeric_id,
            "tokens": list(doc.tokens),
        }
        for doc in ordered
    ]


def build_manifest(
    documents: Sequence[TokenDocument],
    blocks: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    config: PackConfig,
    source_identity: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_pack_config(config)
    bpe.validate_artifact(artifact)
    if not isinstance(source_identity, Mapping):
        raise ShardError("source_identity must be an object")
    doc_records = _document_records(documents)
    ordered_blocks = sorted(
        blocks,
        key=lambda block: (SPLIT_ORDER.index(block["split"]), block["block_index"]),
    )
    logical_digest = _digest_bytes(_canonical_json(doc_records))
    packed_digest = _digest_bytes(_canonical_json(ordered_blocks))
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "pack_config": {
            "length": config.length,
            "offsets": list(config.offsets),
            "max_blocks_per_shard": config.max_blocks_per_shard,
            "max_blocks_total": MAX_TOTAL_BLOCKS,
            "pad": PAD,
            "bos": BOS,
            "eos": EOS,
            "ignore": IGNORE,
        },
        "tokenizer": {
            "artifact_sha256": bpe.artifact_sha256(artifact),
            "actual_vocab_size": artifact["actual_vocab_size"],
            "requested_vocab_size": artifact["requested_vocab_size"],
            "algorithm": artifact["algorithm"],
            "version": artifact["version"],
        },
        "source_identity": dict(source_identity),
        "documents": doc_records,
        "shards": [dict(shard) for shard in shards],
        "counts": compute_counts(documents, ordered_blocks, config),
        "logical_digest": logical_digest,
        "packed_logical_digest": packed_digest,
        "block_count": len(ordered_blocks),
        "shard_count": len(shards),
    }
    manifest["physical_manifest_digest"] = _digest_bytes(_canonical_json(manifest))
    return manifest


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return _canonical_json(manifest)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return _digest_bytes(manifest_bytes(manifest))


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ShardError("shard path must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ShardError(f"unsafe shard path: {value!r}")
    if len(path.parts) != 2 or path.parts[0] != "shards" or not path.parts[1].endswith(".jsonl"):
        raise ShardError(f"shard path is outside the shard directory: {value!r}")
    return path.as_posix()


def _write_dataset_tree(
    root: Path,
    documents: Sequence[TokenDocument],
    artifact: Mapping[str, Any],
    config: PackConfig,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    blocks = pack_documents(documents, artifact, config)
    if not blocks:
        raise ShardError("dataset has no blocks")
    shards = _build_shards(blocks, config)
    manifest = build_manifest(documents, blocks, artifact, config, source_identity, shards)
    if root.exists() or root.is_symlink():
        raise ShardError(f"temporary dataset path already exists: {root}")
    tokenizer_data = bpe.canonical_artifact_bytes(artifact)
    manifest_data = manifest_bytes(manifest)
    total_bytes = (
        len(tokenizer_data)
        + len(manifest_data)
        + sum(shard["bytes"] for shard in shards)
    )
    if total_bytes > MAX_TOTAL_BYTES:
        raise ShardError(f"dataset exceeds {MAX_TOTAL_BYTES} bytes")
    root.mkdir()
    (root / "shards").mkdir()
    (root / "tokenizer.json").write_bytes(tokenizer_data)
    for shard in shards:
        chunk = [
            block
            for block in blocks
            if block["split"] == shard["split"]
            and shard["first_block"] <= block["block_index"] <= shard["last_block"]
        ]
        data = b"".join(_canonical_json(block) + b"\n" for block in chunk)
        if len(data) != shard["bytes"] or _digest_bytes(data) != shard["sha256"]:
            raise ShardError("internal shard metadata mismatch")
        path = root / shard["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "manifest.json").write_bytes(manifest_data)
    return manifest


def materialize_dataset(
    root: Path | str,
    documents: Sequence[TokenDocument],
    artifact: Mapping[str, Any],
    config: PackConfig,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage a dataset in a temporary sibling and atomically promote it."""
    target = Path(root)
    if target.exists() or target.is_symlink():
        raise ShardError(f"destination already exists: {target}")
    parent = target.parent
    if not parent.is_dir():
        raise ShardError(f"destination parent does not exist: {parent}")
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        stage = temp / "dataset"
        manifest = _write_dataset_tree(
            stage, documents, artifact, config, source_identity
        )
        expected = manifest_sha256(manifest)
        load_dataset(stage, expected)
        if target.exists() or target.is_symlink():
            raise ShardError(f"destination appeared during materialization: {target}")
        os.rename(stage, target)
        temp.rmdir()
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _parse_canonical_json(data: bytes, name: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShardError(f"{name} is not valid UTF-8") from exc

    def reject(value: str) -> None:
        raise ShardError(f"{name} contains nonfinite JSON: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ShardError(f"{name} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(text, parse_constant=reject, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ShardError(f"{name} is not valid JSON: {exc}") from exc
    if _canonical_json(parsed) != data:
        raise ShardError(f"{name} is not canonical JSON bytes")
    return parsed


def _read_limited(path: Path, limit: int, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ShardError(f"{name} is missing or not a regular file")
    size = path.stat().st_size
    if size > limit:
        raise ShardError(f"{name} exceeds {limit} bytes")
    return path.read_bytes()


def _walk_files(root: Path) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise ShardError("dataset root must be a real directory")
    files: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        if current.is_symlink():
            raise ShardError("dataset contains a symlink")
        for dirname in dirnames:
            child = current / dirname
            if child.is_symlink():
                raise ShardError("dataset contains a symlink directory")
            if child != root / "shards":
                raise ShardError(f"unexpected dataset directory: {child.relative_to(root)}")
        for filename in filenames:
            child = current / filename
            if child.is_symlink():
                raise ShardError("dataset contains a symlink file")
            files.add(child.relative_to(root).as_posix())
            if len(files) > MAX_FILES:
                raise ShardError(f"dataset exceeds {MAX_FILES} files")
    return files


def _validate_manifest_header(manifest: Any, expected_sha256: str) -> None:
    if not _is_hex64(expected_sha256):
        raise ShardError("expected manifest SHA-256 is invalid")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ShardError("manifest has missing or unexpected keys")
    if manifest["format"] != FORMAT or manifest["version"] != FORMAT_VERSION:
        raise ShardError("manifest format/version is unsupported")
    pack = manifest["pack_config"]
    if not isinstance(pack, dict) or set(pack) != PACK_KEYS:
        raise ShardError("manifest pack_config is invalid")
    if pack["length"] not in ALLOWED_PACK_LENGTHS:
        raise ShardError("manifest pack length is invalid")
    if pack["offsets"] != [1, 2]:
        raise ShardError("manifest offsets are invalid")
    if (
        type(pack["max_blocks_per_shard"]) is not int
        or not 1 <= pack["max_blocks_per_shard"] <= MAX_BLOCKS_PER_SHARD
        or pack["max_blocks_total"] != MAX_TOTAL_BLOCKS
        or pack["pad"] != PAD
        or pack["bos"] != BOS
        or pack["eos"] != EOS
        or pack["ignore"] != IGNORE
    ):
        raise ShardError("manifest pack bounds are invalid")
    tokenizer = manifest["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != TOKENIZER_KEYS:
        raise ShardError("manifest tokenizer metadata is invalid")
    if not _is_hex64(tokenizer["artifact_sha256"]):
        raise ShardError("manifest tokenizer hash is invalid")
    if (
        type(tokenizer["requested_vocab_size"]) is not int
        or tokenizer["requested_vocab_size"] not in bpe.ALLOWED_VOCAB_SIZES
        or type(tokenizer["actual_vocab_size"]) is not int
        or type(tokenizer["version"]) is not int
    ):
        raise ShardError("manifest tokenizer requested size is invalid")
    source = manifest["source_identity"]
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        raise ShardError("manifest source identity is invalid")
    if not _is_hex64(source["corpus_report_digest"]) or not _is_hex64(
        source["bpe_training_digest"]
    ):
        raise ShardError("manifest source digests are invalid")
    if not isinstance(source["seed"], str) or not source["seed"].strip():
        raise ShardError("manifest source seed is invalid")
    split_percentages = source["split_percentages"]
    if (
        not isinstance(split_percentages, dict)
        or set(split_percentages) != {"train", "validation", "test"}
        or any(type(value) is not int or value < 0 for value in split_percentages.values())
        or sum(split_percentages.values()) != 100
    ):
        raise ShardError("manifest split percentages are invalid")
    if not isinstance(source["source_approvals"], list) or not source["source_approvals"]:
        raise ShardError("manifest source approvals are invalid")
    approval_keys: set[tuple[str, str]] = set()
    for entry in source["source_approvals"]:
        if not isinstance(entry, dict) or set(entry) != {"source_id", "revision"}:
            raise ShardError("manifest source approval schema is invalid")
        _require_nonempty_str(entry["source_id"], "source approval ID")
        _require_nonempty_str(entry["revision"], "source approval revision")
        key = (entry["source_id"], entry["revision"])
        if key in approval_keys:
            raise ShardError("duplicate manifest source approval")
        approval_keys.add(key)
    if not isinstance(source["train_doc_ids"], list) or any(
        not isinstance(item, str) or not item for item in source["train_doc_ids"]
    ):
        raise ShardError("manifest train document identity is invalid")
    if source["train_doc_ids"] != sorted(set(source["train_doc_ids"])):
        raise ShardError("manifest train document identity is not sorted/unique")
    for key in ("cluster_splits", "document_splits"):
        if not isinstance(source[key], dict) or any(
            not isinstance(name, str) or split not in SPLIT_ORDER
            for name, split in source[key].items()
        ):
            raise ShardError(f"manifest {key} identity is invalid")
    if not isinstance(source["quarantine"], list):
        raise ShardError("manifest quarantine metadata is invalid")
    for entry in source["quarantine"]:
        if not isinstance(entry, dict) or set(entry) != {
            "cluster_id",
            "reasons",
            "member_ids",
        }:
            raise ShardError("manifest quarantine schema is invalid")
        _require_nonempty_str(entry["cluster_id"], "quarantine cluster ID")
        if not isinstance(entry["reasons"], list) or not isinstance(entry["member_ids"], list):
            raise ShardError("manifest quarantine metadata is invalid")
    docs = manifest["documents"]
    if not isinstance(docs, list) or not docs:
        raise ShardError("manifest documents are invalid")
    if len(docs) > MAX_DOCUMENTS:
        raise ShardError(f"manifest documents exceed {MAX_DOCUMENTS}")
    shards = manifest["shards"]
    if not isinstance(shards, list) or not all(isinstance(shard, dict) for shard in shards):
        raise ShardError("manifest shards are invalid")
    if manifest["shard_count"] != len(shards):
        raise ShardError("manifest shard count is inconsistent")


def _validate_documents(
    docs: Any,
    artifact: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[TokenDocument, ...]:
    if not isinstance(docs, list) or not docs:
        raise ShardError("manifest documents are invalid")
    result: list[TokenDocument] = []
    seen_ids: set[str] = set()
    seen_numeric: set[int] = set()
    for index, item in enumerate(docs):
        if not isinstance(item, dict) or set(item) != DOC_KEYS:
            raise ShardError("manifest document schema is invalid")
        for key in ("doc_id", "cluster_id", "source_id", "source_revision"):
            _require_nonempty_str(item[key], f"document {key}")
        if item["split"] not in SPLIT_ORDER:
            raise ShardError("manifest document split is invalid")
        if item["language"] not in ALLOWED_LANGUAGES:
            raise ShardError("manifest document language is invalid")
        if item["domain"] not in ALLOWED_DOMAINS:
            raise ShardError("manifest document domain is invalid")
        if not _is_hex64(item["content_sha256"]):
            raise ShardError("manifest document content digest is invalid")
        if type(item["numeric_id"]) is not int or item["numeric_id"] != index:
            raise ShardError("manifest numeric document IDs must be contiguous")
        if item["doc_id"] in seen_ids or item["numeric_id"] in seen_numeric:
            raise ShardError("duplicate manifest document identity")
        seen_ids.add(item["doc_id"])
        seen_numeric.add(item["numeric_id"])
        tokens = _validate_tokens(
            item["tokens"], artifact["actual_vocab_size"], f"document {item['doc_id']!r}"
        )
        result.append(
            TokenDocument(
                item["doc_id"],
                item["cluster_id"],
                item["split"],
                item["language"],
                item["domain"],
                item["source_id"],
                item["source_revision"],
                item["content_sha256"],
                item["numeric_id"],
                tokens,
            )
        )
    expected_order = sorted(
        result, key=lambda doc: (SPLIT_ORDER.index(doc.split), doc.doc_id)
    )
    if result != expected_order:
        raise ShardError("manifest documents are not in canonical order")
    if source["train_doc_ids"] != sorted(
        doc.doc_id for doc in result if doc.split == "train"
    ):
        raise ShardError("manifest train document identity is inconsistent")
    if set(source["document_splits"]) != seen_ids:
        raise ShardError("manifest document split identity is incomplete")
    for doc in result:
        if source["document_splits"].get(doc.doc_id) != doc.split:
            raise ShardError("manifest document split mismatch")
        if source["cluster_splits"].get(doc.cluster_id) != doc.split:
            raise ShardError("manifest cluster split mismatch")
    approval_set = {
        (entry["source_id"], entry["revision"]) for entry in source["source_approvals"]
    }
    for doc in result:
        if (doc.source_id, doc.source_revision) not in approval_set:
            raise ShardError("manifest document source is not approved")
    quarantined = {
        member
        for entry in source["quarantine"]
        for member in entry["member_ids"]
        if isinstance(member, str)
    }
    if quarantined & seen_ids:
        raise ShardError("quarantined document was packed")
    return tuple(result)


def _validate_block(
    block: Any,
    artifact: Mapping[str, Any],
    config: PackConfig,
    numeric: Mapping[int, TokenDocument],
) -> None:
    if not isinstance(block, dict) or set(block) != BLOCK_KEYS:
        raise ShardError("block schema is invalid")
    if not isinstance(block["split"], str) or block["split"] not in SPLIT_ORDER:
        raise ShardError("block split is invalid")
    if (
        not isinstance(block["language"], str)
        or block["language"] not in ALLOWED_LANGUAGES
        or not isinstance(block["domain"], str)
        or block["domain"] not in ALLOWED_DOMAINS
    ):
        raise ShardError("block language/domain is invalid")
    if type(block["block_index"]) is not int or block["block_index"] < 0:
        raise ShardError("block index is invalid")
    for key in ("input_ids", "document_ids", "position_ids", "valid"):
        if not isinstance(block[key], list) or len(block[key]) != config.length:
            raise ShardError(f"block {key} has invalid length")
    labels = block.get("labels")
    if not isinstance(labels, dict) or set(labels) != {"1", "2"}:
        raise ShardError("block labels schema is invalid")
    for key in ("1", "2"):
        if not isinstance(labels[key], list) or len(labels[key]) != config.length:
            raise ShardError("block label length is invalid")
    for index in range(config.length):
        token = block["input_ids"][index]
        if type(token) is not int or not 0 <= token < artifact["actual_vocab_size"]:
            raise ShardError("block token ID is out of range")
        doc_id = block["document_ids"][index]
        if type(doc_id) is not int or (doc_id != -1 and doc_id not in numeric):
            raise ShardError("block document ID is invalid")
        if type(block["valid"][index]) is not bool or block["valid"][index] != (doc_id >= 0):
            raise ShardError("block valid/document mask is inconsistent")
        position = block["position_ids"][index]
        if type(position) is not int or position < 0:
            raise ShardError("block position ID is invalid")
        if doc_id >= 0 and (block["language"] != numeric[doc_id].language):
            raise ShardError("block mixes languages")
        if doc_id >= 0 and (block["domain"] != numeric[doc_id].domain):
            raise ShardError("block mixes domains")
    valid_count = sum(1 for value in block["valid"] if value)
    if valid_count == 0:
        raise ShardError("block has no valid tokens")
    if block["valid"] != [True] * valid_count + [False] * (config.length - valid_count):
        raise ShardError("valid mask must be a true prefix followed by padding")
    pads = list(range(valid_count, config.length))
    if pads:
        if any(block["document_ids"][index] != -1 for index in pads):
            raise ShardError("padding document mask is invalid")
        if any(block["position_ids"][index] != 0 for index in pads):
            raise ShardError("padding position IDs are invalid")
        if any(block["input_ids"][index] != PAD for index in pads):
            raise ShardError("padding token IDs are invalid")
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, valid_count + 1):
        if index == valid_count or (
            block["document_ids"][index] != block["document_ids"][index - 1]
        ):
            segments.append((start, index))
            start = index
    seen_segments: set[int] = set()
    for start, end in segments:
        doc_id = block["document_ids"][start]
        if doc_id in seen_segments:
            raise ShardError("document segment appears more than once in a block")
        seen_segments.add(doc_id)
        if block["input_ids"][start] != BOS or block["input_ids"][end - 1] != EOS:
            raise ShardError("document segment lacks exact BOS/EOS boundaries")
        if any(
            token < bpe.BYTE_OFFSET
            for token in block["input_ids"][start + 1:end - 1]
        ):
            raise ShardError("content segment contains a control token")
        if block["position_ids"][start:end] != list(range(end - start)):
            raise ShardError("document positions are not reset and contiguous")
    for offset in (1, 2):
        values = labels[str(offset)]
        for index, value in enumerate(values):
            if value == IGNORE:
                continue
            if type(value) is not int or not 0 <= value < artifact["actual_vocab_size"]:
                raise ShardError("label value is out of vocabulary range")
            if (
                index + offset >= config.length
                or block["document_ids"][index] < 0
                or block["document_ids"][index]
                != block["document_ids"][index + offset]
                or value != block["input_ids"][index + offset]
            ):
                raise ShardError("label crosses a document or padding boundary")



def _validate_shards(
    manifest: Mapping[str, Any],
    root: Path,
    artifact: Mapping[str, Any],
    config: PackConfig,
    numeric: Mapping[int, TokenDocument],
) -> list[dict[str, Any]]:
    shards = manifest["shards"]
    if len(shards) > MAX_FILES:
        raise ShardError("too many shards")
    expected_blocks = 0
    total_bytes = 0
    blocks: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    next_block = {split: 0 for split in SPLIT_ORDER}
    last_split_index = -1
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != SHARD_KEYS:
            raise ShardError("shard metadata schema is invalid")
        path_text = _safe_relative_path(shard["path"])
        if path_text in seen_paths:
            raise ShardError("duplicate shard path")
        seen_paths.add(path_text)
        if shard["split"] not in SPLIT_ORDER:
            raise ShardError("shard split is invalid")
        split_index = SPLIT_ORDER.index(shard["split"])
        if split_index < last_split_index:
            raise ShardError("shards are not in deterministic split order")
        last_split_index = split_index
        if type(shard["bytes"]) is not int or not 0 <= shard["bytes"] <= MAX_SHARD_BYTES:
            raise ShardError("shard byte count is invalid")
        if not _is_hex64(shard["sha256"]):
            raise ShardError("shard hash is invalid")
        if type(shard["blocks"]) is not int or shard["blocks"] < 0:
            raise ShardError("shard block count is invalid")
        if (
            type(shard["first_block"]) is not int
            or type(shard["last_block"]) is not int
            or shard["first_block"] < 0
            or shard["last_block"] < shard["first_block"]
            or shard["blocks"] != shard["last_block"] - shard["first_block"] + 1
        ):
            raise ShardError("shard block range metadata is invalid")
        if shard["first_block"] != next_block[shard["split"]]:
            raise ShardError("shard block range is not contiguous")
        next_block[shard["split"]] = shard["last_block"] + 1
        data = _read_limited(root / path_text, MAX_SHARD_BYTES, f"shard {path_text}")
        if len(data) != shard["bytes"] or _digest_bytes(data) != shard["sha256"]:
            raise ShardError(f"shard hash/size mismatch: {path_text}")
        if not data.endswith(b"\n") or not data:
            raise ShardError(f"shard lacks exact LF terminator: {path_text}")
        lines = data[:-1].split(b"\n")
        if len(lines) != shard["blocks"] or not lines:
            raise ShardError(f"shard block count mismatch: {path_text}")
        shard_blocks: list[dict[str, Any]] = []
        for line in lines:
            if not line:
                raise ShardError(f"shard contains a blank line: {path_text}")
            block = _parse_canonical_json(line, f"shard line {path_text}")
            _validate_block(block, artifact, config, numeric)
            if block["split"] != shard["split"]:
                raise ShardError(f"shard split mismatch: {path_text}")
            shard_blocks.append(block)
        if [block["block_index"] for block in shard_blocks] != list(
            range(shard["first_block"], shard["last_block"] + 1)
        ):
            raise ShardError(f"shard block range mismatch: {path_text}")
        if shard_blocks[0]["block_index"] != shard["first_block"]:
            raise ShardError("shard first block mismatch")
        if shard_blocks[-1]["block_index"] != shard["last_block"]:
            raise ShardError("shard last block mismatch")
        expected_blocks += len(shard_blocks)
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ShardError(f"total shard bytes exceed {MAX_TOTAL_BYTES}")
        blocks.extend(shard_blocks)
    if expected_blocks > MAX_TOTAL_BLOCKS or len(blocks) > MAX_TOTAL_BLOCKS:
        raise ShardError(f"total blocks exceed {MAX_TOTAL_BLOCKS}")
    for split in SPLIT_ORDER:
        actual = sum(1 for block in blocks if block["split"] == split)
        if next_block[split] != actual:
            raise ShardError(f"block indices are not contiguous for {split}")
    if type(manifest["block_count"]) is not int or manifest["block_count"] < 0:
        raise ShardError("manifest block count is invalid")
    if manifest["block_count"] != len(blocks):
        raise ShardError("manifest block count mismatch")
    if manifest["shard_count"] != len(shards):
        raise ShardError("manifest shard count mismatch")
    return blocks


def load_dataset(
    root: Path | str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify the complete physical set and return verified synthetic blocks."""
    base = Path(root)
    if not _is_hex64(expected_manifest_sha256):
        raise ShardError("expected manifest SHA-256 is invalid")
    files = _walk_files(base)
    manifest_data = _read_limited(base / "manifest.json", MAX_MANIFEST_BYTES, "manifest.json")
    if _digest_bytes(manifest_data) != expected_manifest_sha256:
        raise ShardError("manifest SHA-256 mismatch")
    manifest = _parse_canonical_json(manifest_data, "manifest.json")
    _validate_manifest_header(manifest, expected_manifest_sha256)
    tokenizer_data = _read_limited(
        base / "tokenizer.json", MAX_ARTIFACT_BYTES, "tokenizer.json"
    )
    tokenizer_meta = manifest["tokenizer"]
    if _digest_bytes(tokenizer_data) != tokenizer_meta["artifact_sha256"]:
        raise ShardError("tokenizer file hash mismatch")
    artifact = bpe.load_artifact(tokenizer_data, tokenizer_meta["artifact_sha256"])
    if (
        type(tokenizer_meta["actual_vocab_size"]) is not int
        or type(tokenizer_meta["requested_vocab_size"]) is not int
        or type(tokenizer_meta["version"]) is not int
        or artifact["actual_vocab_size"] != tokenizer_meta["actual_vocab_size"]
        or artifact["requested_vocab_size"] != tokenizer_meta["requested_vocab_size"]
        or artifact["algorithm"] != tokenizer_meta["algorithm"]
        or artifact["version"] != tokenizer_meta["version"]
    ):
        raise ShardError("tokenizer metadata mismatch")
    if manifest["source_identity"]["bpe_training_digest"] != artifact["training_digest"]:
        raise ShardError("tokenizer training digest mismatch")
    docs = _validate_documents(manifest["documents"], artifact, manifest["source_identity"])
    numeric = {doc.numeric_id: doc for doc in docs}
    config = PackConfig(
        manifest["pack_config"]["length"],
        manifest["pack_config"]["max_blocks_per_shard"],
        tuple(manifest["pack_config"]["offsets"]),
    )
    _validate_pack_config(config)
    shards = manifest["shards"]
    allowed = {"manifest.json", "tokenizer.json"} | {
        shard["path"] for shard in shards if isinstance(shard, dict)
    }
    if files != allowed:
        raise ShardError("dataset has missing or unexpected files")
    declared_bytes = len(manifest_data) + len(tokenizer_data)
    for shard in shards:
        if not isinstance(shard, dict) or type(shard.get("bytes")) is not int:
            raise ShardError("shard metadata is invalid")
        declared_bytes += shard["bytes"]
    if declared_bytes > MAX_TOTAL_BYTES:
        raise ShardError(f"dataset exceeds {MAX_TOTAL_BYTES} bytes")
    blocks = _validate_shards(manifest, base, artifact, config, numeric)
    if _digest_bytes(_canonical_json(blocks)) != manifest["packed_logical_digest"]:
        raise ShardError("packed logical digest mismatch")
    doc_records = _document_records(docs)
    if _digest_bytes(_canonical_json(doc_records)) != manifest["logical_digest"]:
        raise ShardError("logical digest mismatch")
    if compute_counts(docs, blocks, config) != manifest["counts"]:
        raise ShardError("manifest counts mismatch")
    expected_blocks = pack_documents(docs, artifact, config)
    if _canonical_json(expected_blocks) != _canonical_json(blocks):
        raise ShardError("shards do not match repacked logical blocks")
    physical = {key: value for key, value in manifest.items() if key != "physical_manifest_digest"}
    if _digest_bytes(_canonical_json(physical)) != manifest["physical_manifest_digest"]:
        raise ShardError("physical manifest digest mismatch")
    return {
        "manifest": manifest,
        "artifact": artifact,
        "documents": docs,
        "blocks": tuple(blocks),
    }


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def demo_report() -> dict[str, Any]:
    """Materialize the synthetic demo twice and verify shard invariants."""
    bundle = build_demo_bundle()
    config = PackConfig(length=96, max_blocks_per_shard=MAX_BLOCKS_PER_SHARD)
    with tempfile.TemporaryDirectory(prefix="splot-t0010-demo-") as temp:
        base = Path(temp)
        first = materialize_dataset(
            base / "first",
            bundle["documents"],
            bundle["artifact"],
            config,
            bundle["source_identity"],
        )
        second = materialize_dataset(
            base / "second",
            bundle["documents"],
            bundle["artifact"],
            config,
            bundle["source_identity"],
        )
        first_snapshot = _snapshot(base / "first")
        second_snapshot = _snapshot(base / "second")
        partition = materialize_dataset(
            base / "partition",
            bundle["documents"],
            bundle["artifact"],
            PackConfig(length=96, max_blocks_per_shard=1),
            bundle["source_identity"],
        )
        loaded = load_dataset(base / "first", manifest_sha256(first))
        first_bytes = manifest_bytes(first)
        report = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "pack_length": config.length,
            "documents": len(bundle["documents"]),
            "blocks": first["block_count"],
            "shards": first["shard_count"],
            "logical_digest": first["logical_digest"],
            "packed_logical_digest": first["packed_logical_digest"],
            "physical_manifest_digest": first["physical_manifest_digest"],
            "manifest_sha256": manifest_sha256(first),
            "manifest_bytes": len(first_bytes),
            "artifact_sha256": bundle["artifact_sha256"],
            "artifact_actual_vocab_size": bundle["artifact"]["actual_vocab_size"],
            "corpus_report_digest": bundle["report"]["digest"],
            "bpe_training_digest": bundle["artifact"]["training_digest"],
            "train_doc_ids": bundle["source_identity"]["train_doc_ids"],
            "counts": first["counts"],
            "independent_materializations_identical": first_snapshot == second_snapshot,
            "partition_invariance": {
                "logical_digest_equal": first["logical_digest"] == partition["logical_digest"],
                "packed_logical_digest_equal": first["packed_logical_digest"]
                == partition["packed_logical_digest"],
                "physical_manifest_digest_differs": first["physical_manifest_digest"]
                != partition["physical_manifest_digest"],
                "shard_count": partition["shard_count"],
            },
            "loaded_blocks": len(loaded["blocks"]),
            "manifest": first,
        }
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print the deterministic synthetic tokenized-shard demo report",
    )
    args = parser.parse_args(argv)
    if not args.demo:
        parser.print_help(sys.stderr)
        return 2
    print(
        json.dumps(
            demo_report(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
