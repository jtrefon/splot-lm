"""Deterministic synthetic producer for the standalone splot-pilot-blocks-v2 format.

The producer takes explicit source bytes, document/segment intervals and content
token IDs. It does not acquire data, fit a tokenizer, fetch sources or train a
model. It validates the inputs, packs segment-local blocks, builds a canonical
six-file dataset and verifies it with pilot_v2 before returning/writing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pilot_v2 as v2


class ProducerError(ValueError):
    """Invalid producer input, output path or generated dataset."""


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    family_id: str
    language: str
    split: str
    source_url: str
    revision: str
    rights_evidence_url: str
    raw: bytes
    content: bytes


@dataclass(frozen=True)
class SegmentSpec:
    doc_id: int
    segment_id: str
    source_id: str
    source_byte_start: int
    source_byte_end: int
    segment_byte_start: int
    segment_byte_end: int
    segment_index: int
    segment_count: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ProducerSpec:
    sequence_length: int
    vocab_size: int
    token_bytes: tuple[bytes, ...]
    sources: tuple[SourceSpec, ...]
    segments: tuple[SegmentSpec, ...]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_int(value) -> bool:
    return type(value) is int


def _require_int(value, name, minimum=0, maximum=None):
    if not _is_int(value) or value < minimum:
        raise ProducerError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProducerError(f"{name} exceeds {maximum}")
    return value


def _require_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ProducerError(f"{name} must be nonempty")
    return value


def _require_bytes(value, name):
    if not isinstance(value, bytes) or not value:
        raise ProducerError(f"{name} must be nonempty bytes")
    return value


def _validate_token_table(spec: ProducerSpec) -> tuple[bytes, ...]:
    sequence = _require_int(spec.sequence_length, "sequence_length",
                            v2.SEQUENCE_LENGTH_MIN, v2.SEQUENCE_LENGTH_MAX)
    vocab = _require_int(spec.vocab_size, "vocab_size", v2.VOCAB_MIN, v2.VOCAB_MAX)
    del sequence
    if not isinstance(spec.token_bytes, (tuple, list)):
        raise ProducerError("token_bytes must be a sequence")
    if len(spec.token_bytes) != vocab:
        raise ProducerError("token_bytes length does not match vocab_size")
    table = []
    special = {v2.PAD, v2.BOS, v2.EOS, v2.UNK, *v2.RESERVED}
    seen = {}
    for token_id, value in enumerate(spec.token_bytes):
        if not isinstance(value, bytes):
            raise ProducerError(f"token_bytes[{token_id}] must be bytes")
        if token_id in special:
            if value != b"":
                raise ProducerError("special token bytes must be empty")
            table.append(value)
            continue
        if not value:
            raise ProducerError("content token bytes must be nonempty")
        if len(value) > v2.MAX_SOURCE_BYTES:
            raise ProducerError("content token bytes exceed the bound")
        if value in seen:
            raise ProducerError(
                f"duplicate content token bytes for IDs {seen[value]} and {token_id}")
        seen[value] = token_id
        table.append(value)
    return tuple(table)


def _validate_sources(spec: ProducerSpec) -> tuple[dict, dict, dict]:
    if not isinstance(spec.sources, (tuple, list)):
        raise ProducerError("sources must be a sequence")
    if not 1 <= len(spec.sources) <= v2.MAX_SOURCES:
        raise ProducerError("source count is out of bounds")
    by_id = {}
    family_split = {}
    family_language = {}
    split_seen = set()
    for index, source in enumerate(spec.sources):
        if not isinstance(source, SourceSpec):
            raise ProducerError(f"sources[{index}] must be SourceSpec")
        source_id = _require_str(source.source_id, "source_id")
        family_id = _require_str(source.family_id, "family_id")
        if source.language not in v2.LANGUAGES:
            raise ProducerError("source language must be en or pl")
        if source.split not in v2.SPLIT_ORDER:
            raise ProducerError("source split is invalid")
        _require_str(source.source_url, "source_url")
        _require_str(source.revision, "revision")
        _require_str(source.rights_evidence_url, "rights_evidence_url")
        raw = _require_bytes(source.raw, "raw")
        content = _require_bytes(source.content, "content")
        if len(raw) > v2.MAX_SOURCE_BYTES or len(content) > v2.MAX_SOURCE_BYTES:
            raise ProducerError("source bytes exceed the bound")
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProducerError("source content must be valid UTF-8") from exc
        if source_id in by_id:
            raise ProducerError(f"duplicate source_id: {source_id}")
        if family_id in family_split:
            if (family_split[family_id] != source.split
                    or family_language[family_id] != source.language):
                raise ProducerError("a family crosses split or language")
        else:
            family_split[family_id] = source.split
            family_language[family_id] = source.language
        by_id[source_id] = source
        split_seen.add(source.split)
    if split_seen != set(v2.SPLIT_ORDER):
        raise ProducerError("all three splits need a source")
    return by_id, family_split, family_language


def _build_documents(spec: ProducerSpec, sources: dict, family_split: dict,
                     family_language: dict) -> tuple[list[dict], dict[str, int]]:
    if not isinstance(spec.segments, (tuple, list)):
        raise ProducerError("segments must be a sequence")
    if not 1 <= len(spec.segments) <= v2.MAX_DOCUMENTS:
        raise ProducerError("segment count is out of bounds")
    docs = []
    selected_by_source = {source_id: 0 for source_id in sources}
    doc_ids = set()
    segment_ids = set()
    groups = {}
    for index, segment in enumerate(spec.segments):
        if not isinstance(segment, SegmentSpec):
            raise ProducerError(f"segments[{index}] must be SegmentSpec")
        doc_id = _require_int(segment.doc_id, "doc_id")
        segment_id = _require_str(segment.segment_id, "segment_id")
        if doc_id in doc_ids:
            raise ProducerError(f"duplicate doc_id: {doc_id}")
        if segment_id in segment_ids:
            raise ProducerError(f"duplicate segment_id: {segment_id}")
        doc_ids.add(doc_id)
        segment_ids.add(segment_id)
        source = sources.get(segment.source_id)
        if source is None:
            raise ProducerError(f"unknown source_id: {segment.source_id}")
        if source.split not in v2.SPLIT_ORDER:
            raise ProducerError("source split is invalid")
        content = source.content
        source_start = _require_int(segment.source_byte_start, "source_byte_start")
        source_end = _require_int(segment.source_byte_end, "source_byte_end")
        seg_start = _require_int(segment.segment_byte_start, "segment_byte_start")
        seg_end = _require_int(segment.segment_byte_end, "segment_byte_end")
        if not 0 <= source_start < source_end <= len(content):
            raise ProducerError("source interval is out of bounds")
        if not source_start <= seg_start < seg_end <= source_end:
            raise ProducerError("segment interval is outside the source document")
        if not isinstance(segment.token_ids, (tuple, list)):
            raise ProducerError("token_ids must be a sequence")
        token_ids = tuple(segment.token_ids)
        if not token_ids:
            raise ProducerError("segment needs at least one content token")
        if len(token_ids) + 2 > spec.sequence_length:
            raise ProducerError("segment does not fit in sequence_length")
        if any(not _is_int(token) or not v2.CONTENT_MIN <= token < spec.vocab_size
               for token in token_ids):
            raise ProducerError("token ID is out of content range")
        token_bytes = b"".join(spec.token_bytes[token] for token in token_ids)
        segment_bytes = content[seg_start:seg_end]
        if token_bytes != segment_bytes:
            raise ProducerError("token bytes do not equal the selected segment")
        try:
            segment_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProducerError("selected segment must be valid UTF-8") from exc
        full = content[source_start:source_end]
        try:
            full.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProducerError("full document must be valid UTF-8") from exc
        segment_index = _require_int(segment.segment_index, "segment_index")
        segment_count = _require_int(segment.segment_count, "segment_count", 1)
        if segment_index >= segment_count:
            raise ProducerError("segment_index is outside segment_count")
        truncated = (seg_end - seg_start) < (source_end - source_start)
        doc = {
            "doc_id": doc_id,
            "segment_id": segment_id,
            "source_id": segment.source_id,
            "family_id": source.family_id,
            "language": source.language,
            "split": source.split,
            "source_byte_start": source_start,
            "source_byte_end": source_end,
            "segment_byte_start": seg_start,
            "segment_byte_end": seg_end,
            "full_document_sha256": _sha(full),
            "segment_sha256": _sha(segment_bytes),
            "segment_index": segment_index,
            "segment_count": segment_count,
            "truncated": truncated,
            "selected_bytes": seg_end - seg_start,
            "token_count": len(token_ids),
        }
        if family_split.get(source.family_id) != source.split:
            raise ProducerError("family split mismatch")
        if family_language.get(source.family_id) != source.language:
            raise ProducerError("family language mismatch")
        groups.setdefault((source.source_id, source_start, source_end,
                           doc["full_document_sha256"]), []).append(doc)
        selected_by_source[source.source_id] += doc["selected_bytes"]
        docs.append(doc)
    full_intervals = {source_id: [] for source_id in sources}
    selected_intervals = {source_id: [] for source_id in sources}
    for (source_id, source_start, source_end, _), members in groups.items():
        counts = {item["segment_count"] for item in members}
        if len(counts) != 1:
            raise ProducerError("segment_count differs within a document group")
        segment_count = counts.pop()
        ordered = sorted(members, key=lambda item: (
            item["segment_byte_start"], item["segment_byte_end"], item["segment_index"]))
        indices = [item["segment_index"] for item in ordered]
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ProducerError("segment indices are not ordered by byte start")
        intervals = [(item["segment_byte_start"], item["segment_byte_end"])
                     for item in ordered]
        for (start, end), (next_start, next_end) in zip(intervals, intervals[1:]):
            if end > next_start:
                raise ProducerError("document segments overlap")
        full_intervals[source_id].append((source_start, source_end))
        selected_intervals[source_id].extend(intervals)
        if len(members) == segment_count:
            if indices != list(range(segment_count)):
                raise ProducerError("complete document indices are not ordered")
            if (intervals[0][0] != source_start or intervals[-1][1] != source_end
                    or any(end != next_start for (_, end), (next_start, _)
                           in zip(intervals, intervals[1:]))):
                raise ProducerError("complete document does not partition its source")
            if segment_count == 1 and (ordered[0]["segment_index"] != 0
                                      or ordered[0]["truncated"]):
                raise ProducerError("single full segment must be index 0 and untruncated")
        elif any(not item["truncated"] for item in members):
            raise ProducerError("partial selection must be explicitly truncated")
    for source_id in sources:
        for intervals, label in ((full_intervals[source_id], "full-document"),
                                 (selected_intervals[source_id], "selected")):
            ordered = sorted(intervals)
            for (start, end), (next_start, next_end) in zip(ordered, ordered[1:]):
                if end > next_start:
                    raise ProducerError(f"{label} intervals overlap within a source")
        source = sources[source_id]
        if selected_by_source[source_id] > len(source.content):
            raise ProducerError("selected bytes exceed source content")
    return docs, selected_by_source


def _build_block(segments: list[SegmentSpec], table: tuple[bytes, ...],
                 sequence_length: int) -> dict:
    input_ids = []
    document_ids = []
    position_ids = []
    valid = []
    control = []
    byte_lengths = []
    byte_offsets = []
    labels = {"1": [], "2": []}
    for segment in segments:
        tokens = list(segment.token_ids)
        segment_input = [v2.BOS] + tokens + [v2.EOS]
        length = len(segment_input)
        input_ids.extend(segment_input)
        document_ids.extend([segment.doc_id] * length)
        position_ids.extend(range(length))
        valid.extend([True] * length)
        control.extend([True] + [False] * len(tokens) + [True])
        lengths = [0] + [len(table[token]) for token in tokens] + [0]
        offsets = []
        cursor = 0
        for token in tokens:
            offsets.append(cursor)
            cursor += len(table[token])
        byte_lengths.extend(lengths)
        byte_offsets.extend([0] + offsets + [cursor])
        for offset in (1, 2):
            for index in range(length):
                labels[str(offset)].append(
                    segment_input[index + offset] if index + offset < length else v2.IGNORE)
    if len(input_ids) > sequence_length:
        raise ProducerError("packed block exceeds sequence_length")
    pad = sequence_length - len(input_ids)
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


def _pack_blocks(spec: ProducerSpec, sources: dict, docs: dict[int, dict],
                 table: tuple[bytes, ...]) -> tuple[dict, list[dict]]:
    by_split = {split: [] for split in v2.SPLIT_ORDER}
    for segment in spec.segments:
        by_split[sources[segment.source_id].split].append(segment)
    blocks = []
    shard_records = []
    for split in v2.SPLIT_ORDER:
        packed = []
        current = []
        current_length = 0
        for segment in by_split[split]:
            length = len(segment.token_ids) + 2
            if length > spec.sequence_length:
                raise ProducerError("segment does not fit in sequence_length")
            if current and current_length + length > spec.sequence_length:
                packed.append(current)
                current = []
                current_length = 0
            current.append(segment)
            current_length += length
            if current_length == spec.sequence_length:
                packed.append(current)
                current = []
                current_length = 0
        if current:
            packed.append(current)
        if not packed:
            raise ProducerError(f"split {split} has no segment")
        if len(packed) > {"train": v2.MAX_TRAIN_BLOCKS,
                          "validation": v2.MAX_VALIDATION_BLOCKS,
                          "test": v2.MAX_TEST_BLOCKS}[split]:
            raise ProducerError(f"split {split} block count exceeds the cap")
        split_blocks = []
        split_entity = {"blocks": len(packed), **{key: 0 for key in v2.COUNT_FIELDS}}
        family_tokens = {}
        family_bytes = {}
        language_tokens = {}
        language_bytes = {}
        for block_segments in packed:
            block = _build_block(block_segments, table, spec.sequence_length)
            line = v2._canonical(block)
            if len(line) > v2.MAX_LINE_BYTES:
                raise ProducerError("block line exceeds the byte cap")
            split_blocks.append((block, block_segments))
            for segment in block_segments:
                doc = docs[segment.doc_id]
                content = len(segment.token_ids)
                selected = sum(len(table[token]) for token in segment.token_ids)
                entity = {
                    "valid_tokens": content + 2,
                    "content_tokens": content,
                    "ntp_labels": content + 1,
                    "content_labels": content,
                    "eos_labels": 1,
                    "offset2_labels": content,
                    "content_target_bytes": selected,
                    "selected_bytes": selected,
                }
                for key, value in entity.items():
                    split_entity[key] += value
                family_tokens[doc["family_id"]] = family_tokens.get(doc["family_id"], 0) + content
                family_bytes[doc["family_id"]] = family_bytes.get(doc["family_id"], 0) + selected
                language_tokens[doc["language"]] = language_tokens.get(doc["language"], 0) + content
                language_bytes[doc["language"]] = language_bytes.get(doc["language"], 0) + selected
        payload = b"".join(v2._canonical(block) + b"\n"
                           for block, _ in split_blocks)
        shard_records.append({
            "split": split,
            "path": f"{split}.jsonl",
            "sha256": _sha(payload),
            "bytes": len(payload),
            "blocks": split_entity["blocks"],
            "valid_tokens": split_entity["valid_tokens"],
            "content_tokens": split_entity["content_tokens"],
            "ntp_labels": split_entity["ntp_labels"],
            "content_labels": split_entity["content_labels"],
            "eos_labels": split_entity["eos_labels"],
            "offset2_labels": split_entity["offset2_labels"],
            "content_target_bytes": split_entity["content_target_bytes"],
            "selected_bytes": split_entity["selected_bytes"],
            "family_content_tokens": family_tokens,
            "family_content_bytes": family_bytes,
            "language_content_tokens": language_tokens,
            "language_content_bytes": language_bytes,
        })
        blocks.append((split, split_blocks, payload, split_entity))
    return blocks, shard_records


def build_dataset(spec: ProducerSpec) -> dict:
    """Build and verify the six-file dataset in memory; no writes."""
    table = _validate_token_table(spec)
    sources, family_split, family_language = _validate_sources(spec)
    docs_list, selected_by_source = _build_documents(
        spec, sources, family_split, family_language)
    docs = {doc["doc_id"]: doc for doc in docs_list}
    blocks, shard_records = _pack_blocks(spec, sources, docs, table)
    for shard in shard_records:
        if shard["bytes"] > v2.MAX_TOTAL_BYTES:
            raise ProducerError("shard exceeds the aggregate byte cap")
    if sum(shard["bytes"] for shard in shard_records) > v2.MAX_TOTAL_BYTES:
        raise ProducerError("shards exceed the aggregate byte cap")
    if sum(shard["blocks"] for shard in shard_records) > v2.MAX_TOTAL_BLOCKS:
        raise ProducerError("blocks exceed the aggregate cap")

    source_records = []
    for source in spec.sources:
        source_records.append({
            "source_id": source.source_id,
            "family_id": source.family_id,
            "language": source.language,
            "split": source.split,
            "source_url": source.source_url,
            "revision": source.revision,
            "rights_evidence_url": source.rights_evidence_url,
            "raw_sha256": _sha(source.raw),
            "content_sha256": _sha(source.content),
            "raw_bytes": len(source.raw),
            "content_bytes": len(source.content),
            "selected_bytes": selected_by_source[source.source_id],
        })
    tokenizer = {
        "format": v2.TOKENIZER_FORMAT,
        "version": v2.TOKENIZER_VERSION,
        "actual_vocab_size": spec.vocab_size,
        "special_token_ids": json.loads(json.dumps(v2.SPECIAL_IDS)),
        "token_bytes": [value.hex() for value in table],
    }
    source_bytes = {
        "format": v2.SOURCE_FORMAT,
        "version": v2.SOURCE_VERSION,
        "sources": {
            source.source_id: {"raw": source.raw.hex(), "content": source.content.hex()}
            for source in spec.sources
        },
    }
    tokenizer_bytes = v2._canonical(tokenizer)
    source_bytes_bytes = v2._canonical(source_bytes)
    if len(tokenizer_bytes) > v2.MAX_TOKENIZER_BYTES:
        raise ProducerError("tokenizer artifact exceeds the byte cap")
    if len(source_bytes_bytes) > v2.MAX_SOURCE_BYTES:
        raise ProducerError("source-bytes artifact exceeds the byte cap")

    counts = {"blocks": 0, **{key: 0 for key in v2.COUNT_FIELDS},
              "by_split": {}, "by_language": {}, "by_family": {}}
    for split, _, _, split_entity in blocks:
        counts["by_split"][split] = split_entity
        counts["blocks"] += split_entity["blocks"]
        for key in v2.COUNT_FIELDS:
            counts[key] += split_entity[key]
    for doc in docs_list:
        entity = {
            "valid_tokens": doc["token_count"] + 2,
            "content_tokens": doc["token_count"],
            "ntp_labels": doc["token_count"] + 1,
            "content_labels": doc["token_count"],
            "eos_labels": 1,
            "offset2_labels": doc["token_count"],
            "content_target_bytes": doc["selected_bytes"],
            "selected_bytes": doc["selected_bytes"],
        }
        for group, key in ((counts["by_family"], doc["family_id"]),
                           (counts["by_language"], doc["language"])):
            target = group.setdefault(key, {field: 0 for field in v2.COUNT_FIELDS})
            for field in v2.COUNT_FIELDS:
                target[field] += entity[field]

    manifest = {
        "format": v2.FORMAT,
        "version": v2.VERSION,
        "sequence_length": spec.sequence_length,
        "vocab_size": spec.vocab_size,
        "tokenizer_sha256": _sha(tokenizer_bytes),
        "tokenizer_special_ids": json.loads(json.dumps(v2.SPECIAL_IDS)),
        "source_snapshot_sha256": v2._source_snapshot(source_records),
        "source_bytes_sha256": _sha(source_bytes_bytes),
        "sources": source_records,
        "documents": docs_list,
        "shards": shard_records,
        "counts": counts,
        "logical_sha256": _sha(b"".join(payload for _, _, payload, _ in blocks)),
        "byte_coverage_sha256": v2.byte_coverage_digest(docs_list),
    }
    manifest_bytes = v2._canonical(manifest)
    if len(manifest_bytes) > v2.MAX_MANIFEST_BYTES:
        raise ProducerError("manifest exceeds the byte cap")
    files = {
        "manifest.json": manifest_bytes,
        "tokenizer-bytes.json": tokenizer_bytes,
        "source-bytes.json": source_bytes_bytes,
        **{f"{split}.jsonl": payload for split, _, payload, _ in blocks},
    }
    return {
        "files": files,
        "summary": {
            "schema": "splot-t0030-producer-summary-v1",
            "manifest_sha256": _sha(manifest_bytes),
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "source_bytes_sha256": manifest["source_bytes_sha256"],
            "logical_sha256": manifest["logical_sha256"],
            "byte_coverage_sha256": manifest["byte_coverage_sha256"],
            "counts": counts,
            "sources": source_records,
            "documents": docs_list,
            "tokenizer_scope": v2.TOKENIZER_SCOPE,
        },
    }


def expected_coverage(spec: ProducerSpec) -> dict:
    """Expected counts/hashes computed from explicit specs, before the verifier."""
    return build_dataset(spec)["summary"]


def write_dataset(spec: ProducerSpec, output_dir) -> dict:
    """Write a new six-file dataset; existing paths are refused."""
    built = build_dataset(spec)
    out = Path(output_dir)
    if out.exists() or out.is_symlink():
        raise ProducerError("output path already exists")
    try:
        out.mkdir()
    except OSError as exc:
        raise ProducerError("output directory cannot be created") from exc
    for name in ("manifest.json", "tokenizer-bytes.json", "source-bytes.json",
                 "train.jsonl", "validation.jsonl", "test.jsonl"):
        (out / name).write_bytes(built["files"][name])
    return built["summary"]
