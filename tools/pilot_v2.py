"""Standalone byte-auditable pilot corpus v2 schema and verifier.

This is a synthetic-only prototype. It does not acquire data, fit a tokenizer,
train a model, or integrate with the training CLI. The v1 loader remains
unchanged and is not imported here.

Core invariants:
- canonical JSON manifest and JSONL blocks;
- registered source/family/language/split provenance for every segment;
- explicit token-byte table and per-token byte lengths/offsets;
- content-token target bytes at label offset 1, excluding EOS/control;
- bounded streaming shard reads with explicit v2 caps;
- test shards are integrity-validated but never returned by the public API.

The dataset root is a trusted static root for one verification call. Paths and
symlinks are checked and reads are bounded, but this is not a concurrent-mutation
or race-proof guarantee. The tokenizer table is an explicit synthetic table, not
a production tokenizer; a real byte-level BPE must separately qualify its full
256-byte alphabet, special IDs and train-only fit.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

FORMAT = "splot-pilot-blocks-v2"
VERSION = 2
TOKENIZER_FORMAT = "splot-byte-tokenizer-v2"
TOKENIZER_VERSION = 1
SOURCE_FORMAT = "splot-source-bytes-v2"
SOURCE_VERSION = 1
TOKENIZER_SCOPE = "synthetic-explicit-table"

SEQUENCE_LENGTH_MIN = 8
SEQUENCE_LENGTH_MAX = 512
VOCAB_MIN = 8
VOCAB_MAX = 16384
MAX_SEQUENCE = SEQUENCE_LENGTH_MAX
MAX_VOCAB = VOCAB_MAX

MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BLOCKS = 128
MAX_TRAIN_BLOCKS = 96
MAX_VALIDATION_BLOCKS = 16
MAX_TEST_BLOCKS = 16
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_TOKENIZER_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_LINE_BYTES = 128 * 1024
MAX_SOURCES = 32
MAX_DOCUMENTS = 512

PAD = 0
BOS = 1
EOS = 2
UNK = 3
RESERVED = (4, 5, 6, 7)
CONTENT_MIN = 8
IGNORE = -100
SPLIT_ORDER = ("train", "validation", "test")
LANGUAGES = ("en", "pl")
SPECIAL_IDS = {"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK,
               "reserved": list(RESERVED)}

COUNT_FIELDS = ("valid_tokens", "content_tokens", "ntp_labels", "content_labels",
                "eos_labels", "offset2_labels", "content_target_bytes",
                "selected_bytes")
SOURCE_KEYS = ("source_id", "family_id", "language", "split", "source_url",
               "revision", "rights_evidence_url", "raw_sha256", "content_sha256",
               "raw_bytes", "content_bytes", "selected_bytes")
DOCUMENT_KEYS = ("doc_id", "segment_id", "source_id", "family_id", "language",
                 "split", "source_byte_start", "source_byte_end",
                 "segment_byte_start", "segment_byte_end", "full_document_sha256",
                 "segment_sha256", "segment_index", "segment_count", "truncated",
                 "selected_bytes", "token_count")
SHARD_KEYS = ("split", "path", "sha256", "bytes", "blocks", "valid_tokens",
              "content_tokens", "ntp_labels", "content_labels", "eos_labels",
              "offset2_labels", "content_target_bytes", "selected_bytes",
              "family_content_tokens", "family_content_bytes",
              "language_content_tokens", "language_content_bytes")
COUNT_KEYS = ("blocks",) + COUNT_FIELDS + ("by_split", "by_language", "by_family")
MANIFEST_KEYS = ("format", "version", "sequence_length", "vocab_size",
                 "tokenizer_sha256", "tokenizer_special_ids",
                 "source_snapshot_sha256", "source_bytes_sha256", "sources",
                 "documents", "shards", "counts", "logical_sha256",
                 "byte_coverage_sha256")
TOKENIZER_KEYS = ("format", "version", "actual_vocab_size",
                  "special_token_ids", "token_bytes")
SOURCE_BYTES_KEYS = ("format", "version", "sources")
SOURCE_BLOB_KEYS = ("raw", "content")
BLOCK_KEYS = ("input_ids", "document_ids", "position_ids", "valid",
              "control_mask", "token_byte_lengths", "token_byte_offsets",
              "labels")
_HEX = re.compile(r"^[0-9a-f]+$")


class PilotV2Error(ValueError):
    """Invalid v2 manifest, tokenizer artifact, document, block or shard."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PilotV2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise PilotV2Error(f"nonfinite JSON constant: {value}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex64(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotV2Error(f"{name} must be a nonempty string")
    return value


def _require_int(value: Any, name: str, minimum: int = 0,
                 maximum: int | None = None) -> int:
    if not _is_int(value) or value < minimum:
        raise PilotV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise PilotV2Error(f"{name} exceeds {maximum}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not _is_bool(value):
        raise PilotV2Error(f"{name} must be bool")
    return value


def _require_object(value: Any, name: str, keys: tuple[str, ...]) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise PilotV2Error(f"{name} schema is invalid")
    return value


def _require_exact_list(value: Any, name: str, length: int) -> list:
    if not isinstance(value, list) or len(value) != length:
        raise PilotV2Error(f"{name} must be a list of length {length}")
    return value


def _require_regular(path: Path, label: str, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PilotV2Error(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size > max_bytes:
        raise PilotV2Error(f"{label} exceeds byte bound")
    return path.read_bytes()


def _read_canonical_json(path: Path, label: str, max_bytes: int) -> tuple[Any, bytes]:
    data = _require_regular(path, label, max_bytes)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PilotV2Error(f"{label} is not valid UTF-8") from exc
    try:
        value = json.loads(text, object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except PilotV2Error:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise PilotV2Error(f"{label} is not valid JSON: {exc}") from exc
    if _canonical(value) != data:
        raise PilotV2Error(f"{label} is not canonical JSON")
    return value, data


def _source_snapshot(sources: list[dict]) -> str:
    return _sha256(_canonical(sorted(sources, key=lambda item: item["source_id"])))


def _coverage_records(documents: list[dict]) -> list[dict]:
    keys = ("doc_id", "segment_id", "source_id", "family_id", "language", "split",
            "source_byte_start", "source_byte_end", "segment_byte_start",
            "segment_byte_end", "full_document_sha256", "segment_sha256",
            "segment_index", "segment_count", "truncated", "selected_bytes",
            "token_count")
    return [{key: item[key] for key in keys}
            for item in sorted(documents, key=lambda item: item["doc_id"])]


def byte_coverage_digest(documents: list[dict]) -> str:
    """Canonical digest of the declared byte-coverage records."""
    return _sha256(_canonical(_coverage_records(documents)))


def _validate_sources(sources: Any) -> tuple[dict, dict, dict]:
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise PilotV2Error("sources must be a nonempty bounded list")
    by_id = {}
    family_split = {}
    family_language = {}
    split_seen = set()
    for index, source in enumerate(sources):
        _require_object(source, f"sources[{index}]", SOURCE_KEYS)
        source_id = _require_str(source["source_id"], "source_id")
        family_id = _require_str(source["family_id"], "family_id")
        language = _require_str(source["language"], "language")
        split = _require_str(source["split"], "split")
        if source_id in by_id:
            raise PilotV2Error(f"duplicate source_id: {source_id}")
        if language not in LANGUAGES:
            raise PilotV2Error("source language must be en or pl")
        if split not in SPLIT_ORDER:
            raise PilotV2Error("source split is invalid")
        for key in ("source_url", "revision", "rights_evidence_url"):
            _require_str(source[key], key)
        if not _hex64(source["raw_sha256"]) or not _hex64(source["content_sha256"]):
            raise PilotV2Error("source raw/content digest is invalid")
        raw_bytes = _require_int(source["raw_bytes"], "raw_bytes")
        content_bytes = _require_int(source["content_bytes"], "content_bytes")
        selected_bytes = _require_int(source["selected_bytes"], "selected_bytes")
        if selected_bytes > content_bytes:
            raise PilotV2Error("source selected_bytes exceeds content_bytes")
        if raw_bytes <= 0 or content_bytes <= 0:
            raise PilotV2Error("source byte counts must be positive")
        if family_id in family_split:
            if family_split[family_id] != split or family_language[family_id] != language:
                raise PilotV2Error("a source family crosses split or language")
        else:
            family_split[family_id] = split
            family_language[family_id] = language
        by_id[source_id] = source
        split_seen.add(split)
    if split_seen != set(SPLIT_ORDER):
        raise PilotV2Error("all three splits need a registered source")
    return by_id, family_split, family_language


def _validate_tokenizer(tokenizer: Any, vocab_size: int,
                        manifest_special: Any) -> tuple[tuple[bytes, ...], set[int]]:
    _require_object(tokenizer, "tokenizer", TOKENIZER_KEYS)
    if (tokenizer["format"] != TOKENIZER_FORMAT
            or not _is_int(tokenizer["version"])
            or tokenizer["version"] != TOKENIZER_VERSION):
        raise PilotV2Error("tokenizer format/version is unsupported")
    actual = _require_int(tokenizer["actual_vocab_size"], "actual_vocab_size",
                          VOCAB_MIN, VOCAB_MAX)
    if actual != vocab_size:
        raise PilotV2Error("tokenizer vocabulary does not match manifest")
    special = tokenizer["special_token_ids"]
    if not isinstance(special, dict) or set(special) != set(SPECIAL_IDS):
        raise PilotV2Error("special token ID schema is invalid")
    if (not _is_int(special["pad"]) or special["pad"] != PAD
            or not _is_int(special["bos"]) or special["bos"] != BOS
            or not _is_int(special["eos"]) or special["eos"] != EOS
            or not _is_int(special["unk"]) or special["unk"] != UNK
            or not isinstance(special["reserved"], list)
            or len(special["reserved"]) != len(RESERVED)
            or any(not _is_int(value) for value in special["reserved"])
            or special["reserved"] != list(RESERVED)):
        raise PilotV2Error("special token IDs are not the fixed v2 set")
    if special != manifest_special:
        raise PilotV2Error("tokenizer special IDs do not match manifest")
    entries = _require_exact_list(tokenizer["token_bytes"], "token_bytes", vocab_size)
    result = []
    special_ids = {PAD, BOS, EOS, UNK, *RESERVED}
    content_seen: dict[bytes, int] = {}
    for token_id, entry in enumerate(entries):
        if not isinstance(entry, str):
            raise PilotV2Error(f"token_bytes[{token_id}] must be a string")
        if token_id in special_ids:
            if entry != "":
                raise PilotV2Error("special token byte entry must be empty")
            result.append(b"")
            continue
        if not entry or len(entry) % 2 or not _HEX.fullmatch(entry):
            raise PilotV2Error(f"token_bytes[{token_id}] is not lowercase hex")
        raw = bytes.fromhex(entry)
        if len(raw) > MAX_SOURCE_BYTES:
            raise PilotV2Error(f"token_bytes[{token_id}] exceeds byte bound")
        if raw in content_seen:
            raise PilotV2Error(
                f"duplicate content token bytes for IDs {content_seen[raw]} and {token_id}")
        content_seen[raw] = token_id
        result.append(raw)
    return tuple(result), special_ids


def _hex_bytes(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) % 2 or not _HEX.fullmatch(value):
        raise PilotV2Error(f"{name} must be nonempty lowercase hex bytes")
    return bytes.fromhex(value)


def _validate_source_bytes(artifact: Any, sources: dict[str, dict]) -> dict[str, tuple[bytes, bytes]]:
    _require_object(artifact, "source-bytes", SOURCE_BYTES_KEYS)
    if artifact["format"] != SOURCE_FORMAT or not _is_int(artifact["version"]) \
            or artifact["version"] != SOURCE_VERSION:
        raise PilotV2Error("source-bytes format/version is unsupported")
    blobs = artifact["sources"]
    if not isinstance(blobs, dict) or set(blobs) != set(sources):
        raise PilotV2Error("source-bytes source IDs do not match the manifest")
    result = {}
    for source_id, source in sources.items():
        blob = blobs[source_id]
        _require_object(blob, f"source-bytes.{source_id}", SOURCE_BLOB_KEYS)
        raw = _hex_bytes(blob["raw"], f"source-bytes.{source_id}.raw")
        content = _hex_bytes(blob["content"], f"source-bytes.{source_id}.content")
        if len(raw) > MAX_SOURCE_BYTES or len(content) > MAX_SOURCE_BYTES:
            raise PilotV2Error(f"source bytes exceed the bound for {source_id}")
        if len(raw) != source["raw_bytes"] or _sha256(raw) != source["raw_sha256"]:
            raise PilotV2Error(f"source raw bytes do not match {source_id}")
        if len(content) != source["content_bytes"] or _sha256(content) != source["content_sha256"]:
            raise PilotV2Error(f"source content bytes do not match {source_id}")
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PilotV2Error(f"source content is not valid UTF-8: {source_id}") from exc
        result[source_id] = (raw, content)
    return result


def _validate_documents(documents: Any, sources: dict[str, dict],
                        family_split: dict[str, str],
                        family_language: dict[str, str]) -> tuple[dict[int, dict], dict[str, int]]:
    if not isinstance(documents, list) or not 1 <= len(documents) <= MAX_DOCUMENTS:
        raise PilotV2Error("documents must be a nonempty bounded list")
    by_id = {}
    segment_ids = set()
    groups: dict[tuple, list[dict]] = {}
    selected_by_source = {source_id: 0 for source_id in sources}
    for index, doc in enumerate(documents):
        _require_object(doc, f"documents[{index}]", DOCUMENT_KEYS)
        doc_id = _require_int(doc["doc_id"], "doc_id")
        segment_id = _require_str(doc["segment_id"], "segment_id")
        source_id = _require_str(doc["source_id"], "source_id")
        family_id = _require_str(doc["family_id"], "family_id")
        language = _require_str(doc["language"], "language")
        split = _require_str(doc["split"], "split")
        if doc_id in by_id:
            raise PilotV2Error(f"duplicate doc_id: {doc_id}")
        if segment_id in segment_ids:
            raise PilotV2Error(f"duplicate segment_id: {segment_id}")
        if source_id not in sources:
            raise PilotV2Error(f"unknown source_id: {source_id}")
        source = sources[source_id]
        if (family_id != source["family_id"] or language != source["language"]
                or split != source["split"]):
            raise PilotV2Error("document provenance does not match its source")
        if family_split.get(family_id) != split or family_language.get(family_id) != language:
            raise PilotV2Error("document family crosses split or language")
        source_start = _require_int(doc["source_byte_start"], "source_byte_start")
        source_end = _require_int(doc["source_byte_end"], "source_byte_end")
        segment_start = _require_int(doc["segment_byte_start"], "segment_byte_start")
        segment_end = _require_int(doc["segment_byte_end"], "segment_byte_end")
        if not (0 <= source_start < source_end <= source["content_bytes"]):
            raise PilotV2Error("source byte interval is out of bounds")
        if not (source_start <= segment_start < segment_end <= source_end):
            raise PilotV2Error("segment byte interval is outside the source document")
        selected = _require_int(doc["selected_bytes"], "selected_bytes")
        if selected != segment_end - segment_start:
            raise PilotV2Error("document selected_bytes does not match its interval")
        token_count = _require_int(doc["token_count"], "token_count", 1)
        segment_index = _require_int(doc["segment_index"], "segment_index")
        segment_count = _require_int(doc["segment_count"], "segment_count", 1)
        if segment_index >= segment_count:
            raise PilotV2Error("segment_index is outside segment_count")
        truncated = _require_bool(doc["truncated"], "truncated")
        expected_truncated = (segment_end - segment_start) < (source_end - source_start)
        if truncated != expected_truncated:
            raise PilotV2Error("truncated flag does not match segment coverage")
        for key in ("full_document_sha256", "segment_sha256"):
            if not _hex64(doc[key]):
                raise PilotV2Error(f"{key} is invalid")
        if not truncated and (segment_start != source_start or segment_end != source_end
                              or doc["full_document_sha256"] != doc["segment_sha256"]):
            raise PilotV2Error("untruncated segment must cover the full document")
        groups.setdefault((source_id, source_start, source_end,
                           doc["full_document_sha256"]), []).append(doc)
        selected_by_source[source_id] += selected
        by_id[doc_id] = doc
        segment_ids.add(segment_id)
    full_intervals: dict[str, list[tuple[int, int]]] = {source_id: [] for source_id in sources}
    selected_intervals: dict[str, list[tuple[int, int]]] = {source_id: [] for source_id in sources}
    for (source_id, source_start, source_end, _), members in groups.items():
        counts = {item["segment_count"] for item in members}
        if len(counts) != 1:
            raise PilotV2Error("segment_count differs within a document group")
        segment_count = counts.pop()
        ordered = sorted(members, key=lambda item: (
            item["segment_byte_start"], item["segment_byte_end"], item["segment_index"]))
        indices = [item["segment_index"] for item in ordered]
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise PilotV2Error("segment_index values are not ordered by byte start")
        intervals = [(item["segment_byte_start"], item["segment_byte_end"])
                     for item in ordered]
        for (start, end), (next_start, next_end) in zip(intervals, intervals[1:]):
            if end > next_start:
                raise PilotV2Error("document segments overlap")
        full_intervals[source_id].append((source_start, source_end))
        selected_intervals[source_id].extend(intervals)
        if len(members) == segment_count:
            if indices != list(range(segment_count)):
                raise PilotV2Error("complete document indices are not 0..count-1 in byte order")
            if (intervals[0][0] != source_start or intervals[-1][1] != source_end
                    or any(end != next_start for (_, end), (next_start, _)
                           in zip(intervals, intervals[1:]))):
                raise PilotV2Error("complete document segments do not partition the source")
            if segment_count == 1:
                member = ordered[0]
                if member["segment_index"] != 0 or member["truncated"]:
                    raise PilotV2Error("single full segment requires index 0 and truncated=false")
        else:
            if any(not item["truncated"] for item in members):
                raise PilotV2Error("partial document selection must be explicitly truncated")
    for source_id in sources:
        docs = sorted(full_intervals[source_id])
        for (start, end), (next_start, next_end) in zip(docs, docs[1:]):
            if end > next_start:
                raise PilotV2Error("full-document groups overlap within a source")
        segments = sorted(selected_intervals[source_id])
        for (start, end), (next_start, next_end) in zip(segments, segments[1:]):
            if end > next_start:
                raise PilotV2Error("selected intervals overlap within a source")
    for source_id, source in sources.items():
        if selected_by_source[source_id] != source["selected_bytes"]:
            raise PilotV2Error("source selected_bytes does not match its documents")
    return by_id, selected_by_source


def _check_labels(block: dict, start: int, end: int, sequence_length: int) -> None:
    labels = block["labels"]
    if not isinstance(labels, dict) or set(labels) != {"1", "2"}:
        raise PilotV2Error("block labels schema is invalid")
    for key in ("1", "2"):
        values = _require_exact_list(labels[key], f"labels[{key}]", sequence_length)
        offset = int(key)
        for index in range(start, end):
            value = values[index]
            if not _is_int(value):
                raise PilotV2Error("labels must be integers")
            if index + offset < end and block["document_ids"][index] == block["document_ids"][index + offset]:
                expected = block["input_ids"][index + offset]
                if value != expected:
                    raise PilotV2Error("label does not match the same-document future token")
            elif value != IGNORE:
                raise PilotV2Error("label crosses a document or padding boundary")


def _blank_entity() -> dict:
    return {key: 0 for key in COUNT_FIELDS}


def _add_entity(target: dict, source: dict) -> None:
    for key in COUNT_FIELDS:
        target[key] += source[key]


def _block_counts(block: dict, docs: dict[int, dict],
                  tokenizer_bytes: tuple[bytes, ...], sequence_length: int,
                  vocab_size: int, split: str, seen_doc_ids: set[int]) -> dict:
    _require_object(block, "block", BLOCK_KEYS)
    arrays = {key: _require_exact_list(block[key], key, sequence_length)
              for key in ("input_ids", "document_ids", "position_ids", "valid",
                          "control_mask", "token_byte_lengths", "token_byte_offsets")}
    input_ids = arrays["input_ids"]
    document_ids = arrays["document_ids"]
    position_ids = arrays["position_ids"]
    valid = arrays["valid"]
    control = arrays["control_mask"]
    byte_lengths = arrays["token_byte_lengths"]
    byte_offsets = arrays["token_byte_offsets"]
    if not all(_is_bool(value) for value in valid):
        raise PilotV2Error("valid mask must contain booleans")
    if not all(_is_bool(value) for value in control):
        raise PilotV2Error("control mask must contain booleans")
    for value in input_ids + document_ids + position_ids + byte_lengths + byte_offsets:
        if not _is_int(value):
            raise PilotV2Error("block arrays must contain integers")
    valid_count = sum(valid)
    if valid_count < 3:
        raise PilotV2Error("block needs at least BOS, one content token and EOS")
    if valid != [True] * valid_count + [False] * (sequence_length - valid_count):
        raise PilotV2Error("valid mask must be a true prefix followed by padding")
    for index in range(valid_count, sequence_length):
        if (input_ids[index] != PAD or document_ids[index] != -1
                or position_ids[index] != 0 or control[index] is not True
                or byte_lengths[index] != 0 or byte_offsets[index] != -1):
            raise PilotV2Error("block padding is inconsistent")
    labels = block["labels"]
    if not isinstance(labels, dict) or set(labels) != {"1", "2"}:
        raise PilotV2Error("block labels schema is invalid")
    for key in ("1", "2"):
        _require_exact_list(labels[key], f"labels[{key}]", sequence_length)

    total = {"blocks": 1, **_blank_entity(), "by_family": {}, "by_language": {},
             "documents": []}
    start = 0
    while start < valid_count:
        end = start + 1
        while end < valid_count and document_ids[end] == document_ids[start]:
            end += 1
        doc_id = document_ids[start]
        if not _is_int(doc_id) or doc_id < 0:
            raise PilotV2Error("document ID must be a nonnegative integer")
        if doc_id in seen_doc_ids:
            raise PilotV2Error("document ID is duplicated across blocks or splits")
        if doc_id not in docs:
            raise PilotV2Error(f"block refers to unknown doc_id: {doc_id}")
        doc = docs[doc_id]
        if doc["split"] != split:
            raise PilotV2Error("block document belongs to a different split")
        seen_doc_ids.add(doc_id)
        if end - start < 3:
            raise PilotV2Error("document segment is too short")
        if input_ids[start] != BOS or input_ids[end - 1] != EOS:
            raise PilotV2Error("document segment lacks BOS/EOS boundaries")
        if control[start] is not True or control[end - 1] is not True:
            raise PilotV2Error("BOS/EOS must be marked as control")
        if position_ids[start:end] != list(range(end - start)):
            raise PilotV2Error("document positions are not contiguous")
        if byte_lengths[start] != 0 or byte_offsets[start] != 0:
            raise PilotV2Error("BOS byte coverage must be zero at offset zero")
        if doc["selected_bytes"] > MAX_SOURCE_BYTES:
            raise PilotV2Error("selected segment exceeds the byte bound")
        cumulative = 0
        content_count = 0
        for index in range(start + 1, end - 1):
            token = input_ids[index]
            if not _is_int(token) or not CONTENT_MIN <= token < vocab_size:
                raise PilotV2Error("content token ID is invalid")
            if control[index] is not False:
                raise PilotV2Error("content token must not be control")
            expected = len(tokenizer_bytes[token])
            if expected <= 0:
                raise PilotV2Error("content token has no bytes")
            if expected > doc["selected_bytes"] - cumulative:
                raise PilotV2Error("token reconstruction exceeds declared selected bytes")
            if byte_lengths[index] != expected:
                raise PilotV2Error("token byte length does not match tokenizer table")
            if byte_offsets[index] != cumulative:
                raise PilotV2Error("token byte offsets are not contiguous")
            cumulative += expected
            content_count += 1
        if content_count < 1:
            raise PilotV2Error("segment has no content tokens")
        if byte_lengths[end - 1] != 0 or byte_offsets[end - 1] != cumulative:
            raise PilotV2Error("EOS byte coverage is invalid")
        if cumulative != doc["selected_bytes"]:
            raise PilotV2Error("token byte coverage does not match selected_bytes")
        reconstructed = b"".join(tokenizer_bytes[input_ids[index]]
                                 for index in range(start + 1, end - 1))
        if _sha256(reconstructed) != doc["segment_sha256"]:
            raise PilotV2Error("segment byte hash mismatch")
        try:
            reconstructed.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PilotV2Error("token bytes do not concatenate to valid UTF-8") from exc
        if content_count != doc["token_count"]:
            raise PilotV2Error("document token_count mismatch")
        _check_labels(block, start, end, sequence_length)

        entity = _blank_entity()
        entity["valid_tokens"] = end - start
        entity["content_tokens"] = content_count
        entity["content_labels"] = content_count
        entity["ntp_labels"] = content_count + 1
        entity["eos_labels"] = 1
        entity["offset2_labels"] = content_count
        entity["content_target_bytes"] = cumulative
        entity["selected_bytes"] = cumulative
        if entity["content_target_bytes"] != entity["selected_bytes"]:
            raise PilotV2Error("content target bytes do not cover selected bytes")
        family = doc["family_id"]
        language = doc["language"]
        for container, key in ((total["by_family"], family),
                               (total["by_language"], language)):
            container.setdefault(key, _blank_entity())
            _add_entity(container[key], entity)
        for key in COUNT_FIELDS:
            total[key] += entity[key]
        total["documents"].append(doc_id)
        start = end
    for key in ("1", "2"):
        for index in range(valid_count, sequence_length):
            if not _is_int(labels[key][index]) or labels[key][index] != IGNORE:
                raise PilotV2Error("padding labels must be integer IGNORE")
    if sum(1 for value in control[:valid_count] if value is False) != total["content_tokens"]:
        raise PilotV2Error("control mask content count mismatch")
    converted = dict(block)
    converted["labels"] = {int(key): list(value) for key, value in labels.items()}
    return {"block": converted, "counts": total}


def _stream_shard(path: Path, declared_bytes: int, declared_sha: str):
    if path.is_symlink() or not path.is_file():
        raise PilotV2Error("shard must be a regular non-symlink file")
    if path.stat().st_size != declared_bytes:
        raise PilotV2Error("shard byte count mismatch before read")
    hasher = hashlib.sha256()
    total = 0
    line_no = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            line_no += 1
            if len(line) > MAX_LINE_BYTES:
                raise PilotV2Error("shard line exceeds byte bound")
            total += len(line)
            if total > declared_bytes or total > MAX_TOTAL_BYTES:
                raise PilotV2Error("shard read exceeds declared/aggregate byte bound")
            hasher.update(line)
            if not line.endswith(b"\n"):
                raise PilotV2Error("shard line lacks LF terminator")
            yield line_no, line[:-1], line
    if total != declared_bytes or hasher.hexdigest() != declared_sha:
        raise PilotV2Error("shard hash/byte count mismatch")


def _validate_counts(counts: Any, actual: dict) -> None:
    _require_object(counts, "counts", COUNT_KEYS)
    for key in ("blocks",) + COUNT_FIELDS:
        _require_int(counts[key], f"counts.{key}")
    for group in ("by_split", "by_language", "by_family"):
        value = counts[group]
        if not isinstance(value, dict):
            raise PilotV2Error(f"counts.{group} must be an object")
        for name, entity in value.items():
            _require_str(name, f"counts.{group} key")
            _require_object(entity, f"counts.{group}.{name}",
                            ("blocks",) + COUNT_FIELDS if group == "by_split"
                            else COUNT_FIELDS)
            for key, item in entity.items():
                _require_int(item, f"counts.{group}.{name}.{key}")
    if counts != actual:
        raise PilotV2Error("manifest counts do not reconcile with verified blocks")


def _summarize(counts_by_block: dict[str, list[dict]]) -> dict:
    total = {"blocks": 0, **_blank_entity(), "by_split": {},
             "by_language": {}, "by_family": {}}
    for split in SPLIT_ORDER:
        if split not in counts_by_block:
            continue
        split_entity = {"blocks": 0, **_blank_entity()}
        for item in counts_by_block[split]:
            split_entity["blocks"] += 1
            for key in COUNT_FIELDS:
                split_entity[key] += item[key]
                total[key] += item[key]
            for family, entity in item["by_family"].items():
                target = total["by_family"].setdefault(family, _blank_entity())
                _add_entity(target, entity)
            for language, entity in item["by_language"].items():
                target = total["by_language"].setdefault(language, _blank_entity())
                _add_entity(target, entity)
        total["blocks"] += split_entity["blocks"]
        total["by_split"][split] = split_entity
    return total


def load_v2_dataset(root, expected_manifest_sha256: str) -> dict:
    """Verify a bounded v2 dataset and return train/validation blocks + summary.

    Test shards are fully integrity-checked but never returned as blocks.
    """
    if not _hex64(expected_manifest_sha256):
        raise PilotV2Error("expected manifest SHA-256 is invalid")
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise PilotV2Error("dataset root must be a regular directory")
    expected_files = {"manifest.json", "tokenizer-bytes.json", "source-bytes.json",
                      "train.jsonl", "validation.jsonl", "test.jsonl"}
    for entry in root.iterdir():
        if entry.name not in expected_files or entry.is_symlink() or not entry.is_file():
            raise PilotV2Error("dataset contains an unexpected or nonregular file")
    if {entry.name for entry in root.iterdir()} != expected_files:
        raise PilotV2Error("dataset file set is incomplete")

    manifest, manifest_bytes = _read_canonical_json(root / "manifest.json",
                                                    "manifest.json",
                                                    MAX_MANIFEST_BYTES)
    if _sha256(manifest_bytes) != expected_manifest_sha256:
        raise PilotV2Error("manifest SHA-256 mismatch")
    _require_object(manifest, "manifest", MANIFEST_KEYS)
    if (manifest["format"] != FORMAT or not _is_int(manifest["version"])
            or manifest["version"] != VERSION):
        raise PilotV2Error("v2 format/version is unsupported")
    sequence_length = _require_int(manifest["sequence_length"], "sequence_length",
                                   SEQUENCE_LENGTH_MIN, SEQUENCE_LENGTH_MAX)
    vocab_size = _require_int(manifest["vocab_size"], "vocab_size",
                              VOCAB_MIN, VOCAB_MAX)
    if not _hex64(manifest["tokenizer_sha256"]):
        raise PilotV2Error("tokenizer_sha256 is invalid")
    if not _hex64(manifest["source_bytes_sha256"]):
        raise PilotV2Error("source_bytes_sha256 is invalid")
    if manifest["tokenizer_special_ids"] != SPECIAL_IDS:
        raise PilotV2Error("manifest special token IDs are invalid")
    tokenizer, tokenizer_bytes = _read_canonical_json(root / "tokenizer-bytes.json",
                                                      "tokenizer-bytes.json",
                                                      MAX_TOKENIZER_BYTES)
    if _sha256(tokenizer_bytes) != manifest["tokenizer_sha256"]:
        raise PilotV2Error("tokenizer artifact SHA-256 mismatch")
    tokenizer_table, _ = _validate_tokenizer(tokenizer, vocab_size,
                                             manifest["tokenizer_special_ids"])
    sources, family_split, family_language = _validate_sources(manifest["sources"])
    if _source_snapshot(manifest["sources"]) != manifest["source_snapshot_sha256"]:
        raise PilotV2Error("source snapshot SHA-256 mismatch")
    source_artifact, source_artifact_bytes = _read_canonical_json(
        root / "source-bytes.json", "source-bytes.json", MAX_SOURCE_BYTES)
    if _sha256(source_artifact_bytes) != manifest["source_bytes_sha256"]:
        raise PilotV2Error("source-bytes artifact SHA-256 mismatch")
    source_blobs = _validate_source_bytes(source_artifact, sources)
    documents, _ = _validate_documents(manifest["documents"], sources,
                                       family_split, family_language)
    for doc in documents.values():
        content = source_blobs[doc["source_id"]][1]
        full = content[doc["source_byte_start"]:doc["source_byte_end"]]
        segment = content[doc["segment_byte_start"]:doc["segment_byte_end"]]
        if _sha256(full) != doc["full_document_sha256"]:
            raise PilotV2Error("full document byte hash mismatch")
        if _sha256(segment) != doc["segment_sha256"]:
            raise PilotV2Error("segment byte hash mismatch against source bytes")
        try:
            full.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PilotV2Error("full document is not valid UTF-8") from exc
        try:
            segment.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PilotV2Error("source segment is not valid UTF-8") from exc
    if byte_coverage_digest(manifest["documents"]) != manifest["byte_coverage_sha256"]:
        raise PilotV2Error("byte coverage SHA-256 mismatch")

    shards = manifest["shards"]
    if not isinstance(shards, list) or len(shards) != len(SPLIT_ORDER):
        raise PilotV2Error("shards must be exactly train, validation, test in order")
    if any(not isinstance(shard, dict) for shard in shards):
        raise PilotV2Error("shard entries must be objects")
    if [shard.get("split") for shard in shards] != list(SPLIT_ORDER):
        raise PilotV2Error("shards must be exactly train, validation, test in order")
    total_bytes = 0
    total_blocks = 0
    per_split_limits = {"train": MAX_TRAIN_BLOCKS,
                        "validation": MAX_VALIDATION_BLOCKS,
                        "test": MAX_TEST_BLOCKS}
    for index, shard in enumerate(shards):
        _require_object(shard, f"shards[{index}]", SHARD_KEYS)
        split = shard["split"]
        if shard["path"] != f"{split}.jsonl":
            raise PilotV2Error("shard path is not the fixed split path")
        for key in ("bytes", "blocks") + COUNT_FIELDS:
            _require_int(shard[key], f"shards[{index}].{key}")
        if not _hex64(shard["sha256"]):
            raise PilotV2Error("shard SHA-256 is invalid")
        if shard["blocks"] < 1 or shard["blocks"] > per_split_limits[split]:
            raise PilotV2Error("shard block count is out of bounds")
        for key in ("family_content_tokens", "family_content_bytes",
                    "language_content_tokens", "language_content_bytes"):
            mapping = shard[key]
            if not isinstance(mapping, dict):
                raise PilotV2Error(f"shard {key} must be an object")
            for name, value in mapping.items():
                _require_str(name, f"shard {key} key")
                _require_int(value, f"shard {key}.{name}")
        if shard["content_target_bytes"] != shard["selected_bytes"]:
            raise PilotV2Error("shard content target bytes do not cover selected bytes")
        total_bytes += shard["bytes"]
        total_blocks += shard["blocks"]
    if total_bytes > MAX_TOTAL_BYTES or total_blocks > MAX_TOTAL_BLOCKS:
        raise PilotV2Error("dataset exceeds aggregate v2 caps")
    if (not isinstance(manifest["counts"], dict)
            or not _is_int(manifest["counts"].get("blocks"))):
        raise PilotV2Error("manifest counts schema is invalid")
    if total_blocks != manifest["counts"]["blocks"]:
        raise PilotV2Error("manifest block count does not match shards")

    blocks_by_split = {}
    counts_by_block = {}
    seen_doc_ids: set[int] = set()
    logical = hashlib.sha256()
    for shard in shards:
        split = shard["split"]
        blocks = []
        counts = []
        line_count = 0
        for line_no, payload, raw in _stream_shard(root / shard["path"],
                                                   shard["bytes"], shard["sha256"]):
            logical.update(raw)
            line_count += 1
            try:
                value = json.loads(payload.decode("utf-8"),
                                   object_pairs_hook=_unique_pairs,
                                   parse_constant=_reject_constant)
            except PilotV2Error:
                raise
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise PilotV2Error(f"invalid JSON block at line {line_no}") from exc
            if _canonical(value) != payload:
                raise PilotV2Error(f"block at line {line_no} is not canonical JSON")
            checked = _block_counts(value, documents, tokenizer_table,
                                    sequence_length, vocab_size, split,
                                    seen_doc_ids)
            counts.append(checked["counts"])
            if split != "test":
                blocks.append(checked["block"])
        if line_count != shard["blocks"]:
            raise PilotV2Error("shard block count does not match manifest")
        shard_counts = _summarize({split: counts})["by_split"][split]
        for key in ("blocks",) + COUNT_FIELDS:
            if shard_counts[key] != shard[key]:
                raise PilotV2Error(f"shard {split} {key} mismatch")
        family_tokens = {}
        family_bytes = {}
        language_tokens = {}
        language_bytes = {}
        for item in counts:
            for family, entity in item["by_family"].items():
                family_tokens[family] = family_tokens.get(family, 0) + entity["content_tokens"]
                family_bytes[family] = family_bytes.get(family, 0) + entity["content_target_bytes"]
            for language, entity in item["by_language"].items():
                language_tokens[language] = language_tokens.get(language, 0) + entity["content_tokens"]
                language_bytes[language] = language_bytes.get(language, 0) + entity["content_target_bytes"]
        if family_tokens != shard["family_content_tokens"]:
            raise PilotV2Error("shard family content-token counts mismatch")
        if family_bytes != shard["family_content_bytes"]:
            raise PilotV2Error("shard family content-byte counts mismatch")
        if language_tokens != shard["language_content_tokens"]:
            raise PilotV2Error("shard language content-token counts mismatch")
        if language_bytes != shard["language_content_bytes"]:
            raise PilotV2Error("shard language content-byte counts mismatch")
        blocks_by_split[split] = blocks
        counts_by_block[split] = counts
    if logical.hexdigest() != manifest["logical_sha256"]:
        raise PilotV2Error("logical SHA-256 mismatch")
    if seen_doc_ids != set(documents):
        raise PilotV2Error("orphan or missing document IDs after shard scan")
    document_selected = sum(doc["selected_bytes"] for doc in documents.values())
    shard_selected = sum(shard["selected_bytes"] for shard in shards)
    source_selected = sum(source["selected_bytes"] for source in sources.values())
    if not (document_selected == shard_selected == source_selected):
        raise PilotV2Error("selected-byte reconciliation failed")
    if sum(shard["content_target_bytes"] for shard in shards) != document_selected:
        raise PilotV2Error("content-target byte reconciliation failed")
    actual_counts = _summarize(counts_by_block)
    _validate_counts(manifest["counts"], actual_counts)

    summary = {
        "format": FORMAT, "version": VERSION,
        "manifest_sha256": expected_manifest_sha256,
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "source_bytes_sha256": manifest["source_bytes_sha256"],
        "logical_sha256": manifest["logical_sha256"],
        "byte_coverage_sha256": manifest["byte_coverage_sha256"],
        "sequence_length": sequence_length, "vocab_size": vocab_size,
        "splits": actual_counts["by_split"],
        "counts": actual_counts,
        "sources": manifest["sources"],
        "documents": {doc_id: doc for doc_id, doc in documents.items()
                      if doc["split"] != "test"},
        "test_validated_only": True,
        "test_blocks_exposed": False,
    }
    return {"manifest": summary, "blocks": {"train": blocks_by_split["train"],
                                            "validation": blocks_by_split["validation"]}}


def scoring_targets(dataset: dict, split: str = "validation") -> list[dict]:
    """Yield offset-1 target records with exact byte counts; test is refused."""
    if split not in ("train", "validation"):
        raise PilotV2Error("test targets are not exposed by this API")
    blocks = dataset["blocks"][split]
    docs = dataset["manifest"]["documents"]
    targets = []
    seen = set()
    for block_index, block in enumerate(blocks):
        for position, label in enumerate(block["labels"][1]):
            if not _is_int(label):
                raise PilotV2Error("target label is not an integer")
            if label == IGNORE:
                continue
            key = (block_index, position)
            if key in seen:
                raise PilotV2Error("duplicate target position")
            seen.add(key)
            target_index = position + 1
            if target_index >= len(block["input_ids"]):
                raise PilotV2Error("target position is outside the block")
            if block["document_ids"][target_index] != block["document_ids"][position]:
                raise PilotV2Error("target crosses a document boundary")
            target_id = block["input_ids"][target_index]
            if block["control_mask"][target_index] is False:
                kind = "content"
            elif target_id == EOS:
                kind = "eos"
            else:
                kind = "control"
            doc = docs[block["document_ids"][position]]
            targets.append({
                "target_token_id": target_id,
                "byte_length": block["token_byte_lengths"][target_index],
                "kind": kind,
                "block_index": block_index,
                "position": position,
                "family_id": doc["family_id"],
                "language": doc["language"],
                "split": split,
            })
    return targets


def _nll_value(value: Any, name: str) -> float:
    if type(value) not in (int, float) or _is_bool(value):
        raise PilotV2Error(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise PilotV2Error(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise PilotV2Error(f"{name} must be a finite nonnegative number")
    return result


def _bpb_totals(targets: list[dict]) -> dict:
    """Arithmetic totals only; target identity is not verified here."""
    if not isinstance(targets, list) or not targets:
        raise PilotV2Error("targets must be a nonempty list")
    content_labels = 0
    content_bytes = 0
    content_nll = 0.0
    eos_labels = 0
    eos_nll = 0.0
    control_labels = 0
    control_nll = 0.0
    ordinary_nll = 0.0
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise PilotV2Error(f"targets[{index}] must be an object")
        try:
            nll = _nll_value(target["nll"], f"targets[{index}].nll")
            byte_length = target["byte_length"]
            kind = target["kind"]
        except KeyError as exc:
            raise PilotV2Error(f"targets[{index}] missing {exc.args[0]}") from exc
        if not _is_int(byte_length) or byte_length < 0:
            raise PilotV2Error(f"targets[{index}].byte_length must be a nonnegative integer")
        if kind not in ("content", "eos", "control"):
            raise PilotV2Error(f"targets[{index}].kind is invalid")
        if kind == "content":
            if byte_length <= 0:
                raise PilotV2Error("content target must have positive byte length")
            content_labels += 1
            content_bytes += byte_length
            content_nll += nll
        elif kind == "eos":
            if byte_length != 0:
                raise PilotV2Error("EOS target must not claim bytes")
            eos_labels += 1
            eos_nll += nll
        else:
            if byte_length != 0:
                raise PilotV2Error("control target must not claim bytes")
            control_labels += 1
            control_nll += nll
        ordinary_nll += nll
    for name, value in (("content_nll", content_nll), ("eos_nll", eos_nll),
                        ("control_nll", control_nll), ("ordinary_nll", ordinary_nll)):
        if not math.isfinite(value):
            raise PilotV2Error(f"{name} aggregate is not finite")
    return {
        "content_labels": content_labels,
        "content_nll_sum": content_nll,
        "content_bytes": content_bytes,
        "eos_labels": eos_labels,
        "eos_nll_sum": eos_nll,
        "control_labels": control_labels,
        "control_nll_sum": control_nll,
        "ordinary_ntp_labels": content_labels + eos_labels + control_labels,
        "ordinary_ntp_nll_sum": ordinary_nll,
    }


def aggregate_content_bpb(targets: list[dict]) -> dict:
    """Synthetic arithmetic-only BPB helper; it does not verify target identity."""
    totals = _bpb_totals(targets)
    if totals["content_labels"] == 0 or totals["content_bytes"] == 0:
        raise PilotV2Error("zero content labels or zero-byte denominator")
    totals["content_bpb"] = totals["content_nll_sum"] / (
        math.log(2) * totals["content_bytes"])
    if not math.isfinite(totals["content_bpb"]):
        raise PilotV2Error("content BPB is not finite")
    return totals


def verify_targets(dataset: dict, split: str, targets: list[dict]) -> list[dict]:
    """Verify forged/misordered/duplicate targets against validated blocks."""
    if split not in ("train", "validation"):
        raise PilotV2Error("test targets are not exposed by this API")
    expected = scoring_targets(dataset, split)
    if not isinstance(targets, list) or len(targets) != len(expected):
        raise PilotV2Error("target list length does not match the verified dataset")
    seen = set()
    fields = ("target_token_id", "byte_length", "kind", "block_index", "position",
              "family_id", "language", "split")
    for index, (got, want) in enumerate(zip(targets, expected)):
        if not isinstance(got, dict):
            raise PilotV2Error(f"targets[{index}] must be an object")
        key = (got.get("block_index"), got.get("position"))
        if not (_is_int(key[0]) and _is_int(key[1])):
            raise PilotV2Error(f"targets[{index}] has an invalid identity")
        if key in seen:
            raise PilotV2Error(f"duplicate target identity: {key}")
        seen.add(key)
        for field in fields:
            if got.get(field) != want[field]:
                raise PilotV2Error(f"targets[{index}].{field} does not match the dataset")
        block = dataset["blocks"][split][want["block_index"]]
        position = want["position"]
        if not 0 <= position < len(block["labels"][1]) - 1:
            raise PilotV2Error("target position is outside the block")
        if block["labels"][1][position] == IGNORE:
            raise PilotV2Error("target label is IGNORE")
        target_index = position + 1
        target_id = block["input_ids"][target_index]
        kind = ("content" if block["control_mask"][target_index] is False
                else "eos" if target_id == EOS else "control")
        if (got["target_token_id"] != target_id
                or got["byte_length"] != block["token_byte_lengths"][target_index]
                or got["kind"] != kind):
            raise PilotV2Error("target identity does not match the block label")
    return expected


def aggregate_verified_bpb(dataset: dict, split: str, nlls: list,
                           targets: list[dict] | None = None) -> dict:
    """Dataset-bound aggregation with optional forged-target verification."""
    if split not in ("train", "validation"):
        raise PilotV2Error("test targets are not exposed by this API")
    expected = scoring_targets(dataset, split)
    if targets is not None:
        expected = verify_targets(dataset, split, targets)
    if not isinstance(nlls, list) or len(nlls) != len(expected):
        raise PilotV2Error("NLL list length does not match the verified targets")
    enriched = []
    for target, nll in zip(expected, nlls):
        item = dict(target)
        item["nll"] = nll
        enriched.append(item)
    totals = _bpb_totals(enriched)
    if totals["content_labels"] == 0 or totals["content_bytes"] == 0:
        raise PilotV2Error("zero content labels or zero-byte denominator")
    totals["content_bpb"] = totals["content_nll_sum"] / (
        math.log(2) * totals["content_bytes"])
    if not math.isfinite(totals["content_bpb"]):
        raise PilotV2Error("content BPB is not finite")
    for group_key in ("family_id", "language"):
        grouped = {}
        for item in enriched:
            grouped.setdefault(item[group_key], []).append(item)
        totals["by_" + group_key.replace("_id", "")] = {
            name: _group_bpb(items) for name, items in sorted(grouped.items())}
    return totals


def _group_bpb(items: list[dict]) -> dict:
    totals = _bpb_totals(items)
    if totals["content_labels"] and totals["content_bytes"]:
        totals["content_bpb"] = totals["content_nll_sum"] / (
            math.log(2) * totals["content_bytes"])
        if not math.isfinite(totals["content_bpb"]):
            raise PilotV2Error("content BPB is not finite")
    else:
        totals["content_bpb"] = None
    return totals
