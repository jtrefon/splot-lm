"""Bounded streaming pilot corpus format and verifier (splot-pilot-stream-v1).

New production-oriented prototype; v1 and v2 remain immutable. Source raw/content
bytes live in separate private files, metadata is JSONL, and block shards are
validated with a bounded streaming reader. This is not accepted training input.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pilot_v2 as v2

FORMAT = "splot-pilot-stream-v1"
VERSION = 1
TOKENIZER_FORMAT = "splot-byte-tokenizer-stream-v1"
TOKENIZER_VERSION = 1
SOURCE_FORMAT = "splot-source-registry-stream-v1"
SOURCE_VERSION = 1

SEQUENCE_MIN = 8
SEQUENCE_MAX = 512
VOCAB_MIN = 264  # 8 fixed specials + complete 256-byte alphabet
VOCAB_MAX = 16384
MIN_ALPHABET = 256

MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_TOTAL_POSITIONS = 1_600_000
MAX_TRAIN_VALID = 1_000_000
MAX_VALIDATION_VALID = 50_000
MAX_TEST_VALID = 50_000
MAX_TOTAL_VALID = 1_100_000
MAX_SOURCES = 32
MAX_DOCUMENTS = 20_000
MAX_METADATA_LINE = 128 * 1024
MAX_SHARD_LINE = 256 * 1024
MAX_SHARD_BYTES = 32 * 1024 * 1024
MAX_TOKENIZER_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024

PAD, BOS, EOS, UNK = 0, 1, 2, 3
CONTENT_MIN = 8
RESERVED = (4, 5, 6, 7)
SPECIAL_IDS = {"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK,
               "reserved": list(RESERVED)}
IGNORE = -100
SPLIT_ORDER = ("train", "validation", "test")
LANGUAGES = ("en", "pl")
COUNT_FIELDS = ("valid_tokens", "content_tokens", "ntp_labels", "content_labels",
                "eos_labels", "offset2_labels", "content_target_bytes",
                "selected_bytes")
SOURCE_KEYS = ("source_id", "family_id", "language", "split", "source_url",
               "revision", "rights_evidence_url", "raw_path", "raw_sha256",
               "raw_bytes", "content_path", "content_sha256", "content_bytes",
               "selected_bytes")
DOCUMENT_KEYS = ("doc_id", "segment_id", "source_id", "family_id", "language",
                 "split", "source_byte_start", "source_byte_end",
                 "segment_byte_start", "segment_byte_end",
                 "full_document_sha256", "segment_sha256", "segment_index",
                 "segment_count", "truncated", "selected_bytes", "token_count")
SHARD_KEYS = ("split", "path", "sha256", "bytes", "blocks", "valid_tokens",
              "content_tokens", "ntp_labels", "content_labels", "eos_labels",
              "offset2_labels", "content_target_bytes", "selected_bytes",
              "family_content_tokens", "family_content_bytes",
              "language_content_tokens", "language_content_bytes")
TOKENIZER_KEYS = ("format", "version", "actual_vocab_size",
                  "special_token_ids", "token_bytes")
SOURCES_KEYS = ("format", "version", "sources")
COUNT_KEYS = ("blocks",) + COUNT_FIELDS + ("by_split", "by_language", "by_family")
MANIFEST_KEYS = ("format", "version", "sequence_length", "vocab_size",
                 "tokenizer_sha256", "tokenizer_special_ids",
                 "source_registry_sha256", "document_metadata_sha256",
                 "shards", "counts", "logical_sha256")
_HEX = re.compile(r"^[0-9a-f]+$")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StreamError(ValueError):
    """Invalid stream manifest, registry, document, block or shard."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StreamError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise StreamError(f"nonfinite JSON constant: {value}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex64(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _int(value: Any, name: str, minimum=0, maximum=None) -> int:
    if not _is_int(value) or value < minimum:
        raise StreamError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise StreamError(f"{name} exceeds {maximum}")
    return value


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StreamError(f"{name} must be nonempty")
    return value


def _obj(value: Any, name: str, keys) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise StreamError(f"{name} schema is invalid")
    return value


def _regular(path: Path, label: str, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise StreamError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size > max_bytes:
        raise StreamError(f"{label} exceeds byte bound")
    return path.read_bytes()


def _read_json(path: Path, label: str, max_bytes: int) -> tuple[Any, bytes]:
    data = _regular(path, label, max_bytes)
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except StreamError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StreamError(f"{label} is invalid JSON") from exc
    if _canonical(value) != data:
        raise StreamError(f"{label} is not canonical JSON")
    return value, data


def _tokenizer_data(tokenizer: Any, vocab_size: int) -> tuple[bytes, ...]:
    _obj(tokenizer, "tokenizer", TOKENIZER_KEYS)
    if tokenizer["format"] != TOKENIZER_FORMAT or not _is_int(tokenizer["version"]) \
            or tokenizer["version"] != TOKENIZER_VERSION:
        raise StreamError("tokenizer format/version is unsupported")
    actual = _int(tokenizer["actual_vocab_size"], "actual_vocab_size",
                  VOCAB_MIN, VOCAB_MAX)
    if actual != vocab_size:
        raise StreamError("tokenizer vocabulary does not match manifest")
    special = tokenizer["special_token_ids"]
    if (not isinstance(special, dict) or set(special) != set(SPECIAL_IDS)
            or not _is_int(special["pad"]) or special["pad"] != PAD
            or not _is_int(special["bos"]) or special["bos"] != BOS
            or not _is_int(special["eos"]) or special["eos"] != EOS
            or not _is_int(special["unk"]) or special["unk"] != UNK
            or not isinstance(special["reserved"], list)
            or len(special["reserved"]) != len(RESERVED)
            or any(not _is_int(value) for value in special["reserved"])
            or special["reserved"] != list(RESERVED)):
        raise StreamError("tokenizer special IDs are invalid")
    entries = tokenizer["token_bytes"]
    if not isinstance(entries, list) or len(entries) != vocab_size:
        raise StreamError("token_bytes length is invalid")
    table = []
    single = set()
    seen = {}
    specials = {PAD, BOS, EOS, UNK, *RESERVED}
    for token_id, entry in enumerate(entries):
        if not isinstance(entry, str):
            raise StreamError("token byte entry must be a string")
        if token_id in specials:
            if entry != "":
                raise StreamError("special token bytes must be empty")
            table.append(b"")
            continue
        if not entry or len(entry) % 2 or not _HEX.fullmatch(entry):
            raise StreamError("content token bytes must be lowercase hex")
        raw = bytes.fromhex(entry)
        if not raw or len(raw) > MAX_SOURCE_BYTES:
            raise StreamError("content token bytes are out of bounds")
        if raw in seen:
            raise StreamError("duplicate content token bytes")
        seen[raw] = token_id
        if len(raw) == 1:
            single.add(raw[0])
        table.append(raw)
    if single != set(range(256)):
        raise StreamError("tokenizer must contain the complete 256-byte alphabet")
    return tuple(table)


def _source_data(root: Path, registry: Any) -> tuple[dict, dict, dict, int]:
    _obj(registry, "sources", SOURCES_KEYS)
    if registry["format"] != SOURCE_FORMAT or not _is_int(registry["version"]) \
            or registry["version"] != SOURCE_VERSION:
        raise StreamError("source registry format/version is unsupported")
    sources = registry["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise StreamError("source registry count is out of bounds")
    by_id = {}
    content_map = {}
    family_split = {}
    family_language = {}
    split_seen = set()
    total_source_bytes = 0
    for index, source in enumerate(sources):
        _obj(source, f"sources[{index}]", SOURCE_KEYS)
        source_id = _str(source["source_id"], "source_id")
        family_id = _str(source["family_id"], "family_id")
        if not _SAFE.fullmatch(source_id):
            raise StreamError("source_id is unsafe")
        if source["language"] not in LANGUAGES or source["split"] not in SPLIT_ORDER:
            raise StreamError("source language/split is invalid")
        for key in ("source_url", "revision", "rights_evidence_url"):
            _str(source[key], key)
        if source["raw_path"] != f"sources/{source_id}.raw":
            raise StreamError("raw path is not the fixed source path")
        if source["content_path"] != f"sources/{source_id}.content":
            raise StreamError("content path is not the fixed source path")
        if not _hex64(source["raw_sha256"]) or not _hex64(source["content_sha256"]):
            raise StreamError("source hashes are invalid")
        raw = _regular(root / source["raw_path"], "source raw", MAX_SOURCE_BYTES)
        content = _regular(root / source["content_path"], "source content",
                           MAX_SOURCE_BYTES)
        if len(raw) != source["raw_bytes"] or _sha(raw) != source["raw_sha256"]:
            raise StreamError("source raw bytes do not match")
        if len(content) != source["content_bytes"] or _sha(content) != source["content_sha256"]:
            raise StreamError("source content bytes do not match")
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise StreamError("source content is not valid UTF-8") from exc
        selected = _int(source["selected_bytes"], "selected_bytes")
        if selected > len(content):
            raise StreamError("source selected bytes exceed content")
        if source_id in by_id:
            raise StreamError("duplicate source_id")
        if family_id in family_split:
            if (family_split[family_id] != source["split"]
                    or family_language[family_id] != source["language"]):
                raise StreamError("a family crosses split or language")
        else:
            family_split[family_id] = source["split"]
            family_language[family_id] = source["language"]
        by_id[source_id] = source
        content_map[source_id] = content
        split_seen.add(source["split"])
        total_source_bytes += len(raw) + len(content)
    if split_seen != set(SPLIT_ORDER):
        raise StreamError("all three splits need a source")
    if total_source_bytes > MAX_TOTAL_BYTES:
        raise StreamError("source bytes exceed the aggregate cap")
    return by_id, content_map, family_split, total_source_bytes


def _document_data(root: Path, sources: dict, content_map: dict,
                   family_split: dict,
                   family_language: dict) -> tuple[dict[int, dict], dict[str, int], bytes]:
    path = root / "documents.jsonl"
    if path.is_symlink() or not path.is_file():
        raise StreamError("documents.jsonl must be a regular file")
    by_id = {}
    segment_ids = set()
    groups = {}
    selected_by_source = {source_id: 0 for source_id in sources}
    full_intervals = {source_id: [] for source_id in sources}
    selected_intervals = {source_id: [] for source_id in sources}
    hasher = hashlib.sha256()
    total = 0
    line_no = 0
    last_doc = -1
    with path.open("rb") as handle:
        while True:
            line = handle.readline(MAX_METADATA_LINE + 1)
            if not line:
                break
            line_no += 1
            total += len(line)
            if len(line) > MAX_METADATA_LINE or not line.endswith(b"\n"):
                raise StreamError("document metadata line is invalid")
            hasher.update(line)
            payload = line[:-1]
            try:
                doc = json.loads(payload.decode("utf-8"),
                                 object_pairs_hook=_unique_pairs,
                                 parse_constant=_reject_constant)
            except StreamError:
                raise
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise StreamError("document metadata JSON is invalid") from exc
            if _canonical(doc) != payload:
                raise StreamError("document metadata is not canonical")
            _obj(doc, "document", DOCUMENT_KEYS)
            doc_id = _int(doc["doc_id"], "doc_id")
            if doc_id <= last_doc or doc_id in by_id:
                raise StreamError("document IDs must be strictly ascending and unique")
            last_doc = doc_id
            segment_id = _str(doc["segment_id"], "segment_id")
            if segment_id in segment_ids:
                raise StreamError("duplicate segment_id")
            segment_ids.add(segment_id)
            source_id = _str(doc["source_id"], "source_id")
            source = sources.get(source_id)
            if source is None:
                raise StreamError("unknown source_id")
            if (doc["family_id"] != source["family_id"]
                    or doc["language"] != source["language"]
                    or doc["split"] != source["split"]):
                raise StreamError("document provenance does not match source")
            if family_split[doc["family_id"]] != doc["split"] \
                    or family_language[doc["family_id"]] != doc["language"]:
                raise StreamError("document family crosses split or language")
            source_start = _int(doc["source_byte_start"], "source_byte_start")
            source_end = _int(doc["source_byte_end"], "source_byte_end")
            seg_start = _int(doc["segment_byte_start"], "segment_byte_start")
            seg_end = _int(doc["segment_byte_end"], "segment_byte_end")
            if not 0 <= source_start < source_end <= source["content_bytes"]:
                raise StreamError("source interval is out of bounds")
            if not source_start <= seg_start < seg_end <= source_end:
                raise StreamError("segment interval is outside the document")
            content = content_map[source_id]
            full = content[source_start:source_end]
            segment = content[seg_start:seg_end]
            if _sha(full) != doc["full_document_sha256"]:
                raise StreamError("full document hash mismatch")
            if _sha(segment) != doc["segment_sha256"]:
                raise StreamError("segment hash mismatch")
            try:
                full.decode("utf-8", errors="strict")
                segment.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise StreamError("document bytes are not valid UTF-8") from exc
            selected = _int(doc["selected_bytes"], "selected_bytes")
            if selected != seg_end - seg_start:
                raise StreamError("document selected_bytes mismatch")
            token_count = _int(doc["token_count"], "token_count", 1)
            if len(by_id) >= MAX_DOCUMENTS:
                raise StreamError("document count exceeds cap")
            if total > MAX_TOTAL_BYTES:
                raise StreamError("document metadata exceeds aggregate cap")
            segment_index = _int(doc["segment_index"], "segment_index")
            segment_count = _int(doc["segment_count"], "segment_count", 1)
            if segment_index >= segment_count:
                raise StreamError("segment_index is outside segment_count")
            truncated = doc["truncated"]
            if not _is_bool(truncated):
                raise StreamError("truncated must be bool")
            if truncated != ((seg_end - seg_start) < (source_end - source_start)):
                raise StreamError("truncated flag mismatch")
            for key in ("full_document_sha256", "segment_sha256"):
                if not _hex64(doc[key]):
                    raise StreamError(f"{key} is invalid")
            if not truncated and (seg_start != source_start or seg_end != source_end
                                  or doc["full_document_sha256"] != doc["segment_sha256"]):
                raise StreamError("untruncated segment must cover the full document")
            groups.setdefault((source_id, source_start, source_end,
                               doc["full_document_sha256"]), []).append(doc)
            selected_intervals[source_id].append((seg_start, seg_end))
            selected_by_source[source_id] += selected
            by_id[doc_id] = doc
    if line_no == 0:
        raise StreamError("documents.jsonl is empty")
    if len(by_id) > MAX_DOCUMENTS:
        raise StreamError("document count exceeds the cap")
    for (source_id, source_start, source_end, _), members in groups.items():
        full_intervals[source_id].append((source_start, source_end))
        counts = {item["segment_count"] for item in members}
        if len(counts) != 1:
            raise StreamError("segment_count differs within a group")
        segment_count = counts.pop()
        ordered = sorted(members, key=lambda item: (
            item["segment_byte_start"], item["segment_byte_end"], item["segment_index"]))
        indices = [item["segment_index"] for item in ordered]
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise StreamError("segment indices are not ordered by byte start")
        intervals = [(item["segment_byte_start"], item["segment_byte_end"])
                     for item in ordered]
        for (start, end), (next_start, next_end) in zip(intervals, intervals[1:]):
            if end > next_start:
                raise StreamError("document segments overlap")
        if len(members) == segment_count:
            if indices != list(range(segment_count)):
                raise StreamError("complete document indices are not ordered")
            if (intervals[0][0] != source_start or intervals[-1][1] != source_end
                    or any(end != next_start for (_, end), (next_start, _)
                           in zip(intervals, intervals[1:]))):
                raise StreamError("complete document does not partition source")
            if segment_count == 1 and (ordered[0]["segment_index"] != 0
                                      or ordered[0]["truncated"]):
                raise StreamError("single full segment must be index 0 and untruncated")
        elif any(not item["truncated"] for item in members):
            raise StreamError("partial document selection must be truncated")
    for source_id in sources:
        for intervals, label in ((full_intervals[source_id], "full"),
                                 (selected_intervals[source_id], "selected")):
            ordered = sorted(intervals)
            for (start, end), (next_start, next_end) in zip(ordered, ordered[1:]):
                if end > next_start:
                    raise StreamError(f"{label} intervals overlap")
        if selected_by_source[source_id] != sources[source_id]["selected_bytes"]:
            raise StreamError("source selected_bytes mismatch")
    return by_id, selected_by_source, hasher.hexdigest()


def _stream_shard(path: Path, declared_bytes: int, declared_sha: str):
    if path.is_symlink() or not path.is_file():
        raise StreamError("shard must be a regular non-symlink file")
    if path.stat().st_size != declared_bytes:
        raise StreamError("shard byte count mismatch")
    hasher = hashlib.sha256()
    total = 0
    line_no = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline(MAX_SHARD_LINE + 1)
            if not line:
                break
            line_no += 1
            if len(line) > MAX_SHARD_LINE:
                raise StreamError("shard line exceeds cap")
            total += len(line)
            if total > declared_bytes or total > MAX_SHARD_BYTES:
                raise StreamError("shard read exceeds cap")
            hasher.update(line)
            if not line.endswith(b"\n"):
                raise StreamError("shard line lacks LF")
            yield line_no, line[:-1], line
    if total != declared_bytes or hasher.hexdigest() != declared_sha:
        raise StreamError("shard hash/byte count mismatch")


def _validate_block(block: dict, docs: dict[int, dict], tokenizer: tuple[bytes, ...],
                    sequence_length: int, vocab_size: int, split: str,
                    seen: set[int]) -> dict:
    try:
        return v2._block_counts(block, docs, tokenizer, sequence_length,
                                vocab_size, split, seen)["counts"]
    except v2.PilotV2Error as exc:
        raise StreamError(str(exc)) from exc


def _check_file_set(root: Path, manifest: dict, sources: dict) -> None:
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise StreamError("shard list is invalid")
    shard_paths = set()
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise StreamError("shard entry is invalid")
        path = shard["path"]
        if "/" in path or not _SAFE.fullmatch(path):
            raise StreamError("shard path is unsafe")
        shard_paths.add(path)
    expected = {"manifest.json", "tokenizer-bytes.json", "sources.json",
                "documents.jsonl", "sources"} | shard_paths
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise StreamError("dataset file set is unexpected")
    for name in expected - {"sources"}:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise StreamError("dataset entry is not a regular file")
    source_dir = root / "sources"
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise StreamError("sources directory is invalid")
    expected_sources = set()
    for source in sources.values():
        expected_sources.add(Path(source["raw_path"]).name)
        expected_sources.add(Path(source["content_path"]).name)
    if {entry.name for entry in source_dir.iterdir()} != expected_sources:
        raise StreamError("source file set is unexpected")
    for name in expected_sources:
        path = source_dir / name
        if path.is_symlink() or not path.is_file():
            raise StreamError("source entry is not a regular file")


def _meta(root, manifest_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise StreamError("dataset root must be a regular directory")
    manifest, manifest_bytes = _read_json(root / "manifest.json", "manifest",
                                          MAX_MANIFEST_BYTES)
    if _sha(manifest_bytes) != manifest_sha:
        raise StreamError("manifest SHA-256 mismatch")
    _obj(manifest, "manifest", MANIFEST_KEYS)
    if manifest["format"] != FORMAT or not _is_int(manifest["version"]) \
            or manifest["version"] != VERSION:
        raise StreamError("stream format/version is unsupported")
    sequence = _int(manifest["sequence_length"], "sequence_length",
                    SEQUENCE_MIN, SEQUENCE_MAX)
    vocab = _int(manifest["vocab_size"], "vocab_size", VOCAB_MIN, VOCAB_MAX)
    if manifest["tokenizer_special_ids"] != SPECIAL_IDS:
        raise StreamError("manifest special IDs are invalid")
    tokenizer, tokenizer_bytes = _read_json(root / "tokenizer-bytes.json",
                                            "tokenizer-bytes.json",
                                            MAX_TOKENIZER_BYTES)
    if _sha(tokenizer_bytes) != manifest["tokenizer_sha256"]:
        raise StreamError("tokenizer SHA-256 mismatch")
    table = _tokenizer_data(tokenizer, vocab)
    sources_doc, source_bytes = _read_json(root / "sources.json", "sources.json",
                                          MAX_TOKENIZER_BYTES)
    if _sha(source_bytes) != manifest["source_registry_sha256"]:
        raise StreamError("source registry SHA-256 mismatch")
    sources, content_map, family_split, _ = _source_data(root, sources_doc)
    docs, selected, doc_sha = _document_data(root, sources, content_map, family_split,
                                            {s["family_id"]: s["language"]
                                             for s in sources.values()})
    if doc_sha != manifest["document_metadata_sha256"]:
        raise StreamError("document metadata SHA-256 mismatch")
    _check_file_set(root, manifest, sources)
    return root, manifest, sequence, vocab, table, sources, docs, selected


def _shard_data(manifest, root, sequence, vocab, table, docs, selected):
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise StreamError("shards must be a nonempty list")
    counts = {split: {"blocks": 0, **{k: 0 for k in COUNT_FIELDS},
                      "family_content_tokens": {}, "family_content_bytes": {},
                      "language_content_tokens": {}, "language_content_bytes": {}}
              for split in SPLIT_ORDER}
    family_all = {}
    language_all = {}
    total_bytes = 0
    total_blocks = 0
    total_positions = 0
    total_valid = 0
    seen = set()
    seen_paths = set()
    split_seen = set()
    last_split_index = -1
    for index, shard in enumerate(shards):
        _obj(shard, f"shards[{index}]", SHARD_KEYS)
        split = shard["split"]
        if split not in SPLIT_ORDER:
            raise StreamError("shard split is invalid")
        split_index = SPLIT_ORDER.index(split)
        if split_index < last_split_index:
            raise StreamError("shards are not ordered train/validation/test")
        last_split_index = split_index
        split_seen.add(split)
        path = shard["path"]
        if not _SAFE.fullmatch(path) or "/" in path:
            raise StreamError("shard path is unsafe")
        if path in seen_paths:
            raise StreamError("duplicate shard path")
        seen_paths.add(path)
        if not path.startswith(split + "-") or not path.endswith(".jsonl"):
            raise StreamError("shard path is not split-scoped")
        for key in ("bytes", "blocks") + COUNT_FIELDS:
            _int(shard[key], f"shard {key}")
        if not _hex64(shard["sha256"]):
            raise StreamError("shard hash is invalid")
        if shard["bytes"] > MAX_SHARD_BYTES or shard["blocks"] < 1:
            raise StreamError("shard bounds are invalid")
        total_bytes += shard["bytes"]
        total_blocks += shard["blocks"]
        total_positions += shard["blocks"] * sequence
        total_valid += shard["valid_tokens"]
        if total_bytes > MAX_TOTAL_BYTES or total_positions > MAX_TOTAL_POSITIONS \
                or total_valid > MAX_TOTAL_VALID:
            raise StreamError("dataset exceeds aggregate caps")
        counts[split]["blocks"] += shard["blocks"]
        for key in COUNT_FIELDS:
            counts[split][key] += shard[key]
        split_cap = {"train": MAX_TRAIN_VALID,
                     "validation": MAX_VALIDATION_VALID,
                     "test": MAX_TEST_VALID}[split]
        if counts[split]["valid_tokens"] > split_cap:
            raise StreamError("split valid-token count exceeds cap")
        for key in ("family_content_tokens", "family_content_bytes",
                    "language_content_tokens", "language_content_bytes"):
            if not isinstance(shard[key], dict):
                raise StreamError(f"shard {key} must be an object")
            for name, value in shard[key].items():
                _str(name, f"shard {key} name")
                _int(value, f"shard {key}")
                counts[split][key][name] = counts[split][key].get(name, 0) + value
    if split_seen != set(SPLIT_ORDER):
        raise StreamError("all three splits need at least one shard")
    logical = hashlib.sha256()
    for shard in shards:
        split = shard["split"]
        shard_counts = {"blocks": 0, **{k: 0 for k in COUNT_FIELDS},
                        "family_content_tokens": {}, "family_content_bytes": {},
                        "language_content_tokens": {}, "language_content_bytes": {}}
        for line_no, payload, raw in _stream_shard(root / shard["path"],
                                                   shard["bytes"], shard["sha256"]):
            logical.update(raw)
            try:
                block = json.loads(payload.decode("utf-8"),
                                   object_pairs_hook=_unique_pairs,
                                   parse_constant=_reject_constant)
            except StreamError:
                raise
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise StreamError("block JSON is invalid") from exc
            if _canonical(block) != payload:
                raise StreamError("block is not canonical")
            item = _validate_block(block, docs, table, sequence, vocab, split, seen)
            shard_counts["blocks"] += 1
            for key in COUNT_FIELDS:
                shard_counts[key] += item[key]
            for family, entity in item["by_family"].items():
                target = family_all.setdefault(family, {k: 0 for k in COUNT_FIELDS})
                for key in COUNT_FIELDS:
                    target[key] += entity[key]
                shard_counts["family_content_tokens"][family] = (
                    shard_counts["family_content_tokens"].get(family, 0)
                    + entity["content_tokens"])
                shard_counts["family_content_bytes"][family] = (
                    shard_counts["family_content_bytes"].get(family, 0)
                    + entity["content_target_bytes"])
            for language, entity in item["by_language"].items():
                target = language_all.setdefault(language, {k: 0 for k in COUNT_FIELDS})
                for key in COUNT_FIELDS:
                    target[key] += entity[key]
                shard_counts["language_content_tokens"][language] = (
                    shard_counts["language_content_tokens"].get(language, 0)
                    + entity["content_tokens"])
                shard_counts["language_content_bytes"][language] = (
                    shard_counts["language_content_bytes"].get(language, 0)
                    + entity["content_target_bytes"])
        if shard_counts["blocks"] != shard["blocks"]:
            raise StreamError("shard block count mismatch")
        for key in COUNT_FIELDS:
            if shard_counts[key] != shard[key]:
                raise StreamError(f"shard {key} mismatch")
        for key in ("family_content_tokens", "family_content_bytes",
                    "language_content_tokens", "language_content_bytes"):
            if shard_counts[key] != shard[key]:
                raise StreamError(f"shard {key} mismatch")
    if logical.hexdigest() != manifest["logical_sha256"]:
        raise StreamError("logical SHA-256 mismatch")
    if seen != set(docs):
        raise StreamError("orphan or missing document IDs")
    if sum(s["selected_bytes"] for s in shards) != sum(selected.values()):
        raise StreamError("shard selected-byte reconciliation failed")
    total = {"blocks": total_blocks,
             **{k: sum(c[k] for c in counts.values()) for k in COUNT_FIELDS},
             "by_split": {}, "by_language": language_all, "by_family": family_all}
    for split in SPLIT_ORDER:
        total["by_split"][split] = {k: counts[split][k] for k in ("blocks",) + COUNT_FIELDS}
    try:
        v2._validate_counts(manifest["counts"], total)
    except v2.PilotV2Error as exc:
        raise StreamError(str(exc)) from exc
    return counts, total, shards


def load_stream_dataset(root, manifest_sha: str) -> dict:
    """Verify metadata and all shards; return counts/summary without blocks."""
    root, manifest, sequence, vocab, table, sources, docs, selected = _meta(root, manifest_sha)
    counts, total, shards = _shard_data(manifest, root, sequence, vocab, table, docs, selected)
    if total["valid_tokens"] > MAX_TOTAL_VALID:
        raise StreamError("total valid tokens exceed cap")
    if total["by_split"]["train"]["valid_tokens"] > MAX_TRAIN_VALID:
        raise StreamError("train valid tokens exceed cap")
    if total["by_split"]["validation"]["valid_tokens"] > MAX_VALIDATION_VALID:
        raise StreamError("validation valid tokens exceed cap")
    if total["by_split"]["test"]["valid_tokens"] > MAX_TEST_VALID:
        raise StreamError("test valid tokens exceed cap")
    total_file_bytes = sum(path.stat().st_size for path in root.rglob("*")
                           if path.is_file())
    if total_file_bytes > MAX_TOTAL_BYTES:
        raise StreamError("dataset file bytes exceed cap")
    return {
        "format": FORMAT, "version": VERSION, "manifest_sha256": manifest_sha,
        "sequence_length": sequence, "vocab_size": vocab,
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "source_registry_sha256": manifest["source_registry_sha256"],
        "document_metadata_sha256": manifest["document_metadata_sha256"],
        "logical_sha256": manifest["logical_sha256"],
        "counts": total, "splits": counts, "shards": shards,
        "sources": [sources[k] for k in sorted(sources)],
        "documents": {k: v for k, v in docs.items() if v["split"] != "test"},
        "test_validated_only": True, "test_blocks_exposed": False,
    }


def iter_blocks(root, manifest_sha: str, split: str = "validation"):
    """Yield validated train/validation blocks; test is never exposed."""
    if split not in ("train", "validation"):
        raise StreamError("test blocks are not exposed")
    load_stream_dataset(root, manifest_sha)
    root, manifest, sequence, vocab, table, sources, docs, selected = _meta(root, manifest_sha)
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        for _, payload, _ in _stream_shard(Path(root) / shard["path"],
                                           shard["bytes"], shard["sha256"]):
            block = json.loads(payload.decode("utf-8"),
                               object_pairs_hook=_unique_pairs,
                               parse_constant=_reject_constant)
            block["labels"] = {int(key): list(value)
                               for key, value in block["labels"].items()}
            yield block


def iter_targets(root, manifest_sha: str, split: str = "validation"):
    if split not in ("train", "validation"):
        raise StreamError("test targets are not exposed")
    dataset = load_stream_dataset(root, manifest_sha)
    root, manifest, sequence, vocab, table, sources, docs, selected = _meta(root, manifest_sha)
    for block_index, block in enumerate(iter_blocks(root, manifest_sha, split)):
        for position, label in enumerate(block["labels"][1]):
            if label == IGNORE:
                continue
            target_index = position + 1
            target = block["input_ids"][target_index]
            kind = ("content" if block["control_mask"][target_index] is False
                    else "eos" if target == EOS else "control")
            doc = docs[block["document_ids"][position]]
            yield {"target_token_id": target, "byte_length": block["token_byte_lengths"][target_index],
                   "kind": kind, "block_index": block_index, "position": position,
                   "family_id": doc["family_id"], "language": doc["language"],
                   "split": split}


def verify_targets(root, manifest_sha: str, split: str, targets: list) -> list[dict]:
    """Verify caller-supplied target records against the validated stream."""
    if split not in ("train", "validation"):
        raise StreamError("test targets are not exposed")
    expected = list(iter_targets(root, manifest_sha, split))
    if not isinstance(targets, list) or len(targets) != len(expected):
        raise StreamError("target list length mismatch")
    seen = set()
    fields = ("target_token_id", "byte_length", "kind", "block_index", "position",
              "family_id", "language", "split")
    for index, (got, want) in enumerate(zip(targets, expected)):
        if not isinstance(got, dict):
            raise StreamError("target must be an object")
        key = (got.get("block_index"), got.get("position"))
        if key in seen:
            raise StreamError("duplicate target identity")
        seen.add(key)
        for field in fields:
            if got.get(field) != want[field]:
                raise StreamError(f"target {field} mismatch")
    return expected


def aggregate_bpb(targets) -> dict:
    """Content-only BPB arithmetic over stream target records."""
    targets = list(targets)
    if not targets:
        raise StreamError("targets must be nonempty")
    content_labels = content_bytes = eos_labels = control_labels = 0
    content_nll = eos_nll = control_nll = ordinary = 0.0
    for index, target in enumerate(targets):
        try:
            nll = target["nll"]
            byte_length = target["byte_length"]
            kind = target["kind"]
        except (TypeError, KeyError) as exc:
            raise StreamError("target schema is invalid") from exc
        if type(nll) not in (int, float) or _is_bool(nll):
            raise StreamError("NLL must be numeric")
        try:
            nll = float(nll)
        except (OverflowError, ValueError) as exc:
            raise StreamError("NLL must be finite") from exc
        if not math.isfinite(nll) or nll < 0 or not _is_int(byte_length) or byte_length < 0:
            raise StreamError("NLL/bytes are invalid")
        if kind == "content":
            if byte_length <= 0:
                raise StreamError("content target has zero bytes")
            content_labels += 1
            content_bytes += byte_length
            content_nll += nll
        elif kind == "eos":
            if byte_length:
                raise StreamError("EOS target claims bytes")
            eos_labels += 1
            eos_nll += nll
        elif kind == "control":
            if byte_length:
                raise StreamError("control target claims bytes")
            control_labels += 1
            control_nll += nll
        else:
            raise StreamError("target kind is invalid")
        ordinary += nll
    for value in (content_nll, eos_nll, control_nll, ordinary):
        if not math.isfinite(value):
            raise StreamError("aggregate NLL is not finite")
    if not content_labels or not content_bytes:
        raise StreamError("zero content denominator")
    return {"content_labels": content_labels, "content_nll_sum": content_nll,
            "content_bytes": content_bytes,
            "content_bpb": content_nll / (math.log(2) * content_bytes),
            "eos_labels": eos_labels, "eos_nll_sum": eos_nll,
            "control_labels": control_labels, "control_nll_sum": control_nll,
            "ordinary_ntp_labels": content_labels + eos_labels + control_labels,
            "ordinary_ntp_nll_sum": ordinary}
