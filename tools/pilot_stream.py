"""Bounded streaming pilot corpus format and verifier (splot-pilot-stream-v1).

New production-oriented prototype; v1 and v2 remain immutable. Source raw/content
bytes live in separate private files, metadata is JSONL, and block shards are
validated with a bounded streaming reader. This is not accepted training input.

The dataset root is a trusted static private directory for one verification call.
Paths and symlinks are checked before source reads, and source/document/shard
bytes are read incrementally under explicit per-file and aggregate caps. This is
not a concurrent-mutation or race-proof guarantee.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

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
CHUNK_BYTES = 64 * 1024

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
TARGET_IDENTITY_FIELDS = ("manifest_sha256", "target_token_id", "byte_length",
                          "kind", "block_index", "position", "family_id",
                          "language", "split")
TARGET_NLL_FIELDS = TARGET_IDENTITY_FIELDS + ("nll",)
TARGET_KINDS = ("content", "eos", "control")
_HEX = re.compile(r"^[0-9a-f]+$")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StreamError(ValueError):
    """Invalid stream manifest, registry, document, block or shard."""


@dataclass(frozen=True)
class ShardIdentity:
    split: str
    path: str
    sha256: str
    bytes: int
    blocks: int


@dataclass(frozen=True)
class StreamIdentity:
    """Immutable identity of one fully verified stream dataset."""

    format: str
    version: int
    manifest_sha256: str
    sequence_length: int
    vocab_size: int
    tokenizer_sha256: str
    source_registry_sha256: str
    document_metadata_sha256: str
    logical_sha256: str
    counts_json: str
    shards: tuple[ShardIdentity, ...]

    @property
    def counts(self) -> dict:
        return json.loads(self.counts_json)


@dataclass
class _VerifiedDataset:
    root: Path
    manifest: dict
    identity: StreamIdentity
    sequence: int
    vocab: int
    table: tuple[bytes, ...]
    sources: dict
    docs: dict[int, dict]
    shards: list[dict]
    counts: dict
    total: dict
    total_file_bytes: int


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


def _read_bounded(path: Path, label: str, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise StreamError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size > max_bytes:
        raise StreamError(f"{label} exceeds byte bound")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise StreamError(f"{label} exceeds byte bound during read")
    return data


def _read_json(path: Path, label: str, max_bytes: int) -> tuple[Any, bytes]:
    data = _read_bounded(path, label, max_bytes)
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


def _stream_hash(path: Path, declared_bytes: int, declared_sha: str, *,
                 utf8: bool = False, label: str = "file") -> None:
    """Hash one file incrementally; never materialize its full contents."""
    if path.is_symlink() or not path.is_file():
        raise StreamError(f"{label} must be a regular non-symlink file")
    if path.stat().st_size != declared_bytes:
        raise StreamError(f"{label} byte count mismatch before read")
    hasher = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict") if utf8 else None
    total = 0
    try:
        with path.open("rb") as handle:
            while total < declared_bytes:
                chunk = handle.read(min(CHUNK_BYTES, declared_bytes - total))
                if not chunk:
                    raise StreamError(f"{label} ended before declared bytes")
                total += len(chunk)
                hasher.update(chunk)
                if decoder is not None:
                    decoder.decode(chunk)
            if handle.read(1):
                raise StreamError(f"{label} grew beyond declared bytes")
        if decoder is not None:
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise StreamError(f"{label} is not valid UTF-8") from exc
    if total != declared_bytes or hasher.hexdigest() != declared_sha:
        raise StreamError(f"{label} hash/byte count mismatch")


def _hash_interval(path: Path, start: int, end: int, *, utf8: bool = False,
                   label: str = "interval") -> str:
    """Hash a bounded slice incrementally; never read the whole source file."""
    if not 0 <= start <= end:
        raise StreamError(f"{label} interval is invalid")
    if path.is_symlink() or not path.is_file():
        raise StreamError(f"{label} source must be a regular file")
    if end > path.stat().st_size:
        raise StreamError(f"{label} interval exceeds source bytes")
    hasher = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict") if utf8 else None
    total = 0
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start
            while total < remaining:
                chunk = handle.read(min(CHUNK_BYTES, remaining - total))
                if not chunk:
                    raise StreamError(f"{label} ended before interval end")
                total += len(chunk)
                hasher.update(chunk)
                if decoder is not None:
                    decoder.decode(chunk)
        if decoder is not None:
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise StreamError(f"{label} bytes are not valid UTF-8") from exc
    if total != remaining:
        raise StreamError(f"{label} interval byte count mismatch")
    return hasher.hexdigest()


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


def _source_metadata(registry: Any) -> tuple[dict, dict, dict, dict, int]:
    """Validate source metadata only; files are not read here."""
    _obj(registry, "sources", SOURCES_KEYS)
    if registry["format"] != SOURCE_FORMAT or not _is_int(registry["version"]) \
            or registry["version"] != SOURCE_VERSION:
        raise StreamError("source registry format/version is unsupported")
    sources = registry["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise StreamError("source registry count is out of bounds")
    by_id = {}
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
        raw_bytes = _int(source["raw_bytes"], "raw_bytes", 1, MAX_SOURCE_BYTES)
        content_bytes = _int(source["content_bytes"], "content_bytes", 1,
                             MAX_SOURCE_BYTES)
        selected = _int(source["selected_bytes"], "selected_bytes")
        if selected > content_bytes:
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
        split_seen.add(source["split"])
        total_source_bytes += raw_bytes + content_bytes
        if total_source_bytes > MAX_TOTAL_BYTES:
            raise StreamError("source bytes exceed the aggregate cap")
    if split_seen != set(SPLIT_ORDER):
        raise StreamError("all three splits need a source")
    files = {}
    for source_id, source in by_id.items():
        files[source_id] = {
            "raw_path": source["raw_path"], "raw_bytes": source["raw_bytes"],
            "raw_sha256": source["raw_sha256"],
            "content_path": source["content_path"],
            "content_bytes": source["content_bytes"],
            "content_sha256": source["content_sha256"],
        }
    return by_id, family_split, family_language, files, total_source_bytes


def _verify_sources(root: Path, sources: dict) -> None:
    for source_id in sorted(sources):
        source = sources[source_id]
        _stream_hash(root / source["raw_path"], source["raw_bytes"],
                     source["raw_sha256"], utf8=False,
                     label=f"source raw {source_id}")
        _stream_hash(root / source["content_path"], source["content_bytes"],
                     source["content_sha256"], utf8=True,
                     label=f"source content {source_id}")


def _check_file_set(root: Path, shards: list[dict],
                    source_files: dict) -> dict[str, int]:
    """Check the exact file set and parent paths before reading source bytes."""
    seen = set()
    for path in (shard["path"] for shard in shards):
        if not _SAFE.fullmatch(path) or "/" in path:
            raise StreamError("shard path is unsafe")
        if path in seen:
            raise StreamError("duplicate shard path")
        seen.add(path)
    expected = {"manifest.json", "tokenizer-bytes.json", "sources.json",
                "documents.jsonl", "sources"} | seen
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise StreamError("dataset file set is unexpected")
    root_real = root.resolve(strict=True)
    sizes = {}
    for name in expected - {"sources"}:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise StreamError("dataset entry is not a regular file")
        resolved = path.resolve(strict=True)
        if resolved.parent != root_real:
            raise StreamError("dataset entry escapes the root")
        sizes[name] = path.stat().st_size
    source_dir = root / "sources"
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise StreamError("sources directory is invalid")
    source_real = source_dir.resolve(strict=True)
    if source_real.parent != root_real:
        raise StreamError("sources directory escapes the root")
    expected_sources = set()
    for source in source_files.values():
        expected_sources.add(Path(source["raw_path"]).name)
        expected_sources.add(Path(source["content_path"]).name)
    if {entry.name for entry in source_dir.iterdir()} != expected_sources:
        raise StreamError("source file set is unexpected")
    for name in expected_sources:
        path = source_dir / name
        if path.is_symlink() or not path.is_file():
            raise StreamError("source entry is not a regular file")
        resolved = path.resolve(strict=True)
        if resolved.parent != source_real:
            raise StreamError("source entry escapes the source directory")
        sizes[f"sources/{name}"] = path.stat().st_size
    for source in source_files.values():
        if sizes[f"sources/{Path(source['raw_path']).name}"] != source["raw_bytes"]:
            raise StreamError("source raw byte count mismatch before read")
        if sizes[f"sources/{Path(source['content_path']).name}"] != source["content_bytes"]:
            raise StreamError("source content byte count mismatch before read")
    declared = {shard["path"]: shard["bytes"] for shard in shards}
    for path, expected_bytes in declared.items():
        actual_bytes = sizes[path]
        if actual_bytes != expected_bytes:
            raise StreamError("shard byte count mismatch before read")
        if actual_bytes > MAX_SHARD_BYTES:
            raise StreamError("shard exceeds byte bound before read")
    if sum(sizes.values()) > MAX_TOTAL_BYTES:
        raise StreamError("dataset file bytes exceed the aggregate cap")
    return sizes


def _validate_manifest(manifest: Any, manifest_sha: str,
                       manifest_bytes: bytes) -> tuple[list[dict], dict]:
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
    for key in ("tokenizer_sha256", "source_registry_sha256",
                "document_metadata_sha256", "logical_sha256"):
        if not _hex64(manifest[key]):
            raise StreamError(f"{key} is invalid")
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise StreamError("shard list is invalid")
    seen_paths = set()
    split_seen = set()
    last_split_index = -1
    total_bytes = 0
    total_blocks = 0
    total_valid = 0
    split_valid = {split: 0 for split in SPLIT_ORDER}
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
        if shard["valid_tokens"] > shard["blocks"] * sequence:
            raise StreamError("shard valid tokens exceed positions")
        if shard["content_tokens"] > shard["valid_tokens"]:
            raise StreamError("shard content tokens exceed valid tokens")
        total_bytes += shard["bytes"]
        total_blocks += shard["blocks"]
        total_valid += shard["valid_tokens"]
        split_valid[split] += shard["valid_tokens"]
        if total_bytes > MAX_TOTAL_BYTES or total_blocks * sequence > MAX_TOTAL_POSITIONS \
                or total_valid > MAX_TOTAL_VALID:
            raise StreamError("dataset exceeds aggregate caps")
        split_cap = {"train": MAX_TRAIN_VALID,
                     "validation": MAX_VALIDATION_VALID,
                     "test": MAX_TEST_VALID}[split]
        if split_valid[split] > split_cap:
            raise StreamError("split valid-token count exceeds cap")
    if split_seen != set(SPLIT_ORDER):
        raise StreamError("all three splits need at least one shard")
    return shards, {"bytes": total_bytes, "blocks": total_blocks,
                    "valid_tokens": total_valid}


def _verify_document_slice(path: Path, source_bytes: int,
                           source_start: int, source_end: int,
                           seg_start: int, seg_end: int,
                           full_sha: str, segment_sha: str,
                           verified_full: set) -> None:
    if not 0 <= source_start < source_end <= source_bytes:
        raise StreamError("source interval is out of bounds")
    if not source_start <= seg_start < seg_end <= source_end:
        raise StreamError("segment interval is outside the document")
    full_key = (str(path), source_start, source_end, full_sha)
    if full_key not in verified_full:
        got = _hash_interval(path, source_start, source_end, utf8=True,
                             label="full document")
        if got != full_sha:
            raise StreamError("full document hash mismatch")
        verified_full.add(full_key)
    if seg_start == source_start and seg_end == source_end and segment_sha == full_sha:
        return
    got = _hash_interval(path, seg_start, seg_end, utf8=True,
                         label="document segment")
    if got != segment_sha:
        raise StreamError("segment hash mismatch")


def _document_data(root: Path, sources: dict, source_files: dict,
                   family_split: dict, family_language: dict,
                   sequence: int) -> tuple[dict[int, dict], dict[str, int], str]:
    path = root / "documents.jsonl"
    if path.is_symlink() or not path.is_file():
        raise StreamError("documents.jsonl must be a regular file")
    by_id = {}
    segment_ids = set()
    groups = {}
    selected_by_source = {source_id: 0 for source_id in sources}
    full_intervals = {source_id: [] for source_id in sources}
    selected_intervals = {source_id: [] for source_id in sources}
    verified_full = set()
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
            selected = _int(doc["selected_bytes"], "selected_bytes")
            if selected != seg_end - seg_start:
                raise StreamError("document selected_bytes mismatch")
            token_count = _int(doc["token_count"], "token_count", 1,
                               sequence - 2)
            if not _hex64(doc["full_document_sha256"]) \
                    or not _hex64(doc["segment_sha256"]):
                raise StreamError("document hashes are invalid")
            _verify_document_slice(
                root / source_files[source_id]["content_path"],
                source["content_bytes"], source_start, source_end,
                seg_start, seg_end, doc["full_document_sha256"],
                doc["segment_sha256"], verified_full)
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


def _verify_shards(root: Path, shards: list[dict], sequence: int, vocab: int,
                   table: tuple[bytes, ...], docs: dict[int, dict],
                   selected: dict[str, int]) -> tuple[dict, dict]:
    counts = {split: {"blocks": 0, **{k: 0 for k in COUNT_FIELDS},
                      "family_content_tokens": {}, "family_content_bytes": {},
                      "language_content_tokens": {}, "language_content_bytes": {}}
              for split in SPLIT_ORDER}
    family_all = {}
    language_all = {}
    seen = set()
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
        counts[split]["blocks"] += shard_counts["blocks"]
        for key in COUNT_FIELDS:
            counts[split][key] += shard_counts[key]
        for key in ("family_content_tokens", "family_content_bytes",
                    "language_content_tokens", "language_content_bytes"):
            for name, value in shard_counts[key].items():
                counts[split][key][name] = counts[split][key].get(name, 0) + value
    if len(seen) != len(docs):
        raise StreamError("orphan or missing document IDs")
    if seen != set(docs):
        raise StreamError("orphan or missing document IDs")
    if sum(shard["selected_bytes"] for shard in shards) != sum(selected.values()):
        raise StreamError("shard selected-byte reconciliation failed")
    total = {"blocks": sum(c["blocks"] for c in counts.values()),
             **{k: sum(c[k] for c in counts.values()) for k in COUNT_FIELDS},
             "by_split": {}, "by_language": language_all, "by_family": family_all}
    for split in SPLIT_ORDER:
        total["by_split"][split] = {k: counts[split][k] for k in ("blocks",) + COUNT_FIELDS}
    return counts, total


def _verify_dataset(root, manifest_sha: str) -> _VerifiedDataset:
    if not _hex64(manifest_sha):
        raise StreamError("expected manifest SHA-256 is invalid")
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise StreamError("dataset root must be a regular directory")
    if not root.parent.is_dir():
        raise StreamError("dataset parent must be a directory")
    manifest, manifest_bytes = _read_json(root / "manifest.json", "manifest",
                                          MAX_MANIFEST_BYTES)
    shards, _declared = _validate_manifest(manifest, manifest_sha, manifest_bytes)
    sequence = manifest["sequence_length"]
    vocab = manifest["vocab_size"]
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
    sources, family_split, family_language, source_files, _total_source = \
        _source_metadata(sources_doc)
    sizes = _check_file_set(root, shards, source_files)
    _verify_sources(root, sources)
    docs, selected, doc_sha = _document_data(
        root, sources, source_files, family_split, family_language, sequence)
    if doc_sha != manifest["document_metadata_sha256"]:
        raise StreamError("document metadata SHA-256 mismatch")
    counts, total = _verify_shards(root, shards, sequence, vocab, table, docs,
                                   selected)
    try:
        v2._validate_counts(manifest["counts"], total)
    except v2.PilotV2Error as exc:
        raise StreamError(str(exc)) from exc
    if total["valid_tokens"] > MAX_TOTAL_VALID:
        raise StreamError("total valid tokens exceed cap")
    if total["by_split"]["train"]["valid_tokens"] > MAX_TRAIN_VALID:
        raise StreamError("train valid tokens exceed cap")
    if total["by_split"]["validation"]["valid_tokens"] > MAX_VALIDATION_VALID:
        raise StreamError("validation valid tokens exceed cap")
    if total["by_split"]["test"]["valid_tokens"] > MAX_TEST_VALID:
        raise StreamError("test valid tokens exceed cap")
    total_file_bytes = sum(sizes.values())
    if total_file_bytes > MAX_TOTAL_BYTES:
        raise StreamError("dataset file bytes exceed cap")
    identity = StreamIdentity(
        format=FORMAT, version=VERSION, manifest_sha256=manifest_sha,
        sequence_length=sequence, vocab_size=vocab,
        tokenizer_sha256=manifest["tokenizer_sha256"],
        source_registry_sha256=manifest["source_registry_sha256"],
        document_metadata_sha256=manifest["document_metadata_sha256"],
        logical_sha256=manifest["logical_sha256"],
        counts_json=_canonical(total).decode("utf-8"),
        shards=tuple(ShardIdentity(split=shard["split"], path=shard["path"],
                                   sha256=shard["sha256"], bytes=shard["bytes"],
                                   blocks=shard["blocks"])
                     for shard in shards))
    return _VerifiedDataset(root=root, manifest=manifest, identity=identity,
                            sequence=sequence, vocab=vocab, table=table,
                            sources=sources, docs=docs, shards=shards,
                            counts=counts, total=total,
                            total_file_bytes=total_file_bytes)


def load_stream_dataset(root, manifest_sha: str) -> dict:
    """Verify metadata and all shards; return counts/summary without blocks."""
    dataset = _verify_dataset(root, manifest_sha)
    return {
        "format": FORMAT, "version": VERSION,
        "manifest_sha256": manifest_sha, "identity": dataset.identity,
        "sequence_length": dataset.sequence, "vocab_size": dataset.vocab,
        "tokenizer_sha256": dataset.manifest["tokenizer_sha256"],
        "source_registry_sha256": dataset.manifest["source_registry_sha256"],
        "document_metadata_sha256": dataset.manifest["document_metadata_sha256"],
        "logical_sha256": dataset.manifest["logical_sha256"],
        "counts": dataset.total, "splits": dataset.counts,
        "shards": dataset.shards,
        "sources": [dataset.sources[k] for k in sorted(dataset.sources)],
        "documents": {k: v for k, v in dataset.docs.items()
                      if v["split"] != "test"},
        "test_validated_only": True, "test_blocks_exposed": False,
        "target_identity_fields": list(TARGET_IDENTITY_FIELDS),
        "total_file_bytes": dataset.total_file_bytes,
    }


def _iter_blocks(dataset: _VerifiedDataset, split: str) -> Iterator[dict]:
    for shard in dataset.shards:
        if shard["split"] != split:
            continue
        for _, payload, _ in _stream_shard(dataset.root / shard["path"],
                                           shard["bytes"], shard["sha256"]):
            block = json.loads(payload.decode("utf-8"),
                               object_pairs_hook=_unique_pairs,
                               parse_constant=_reject_constant)
            block["labels"] = {int(key): list(value)
                               for key, value in block["labels"].items()}
            yield block


def iter_blocks(root, manifest_sha: str, split: str = "validation"):
    """Yield validated train/validation blocks; test is never exposed."""
    if split not in ("train", "validation"):
        raise StreamError("test blocks are not exposed")
    dataset = _verify_dataset(root, manifest_sha)
    yield from _iter_blocks(dataset, split)


def _iter_targets(dataset: _VerifiedDataset, split: str) -> Iterator[dict]:
    for block_index, block in enumerate(_iter_blocks(dataset, split)):
        for position, label in enumerate(block["labels"][1]):
            if label == IGNORE:
                continue
            target_index = position + 1
            target = block["input_ids"][target_index]
            kind = ("content" if block["control_mask"][target_index] is False
                    else "eos" if target == EOS else "control")
            doc = dataset.docs[block["document_ids"][position]]
            yield {"manifest_sha256": dataset.identity.manifest_sha256,
                   "target_token_id": target,
                   "byte_length": block["token_byte_lengths"][target_index],
                   "kind": kind, "block_index": block_index,
                   "position": position, "family_id": doc["family_id"],
                   "language": doc["language"], "split": split}


def iter_targets(root, manifest_sha: str, split: str = "validation"):
    if split not in ("train", "validation"):
        raise StreamError("test targets are not exposed")
    dataset = _verify_dataset(root, manifest_sha)
    yield from _iter_targets(dataset, split)


def _check_target_identity(record: Any, expected: dict, dataset: _VerifiedDataset,
                           fields) -> None:
    if not isinstance(record, dict) or set(record) != set(fields):
        raise StreamError("target schema is invalid")
    if record["manifest_sha256"] != dataset.identity.manifest_sha256:
        raise StreamError("target manifest identity mismatch")
    token = record["target_token_id"]
    if not _is_int(token) or not 0 <= token < dataset.vocab:
        raise StreamError("target token ID is invalid")
    byte_length = record["byte_length"]
    if not _is_int(byte_length) or not 0 <= byte_length <= MAX_SOURCE_BYTES:
        raise StreamError("target byte length is invalid")
    kind = record["kind"]
    if not isinstance(kind, str) or kind not in TARGET_KINDS:
        raise StreamError("target kind is invalid")
    for key in ("block_index", "position"):
        if not _is_int(record[key]) or record[key] < 0:
            raise StreamError(f"target {key} is invalid")
    for key in ("family_id", "language", "split"):
        if not isinstance(record[key], str) or not record[key]:
            raise StreamError(f"target {key} is invalid")
    for key in fields:
        if key == "nll":
            continue
        if record[key] != expected[key]:
            raise StreamError(f"target {key} mismatch")


class _BpbAccumulator:
    __slots__ = ("content_labels", "content_bytes", "eos_labels", "control_labels",
                 "content_nll", "eos_nll", "control_nll", "ordinary", "count")

    def __init__(self):
        self.content_labels = 0
        self.content_bytes = 0
        self.eos_labels = 0
        self.control_labels = 0
        self.content_nll = 0.0
        self.eos_nll = 0.0
        self.control_nll = 0.0
        self.ordinary = 0.0
        self.count = 0

    def add(self, target: Any) -> None:
        if not isinstance(target, dict):
            raise StreamError("target must be an object")
        try:
            nll = target["nll"]
            byte_length = target["byte_length"]
            kind = target["kind"]
        except KeyError as exc:
            raise StreamError("target schema is invalid") from exc
        if type(nll) not in (int, float) or _is_bool(nll):
            raise StreamError("NLL must be numeric")
        try:
            nll = float(nll)
        except (OverflowError, ValueError) as exc:
            raise StreamError("NLL must be finite") from exc
        if not math.isfinite(nll) or nll < 0:
            raise StreamError("NLL/bytes are invalid")
        if not _is_int(byte_length) or byte_length < 0:
            raise StreamError("NLL/bytes are invalid")
        if not isinstance(kind, str):
            raise StreamError("target kind is invalid")
        if kind == "content":
            if byte_length <= 0:
                raise StreamError("content target has zero bytes")
            self.content_labels += 1
            self.content_bytes += byte_length
            self.content_nll += nll
        elif kind == "eos":
            if byte_length:
                raise StreamError("EOS target claims bytes")
            self.eos_labels += 1
            self.eos_nll += nll
        elif kind == "control":
            if byte_length:
                raise StreamError("control target claims bytes")
            self.control_labels += 1
            self.control_nll += nll
        else:
            raise StreamError("target kind is invalid")
        self.ordinary += nll
        self.count += 1

    def result(self) -> dict:
        for value in (self.content_nll, self.eos_nll, self.control_nll,
                      self.ordinary):
            if not math.isfinite(value):
                raise StreamError("aggregate NLL is not finite")
        if not self.content_labels or not self.content_bytes:
            raise StreamError("zero content denominator")
        return {"content_labels": self.content_labels,
                "content_nll_sum": self.content_nll,
                "content_bytes": self.content_bytes,
                "content_bpb": self.content_nll / (math.log(2) * self.content_bytes),
                "eos_labels": self.eos_labels, "eos_nll_sum": self.eos_nll,
                "control_labels": self.control_labels,
                "control_nll_sum": self.control_nll,
                "ordinary_ntp_labels": self.content_labels + self.eos_labels
                + self.control_labels,
                "ordinary_ntp_nll_sum": self.ordinary}


def aggregate_bpb(targets) -> dict:
    """Content-only BPB arithmetic over a one-pass stream target iterator."""
    try:
        iterator = iter(targets)
    except TypeError as exc:
        raise StreamError("targets must be iterable") from exc
    accumulator = _BpbAccumulator()
    allowed = ({"nll", "byte_length", "kind"}, set(TARGET_NLL_FIELDS))
    for target in iterator:
        if not isinstance(target, dict) or set(target) not in allowed:
            raise StreamError("target schema is invalid")
        accumulator.add(target)
    if not accumulator.count:
        raise StreamError("targets must be nonempty")
    return accumulator.result()


def _target_summary(manifest_sha: str, counts: dict) -> dict:
    return {"manifest_sha256": manifest_sha,
            "verified_targets": sum(counts.values()),
            "content": counts["content"], "eos": counts["eos"],
            "control": counts["control"]}


def verify_targets(root, manifest_sha: str, split: str, targets) -> dict:
    """Verify caller target records against expected records in lockstep."""
    if split not in ("train", "validation"):
        raise StreamError("test targets are not exposed")
    dataset = _verify_dataset(root, manifest_sha)
    try:
        provided = iter(targets)
    except TypeError as exc:
        raise StreamError("targets must be iterable") from exc
    sentinel = object()
    counts = {"content": 0, "eos": 0, "control": 0}
    verified = 0
    for expected in _iter_targets(dataset, split):
        got = next(provided, sentinel)
        if got is sentinel:
            raise StreamError("target list ended early")
        _check_target_identity(got, expected, dataset, TARGET_IDENTITY_FIELDS)
        counts[got["kind"]] += 1
        verified += 1
    if next(provided, sentinel) is not sentinel:
        raise StreamError("target list has extra records")
    if not verified:
        raise StreamError("targets must be nonempty")
    return _target_summary(dataset.identity.manifest_sha256, counts)


def verify_and_aggregate_bpb(root, manifest_sha: str, split: str, targets) -> dict:
    """Stream target identity verification and BPB aggregation in one pass."""
    if split not in ("train", "validation"):
        raise StreamError("test targets are not exposed")
    dataset = _verify_dataset(root, manifest_sha)
    try:
        provided = iter(targets)
    except TypeError as exc:
        raise StreamError("targets must be iterable") from exc
    sentinel = object()
    counts = {"content": 0, "eos": 0, "control": 0}
    verified = 0
    accumulator = _BpbAccumulator()
    for expected in _iter_targets(dataset, split):
        got = next(provided, sentinel)
        if got is sentinel:
            raise StreamError("target list ended early")
        _check_target_identity(got, expected, dataset, TARGET_NLL_FIELDS)
        accumulator.add(got)
        counts[got["kind"]] += 1
        verified += 1
    if next(provided, sentinel) is not sentinel:
        raise StreamError("target list has extra records")
    if not verified:
        raise StreamError("targets must be nonempty")
    result = _target_summary(dataset.identity.manifest_sha256, counts)
    result.update(accumulator.result())
    return result
