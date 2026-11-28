"""Independent auditor for splot-pilot-stream-v1.

This script deliberately does not import pilot_stream or pilot_stream_producer.
It re-parses the canonical artifacts and recomputes coverage/counts/hashes with
stdlib code for an independent check. Source, document and shard bytes are read
incrementally under explicit caps. It never returns test token arrays.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import re
import sys
from pathlib import Path

FORMAT = "splot-pilot-stream-v1"
VERSION = 1
SPLITS = ("train", "validation", "test")
PAD, BOS, EOS, IGNORE = 0, 1, 2, -100
CONTENT_MIN = 8
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
BLOCK_KEYS = ("input_ids", "document_ids", "position_ids", "valid",
              "control_mask", "token_byte_lengths", "token_byte_offsets",
              "labels")
MANIFEST_KEYS = ("format", "version", "sequence_length", "vocab_size",
                 "tokenizer_sha256", "tokenizer_special_ids",
                 "source_registry_sha256", "document_metadata_sha256",
                 "shards", "counts", "logical_sha256")
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_TOTAL_POSITIONS = 1_600_000
MAX_TRAIN_VALID = 1_000_000
MAX_VALIDATION_VALID = 50_000
MAX_TEST_VALID = 50_000
MAX_TOTAL_VALID = 1_100_000
MAX_DOCUMENTS = 20_000
MAX_SOURCES = 32
MAX_METADATA_LINE = 128 * 1024
MAX_SHARD_LINE = 256 * 1024
MAX_SHARD_BYTES = 32 * 1024 * 1024
MAX_TOKENIZER_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
HEX = re.compile(r"^[0-9a-f]+$")
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AuditError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def is_int(value):
    return type(value) is int


def is_bool(value):
    return type(value) is bool


def need_int(value, name, minimum=0, maximum=None):
    if not is_int(value) or value < minimum:
        raise AuditError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise AuditError(f"{name} exceeds {maximum}")
    return value


def need_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{name} must be nonempty")
    return value


def need_obj(value, name, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise AuditError(f"{name} schema invalid")
    return value


def need_hex(value, name):
    if not isinstance(value, str) or len(value) != 64 or not HEX.fullmatch(value):
        raise AuditError(f"{name} hash invalid")


def read_bounded(path, name, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{name} not regular")
    if path.stat().st_size > limit:
        raise AuditError(f"{name} exceeds cap")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise AuditError(f"{name} exceeds cap during read")
    return data


def read_json(path, name, limit):
    data = read_bounded(path, name, limit)
    value = json.loads(data.decode())
    if canonical(value) != data:
        raise AuditError(f"{name} not canonical")
    return value, data


def stream_hash(path, declared_bytes, declared_sha, *, utf8=False, label="file"):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{label} not regular")
    if path.stat().st_size != declared_bytes:
        raise AuditError(f"{label} byte count before read")
    hasher = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict") if utf8 else None
    total = 0
    try:
        with path.open("rb") as handle:
            while total < declared_bytes:
                chunk = handle.read(min(CHUNK_BYTES, declared_bytes - total))
                if not chunk:
                    raise AuditError(f"{label} ended early")
                total += len(chunk)
                hasher.update(chunk)
                if decoder is not None:
                    decoder.decode(chunk)
            if handle.read(1):
                raise AuditError(f"{label} grew during read")
        if decoder is not None:
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label} not UTF-8") from exc
    if total != declared_bytes or hasher.hexdigest() != declared_sha:
        raise AuditError(f"{label} hash")


def hash_interval(path, start, end, *, utf8=False, label="interval"):
    if not 0 <= start <= end:
        raise AuditError(f"{label} interval invalid")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{label} source not regular")
    if end > path.stat().st_size:
        raise AuditError(f"{label} interval exceeds source")
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
                    raise AuditError(f"{label} ended early")
                total += len(chunk)
                hasher.update(chunk)
                if decoder is not None:
                    decoder.decode(chunk)
        if decoder is not None:
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label} not UTF-8") from exc
    if total != remaining:
        raise AuditError(f"{label} byte count")
    return hasher.hexdigest()


def tokenizer(root, manifest):
    value, data = read_json(root / "tokenizer-bytes.json", "tokenizer",
                            MAX_TOKENIZER_BYTES)
    if sha(data) != manifest["tokenizer_sha256"]:
        raise AuditError("tokenizer hash")
    need_obj(value, "tokenizer", ("format", "version", "actual_vocab_size",
                                  "special_token_ids", "token_bytes"))
    if value["format"] != "splot-byte-tokenizer-stream-v1" \
            or not is_int(value["version"]) or value["version"] != 1:
        raise AuditError("tokenizer format")
    vocab = need_int(value["actual_vocab_size"], "actual_vocab_size", 264, 16384)
    if vocab != manifest["vocab_size"]:
        raise AuditError("tokenizer vocab")
    special = value["special_token_ids"]
    need_obj(special, "special_ids", ("pad", "bos", "eos", "unk", "reserved"))
    if (special["pad"], special["bos"], special["eos"], special["unk"]) != (0, 1, 2, 3) \
            or special["reserved"] != [4, 5, 6, 7]:
        raise AuditError("special ids")
    table = value["token_bytes"]
    if not isinstance(table, list) or len(table) != vocab:
        raise AuditError("tokenizer length")
    special_ids = {0, 1, 2, 3, 4, 5, 6, 7}
    seen = set()
    single = set()
    for index, entry in enumerate(table):
        if not isinstance(entry, str):
            raise AuditError("token entry type")
        if index in special_ids:
            if entry != "":
                raise AuditError("special bytes")
            continue
        if not entry or len(entry) % 2 or not HEX.fullmatch(entry):
            raise AuditError("token bytes hex")
        raw = bytes.fromhex(entry)
        if not raw or len(raw) > MAX_SOURCE_BYTES or raw in seen:
            raise AuditError("token bytes duplicate/empty")
        seen.add(raw)
        if len(raw) == 1:
            single.add(raw[0])
    if single != set(range(256)):
        raise AuditError("incomplete alphabet")
    return tuple(b"" if i in special_ids else bytes.fromhex(table[i])
                 for i in range(len(table)))


def preflight_manifest(manifest, manifest_bytes, manifest_sha):
    if sha(manifest_bytes) != manifest_sha:
        raise AuditError("manifest hash")
    value = manifest
    need_obj(value, "manifest", MANIFEST_KEYS)
    if value["format"] != FORMAT or not is_int(value["version"]) \
            or value["version"] != VERSION:
        raise AuditError("manifest format")
    sequence = need_int(value["sequence_length"], "sequence_length", 8, 512)
    need_int(value["vocab_size"], "vocab_size", 264, 16384)
    if value["tokenizer_special_ids"] != {"pad": 0, "bos": 1, "eos": 2,
                                          "unk": 3, "reserved": [4, 5, 6, 7]}:
        raise AuditError("manifest special ids")
    for key in ("tokenizer_sha256", "source_registry_sha256",
                "document_metadata_sha256", "logical_sha256"):
        need_hex(value[key], key)
    shards = value["shards"]
    if not isinstance(shards, list) or not shards:
        raise AuditError("shards")
    seen = set()
    split_seen = set()
    last = -1
    total_bytes = 0
    total_valid = 0
    total_blocks = 0
    split_valid = {split: 0 for split in SPLITS}
    for shard in shards:
        need_obj(shard, "shard", SHARD_KEYS)
        split = shard["split"]
        if split not in SPLITS:
            raise AuditError("shard split")
        index = SPLITS.index(split)
        if index < last:
            raise AuditError("shard order")
        last = index
        split_seen.add(split)
        path = shard["path"]
        if not isinstance(path, str) or not SAFE.fullmatch(path) or "/" in path:
            raise AuditError("shard path")
        if path in seen or not path.startswith(split + "-") \
                or not path.endswith(".jsonl"):
            raise AuditError("shard path duplicate/scoped")
        seen.add(path)
        need_hex(shard["sha256"], "shard sha")
        for key in ("bytes", "blocks") + COUNT_FIELDS:
            need_int(shard[key], f"shard {key}")
        if shard["bytes"] > MAX_SHARD_BYTES or shard["blocks"] < 1:
            raise AuditError("shard bounds")
        if shard["valid_tokens"] > shard["blocks"] * sequence \
                or shard["content_tokens"] > shard["valid_tokens"]:
            raise AuditError("shard token bounds")
        total_bytes += shard["bytes"]
        total_blocks += shard["blocks"]
        total_valid += shard["valid_tokens"]
        split_valid[split] += shard["valid_tokens"]
        if total_bytes > MAX_TOTAL_BYTES \
                or total_blocks * sequence > MAX_TOTAL_POSITIONS \
                or total_valid > MAX_TOTAL_VALID:
            raise AuditError("aggregate caps")
        cap = {"train": MAX_TRAIN_VALID, "validation": MAX_VALIDATION_VALID,
               "test": MAX_TEST_VALID}[split]
        if split_valid[split] > cap:
            raise AuditError("split cap")
    if split_seen != set(SPLITS):
        raise AuditError("missing split")
    return value, sequence, shards


def source_metadata(root, manifest):
    value, data = read_json(root / "sources.json", "sources", MAX_TOKENIZER_BYTES)
    if sha(data) != manifest["source_registry_sha256"]:
        raise AuditError("source registry hash")
    need_obj(value, "sources", ("format", "version", "sources"))
    if value["format"] != "splot-source-registry-stream-v1" \
            or not is_int(value["version"]) or value["version"] != 1:
        raise AuditError("source registry format")
    records = value["sources"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_SOURCES:
        raise AuditError("source count")
    by_id = {}
    family_split = {}
    family_language = {}
    split_seen = set()
    total = 0
    for source in records:
        need_obj(source, "source", SOURCE_KEYS)
        sid = need_str(source["source_id"], "source_id")
        if not SAFE.fullmatch(sid):
            raise AuditError("source id unsafe")
        family = need_str(source["family_id"], "family_id")
        if source["language"] not in LANGUAGES or source["split"] not in SPLITS:
            raise AuditError("source language/split")
        for key in ("source_url", "revision", "rights_evidence_url"):
            need_str(source[key], key)
        if source["raw_path"] != f"sources/{sid}.raw" \
                or source["content_path"] != f"sources/{sid}.content":
            raise AuditError("source path")
        need_hex(source["raw_sha256"], "raw sha")
        need_hex(source["content_sha256"], "content sha")
        raw_bytes = need_int(source["raw_bytes"], "raw_bytes", 1, MAX_SOURCE_BYTES)
        content_bytes = need_int(source["content_bytes"], "content_bytes", 1,
                                 MAX_SOURCE_BYTES)
        selected = need_int(source["selected_bytes"], "selected_bytes")
        if selected > content_bytes:
            raise AuditError("source selected")
        if sid in by_id:
            raise AuditError("duplicate source")
        if family in family_split and (family_split[family] != source["split"]
                                       or family_language[family] != source["language"]):
            raise AuditError("family split/language")
        family_split[family] = source["split"]
        family_language[family] = source["language"]
        by_id[sid] = source
        split_seen.add(source["split"])
        total += raw_bytes + content_bytes
        if total > MAX_TOTAL_BYTES:
            raise AuditError("source bytes cap")
    if split_seen != set(SPLITS):
        raise AuditError("source splits")
    return records, by_id, family_split, family_language


def check_file_set(root, shards, sources):
    expected = {"manifest.json", "tokenizer-bytes.json", "sources.json",
                "documents.jsonl", "sources"} | {s["path"] for s in shards}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise AuditError("file set")
    root_real = root.resolve(strict=True)
    sizes = {}
    for name in expected - {"sources"}:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise AuditError("entry not regular")
        if path.resolve(strict=True).parent != root_real:
            raise AuditError("entry escapes root")
        sizes[name] = path.stat().st_size
    source_dir = root / "sources"
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise AuditError("sources dir")
    source_real = source_dir.resolve(strict=True)
    if source_real.parent != root_real:
        raise AuditError("sources dir escapes root")
    names = set()
    for source in sources:
        names.add(Path(source["raw_path"]).name)
        names.add(Path(source["content_path"]).name)
    if {entry.name for entry in source_dir.iterdir()} != names:
        raise AuditError("source file set")
    for name in names:
        path = source_dir / name
        if path.is_symlink() or not path.is_file():
            raise AuditError("source not regular")
        if path.resolve(strict=True).parent != source_real:
            raise AuditError("source escapes dir")
        sizes[f"sources/{name}"] = path.stat().st_size
    for source in sources:
        if sizes[f"sources/{Path(source['raw_path']).name}"] != source["raw_bytes"] \
                or sizes[f"sources/{Path(source['content_path']).name}"] != source["content_bytes"]:
            raise AuditError("source byte count before read")
    declared = {s["path"]: s["bytes"] for s in shards}
    for path, expected_bytes in declared.items():
        if sizes[path] != expected_bytes:
            raise AuditError("shard byte count before read")
    if sum(sizes.values()) > MAX_TOTAL_BYTES:
        raise AuditError("file bytes cap")
    return sizes


def verify_sources(root, sources):
    for source in sources:
        stream_hash(root / source["raw_path"], source["raw_bytes"],
                    source["raw_sha256"], utf8=False,
                    label=f"raw {source['source_id']}")
        stream_hash(root / source["content_path"], source["content_bytes"],
                    source["content_sha256"], utf8=True,
                    label=f"content {source['source_id']}")


def verify_slice(path, source_bytes, s0, s1, g0, g1, full_sha, segment_sha,
                 verified_full):
    if not 0 <= s0 < s1 <= source_bytes or not s0 <= g0 < g1 <= s1:
        raise AuditError("doc interval")
    key = (str(path), s0, s1, full_sha)
    if key not in verified_full:
        if hash_interval(path, s0, s1, utf8=True, label="full doc") != full_sha:
            raise AuditError("full doc hash")
        verified_full.add(key)
    if g0 == s0 and g1 == s1 and segment_sha == full_sha:
        return
    if hash_interval(path, g0, g1, utf8=True, label="segment") != segment_sha:
        raise AuditError("segment hash")


def documents(root, manifest, source_map, sequence):
    path = root / "documents.jsonl"
    if path.is_symlink() or not path.is_file():
        raise AuditError("documents not regular")
    hasher = hashlib.sha256()
    docs = {}
    selected = {sid: 0 for sid in source_map}
    groups = {}
    full_intervals = {sid: [] for sid in source_map}
    selected_intervals = {sid: [] for sid in source_map}
    verified_full = set()
    last = -1
    total = 0
    lines = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline(MAX_METADATA_LINE + 1)
            if not line:
                break
            lines += 1
            total += len(line)
            if len(line) > MAX_METADATA_LINE or not line.endswith(b"\n"):
                raise AuditError("document line")
            hasher.update(line)
            payload = line[:-1]
            try:
                doc = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise AuditError("document JSON") from exc
            if canonical(doc) != payload:
                raise AuditError("document canonical")
            need_obj(doc, "document", DOCUMENT_KEYS)
            doc_id = need_int(doc["doc_id"], "doc_id")
            if doc_id <= last or doc_id in docs:
                raise AuditError("doc order/duplicate")
            last = doc_id
            need_str(doc["segment_id"], "segment_id")
            source = source_map.get(doc["source_id"])
            if source is None:
                raise AuditError("unknown source")
            if (doc["family_id"] != source["family_id"]
                    or doc["language"] != source["language"]
                    or doc["split"] != source["split"]):
                raise AuditError("doc provenance")
            s0 = need_int(doc["source_byte_start"], "source_byte_start")
            s1 = need_int(doc["source_byte_end"], "source_byte_end")
            g0 = need_int(doc["segment_byte_start"], "segment_byte_start")
            g1 = need_int(doc["segment_byte_end"], "segment_byte_end")
            need_hex(doc["full_document_sha256"], "full sha")
            need_hex(doc["segment_sha256"], "segment sha")
            selected_bytes = need_int(doc["selected_bytes"], "selected_bytes")
            if selected_bytes != g1 - g0:
                raise AuditError("doc selected")
            need_int(doc["token_count"], "token_count", 1, sequence - 2)
            verify_slice(root / source["content_path"], source["content_bytes"],
                         s0, s1, g0, g1, doc["full_document_sha256"],
                         doc["segment_sha256"], verified_full)
            index = need_int(doc["segment_index"], "segment_index")
            count = need_int(doc["segment_count"], "segment_count", 1)
            if index >= count or not is_bool(doc["truncated"]):
                raise AuditError("segment index/truncated")
            if doc["truncated"] != ((g1 - g0) < (s1 - s0)):
                raise AuditError("truncated mismatch")
            if not doc["truncated"] and (g0 != s0 or g1 != s1
                                         or doc["full_document_sha256"] != doc["segment_sha256"]):
                raise AuditError("full segment")
            groups.setdefault((doc["source_id"], s0, s1,
                               doc["full_document_sha256"]), []).append(doc)
            selected_intervals[doc["source_id"]].append((g0, g1))
            selected[doc["source_id"]] += g1 - g0
            docs[doc_id] = doc
            if len(docs) > MAX_DOCUMENTS or total > MAX_TOTAL_BYTES:
                raise AuditError("doc count/bytes")
    if not lines:
        raise AuditError("documents empty")
    if hasher.hexdigest() != manifest["document_metadata_sha256"]:
        raise AuditError("document metadata hash")
    for (sid, s0, s1, _), members in groups.items():
        full_intervals[sid].append((s0, s1))
        count = {m["segment_count"] for m in members}
        if len(count) != 1:
            raise AuditError("segment_count")
        count = count.pop()
        ordered = sorted(members, key=lambda d: (
            d["segment_byte_start"], d["segment_byte_end"], d["segment_index"]))
        indices = [d["segment_index"] for d in ordered]
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise AuditError("segment order")
        intervals = [(d["segment_byte_start"], d["segment_byte_end"])
                     for d in ordered]
        for (a, b), (c, d) in zip(intervals, intervals[1:]):
            if b > c:
                raise AuditError("segment overlap")
        if len(members) == count:
            if indices != list(range(count)):
                raise AuditError("complete indices")
            if intervals[0][0] != s0 or intervals[-1][1] != s1 \
                    or any(b != c for (_, b), (c, _) in zip(intervals, intervals[1:])):
                raise AuditError("complete partition")
            if count == 1 and (ordered[0]["segment_index"] != 0
                               or ordered[0]["truncated"]):
                raise AuditError("single full segment")
        elif any(not d["truncated"] for d in members):
            raise AuditError("partial not truncated")
    for sid in source_map:
        for intervals in (full_intervals[sid], selected_intervals[sid]):
            ordered = sorted(intervals)
            for (a, b), (c, d) in zip(ordered, ordered[1:]):
                if b > c:
                    raise AuditError("source interval overlap")
        if selected[sid] != source_map[sid]["selected_bytes"]:
            raise AuditError("source selected")
    return docs, selected


def block_counts(block, docs, table, sequence, split, seen):
    need_obj(block, "block", BLOCK_KEYS)
    input_ids = block["input_ids"]
    document_ids = block["document_ids"]
    positions = block["position_ids"]
    valid = block["valid"]
    control = block["control_mask"]
    lengths = block["token_byte_lengths"]
    offsets = block["token_byte_offsets"]
    labels = block["labels"]
    for key in ("input_ids", "document_ids", "position_ids", "valid",
                "control_mask", "token_byte_lengths", "token_byte_offsets"):
        value = block[key]
        if not isinstance(value, list) or len(value) != sequence:
            raise AuditError("block array length")
    for value in valid + control:
        if not is_bool(value):
            raise AuditError("mask type")
    for value in input_ids + document_ids + positions + lengths + offsets:
        if not is_int(value):
            raise AuditError("block integer type")
    valid_n = sum(valid)
    if valid_n < 3 or valid != [True] * valid_n + [False] * (sequence - valid_n):
        raise AuditError("valid mask")
    for index in range(valid_n, sequence):
        if (input_ids[index] != PAD or document_ids[index] != -1
                or positions[index] != 0 or valid[index] is not False
                or control[index] is not True or lengths[index] != 0
                or offsets[index] != -1):
            raise AuditError("padding")
    need_obj(labels, "labels", ("1", "2"))
    for key in ("1", "2"):
        value = labels[key]
        if not isinstance(value, list) or len(value) != sequence:
            raise AuditError("label length")
        if any(not is_int(item) for item in value):
            raise AuditError("label type")
    total = {"blocks": 1, **{k: 0 for k in COUNT_FIELDS},
             "by_family": {}, "by_language": {}}
    start = 0
    while start < valid_n:
        end = start + 1
        while end < valid_n and document_ids[end] == document_ids[start]:
            end += 1
        doc_id = document_ids[start]
        if doc_id in seen or doc_id not in docs:
            raise AuditError("doc id")
        seen.add(doc_id)
        doc = docs[doc_id]
        if doc["split"] != split:
            raise AuditError("split")
        if end - start < 3 or input_ids[start] != BOS or input_ids[end - 1] != EOS:
            raise AuditError("segment boundary")
        if positions[start:end] != list(range(end - start)):
            raise AuditError("positions")
        if control[start] is not True or control[end - 1] is not True:
            raise AuditError("control boundary")
        cumulative = 0
        content_count = 0
        reconstructed = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            for index in range(start + 1, end - 1):
                token = input_ids[index]
                if not is_int(token) or not CONTENT_MIN <= token < len(table):
                    raise AuditError("content token")
                if control[index] is not False or lengths[index] != len(table[token]):
                    raise AuditError("content coverage")
                if offsets[index] != cumulative:
                    raise AuditError("offset")
                raw = table[token]
                cumulative += len(raw)
                if cumulative > doc["selected_bytes"]:
                    raise AuditError("token reconstruction cap")
                reconstructed.update(raw)
                decoder.decode(raw)
                content_count += 1
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise AuditError("segment not UTF-8") from exc
        if cumulative != doc["selected_bytes"] or content_count != doc["token_count"]:
            raise AuditError("segment coverage")
        if reconstructed.hexdigest() != doc["segment_sha256"]:
            raise AuditError("segment hash")
        for key, offset in (("1", 1), ("2", 2)):
            for index in range(start, end):
                expected = input_ids[index + offset] if index + offset < end else IGNORE
                if labels[key][index] != expected:
                    raise AuditError("label value")
        for key in ("1", "2"):
            for index in range(valid_n, sequence):
                if labels[key][index] != IGNORE:
                    raise AuditError("padding label")
        entity = {"valid_tokens": end - start, "content_tokens": content_count,
                  "ntp_labels": content_count + 1, "content_labels": content_count,
                  "eos_labels": 1, "offset2_labels": content_count,
                  "content_target_bytes": cumulative, "selected_bytes": cumulative}
        for key in COUNT_FIELDS:
            total[key] += entity[key]
        group = total["by_family"].setdefault(doc["family_id"],
                                              {k: 0 for k in COUNT_FIELDS})
        lang = total["by_language"].setdefault(doc["language"],
                                               {k: 0 for k in COUNT_FIELDS})
        for key in COUNT_FIELDS:
            group[key] += entity[key]
            lang[key] += entity[key]
        start = end
    return total


def verify_shards(root, manifest, shards, docs, table):
    sequence = manifest["sequence_length"]
    counts = {split: {"blocks": 0, **{k: 0 for k in COUNT_FIELDS},
                      "family_content_tokens": {}, "family_content_bytes": {},
                      "language_content_tokens": {}, "language_content_bytes": {}}
              for split in SPLITS}
    family_all = {}
    language_all = {}
    seen = set()
    logical = hashlib.sha256()
    for shard in shards:
        split = shard["split"]
        path = root / shard["path"]
        if path.is_symlink() or not path.is_file():
            raise AuditError("shard not regular")
        hasher = hashlib.sha256()
        total_bytes = 0
        sc = {"blocks": 0, **{k: 0 for k in COUNT_FIELDS},
              "family_content_tokens": {}, "family_content_bytes": {},
              "language_content_tokens": {}, "language_content_bytes": {}}
        with path.open("rb") as handle:
            while True:
                line = handle.readline(MAX_SHARD_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_SHARD_LINE:
                    raise AuditError("shard line")
                total_bytes += len(line)
                if total_bytes > shard["bytes"] or total_bytes > MAX_SHARD_BYTES:
                    raise AuditError("shard bytes")
                hasher.update(line)
                logical.update(line)
                try:
                    block = json.loads(line[:-1])
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise AuditError("block JSON") from exc
                if canonical(block) != line[:-1]:
                    raise AuditError("block canonical")
                item = block_counts(block, docs, table, sequence, split, seen)
                sc["blocks"] += 1
                for key in COUNT_FIELDS:
                    sc[key] += item[key]
                for family, entity in item["by_family"].items():
                    target = family_all.setdefault(family,
                                                   {k: 0 for k in COUNT_FIELDS})
                    for key in COUNT_FIELDS:
                        target[key] += entity[key]
                    sc["family_content_tokens"][family] = (
                        sc["family_content_tokens"].get(family, 0)
                        + entity["content_tokens"])
                    sc["family_content_bytes"][family] = (
                        sc["family_content_bytes"].get(family, 0)
                        + entity["content_target_bytes"])
                for language, entity in item["by_language"].items():
                    target = language_all.setdefault(language,
                                                     {k: 0 for k in COUNT_FIELDS})
                    for key in COUNT_FIELDS:
                        target[key] += entity[key]
                    sc["language_content_tokens"][language] = (
                        sc["language_content_tokens"].get(language, 0)
                        + entity["content_tokens"])
                    sc["language_content_bytes"][language] = (
                        sc["language_content_bytes"].get(language, 0)
                        + entity["content_target_bytes"])
        if total_bytes != shard["bytes"] or hasher.hexdigest() != shard["sha256"]:
            raise AuditError("shard hash")
        for key in ("blocks",) + COUNT_FIELDS:
            if sc[key] != shard[key]:
                raise AuditError("shard count")
        for key in ("family_content_tokens", "family_content_bytes",
                    "language_content_tokens", "language_content_bytes"):
            if sc[key] != shard[key]:
                raise AuditError("shard map")
        counts[split]["blocks"] += sc["blocks"]
        for key in COUNT_FIELDS:
            counts[split][key] += sc[key]
        for key in ("family_content_tokens", "family_content_bytes",
                    "language_content_tokens", "language_content_bytes"):
            for name, value in sc[key].items():
                counts[split][key][name] = counts[split][key].get(name, 0) + value
    if logical.hexdigest() != manifest["logical_sha256"]:
        raise AuditError("logical hash")
    if seen != set(docs):
        raise AuditError("orphan/missing docs")
    total = {"blocks": sum(c["blocks"] for c in counts.values()),
             **{k: sum(c[k] for c in counts.values()) for k in COUNT_FIELDS},
             "by_split": {s: {k: counts[s][k] for k in ("blocks",) + COUNT_FIELDS}
                          for s in SPLITS},
             "by_family": family_all, "by_language": language_all}
    if total != manifest["counts"]:
        raise AuditError("manifest counts")
    if counts["train"]["valid_tokens"] > MAX_TRAIN_VALID \
            or counts["validation"]["valid_tokens"] > MAX_VALIDATION_VALID \
            or counts["test"]["valid_tokens"] > MAX_TEST_VALID \
            or total["valid_tokens"] > MAX_TOTAL_VALID:
        raise AuditError("valid caps")
    return counts, total


def audit(root, manifest_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise AuditError("root")
    if not root.parent.is_dir():
        raise AuditError("root parent")
    manifest, manifest_bytes = read_json(root / "manifest.json", "manifest",
                                         MAX_MANIFEST_BYTES)
    manifest, sequence, shards = preflight_manifest(manifest, manifest_bytes,
                                                    manifest_sha)
    table = tokenizer(root, manifest)
    source_records, source_map, _, _ = source_metadata(root, manifest)
    sizes = check_file_set(root, shards, source_records)
    verify_sources(root, source_records)
    docs, _selected = documents(root, manifest, source_map, sequence)
    counts, total = verify_shards(root, manifest, shards, docs, table)
    return {
        "schema": "splot-stream-audit-v1",
        "manifest_sha256": manifest_sha,
        "file_bytes": sum(sizes.values()),
        "counts": manifest["counts"],
        "splits": {s: counts[s] for s in SPLITS},
        "sources": len(source_map),
        "documents": len(docs),
        "test_blocks_audited": counts["test"]["blocks"],
        "test_blocks_exposed": False,
        "bounded_io": {"mode": "incremental", "chunk_bytes": CHUNK_BYTES,
                       "full_source_reads": 0, "full_shard_reads": 0},
        "hashes": {
            "manifest": manifest_sha,
            "tokenizer": manifest["tokenizer_sha256"],
            "source_registry": manifest["source_registry_sha256"],
            "document_metadata": manifest["document_metadata_sha256"],
            "logical": manifest["logical_sha256"],
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: pilot_stream_audit.py <dataset-dir> <manifest-sha>")
    print(json.dumps(audit(sys.argv[1], sys.argv[2]), indent=2, sort_keys=True,
                     allow_nan=False))
