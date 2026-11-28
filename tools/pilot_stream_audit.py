"""Independent auditor for splot-pilot-stream-v1.

This script deliberately does not import pilot_stream or pilot_stream_producer.
It re-parses the canonical artifacts and recomputes coverage/counts/hashes with
stdlib code for an independent check. It never returns test token arrays.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

FORMAT = "splot-pilot-stream-v1"
VERSION = 1
SPLITS = ("train", "validation", "test")
PAD, BOS, EOS, IGNORE = 0, 1, 2, -100
CONTENT_MIN = 8
COUNT_FIELDS = ("valid_tokens", "content_tokens", "ntp_labels", "content_labels",
                "eos_labels", "offset2_labels", "content_target_bytes",
                "selected_bytes")
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_TOTAL_POSITIONS = 1_600_000
MAX_TRAIN_VALID = 1_000_000
MAX_VALIDATION_VALID = 50_000
MAX_TEST_VALID = 50_000
MAX_TOTAL_VALID = 1_100_000
MAX_DOCUMENTS = 20_000
MAX_SOURCES = 32
HEX = re.compile(r"^[0-9a-f]+$")


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


def need_obj(value, name, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise AuditError(f"{name} schema invalid")
    return value


def need_hex(value, name):
    if not isinstance(value, str) or len(value) != 64 or not HEX.fullmatch(value):
        raise AuditError(f"{name} hash invalid")


def read_json(path, name, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{name} not regular")
    if path.stat().st_size > limit:
        raise AuditError(f"{name} exceeds cap")
    data = path.read_bytes()
    value = json.loads(data.decode())
    if canonical(value) != data:
        raise AuditError(f"{name} not canonical")
    return value, data


def tokenizer(root, manifest):
    value, data = read_json(root / "tokenizer-bytes.json", "tokenizer", 4 * 1024 * 1024)
    if sha(data) != manifest["tokenizer_sha256"]:
        raise AuditError("tokenizer hash")
    table = value["token_bytes"]
    if len(table) != manifest["vocab_size"]:
        raise AuditError("tokenizer length")
    special = {0, 1, 2, 3, 4, 5, 6, 7}
    seen = set()
    single = set()
    for index, entry in enumerate(table):
        if index in special:
            if entry != "":
                raise AuditError("special bytes")
            continue
        raw = bytes.fromhex(entry)
        if not raw or raw in seen:
            raise AuditError("token bytes duplicate/empty")
        seen.add(raw)
        if len(raw) == 1:
            single.add(raw[0])
    if single != set(range(256)):
        raise AuditError("incomplete alphabet")
    return tuple(b"" if i in special else bytes.fromhex(table[i])
                 for i in range(len(table)))


def sources(root, manifest):
    value, data = read_json(root / "sources.json", "sources", 4 * 1024 * 1024)
    if sha(data) != manifest["source_registry_sha256"]:
        raise AuditError("source registry hash")
    records = value["sources"]
    if not 1 <= len(records) <= MAX_SOURCES:
        raise AuditError("source count")
    by_id = {}
    content = {}
    family_split = {}
    total = 0
    for source in records:
        sid = source["source_id"]
        if sid in by_id:
            raise AuditError("duplicate source")
        if source["raw_path"] != f"sources/{sid}.raw" or source["content_path"] != f"sources/{sid}.content":
            raise AuditError("source path")
        raw = (root / source["raw_path"]).read_bytes()
        body = (root / source["content_path"]).read_bytes()
        if len(raw) != source["raw_bytes"] or sha(raw) != source["raw_sha256"]:
            raise AuditError("raw hash")
        if len(body) != source["content_bytes"] or sha(body) != source["content_sha256"]:
            raise AuditError("content hash")
        body.decode("utf-8")
        family_split[source["family_id"]] = source["split"]
        by_id[sid] = source
        content[sid] = body
        total += len(raw) + len(body)
    if total > MAX_TOTAL_BYTES:
        raise AuditError("source bytes cap")
    return by_id, content, family_split


def documents(root, manifest, source_map, content_map):
    path = root / "documents.jsonl"
    data = path.read_bytes()
    if sha(data) != manifest["document_metadata_sha256"]:
        raise AuditError("document metadata hash")
    docs = {}
    selected = {sid: 0 for sid in source_map}
    groups = {}
    full_intervals = {sid: [] for sid in source_map}
    selected_intervals = {sid: [] for sid in source_map}
    last = -1
    for line in data.splitlines():
        doc = json.loads(line)
        if canonical(doc) != line:
            raise AuditError("document canonical")
        doc_id = need_int(doc["doc_id"], "doc_id")
        if doc_id <= last or doc_id in docs:
            raise AuditError("doc order/duplicate")
        last = doc_id
        source = source_map.get(doc["source_id"])
        if source is None:
            raise AuditError("unknown source")
        if (doc["family_id"] != source["family_id"] or doc["language"] != source["language"]
                or doc["split"] != source["split"]):
            raise AuditError("doc provenance")
        s0, s1 = doc["source_byte_start"], doc["source_byte_end"]
        g0, g1 = doc["segment_byte_start"], doc["segment_byte_end"]
        body = content_map[doc["source_id"]]
        if not 0 <= s0 < s1 <= len(body) or not s0 <= g0 < g1 <= s1:
            raise AuditError("doc interval")
        full = body[s0:s1]
        segment = body[g0:g1]
        if sha(full) != doc["full_document_sha256"] or sha(segment) != doc["segment_sha256"]:
            raise AuditError("doc hash")
        full.decode("utf-8")
        segment.decode("utf-8")
        if doc["selected_bytes"] != g1 - g0 or doc["token_count"] < 1:
            raise AuditError("doc selected/token count")
        groups.setdefault((doc["source_id"], s0, s1, doc["full_document_sha256"]), []).append(doc)
        selected_intervals[doc["source_id"]].append((g0, g1))
        selected[doc["source_id"]] += g1 - g0
        docs[doc_id] = doc
    if len(docs) > MAX_DOCUMENTS:
        raise AuditError("doc count")
    for (sid, s0, s1, _), members in groups.items():
        full_intervals[sid].append((s0, s1))
        count = {m["segment_count"] for m in members}
        if len(count) != 1:
            raise AuditError("segment_count")
        count = count.pop()
        ordered = sorted(members, key=lambda d: (d["segment_byte_start"], d["segment_index"]))
        indices = [d["segment_index"] for d in ordered]
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise AuditError("segment order")
        intervals = [(d["segment_byte_start"], d["segment_byte_end"]) for d in ordered]
        for (a, b), (c, d) in zip(intervals, intervals[1:]):
            if b > c:
                raise AuditError("segment overlap")
        if len(members) == count:
            if indices != list(range(count)):
                raise AuditError("complete indices")
            if intervals[0][0] != s0 or intervals[-1][1] != s1 \
                    or any(b != c for (_, b), (c, _) in zip(intervals, intervals[1:])):
                raise AuditError("complete partition")
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
        if not isinstance(block[key], list) or len(block[key]) != sequence:
            raise AuditError("block array length")
    valid_n = sum(valid)
    if valid_n < 3 or valid != [True] * valid_n + [False] * (sequence - valid_n):
        raise AuditError("valid mask")
    for index in range(valid_n, sequence):
        if (input_ids[index] != PAD or document_ids[index] != -1 or positions[index] != 0
                or valid[index] is not False or control[index] is not True
                or lengths[index] != 0 or offsets[index] != -1):
            raise AuditError("padding")
    for key in ("1", "2"):
        if not isinstance(labels[key], list) or len(labels[key]) != sequence:
            raise AuditError("label length")
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
        for index in range(start + 1, end - 1):
            token = input_ids[index]
            if not is_int(token) or not CONTENT_MIN <= token < len(table):
                raise AuditError("content token")
            if control[index] is not False or lengths[index] != len(table[token]):
                raise AuditError("content coverage")
            if offsets[index] != cumulative:
                raise AuditError("offset")
            cumulative += len(table[token])
            content_count += 1
        if cumulative != doc["selected_bytes"] or content_count != doc["token_count"]:
            raise AuditError("segment coverage")
        payload = b"".join(table[input_ids[index]] for index in range(start + 1, end - 1))
        if sha(payload) != doc["segment_sha256"]:
            raise AuditError("segment hash")
        payload.decode("utf-8")
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
        group = total["by_family"].setdefault(doc["family_id"], {k: 0 for k in COUNT_FIELDS})
        lang = total["by_language"].setdefault(doc["language"], {k: 0 for k in COUNT_FIELDS})
        for key in COUNT_FIELDS:
            group[key] += entity[key]
            lang[key] += entity[key]
        start = end
    return total


def audit(root, manifest_sha):
    root = Path(root)
    manifest, manifest_bytes = read_json(root / "manifest.json", "manifest", 2 * 1024 * 1024)
    if sha(manifest_bytes) != manifest_sha:
        raise AuditError("manifest hash")
    table = tokenizer(root, manifest)
    source_map, content_map, _ = sources(root, manifest)
    docs, selected = documents(root, manifest, source_map, content_map)
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise AuditError("shards")
    counts = {split: {"blocks": 0, **{k: 0 for k in COUNT_FIELDS},
                      "family_content_tokens": {}, "family_content_bytes": {},
                      "language_content_tokens": {}, "language_content_bytes": {}}
              for split in SPLITS}
    total_bytes = 0
    total_blocks = 0
    total_positions = 0
    seen = set()
    family_all = {}
    language_all = {}
    logical = hashlib.sha256()
    split_seen = set()
    last_split = -1
    for shard in shards:
        split = shard["split"]
        if split not in SPLITS:
            raise AuditError("shard split")
        idx = SPLITS.index(split)
        if idx < last_split:
            raise AuditError("shard order")
        last_split = idx
        split_seen.add(split)
        path = root / shard["path"]
        data = path.read_bytes()
        if len(data) != shard["bytes"] or sha(data) != shard["sha256"]:
            raise AuditError("shard hash")
        logical.update(data)
        total_bytes += len(data)
        total_blocks += shard["blocks"]
        total_positions += shard["blocks"] * manifest["sequence_length"]
        if total_bytes > MAX_TOTAL_BYTES or total_positions > MAX_TOTAL_POSITIONS:
            raise AuditError("aggregate cap")
        sc = {"blocks": 0, **{k: 0 for k in COUNT_FIELDS},
              "family_content_tokens": {}, "family_content_bytes": {},
              "language_content_tokens": {}, "language_content_bytes": {}}
        for line in data.splitlines():
            block = json.loads(line)
            if canonical(block) != line:
                raise AuditError("block canonical")
            item = block_counts(block, docs, table, manifest["sequence_length"],
                                split, seen)
            sc["blocks"] += 1
            for key in COUNT_FIELDS:
                sc[key] += item[key]
            for family, entity in item["by_family"].items():
                target = family_all.setdefault(family, {k: 0 for k in COUNT_FIELDS})
                for key in COUNT_FIELDS:
                    target[key] += entity[key]
                sc["family_content_tokens"][family] = (
                    sc["family_content_tokens"].get(family, 0) + entity["content_tokens"])
                sc["family_content_bytes"][family] = (
                    sc["family_content_bytes"].get(family, 0)
                    + entity["content_target_bytes"])
            for language, entity in item["by_language"].items():
                target = language_all.setdefault(language, {k: 0 for k in COUNT_FIELDS})
                for key in COUNT_FIELDS:
                    target[key] += entity[key]
                sc["language_content_tokens"][language] = (
                    sc["language_content_tokens"].get(language, 0) + entity["content_tokens"])
                sc["language_content_bytes"][language] = (
                    sc["language_content_bytes"].get(language, 0)
                    + entity["content_target_bytes"])
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
        for family, value in sc["family_content_tokens"].items():
            counts[split]["family_content_tokens"][family] = (
                counts[split]["family_content_tokens"].get(family, 0) + value)
        for family, value in sc["family_content_bytes"].items():
            counts[split]["family_content_bytes"][family] = (
                counts[split]["family_content_bytes"].get(family, 0) + value)
        for language, value in sc["language_content_tokens"].items():
            counts[split]["language_content_tokens"][language] = (
                counts[split]["language_content_tokens"].get(language, 0) + value)
        for language, value in sc["language_content_bytes"].items():
            counts[split]["language_content_bytes"][language] = (
                counts[split]["language_content_bytes"].get(language, 0) + value)
    if split_seen != set(SPLITS):
        raise AuditError("missing split")
    if logical.hexdigest() != manifest["logical_sha256"]:
        raise AuditError("logical hash")
    if seen != set(docs):
        raise AuditError("orphan/missing docs")
    if sum(shard["selected_bytes"] for shard in shards) != sum(selected.values()):
        raise AuditError("selected reconciliation")
    if counts["train"]["valid_tokens"] > MAX_TRAIN_VALID \
            or counts["validation"]["valid_tokens"] > MAX_VALIDATION_VALID \
            or counts["test"]["valid_tokens"] > MAX_TEST_VALID \
            or sum(c["valid_tokens"] for c in counts.values()) > MAX_TOTAL_VALID:
        raise AuditError("valid caps")
    total = {"blocks": sum(c["blocks"] for c in counts.values()),
             **{k: sum(c[k] for c in counts.values()) for k in COUNT_FIELDS},
             "by_split": {s: {k: counts[s][k] for k in ("blocks",) + COUNT_FIELDS}
                          for s in SPLITS},
             "by_family": family_all, "by_language": language_all}
    if total != manifest["counts"]:
        raise AuditError("manifest counts")
    return {
        "schema": "splot-stream-audit-v1",
        "manifest_sha256": manifest_sha,
        "file_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "counts": manifest["counts"],
        "splits": {s: counts[s] for s in SPLITS},
        "sources": len(source_map),
        "documents": len(docs),
        "test_blocks_audited": counts["test"]["blocks"],
        "test_blocks_exposed": False,
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
