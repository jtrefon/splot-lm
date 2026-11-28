"""Deterministic synthetic producer for splot-pilot-stream-v1.

Writes source raw/content files, canonical metadata and block shards into a new
private staging directory, verifies the result, then atomically publishes it.
No acquisition, tokenizer fitting or training.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pilot_stream as stream


class ProducerError(ValueError):
    """Invalid producer input or output path."""


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
class StreamSpec:
    sequence_length: int
    vocab_size: int
    token_bytes: tuple[bytes, ...]
    sources: tuple[SourceSpec, ...]
    segments: tuple[SegmentSpec, ...]
    shards_per_split: dict


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _int(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum:
        raise ProducerError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProducerError(f"{name} exceeds {maximum}")
    return value


def _str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ProducerError(f"{name} must be nonempty")
    return value


def _tokenizer(spec: StreamSpec) -> tuple[bytes, ...]:
    sequence = _int(spec.sequence_length, "sequence_length",
                    stream.SEQUENCE_MIN, stream.SEQUENCE_MAX)
    del sequence
    vocab = _int(spec.vocab_size, "vocab_size", stream.VOCAB_MIN, stream.VOCAB_MAX)
    if not isinstance(spec.token_bytes, (tuple, list)):
        raise ProducerError("token_bytes must be a sequence")
    if len(spec.token_bytes) != vocab:
        raise ProducerError("token_bytes length does not match vocab_size")
    table = []
    seen = {}
    single = set()
    special = {stream.PAD, stream.BOS, stream.EOS, stream.UNK, *stream.RESERVED}
    for token_id, value in enumerate(spec.token_bytes):
        if not isinstance(value, bytes):
            raise ProducerError(f"token_bytes[{token_id}] must be bytes")
        if token_id in special:
            if value != b"":
                raise ProducerError("special token bytes must be empty")
            table.append(value)
            continue
        if not value or len(value) > stream.MAX_SOURCE_BYTES:
            raise ProducerError("content token bytes are invalid")
        if value in seen:
            raise ProducerError("duplicate content token bytes")
        seen[value] = token_id
        if len(value) == 1:
            single.add(value[0])
        table.append(value)
    if single != set(range(256)):
        raise ProducerError("tokenizer needs the complete 256-byte alphabet")
    return tuple(table)


def _sources(spec: StreamSpec) -> tuple[dict, dict, dict]:
    if not isinstance(spec.sources, (tuple, list)) or not 1 <= len(spec.sources) <= stream.MAX_SOURCES:
        raise ProducerError("source count is invalid")
    by_id = {}
    family_split = {}
    family_language = {}
    split_seen = set()
    for source in spec.sources:
        if not isinstance(source, SourceSpec):
            raise ProducerError("sources must be SourceSpec")
        source_id = _str(source.source_id, "source_id")
        if not stream._SAFE.fullmatch(source_id):
            raise ProducerError("source_id is unsafe")
        family = _str(source.family_id, "family_id")
        if source.language not in stream.LANGUAGES or source.split not in stream.SPLIT_ORDER:
            raise ProducerError("source language/split is invalid")
        for key in ("source_url", "revision", "rights_evidence_url"):
            _str(getattr(source, key), key)
        raw = source.raw
        content = source.content
        if not isinstance(raw, bytes) or not raw or not isinstance(content, bytes) or not content:
            raise ProducerError("source bytes are empty or invalid")
        if len(raw) > stream.MAX_SOURCE_BYTES or len(content) > stream.MAX_SOURCE_BYTES:
            raise ProducerError("source bytes exceed cap")
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProducerError("source content must be UTF-8") from exc
        if source_id in by_id:
            raise ProducerError("duplicate source_id")
        if family in family_split and (family_split[family] != source.split
                                      or family_language[family] != source.language):
            raise ProducerError("family crosses split or language")
        family_split[family] = source.split
        family_language[family] = source.language
        by_id[source_id] = source
        split_seen.add(source.split)
    if split_seen != set(stream.SPLIT_ORDER):
        raise ProducerError("all three splits need a source")
    return by_id, family_split, family_language


def _documents(spec: StreamSpec, sources: dict, family_split: dict) -> tuple[list[dict], dict]:
    if not isinstance(spec.segments, (tuple, list)) or not 1 <= len(spec.segments) <= stream.MAX_DOCUMENTS:
        raise ProducerError("segment count is invalid")
    docs = []
    selected = {source_id: 0 for source_id in sources}
    seen_doc = set()
    seen_seg = set()
    groups = {}
    for segment in spec.segments:
        if not isinstance(segment, SegmentSpec):
            raise ProducerError("segments must be SegmentSpec")
        doc_id = _int(segment.doc_id, "doc_id")
        segment_id = _str(segment.segment_id, "segment_id")
        if doc_id in seen_doc or segment_id in seen_seg:
            raise ProducerError("duplicate document/segment ID")
        seen_doc.add(doc_id)
        seen_seg.add(segment_id)
        source = sources.get(segment.source_id)
        if source is None:
            raise ProducerError("unknown source_id")
        content = source.content
        s0 = _int(segment.source_byte_start, "source_byte_start")
        s1 = _int(segment.source_byte_end, "source_byte_end")
        g0 = _int(segment.segment_byte_start, "segment_byte_start")
        g1 = _int(segment.segment_byte_end, "segment_byte_end")
        if not 0 <= s0 < s1 <= len(content) or not s0 <= g0 < g1 <= s1:
            raise ProducerError("document interval is invalid")
        if not isinstance(segment.token_ids, (tuple, list)) or not segment.token_ids:
            raise ProducerError("token_ids must be a nonempty sequence")
        token_ids = tuple(segment.token_ids)
        if len(token_ids) + 2 > spec.sequence_length:
            raise ProducerError("segment does not fit sequence_length")
        if any(type(t) is not int or not stream.CONTENT_MIN <= t < spec.vocab_size
               for t in token_ids):
            raise ProducerError("token ID is out of range")
        full = content[s0:s1]
        segment_bytes = content[g0:g1]
        if b"".join(spec.token_bytes[t] for t in token_ids) != segment_bytes:
            raise ProducerError("token bytes do not equal segment")
        try:
            full.decode("utf-8", errors="strict")
            segment_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProducerError("document bytes are not UTF-8") from exc
        index = _int(segment.segment_index, "segment_index")
        count = _int(segment.segment_count, "segment_count", 1)
        if index >= count:
            raise ProducerError("segment_index is outside segment_count")
        doc = {
            "doc_id": doc_id,
            "segment_id": segment_id,
            "source_id": source.source_id,
            "family_id": source.family_id,
            "language": source.language,
            "split": source.split,
            "source_byte_start": s0,
            "source_byte_end": s1,
            "segment_byte_start": g0,
            "segment_byte_end": g1,
            "full_document_sha256": _sha(full),
            "segment_sha256": _sha(segment_bytes),
            "segment_index": index,
            "segment_count": count,
            "truncated": (g1 - g0) < (s1 - s0),
            "selected_bytes": g1 - g0,
            "token_count": len(token_ids),
        }
        if family_split[source.family_id] != source.split:
            raise ProducerError("family split mismatch")
        groups.setdefault((source.source_id, s0, s1, doc["full_document_sha256"]), []).append(doc)
        selected[source.source_id] += doc["selected_bytes"]
        docs.append(doc)
    full_intervals = {source_id: [] for source_id in sources}
    selected_intervals = {source_id: [] for source_id in sources}
    for (source_id, s0, s1, _), members in groups.items():
        full_intervals[source_id].append((s0, s1))
        counts = {d["segment_count"] for d in members}
        if len(counts) != 1:
            raise ProducerError("segment_count differs in group")
        count = counts.pop()
        ordered = sorted(members, key=lambda d: (
            d["segment_byte_start"], d["segment_byte_end"], d["segment_index"]))
        indices = [d["segment_index"] for d in ordered]
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise ProducerError("segment indices are not ordered")
        intervals = [(d["segment_byte_start"], d["segment_byte_end"]) for d in ordered]
        for (a, b), (c, d) in zip(intervals, intervals[1:]):
            if b > c:
                raise ProducerError("document segments overlap")
        selected_intervals[source_id].extend(intervals)
        if len(members) == count:
            if indices != list(range(count)):
                raise ProducerError("complete indices are not ordered")
            if intervals[0][0] != s0 or intervals[-1][1] != s1 \
                    or any(b != c for (_, b), (c, _) in zip(intervals, intervals[1:])):
                raise ProducerError("complete document does not partition source")
            if count == 1 and (ordered[0]["segment_index"] != 0 or ordered[0]["truncated"]):
                raise ProducerError("single full segment must be index 0")
        elif any(not d["truncated"] for d in members):
            raise ProducerError("partial selection must be truncated")
    for source_id in sources:
        for intervals in (full_intervals[source_id], selected_intervals[source_id]):
            ordered = sorted(intervals)
            for (a, b), (c, d) in zip(ordered, ordered[1:]):
                if b > c:
                    raise ProducerError("source intervals overlap")
    return docs, selected


def _block(segments: list[SegmentSpec], docs: dict[int, dict],
           table: tuple[bytes, ...], sequence: int) -> dict:
    arrays = {key: [] for key in ("input_ids", "document_ids", "position_ids",
                                  "valid", "control_mask", "token_byte_lengths",
                                  "token_byte_offsets")}
    labels = {"1": [], "2": []}
    for segment in segments:
        tokens = list(segment.token_ids)
        seg_input = [stream.BOS] + tokens + [stream.EOS]
        length = len(seg_input)
        arrays["input_ids"].extend(seg_input)
        arrays["document_ids"].extend([segment.doc_id] * length)
        arrays["position_ids"].extend(range(length))
        arrays["valid"].extend([True] * length)
        arrays["control_mask"].extend([True] + [False] * len(tokens) + [True])
        lengths = [0] + [len(table[t]) for t in tokens] + [0]
        offsets = []
        cursor = 0
        for token in tokens:
            offsets.append(cursor)
            cursor += len(table[token])
        arrays["token_byte_lengths"].extend(lengths)
        arrays["token_byte_offsets"].extend([0] + offsets + [cursor])
        for offset in (1, 2):
            for index in range(length):
                labels[str(offset)].append(
                    seg_input[index + offset] if index + offset < length else stream.IGNORE)
    if len(arrays["input_ids"]) > sequence:
        raise ProducerError("block exceeds sequence")
    pad = sequence - len(arrays["input_ids"])
    arrays["input_ids"].extend([stream.PAD] * pad)
    arrays["document_ids"].extend([-1] * pad)
    arrays["position_ids"].extend([0] * pad)
    arrays["valid"].extend([False] * pad)
    arrays["control_mask"].extend([True] * pad)
    arrays["token_byte_lengths"].extend([0] * pad)
    arrays["token_byte_offsets"].extend([-1] * pad)
    for offset in (1, 2):
        labels[str(offset)].extend([stream.IGNORE] * pad)
    return {**arrays, "labels": labels}


def _pack(spec: StreamSpec, sources: dict) -> dict:
    by_split = {split: [] for split in stream.SPLIT_ORDER}
    for segment in spec.segments:
        by_split[sources[segment.source_id].split].append(segment)
    packed = {}
    for split in stream.SPLIT_ORDER:
        blocks = []
        current = []
        length = 0
        for segment in by_split[split]:
            size = len(segment.token_ids) + 2
            if current and length + size > spec.sequence_length:
                blocks.append(current)
                current = []
                length = 0
            current.append(segment)
            length += size
            if length == spec.sequence_length:
                blocks.append(current)
                current = []
                length = 0
        if current:
            blocks.append(current)
        if not blocks:
            raise ProducerError(f"split {split} has no blocks")
        packed[split] = blocks
    return packed


def _entity(segment: SegmentSpec, doc: dict, table: tuple[bytes, ...]) -> dict:
    content = len(segment.token_ids)
    selected = sum(len(table[t]) for t in segment.token_ids)
    return {"valid_tokens": content + 2, "content_tokens": content,
            "ntp_labels": content + 1, "content_labels": content,
            "eos_labels": 1, "offset2_labels": content,
            "content_target_bytes": selected, "selected_bytes": selected}


def _counts(packed: dict, docs: dict[int, dict], table: tuple[bytes, ...]):
    by_split = {}
    by_family = {}
    by_language = {}
    total = {"blocks": 0, **{k: 0 for k in stream.COUNT_FIELDS},
             "by_split": {}, "by_language": {}, "by_family": {}}
    for split in stream.SPLIT_ORDER:
        entity = {"blocks": len(packed[split]), **{k: 0 for k in stream.COUNT_FIELDS}}
        for block_segments in packed[split]:
            for segment in block_segments:
                doc = docs[segment.doc_id]
                item = _entity(segment, doc, table)
                for key in stream.COUNT_FIELDS:
                    entity[key] += item[key]
                for group, name in ((by_family, doc["family_id"]),
                                    (by_language, doc["language"])):
                    target = group.setdefault(name, {k: 0 for k in stream.COUNT_FIELDS})
                    for key in stream.COUNT_FIELDS:
                        target[key] += item[key]
        total["blocks"] += entity["blocks"]
        for key in stream.COUNT_FIELDS:
            total[key] += entity[key]
        by_split[split] = entity
    total["by_split"] = by_split
    total["by_family"] = by_family
    total["by_language"] = by_language
    return total


def _text_bytes(pattern: str, length: int) -> bytes:
    text = pattern
    while len(text.encode("utf-8")) < length:
        text += pattern
    while len(text.encode("utf-8")) > length:
        text = text[:-1]
    data = text.encode("utf-8")
    if len(data) < length:
        data += b" " * (length - len(data))
    return data


def make_scale_spec() -> StreamSpec:
    """Deterministic near-1M synthetic plan: 8/4/4 families, 512 positions."""
    table = tuple([b""] * 8 + [bytes([value]) for value in range(256)])
    en_pattern = "the quick brown fox jumps over the lazy dog. "
    pl_pattern = "Zażółć gęślą jaźń. "
    families = []
    for split, count in (("train", 8), ("validation", 4), ("test", 4)):
        for index in range(count):
            language = "en" if index < count // 2 else "pl"
            families.append((f"fam-{split}-{index:02d}", split, language))
    sources = []
    segments = []
    doc_id = 0
    for family_id, split, language in families:
        if split == "train":
            segment_target = 1400
        else:
            segment_target = 140
        pattern = en_pattern if language == "en" else pl_pattern
        full_length = 120
        split_length = 60
        content_parts = []
        offset = 0
        produced = 0
        while produced < segment_target:
            remaining = segment_target - produced
            if remaining >= 3:
                # One full document and one two-segment document.
                for kind in ("full", "partial"):
                    if kind == "full":
                        doc_bytes = _text_bytes(pattern, full_length)
                        content_parts.append(doc_bytes)
                        segments.append(SegmentSpec(
                            doc_id=doc_id,
                            segment_id=f"seg-{doc_id:08d}-0",
                            source_id=f"src-{family_id}",
                            source_byte_start=offset,
                            source_byte_end=offset + len(doc_bytes),
                            segment_byte_start=offset,
                            segment_byte_end=offset + len(doc_bytes),
                            segment_index=0,
                            segment_count=1,
                            token_ids=tuple(8 + b for b in doc_bytes),
                        ))
                        offset += len(doc_bytes)
                        doc_id += 1
                        produced += 1
                    else:
                        doc_bytes = (_text_bytes(pattern, split_length)
                                     + _text_bytes(pattern, split_length))
                        content_parts.append(doc_bytes)
                        first = doc_bytes[:split_length]
                        second = doc_bytes[split_length:]
                        assert len(first) == split_length and len(second) == split_length
                        for index, chunk in enumerate((first, second)):
                            segments.append(SegmentSpec(
                                doc_id=doc_id,
                                segment_id=f"seg-{doc_id:08d}-{index}",
                                source_id=f"src-{family_id}",
                                source_byte_start=offset,
                                source_byte_end=offset + len(doc_bytes),
                                segment_byte_start=offset + index * split_length,
                                segment_byte_end=offset + (index + 1) * split_length,
                                segment_index=index,
                                segment_count=2,
                                token_ids=tuple(8 + b for b in chunk),
                            ))
                            doc_id += 1
                        offset += len(doc_bytes)
                        produced += 2
            else:
                doc_bytes = _text_bytes(pattern, full_length)
                content_parts.append(doc_bytes)
                segments.append(SegmentSpec(
                    doc_id=doc_id,
                    segment_id=f"seg-{doc_id:08d}-0",
                    source_id=f"src-{family_id}",
                    source_byte_start=offset,
                    source_byte_end=offset + len(doc_bytes),
                    segment_byte_start=offset,
                    segment_byte_end=offset + len(doc_bytes),
                    segment_index=0,
                    segment_count=1,
                    token_ids=tuple(8 + b for b in doc_bytes),
                ))
                offset += len(doc_bytes)
                doc_id += 1
                produced += 1
        content = b"".join(content_parts)
        sources.append(SourceSpec(
            source_id=f"src-{family_id}",
            family_id=family_id,
            language=language,
            split=split,
            source_url=f"https://example.invalid/{family_id}",
            revision="synthetic-stream-v1",
            rights_evidence_url="https://example.invalid/rights",
            raw=b"RAW:" + content,
            content=content,
        ))
    return StreamSpec(
        sequence_length=512,
        vocab_size=264,
        token_bytes=table,
        sources=tuple(sources),
        segments=tuple(segments),
        shards_per_split={"train": 4, "validation": 1, "test": 1},
    )


def write_dataset(spec: StreamSpec, output_dir, *, verify: bool = True) -> dict:
    """Build in a new staging directory, verify, then publish atomically."""
    table = _tokenizer(spec)
    sources, family_split, _ = _sources(spec)
    docs_list, selected = _documents(spec, sources, family_split)
    docs = {doc["doc_id"]: doc for doc in docs_list}
    packed = _pack(spec, sources)
    counts = _counts(packed, docs, table)
    if counts["by_split"]["train"]["valid_tokens"] > stream.MAX_TRAIN_VALID:
        raise ProducerError("train valid tokens exceed cap")
    if counts["by_split"]["validation"]["valid_tokens"] > stream.MAX_VALIDATION_VALID:
        raise ProducerError("validation valid tokens exceed cap")
    if counts["by_split"]["test"]["valid_tokens"] > stream.MAX_TEST_VALID:
        raise ProducerError("test valid tokens exceed cap")
    if counts["valid_tokens"] > stream.MAX_TOTAL_VALID:
        raise ProducerError("total valid tokens exceed cap")
    total_positions = sum(counts["by_split"][split]["blocks"] * spec.sequence_length
                          for split in stream.SPLIT_ORDER)
    if total_positions > stream.MAX_TOTAL_POSITIONS:
        raise ProducerError("block positions exceed cap")
    if sum(len(source.raw) + len(source.content) for source in spec.sources) > stream.MAX_TOTAL_BYTES:
        raise ProducerError("source bytes exceed aggregate cap")
    out = Path(output_dir)
    if out.exists() or out.is_symlink():
        raise ProducerError("output path already exists")
    staging = out.parent / (out.name + ".staging")
    if staging.exists() or staging.is_symlink():
        raise ProducerError("staging path already exists")
    try:
        staging.mkdir()
        (staging / "sources").mkdir()
        source_records = []
        for source in sorted(spec.sources, key=lambda s: s.source_id):
            raw_name = f"sources/{source.source_id}.raw"
            content_name = f"sources/{source.source_id}.content"
            (staging / raw_name).write_bytes(source.raw)
            (staging / content_name).write_bytes(source.content)
            source_records.append({
                "source_id": source.source_id, "family_id": source.family_id,
                "language": source.language, "split": source.split,
                "source_url": source.source_url, "revision": source.revision,
                "rights_evidence_url": source.rights_evidence_url,
                "raw_path": raw_name, "raw_sha256": _sha(source.raw),
                "raw_bytes": len(source.raw), "content_path": content_name,
                "content_sha256": _sha(source.content),
                "content_bytes": len(source.content),
                "selected_bytes": selected[source.source_id],
            })
        source_registry = {"format": stream.SOURCE_FORMAT,
                           "version": stream.SOURCE_VERSION,
                           "sources": source_records}
        source_bytes = stream._canonical(source_registry)
        (staging / "sources.json").write_bytes(source_bytes)
        doc_bytes = b"".join(stream._canonical(doc) + b"\n"
                             for doc in sorted(docs_list, key=lambda d: d["doc_id"]))
        (staging / "documents.jsonl").write_bytes(doc_bytes)
        tokenizer = {"format": stream.TOKENIZER_FORMAT,
                     "version": stream.TOKENIZER_VERSION,
                     "actual_vocab_size": spec.vocab_size,
                     "special_token_ids": json.loads(json.dumps(stream.SPECIAL_IDS)),
                     "token_bytes": [value.hex() for value in table]}
        tokenizer_bytes = stream._canonical(tokenizer)
        (staging / "tokenizer-bytes.json").write_bytes(tokenizer_bytes)
        shards = []
        shard_payloads = {}
        for split in stream.SPLIT_ORDER:
            blocks = packed[split]
            n_shards = spec.shards_per_split.get(split, 1)
            if type(n_shards) is not int or n_shards < 1 or n_shards > len(blocks):
                raise ProducerError("invalid shard count")
            per = (len(blocks) + n_shards - 1) // n_shards
            for shard_index in range(n_shards):
                chunk = blocks[shard_index * per:(shard_index + 1) * per]
                if not chunk:
                    continue
                path = f"{split}-{shard_index:03d}.jsonl"
                shard_entity = {"blocks": len(chunk), **{k: 0 for k in stream.COUNT_FIELDS}}
                family_tokens = {}
                family_bytes = {}
                language_tokens = {}
                language_bytes = {}
                file = staging / path
                hasher = hashlib.sha256()
                with file.open("wb") as handle:
                    for block_segments in chunk:
                        block = _block(block_segments, docs, table, spec.sequence_length)
                        line = stream._canonical(block) + b"\n"
                        if len(line) > stream.MAX_SHARD_LINE:
                            raise ProducerError("block line exceeds cap")
                        handle.write(line)
                        hasher.update(line)
                        for segment in block_segments:
                            doc = docs[segment.doc_id]
                            item = _entity(segment, doc, table)
                            for key in stream.COUNT_FIELDS:
                                shard_entity[key] += item[key]
                            family_tokens[doc["family_id"]] = (
                                family_tokens.get(doc["family_id"], 0) + item["content_tokens"])
                            family_bytes[doc["family_id"]] = (
                                family_bytes.get(doc["family_id"], 0) + item["content_target_bytes"])
                            language_tokens[doc["language"]] = (
                                language_tokens.get(doc["language"], 0) + item["content_tokens"])
                            language_bytes[doc["language"]] = (
                                language_bytes.get(doc["language"], 0) + item["content_target_bytes"])
                payload = file.read_bytes()
                shard_payloads[path] = payload
                shards.append({
                    "split": split, "path": path, "sha256": hasher.hexdigest(),
                    "bytes": len(payload), "blocks": shard_entity["blocks"],
                    **{key: shard_entity[key] for key in stream.COUNT_FIELDS},
                    "family_content_tokens": family_tokens,
                    "family_content_bytes": family_bytes,
                    "language_content_tokens": language_tokens,
                    "language_content_bytes": language_bytes,
                })
        manifest = {
            "format": stream.FORMAT, "version": stream.VERSION,
            "sequence_length": spec.sequence_length, "vocab_size": spec.vocab_size,
            "tokenizer_sha256": _sha(tokenizer_bytes),
            "tokenizer_special_ids": json.loads(json.dumps(stream.SPECIAL_IDS)),
            "source_registry_sha256": _sha(source_bytes),
            "document_metadata_sha256": _sha(doc_bytes),
            "shards": shards, "counts": counts,
            "logical_sha256": _sha(b"".join(shard_payloads[s["path"]] for s in shards)),
        }
        manifest_bytes = stream._canonical(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        total_files = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        if total_files > stream.MAX_TOTAL_BYTES:
            raise ProducerError("generated files exceed cap")
        manifest_sha = _sha(manifest_bytes)
        if verify:
            stream.load_stream_dataset(staging, manifest_sha)
        if out.exists() or out.is_symlink():
            raise ProducerError("output path appeared during build")
        os.replace(staging, out)
        return {"manifest_sha256": manifest_sha, "counts": counts,
                "shards": shards, "sources": source_records,
                "documents": docs_list,
                "generated_bytes": total_files}
    except Exception:
        if staging.exists() and not out.exists():
            shutil.rmtree(staging)
        raise
