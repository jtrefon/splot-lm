"""Bounded real bilingual pilot acquisition and materialization (T0016).

All raw text, tokenizer files and token blocks are local-only. The checked-in
registry contains source decisions, official URLs, pins and hashes, never text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

import production_tokenizer as pt

FORMAT = "splot-pilot-blocks-v1"
VERSION = 1
SEQUENCE_LENGTH = 512
PAD = 0
BOS = 1
EOS = 2
IGNORE = -100
CONTENT_START = 8
SPLITS = ("train", "validation", "test")
MAX_EXTRACTED_BYTES = 10 * 1024 * 1024
MAX_WORK_BYTES = 2621440
MAX_NETWORK_BYTES = 50 * 1024 * 1024
MAX_LOCAL_BYTES = 512 * 1024 * 1024
MAX_SEGMENTS = 10000
MAX_TEXT_SEGMENT_BYTES = 4096
MAX_BLOCK_CONTENT_TOKENS = 510
TOKENIZER_TRAIN_BYTES = 4 * 1024 * 1024
TRAIN_BLOCKS = 128
VALIDATION_BLOCKS = 16
TEST_BLOCKS = 16
MODEL_VOCAB_SIZE = 32768
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_TOKENIZER_BYTES = 8 * 1024 * 1024
MAX_SHARD_BYTES = 16 * 1024 * 1024
MAX_BLOCKS_PER_SPLIT = 128
ALLOWED_HOSTS = frozenset({"raw.githubusercontent.com", "wolnelektury.pl"})
SPECIAL_TOKENS = pt.SPECIAL_TOKENS
SOURCE_KEYS = frozenset(
    {
        "family_id",
        "language",
        "split",
        "content_sha256",
        "raw_sha256",
        "revision",
        "rights_evidence_url",
        "source_url",
    }
)
SHARD_KEYS = frozenset(
    {
        "split",
        "path",
        "sha256",
        "bytes",
        "blocks",
        "valid_tokens",
        "ntp_labels",
        "offset2_labels",
    }
)
BLOCK_KEYS = frozenset(
    {"input_ids", "document_ids", "position_ids", "valid", "labels"}
)
MANIFEST_KEYS = frozenset(
    {
        "format",
        "version",
        "sequence_length",
        "vocab_size",
        "tokenizer_sha256",
        "source_snapshot_sha256",
        "sources",
        "shards",
        "logical_sha256",
    }
)
SPLIT_ORDER = {split: index for index, split in enumerate(SPLITS)}
FAMILY_ORDER = {
    "train": (
        "se-pride-and-prejudice",
        "wl-lalka",
        "se-frankenstein",
        "wl-chlopi",
    ),
    "validation": ("se-alices-adventures", "wl-quo-vadis"),
    "test": ("se-moby-dick", "wl-faraon"),
}
FAMILY_SPLIT = {
    "se-pride-and-prejudice": "train",
    "se-frankenstein": "train",
    "wl-lalka": "train",
    "wl-chlopi": "train",
    "se-alices-adventures": "validation",
    "wl-quo-vadis": "validation",
    "se-moby-dick": "test",
    "wl-faraon": "test",
}
RESERVED_STRINGS = tuple(SPECIAL_TOKENS.values())
SKIP_XHTML_TAGS = frozenset(
    {"head", "script", "style", "nav", "figure", "figcaption", "svg", "math", "aside"}
)
BLOCK_XHTML_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "tr",
        "table",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "pre",
    }
)
WOLNE_BLOCK_TAGS = frozenset(
    {
        "akap",
        "akap_cd",
        "akap_dialog",
        "naglowek_czesc",
        "naglowek_rozdzial",
        "tytul_dziela",
        "strofa",
    }
)
WOLNE_SKIP_TAGS = frozenset(
    {
        "motyw",
        "pr",
        "uwaga",
        "pe",
        "begin",
        "end",
        "slowo_obce",
        "ilustracja",
        "abstrakt",
        "nota_red",
        "nota",
        "poslowie",
        "bibliografia",
        "indeks",
        "przypisy",
    }
)


class PilotDataError(ValueError):
    """Invalid source, text, block, manifest or dataset."""


class SourceExcluded(PilotDataError):
    """A source cannot be used under the reviewed rules."""


def _utf8(text: str, name: str = "text") -> bytes:
    if not isinstance(text, str):
        raise PilotDataError(f"{name} must be a string")
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise PilotDataError(f"{name} contains an invalid surrogate") from exc


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PilotDataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PilotDataError(f"nonfinite JSON constant: {value}")


def _parse_json(raw: bytes, name: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PilotDataError(f"{name} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_pairs,
        )
    except json.JSONDecodeError as exc:
        raise PilotDataError(f"{name} JSON is invalid: {exc}") from exc


def _safe_part_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise PilotDataError("part path must be nonempty")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise PilotDataError(f"unsafe part path: {value!r}")
    if "\\" in value or ":" in value:
        raise PilotDataError(f"unsafe part path: {value!r}")
    return path.as_posix()


def _require_https(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise PilotDataError(f"unsupported source URL: {url!r}")
    if parsed.username or parsed.password:
        raise PilotDataError("source URL must not contain credentials")
    return url


def _require_download_url(url: str) -> str:
    _require_https(url)
    if urllib.parse.urlsplit(url).hostname not in ALLOWED_HOSTS:
        raise PilotDataError(f"unsupported acquisition host: {url!r}")
    return url


def validate_registry(registry: Any) -> dict[str, Any]:
    if not isinstance(registry, dict):
        raise PilotDataError("registry must be an object")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PilotDataError("registry sources must be a nonempty list")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise PilotDataError("registry source must be an object")
        family = source.get("family_id")
        if not isinstance(family, str) or not family:
            raise PilotDataError("source family_id must be nonempty")
        if family in seen:
            raise PilotDataError(f"duplicate source family: {family!r}")
        seen.add(family)
        if source.get("language") not in ("en", "pl"):
            raise PilotDataError(f"unsupported language for {family!r}")
        if source.get("split") not in SPLITS:
            raise PilotDataError(f"invalid split for {family!r}")
        if source.get("split") != FAMILY_SPLIT.get(family):
            raise PilotDataError(f"family split decision mismatch for {family!r}")
        for key in (
            "title",
            "source_name",
            "item_id",
            "revision",
            "source_url",
            "rights_evidence_url",
            "authority_url",
        ):
            if not isinstance(source.get(key), str) or not source[key]:
                raise PilotDataError(f"{family!r} missing {key}")
        _require_https(source["source_url"])
        _require_https(source["rights_evidence_url"])
        _require_https(source["authority_url"])
        parts = source.get("parts")
        if not isinstance(parts, list) or not parts:
            raise PilotDataError(f"{family!r} parts must be nonempty")
        part_paths: set[str] = set()
        for part in parts:
            if not isinstance(part, dict):
                raise PilotDataError(f"{family!r} part must be an object")
            path = _safe_part_path(part.get("path"))
            if path in part_paths:
                raise PilotDataError(f"{family!r} duplicate part path: {path!r}")
            part_paths.add(path)
            if part.get("kind") not in ("xhtml", "xml"):
                raise PilotDataError(f"{family!r} part kind is invalid")
            _require_download_url(part.get("url"))
        excluded_paths = source.get("excluded_paths", [])
        if not isinstance(excluded_paths, list):
            raise PilotDataError(f"{family!r} excluded_paths must be a list")
        for path in excluded_paths:
            _safe_part_path(path)
    expected = set(FAMILY_SPLIT)
    if seen != expected:
        raise PilotDataError(
            f"registry families must be exactly {sorted(expected)}; got {sorted(seen)}"
        )
    for split in SPLITS:
        languages = {
            source["language"] for source in sources if source["split"] == split
        }
        if split != "test" and not {"en", "pl"} <= languages:
            raise PilotDataError(f"{split} must contain both languages")
        if split == "test" and not languages:
            raise PilotDataError("test split must contain at least one family")
    excluded = registry.get("excluded_candidates")
    if not isinstance(excluded, list) or not any(
        item.get("family_id") == "wl-pan-tadeusz"
        and item.get("genre") == "verse_epic"
        for item in excluded
        if isinstance(item, dict)
    ):
        raise PilotDataError("Pan Tadeusz must be explicitly excluded as verse")
    return registry


def load_registry(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise PilotDataError("registry is too large")
    return validate_registry(_parse_json(raw, "registry"))


def _fetch(url: str, limit: int, timeout: int = 60) -> bytes:
    _require_download_url(url)
    last_error: Exception | None = None
    for attempt in (1, 2):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "splot-t0016-acquisition/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final = response.geturl()
                _require_download_url(final)
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        announced = int(length)
                    except ValueError as exc:
                        raise PilotDataError(f"invalid Content-Length for {url}") from exc
                    if announced > limit:
                        raise PilotDataError(
                            f"announced size {announced} exceeds {limit}: {url}"
                        )
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise PilotDataError(f"download exceeds {limit} bytes: {url}")
                return data
        except urllib.error.HTTPError as exc:
            last_error = exc
            if attempt == 1 and exc.code in (500, 502, 503, 504):
                time.sleep(2)
                continue
            raise PilotDataError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == 1:
                time.sleep(2)
                continue
            raise PilotDataError(f"network failure for {url}: {exc}") from exc
    raise PilotDataError(f"network failure for {url}: {last_error}")


def _check_signature(data: bytes, kind: str, url: str) -> None:
    try:
        text = data[:512].decode("utf-8", errors="strict").lstrip("\ufeff \t\r\n")
    except UnicodeDecodeError as exc:
        raise PilotDataError(f"source is not UTF-8: {url}") from exc
    if kind == "xhtml" and not (text.startswith("<?xml") or text.startswith("<html")):
        raise PilotDataError(f"unexpected XHTML signature: {url}")
    if kind == "xml" and not (text.startswith("<?xml") or text.startswith("<utwor")):
        raise PilotDataError(f"unexpected XML signature: {url}")


def _write_acquisition(staging: Path, records: list[dict[str, Any]]) -> None:
    total = sum(
        part["bytes"] for source in records for part in source["parts"]
    )
    payload = {
        "schema_version": 1,
        "total_network_bytes": total,
        "network_limit_bytes": MAX_NETWORK_BYTES,
        "sources": sorted(records, key=lambda item: item["family_id"]),
    }
    (staging / "acquisition.json").write_bytes(_canonical_json(payload) + b"\n")


def acquire(
    registry_path: str | Path,
    staging_dir: str | Path,
    *,
    families: Iterable[str] | None = None,
    max_parts: int | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    requested = set(families) if families is not None else set(FAMILY_SPLIT)
    if not requested <= set(FAMILY_SPLIT):
        raise PilotDataError(f"unknown acquisition family: {sorted(requested - set(FAMILY_SPLIT))}")
    if max_parts is not None and (type(max_parts) is not int or max_parts < 1):
        raise PilotDataError("max_parts must be a positive integer")
    staging = Path(staging_dir)
    if staging.exists():
        existing = load_acquisition(staging)
        records = {source["family_id"]: dict(source) for source in existing["sources"]}
    else:
        staging.mkdir(parents=True)
        (staging / "raw").mkdir()
        records = {}
    raw_root = staging / "raw"
    new_parts = 0
    for source in registry["sources"]:
        family = source["family_id"]
        if family not in requested:
            continue
        existing_source = records.get(family)
        existing_parts = {
            part["path"]: part for part in existing_source["parts"]
        } if existing_source else {}
        family_root = raw_root / family
        family_root.mkdir(exist_ok=True)
        parts_by_path: dict[str, dict[str, Any]] = dict(existing_parts)
        for index, part in enumerate(source["parts"]):
            if max_parts is not None and new_parts >= max_parts:
                break
            if part["path"] in parts_by_path:
                continue
            existing_bytes = sum(
                item["bytes"] for record in records.values() for item in record["parts"]
            )
            new_source_bytes = sum(
                item["bytes"]
                for path, item in parts_by_path.items()
                if path not in existing_parts
            )
            remaining = MAX_NETWORK_BYTES - existing_bytes - new_source_bytes
            if remaining <= 0:
                raise PilotDataError("network budget exhausted")
            data = _fetch(part["url"], min(remaining, 8 * 1024 * 1024))
            _check_signature(data, part["kind"], part["url"])
            name = PurePosixPath(part["path"]).name
            local = family_root / f"{index:03d}-{name}"
            local.write_bytes(data)
            parts_by_path[part["path"]] = {
                "path": part["path"],
                "url": part["url"],
                "kind": part["kind"],
                "sha256": _sha(data),
                "bytes": len(data),
                "local_path": local.relative_to(staging).as_posix(),
            }
            new_parts += 1
        ordered_parts = [
            parts_by_path[part["path"]]
            for part in source["parts"]
            if part["path"] in parts_by_path
        ]
        if ordered_parts:
            records[family] = {
                "family_id": family,
                "language": source["language"],
                "split": source["split"],
                "parts": ordered_parts,
            }
        _write_acquisition(staging, list(records.values()))
        if max_parts is not None and new_parts >= max_parts:
            break
    _write_acquisition(staging, list(records.values()))
    result = load_acquisition(staging)
    return result


def load_acquisition(staging_dir: str | Path) -> dict[str, Any]:
    staging = Path(staging_dir)
    p = staging / "acquisition.json"
    if not p.is_file() or p.is_symlink():
        raise PilotDataError("acquisition.json is required; run acquire first")
    raw = p.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise PilotDataError("acquisition metadata is too large")
    data = _parse_json(raw, "acquisition")
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise PilotDataError("acquisition schema is invalid")
    total = data.get("total_network_bytes")
    if type(total) is not int or not 0 <= total <= MAX_NETWORK_BYTES:
        raise PilotDataError("acquisition network total is invalid")
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise PilotDataError("acquisition sources are invalid")
    for source in sources:
        if source.get("family_id") not in FAMILY_SPLIT:
            raise PilotDataError("acquisition family is invalid")
        if source.get("split") != FAMILY_SPLIT[source["family_id"]]:
            raise PilotDataError("acquisition split is invalid")
        parts = source.get("parts")
        if not isinstance(parts, list) or not parts:
            raise PilotDataError("acquisition parts are missing")
        for part in parts:
            if not isinstance(part, dict) or not _is_hex64(part.get("sha256")):
                raise PilotDataError("acquisition part hash is invalid")
            local = _safe_part_path(part.get("local_path"))
            if not local.startswith("raw/"):
                raise PilotDataError("acquisition local path is invalid")
            if type(part.get("bytes")) is not int or part["bytes"] < 0:
                raise PilotDataError("acquisition part bytes are invalid")
    return data


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise PilotDataError("text must be a string")
    if "\x00" in text:
        raise PilotDataError("text contains NUL")
    _utf8(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip(" ") for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")


class _XhtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.depth += 1
        if self.depth > 256:
            raise PilotDataError("XHTML nesting is too deep")
        if tag in SKIP_XHTML_TAGS:
            self.skip += 1
        elif not self.skip:
            if tag in BLOCK_XHTML_TAGS:
                self.parts.append("\n\n")
            elif tag == "br":
                self.parts.append("\n")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() == "br" and not self.skip:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_XHTML_TAGS:
            self.skip = max(0, self.skip - 1)
        elif not self.skip and tag in BLOCK_XHTML_TAGS:
            self.parts.append("\n\n")
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def extract_xhtml(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PilotDataError("XHTML is not valid UTF-8") from exc
    parser = _XhtmlText()
    try:
        parser.feed(text)
        parser.close()
    except PilotDataError:
        raise
    except Exception as exc:
        raise PilotDataError(f"XHTML parsing failed: {exc}") from exc
    result = normalize_text("".join(parser.parts))
    if not result:
        raise PilotDataError("XHTML contains no main text")
    return result


def _wolne_text(element: ET.Element, depth: int = 0) -> str:
    if depth > 64:
        raise PilotDataError("XML nesting is too deep")
    if element.tag in WOLNE_SKIP_TAGS:
        return ""
    parts = [element.text or ""]
    for child in element:
        parts.append(_wolne_text(child, depth + 1))
        parts.append(child.tail or "")
    return "".join(parts)


def extract_wolne_xml(data: bytes) -> str:
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise PilotDataError("XML DTD/entities are not allowed")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise PilotDataError(f"XML parsing failed: {exc}") from exc
    container = next((element for element in root.iter() if element.tag == "powiesc"), None)
    if container is None:
        raise PilotDataError("Wolne Lektury XML has no prose container")
    parent = {child: element for element in container.iter() for child in element}

    def inside_skipped(element: ET.Element) -> bool:
        current = parent.get(element)
        while current is not None:
            if current.tag in WOLNE_SKIP_TAGS:
                return True
            current = parent.get(current)
        return False

    blocks: list[str] = []
    for element in container.iter():
        if element.tag not in WOLNE_BLOCK_TAGS:
            continue
        if inside_skipped(element):
            continue
        owner = parent.get(element)
        if owner is not None and owner.tag in WOLNE_BLOCK_TAGS:
            continue
        text = normalize_text(_wolne_text(element))
        if text:
            blocks.append(text)
    result = "\n\n".join(blocks)
    if not result:
        raise PilotDataError("Wolne Lektury XML contains no prose text")
    return result


@dataclass(frozen=True)
class RawPart:
    path: str
    url: str
    kind: str
    sha256: str
    bytes: int
    local_path: Path


@dataclass(frozen=True)
class TextSegment:
    family_id: str
    language: str
    split: str
    start: int
    end: int
    text: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ExtractedSource:
    family_id: str
    title: str
    language: str
    split: str
    revision: str
    source_url: str
    rights_evidence_url: str
    authority_url: str
    content_sha256: str
    raw_sha256: str
    raw_bytes: int
    content_bytes: int
    parts: tuple[RawPart, ...]
    segments: tuple[TextSegment, ...]


@dataclass(frozen=True)
class BlockCandidate:
    family_id: str
    language: str
    split: str
    start: int
    end: int
    content_ids: tuple[int, ...]
    text_bytes: int
    text_sha256: str


def source_raw_sha(parts: Sequence[RawPart]) -> str:
    payload = [
        {"path": part.path, "sha256": part.sha256, "bytes": part.bytes}
        for part in parts
    ]
    return _sha(_canonical_json(payload))


def segment_text(
    text: str,
    *,
    max_bytes: int = MAX_TEXT_SEGMENT_BYTES,
) -> tuple[TextSegment, ...]:
    if max_bytes < 1 or max_bytes > MAX_TEXT_SEGMENT_BYTES:
        raise PilotDataError("segment byte limit is invalid")
    if not text:
        return ()
    segments: list[TextSegment] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(total, start + max_bytes)
        chunk = text[start:end]
        while len(_utf8(chunk)) > max_bytes:
            excess = len(_utf8(chunk)) - max_bytes
            end -= max(1, excess)
            if end <= start:
                raise PilotDataError("cannot make a UTF-8 segment fit")
            chunk = text[start:end]
        paragraph = chunk.rfind("\n\n")
        if paragraph >= len(chunk) // 2:
            end = start + paragraph + 2
        else:
            floor = max(0, len(chunk) - max(1, len(chunk) // 4))
            for index in range(len(chunk) - 1, floor - 1, -1):
                if chunk[index].isspace():
                    end = start + index + 1
                    break
        if end <= start:
            end = min(total, start + 1)
        segment = text[start:end]
        if not segment:
            raise PilotDataError("empty text segment")
        data = _utf8(segment)
        if len(data) > max_bytes:
            raise PilotDataError("text segment exceeds byte limit")
        segments.append(
            TextSegment(
                "",
                "",
                "",
                start,
                end,
                segment,
                len(data),
                _sha(data),
            )
        )
        start = end
    if "".join(segment.text for segment in segments) != text:
        raise PilotDataError("text segmentation is not lossless")
    return tuple(segments)


def _bind_segments(
    family_id: str,
    language: str,
    split: str,
    segments: Sequence[TextSegment],
) -> tuple[TextSegment, ...]:
    return tuple(
        TextSegment(
            family_id,
            language,
            split,
            segment.start,
            segment.end,
            segment.text,
            segment.bytes,
            segment.sha256,
        )
        for segment in segments
    )


def _extract_source(
    source: Mapping[str, Any],
    acquisition_source: Mapping[str, Any],
    staging: Path,
) -> ExtractedSource:
    parts: list[RawPart] = []
    texts: list[str] = []
    for part in acquisition_source["parts"]:
        local = staging / part["local_path"]
        if not local.is_file() or local.is_symlink():
            raise PilotDataError(f"raw part is missing or not regular: {part['local_path']}")
        data = local.read_bytes()
        if len(data) != part["bytes"] or _sha(data) != part["sha256"]:
            raise PilotDataError(f"raw part hash mismatch: {part['local_path']}")
        kind = part["kind"]
        if kind == "xhtml":
            text = extract_xhtml(data)
        elif kind == "xml":
            text = extract_wolne_xml(data)
        else:
            raise PilotDataError(f"unsupported part kind: {kind!r}")
        texts.append(text)
        parts.append(
            RawPart(
                part["path"],
                part["url"],
                kind,
                part["sha256"],
                part["bytes"],
                local,
            )
        )
    text = normalize_text("\n\n".join(texts))
    if not text:
        raise SourceExcluded(f"{source['family_id']}: no extracted text")
    for reserved in RESERVED_STRINGS:
        if reserved in text:
            raise SourceExcluded(
                f"{source['family_id']}: reserved special spelling {reserved!r}"
            )
    content = _utf8(text)
    if len(content) > MAX_WORK_BYTES:
        raise SourceExcluded(
            f"{source['family_id']}: {len(content)} extracted bytes exceed {MAX_WORK_BYTES}"
        )
    raw_sha = source_raw_sha(parts)
    revision = source["revision"]
    if not revision.startswith("git:"):
        revision = f"local-sha256:{raw_sha}"
    segments = _bind_segments(
        source["family_id"],
        source["language"],
        source["split"],
        segment_text(text),
    )
    return ExtractedSource(
        source["family_id"],
        source["title"],
        source["language"],
        source["split"],
        revision,
        source["source_url"],
        source["rights_evidence_url"],
        source["authority_url"],
        _sha(content),
        raw_sha,
        sum(part.bytes for part in parts),
        len(content),
        tuple(parts),
        segments,
    )


def extract_all(
    registry: Mapping[str, Any],
    staging_dir: str | Path,
) -> tuple[list[ExtractedSource], list[dict[str, str]]]:
    staging = Path(staging_dir)
    acquisition = load_acquisition(staging)
    acquired = {item["family_id"]: item for item in acquisition["sources"]}
    if set(acquired) != set(FAMILY_SPLIT):
        raise PilotDataError("acquisition is incomplete; run acquire for all families")
    for source in registry["sources"]:
        family = source["family_id"]
        expected_paths = [part["path"] for part in source["parts"]]
        actual_paths = [part["path"] for part in acquired[family]["parts"]]
        if actual_paths != expected_paths:
            raise PilotDataError(f"acquisition parts are incomplete for {family!r}")
    extracted: list[ExtractedSource] = []
    excluded: list[dict[str, str]] = []
    for source in registry["sources"]:
        family = source["family_id"]
        try:
            extracted.append(_extract_source(source, acquired[family], staging))
        except SourceExcluded as exc:
            excluded.append({"family_id": family, "reason": str(exc)})
    total = sum(source.content_bytes for source in extracted)
    if total > MAX_EXTRACTED_BYTES:
        raise PilotDataError(
            f"extracted text exceeds {MAX_EXTRACTED_BYTES} bytes: {total}"
        )
    segments = sum(len(source.segments) for source in extracted)
    if segments > MAX_SEGMENTS:
        raise PilotDataError(f"segment count exceeds {MAX_SEGMENTS}: {segments}")
    languages = {split: set() for split in SPLITS}
    for source in extracted:
        languages[source.split].add(source.language)
    if not {"en", "pl"} <= languages["train"]:
        raise PilotDataError("usable train data lacks both languages")
    if not {"en", "pl"} <= languages["validation"]:
        raise PilotDataError("usable validation data lacks both languages")
    if not languages["test"]:
        raise PilotDataError("usable test data is absent")
    return extracted, excluded


def select_tokenizer_segments(
    sources: Sequence[ExtractedSource],
    *,
    max_bytes: int = TOKENIZER_TRAIN_BYTES,
) -> tuple[list[TextSegment], bool]:
    train = [source for source in sources if source.split == "train"]
    total = sum(segment.bytes for source in train for segment in source.segments)
    if total <= max_bytes:
        selected = [
            segment
            for source in sorted(train, key=lambda item: item.family_id)
            for segment in source.segments
        ]
        return selected, False
    queues = {
        source.family_id: deque(source.segments)
        for source in sorted(train, key=lambda item: item.family_id)
    }
    order = [
        family
        for family in FAMILY_ORDER["train"]
        if family in queues
    ]
    selected: list[TextSegment] = []
    used = 0
    while used < max_bytes:
        progressed = False
        for family in order:
            queue = queues[family]
            while queue:
                segment = queue.popleft()
                if used + segment.bytes <= max_bytes:
                    selected.append(segment)
                    used += segment.bytes
                    progressed = True
                    break
        if not progressed:
            break
    if not selected:
        raise PilotDataError("no tokenizer training segments selected")
    return selected, True


def train_tokenizer_from_segments(
    segments: Sequence[TextSegment],
) -> dict[str, Any]:
    documents = [
        pt.TrainingDocument(
            f"{segment.family_id}:{segment.start:08d}:{segment.end:08d}",
            segment.text,
            "train",
            segment.language,
            "prose",
        )
        for segment in segments
    ]
    return pt.train_real_pilot_tokenizer(documents, 16384)


def _split_text_for_block(
    session: pt.TokenizerSession,
    text: str,
    start: int,
    end: int,
    *,
    max_tokens: int = MAX_BLOCK_CONTENT_TOKENS,
    depth: int = 0,
) -> list[tuple[int, int, tuple[int, ...]]]:
    if depth > 32:
        raise PilotDataError("token segmentation recursion is too deep")
    chunk = text[start:end]
    ids = tuple(session.encode(chunk))
    if len(ids) <= max_tokens:
        return [(start, end, ids)]
    if end - start <= 1:
        raise PilotDataError("single character exceeds block token limit")
    middle = start + (end - start) // 2
    cut = text.rfind("\n\n", start + 1, middle)
    if cut < 0:
        cut = text.find("\n\n", middle, end - 1)
    if cut < 0:
        left = text.rfind(" ", start + 1, middle)
        right = text.find(" ", middle, end - 1)
        candidates = [value for value in (left, right) if value >= 0]
        cut = min(candidates, key=lambda value: abs(value - middle)) if candidates else middle
    if cut <= start or cut >= end:
        cut = middle
    left = _split_text_for_block(
        session, text, start, cut, max_tokens=max_tokens, depth=depth + 1
    )
    right = _split_text_for_block(
        session, text, cut, end, max_tokens=max_tokens, depth=depth + 1
    )
    return left + right


def build_block_candidates(
    sources: Sequence[ExtractedSource],
    session: pt.TokenizerSession,
) -> dict[str, list[BlockCandidate]]:
    result: dict[str, list[BlockCandidate]] = {split: [] for split in SPLITS}
    for source in sources:
        for segment in source.segments:
            pieces = _split_text_for_block(
                session,
                segment.text,
                0,
                len(segment.text),
            )
            for start, end, ids in pieces:
                text = segment.text[start:end]
                data = _utf8(text)
                if not 1 <= len(ids) <= MAX_BLOCK_CONTENT_TOKENS:
                    raise PilotDataError("block candidate has invalid content length")
                result[source.split].append(
                    BlockCandidate(
                        source.family_id,
                        source.language,
                        source.split,
                        segment.start + start,
                        segment.start + end,
                        ids,
                        len(data),
                        _sha(data),
                    )
                )
    for split in SPLITS:
        result[split].sort(
            key=lambda item: (item.family_id, item.start, item.end)
        )
    return result


def select_block_candidates(
    candidates: Mapping[str, Sequence[BlockCandidate]],
) -> dict[str, list[BlockCandidate]]:
    caps = {
        "train": TRAIN_BLOCKS,
        "validation": VALIDATION_BLOCKS,
        "test": TEST_BLOCKS,
    }
    selected: dict[str, list[BlockCandidate]] = {}
    for split in SPLITS:
        queues = {
            family: deque(
                item for item in candidates.get(split, ()) if item.family_id == family
            )
            for family in FAMILY_ORDER[split]
        }
        chosen: list[BlockCandidate] = []
        while len(chosen) < caps[split]:
            progressed = False
            for family in FAMILY_ORDER[split]:
                if queues[family]:
                    chosen.append(queues[family].popleft())
                    progressed = True
                    if len(chosen) >= caps[split]:
                        break
            if not progressed:
                break
        if not chosen:
            raise PilotDataError(f"no block candidates for split {split!r}")
        selected[split] = chosen
    if len(selected["train"]) < 32:
        raise PilotDataError("train split has fewer than 32 blocks")
    if len(selected["validation"]) < 2:
        raise PilotDataError("validation split has fewer than 2 blocks")
    return selected


def _block_labels(input_ids: Sequence[int], document_ids: Sequence[int]) -> dict[str, list[int]]:
    labels: dict[str, list[int]] = {}
    for offset in (1, 2):
        labels[str(offset)] = [
            input_ids[index + offset]
            if index + offset < SEQUENCE_LENGTH
            and document_ids[index] >= 0
            and document_ids[index] == document_ids[index + offset]
            else IGNORE
            for index in range(SEQUENCE_LENGTH)
        ]
    return labels


def pack_blocks(
    selected: Mapping[str, Sequence[BlockCandidate]],
    *,
    vocab_size: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, BlockCandidate]]:
    ordered: list[BlockCandidate] = []
    for split in SPLITS:
        ordered.extend(
            sorted(
                selected[split],
                key=lambda item: (item.family_id, item.start, item.end),
            )
        )
    mapping: dict[int, BlockCandidate] = {}
    for doc_id, candidate in enumerate(ordered):
        mapping[doc_id] = candidate
    blocks: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for doc_id, candidate in mapping.items():
        IDs = list(candidate.content_ids)
        length = len(IDs) + 2
        if length > SEQUENCE_LENGTH:
            raise PilotDataError("block exceeds sequence length")
        pad = SEQUENCE_LENGTH - length
        input_ids = [BOS, *IDs, EOS] + [PAD] * pad
        document_ids = [doc_id] * length + [-1] * pad
        position_ids = list(range(length)) + [0] * pad
        valid = [True] * length + [False] * pad
        block = {
            "input_ids": input_ids,
            "document_ids": document_ids,
            "position_ids": position_ids,
            "valid": valid,
            "labels": _block_labels(input_ids, document_ids),
        }
        validate_block(block, vocab_size)
        blocks[candidate.split].append(block)
    return blocks, mapping


def validate_block(
    block: Any,
    vocab_size: int,
    *,
    expected_length: int = SEQUENCE_LENGTH,
) -> tuple[int, int, int]:
    if not isinstance(block, dict) or set(block) != BLOCK_KEYS:
        raise PilotDataError("block schema is invalid")
    if type(vocab_size) is not int or not 8 <= vocab_size <= 16384:
        raise PilotDataError("block vocabulary size is invalid")
    arrays = {}
    for key in ("input_ids", "document_ids", "position_ids", "valid"):
        value = block[key]
        if not isinstance(value, list) or len(value) != expected_length:
            raise PilotDataError(f"block {key} length is invalid")
        arrays[key] = value
    labels = block["labels"]
    if not isinstance(labels, dict) or set(labels) != {"1", "2"}:
        raise PilotDataError("block labels keys are invalid")
    for key in ("1", "2"):
        value = labels[key]
        if not isinstance(value, list) or len(value) != expected_length:
            raise PilotDataError("block label length is invalid")
    input_ids = arrays["input_ids"]
    document_ids = arrays["document_ids"]
    position_ids = arrays["position_ids"]
    valid = arrays["valid"]
    if any(type(value) is not int for value in input_ids):
        raise PilotDataError("input IDs must be integers")
    if input_ids[0] != BOS:
        raise PilotDataError("block must start with BOS")
    content_end = 0
    for index in range(1, expected_length):
        value = input_ids[index]
        if value == PAD:
            continue
        if value == EOS:
            content_end = index + 1
            break
        if value < CONTENT_START or value >= vocab_size:
            raise PilotDataError(f"invalid content token: {value}")
        content_end = index + 1
    if content_end < 3 or input_ids[content_end - 1] != EOS:
        raise PilotDataError("block must contain a terminal EOS")
    if any(value != PAD for value in input_ids[content_end:]):
        raise PilotDataError("padding must follow EOS")
    if any(value < CONTENT_START or value >= vocab_size for value in input_ids[1:content_end - 1]):
        raise PilotDataError("control token inside content")
    if any(type(value) is not bool for value in valid):
        raise PilotDataError("valid mask must be boolean")
    if valid[:content_end] != [True] * content_end or any(valid[content_end:]):
        raise PilotDataError("valid mask does not match content")
    doc_id = document_ids[0]
    if type(doc_id) is not int or doc_id < 0:
        raise PilotDataError("document ID must be nonnegative")
    if any(value != doc_id for value in document_ids[:content_end]):
        raise PilotDataError("block mixes document IDs")
    if any(value != -1 for value in document_ids[content_end:]):
        raise PilotDataError("padding document IDs must be -1")
    if position_ids[:content_end] != list(range(content_end)):
        raise PilotDataError("position IDs do not reset at segment start")
    if any(value != 0 for value in position_ids[content_end:]):
        raise PilotDataError("padding positions must be zero")
    ntp = 0
    offset2 = 0
    for index in range(expected_length):
        for offset, counter in ((1, "ntp"), (2, "offset2")):
            expected = (
                input_ids[index + offset]
                if index + offset < expected_length
                and document_ids[index] >= 0
                and document_ids[index] == document_ids[index + offset]
                else IGNORE
            )
            actual = labels[str(offset)][index]
            if actual != expected:
                raise PilotDataError("label does not match future-token rule")
            if actual != IGNORE:
                if counter == "ntp":
                    ntp += 1
                else:
                    offset2 += 1
    if ntp == 0:
        raise PilotDataError("block has no NTP labels")
    return content_end, ntp, offset2


def _source_records(sources: Sequence[ExtractedSource]) -> list[dict[str, str]]:
    records = [
        {
            "family_id": source.family_id,
            "language": source.language,
            "split": source.split,
            "content_sha256": source.content_sha256,
            "raw_sha256": source.raw_sha256,
            "revision": source.revision,
            "rights_evidence_url": source.rights_evidence_url,
            "source_url": source.source_url,
        }
        for source in sorted(sources, key=lambda item: item.family_id)
    ]
    return records


def _validate_sources(sources: Any, vocab_size: int) -> list[dict[str, str]]:
    if not isinstance(sources, list) or not sources:
        raise PilotDataError("manifest sources must be nonempty")
    result = []
    seen = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
            raise PilotDataError("manifest source schema is invalid")
        family = source["family_id"]
        if not isinstance(family, str) or not family or family in seen:
            raise PilotDataError("manifest source family is invalid")
        seen.add(family)
        if source["language"] not in ("en", "pl"):
            raise PilotDataError("manifest source language is invalid")
        if source["split"] not in SPLITS:
            raise PilotDataError("manifest source split is invalid")
        if source["split"] != FAMILY_SPLIT.get(family):
            raise PilotDataError("manifest source split decision is invalid")
        if not _is_hex64(source["content_sha256"]) or not _is_hex64(source["raw_sha256"]):
            raise PilotDataError("manifest source hash is invalid")
        for key in ("revision", "rights_evidence_url", "source_url"):
            if not isinstance(source[key], str) or not source[key]:
                raise PilotDataError(f"manifest source {key} is invalid")
        if not source["rights_evidence_url"].startswith("https://"):
            raise PilotDataError("manifest rights URL is invalid")
        if not source["source_url"].startswith("https://"):
            raise PilotDataError("manifest source URL is invalid")
        result.append(source)
    if result != sorted(result, key=lambda item: item["family_id"]):
        raise PilotDataError("manifest sources are not sorted")
    if set(seen) != set(FAMILY_SPLIT):
        raise PilotDataError("manifest source families are incomplete")
    for split in SPLITS:
        languages = {
            source["language"] for source in result if source["split"] == split
        }
        if split != "test" and not {"en", "pl"} <= languages:
            raise PilotDataError(f"manifest {split} lacks a language")
        if split == "test" and not languages:
            raise PilotDataError("manifest test split is empty")
    return result


def _read_canonical_json(path: Path, limit: int, name: str) -> tuple[Any, bytes]:
    if not path.is_file() or path.is_symlink():
        raise PilotDataError(f"{name} is missing or not regular")
    if path.stat().st_size > limit:
        raise PilotDataError(f"{name} is too large")
    raw = path.read_bytes()
    value = _parse_json(raw, name)
    if _canonical_json(value) != raw:
        raise PilotDataError(f"{name} is not canonical JSON")
    return value, raw


def _validate_tokenizer_json(data: bytes, vocab_size: int) -> dict[str, Any]:
    if len(data) > MAX_TOKENIZER_BYTES:
        raise PilotDataError("tokenizer JSON is too large")
    value = _parse_json(data, "tokenizer.json")
    if not isinstance(value, dict):
        raise PilotDataError("tokenizer JSON must be an object")
    model = value.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise PilotDataError("tokenizer model vocabulary is missing")
    if len(model["vocab"]) != vocab_size:
        raise PilotDataError("tokenizer vocabulary size mismatch")
    added = value.get("added_tokens")
    if not isinstance(added, list):
        raise PilotDataError("tokenizer added_tokens is missing")
    mapping = {}
    for item in added:
        if not isinstance(item, dict) or type(item.get("id")) is not int:
            raise PilotDataError("tokenizer added token is invalid")
        mapping[item["id"]] = item.get("content")
    for token_id, token in SPECIAL_TOKENS.items():
        if mapping.get(token_id) != token:
            raise PilotDataError("tokenizer special token mapping is invalid")
    return value


def load_pilot_dataset(
    dataset_dir: str | Path,
    manifest_sha256: str,
    *,
    expected_vocab: int | None = None,
    expected_length: int = SEQUENCE_LENGTH,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    if not root.is_dir() or root.is_symlink():
        raise PilotDataError("dataset directory is missing or not regular")
    if not _is_hex64(manifest_sha256):
        raise PilotDataError("manifest SHA-256 is invalid")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PilotDataError("manifest.json is required")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise PilotDataError("manifest is too large")
    raw_manifest = manifest_path.read_bytes()
    if _sha(raw_manifest) != manifest_sha256:
        raise PilotDataError("manifest SHA-256 mismatch")
    manifest = _parse_json(raw_manifest, "manifest")
    if _canonical_json(manifest) != raw_manifest:
        raise PilotDataError("manifest is not canonical JSON")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise PilotDataError("manifest schema is invalid")
    if manifest["format"] != FORMAT or manifest["version"] != VERSION:
        raise PilotDataError("manifest format/version is invalid")
    if manifest["sequence_length"] != expected_length:
        raise PilotDataError("manifest sequence length is invalid")
    vocab_size = manifest["vocab_size"]
    if type(vocab_size) is not int or not 8 <= vocab_size <= 16384:
        raise PilotDataError("manifest vocabulary size is invalid")
    if expected_vocab is not None and vocab_size != expected_vocab:
        raise PilotDataError("manifest vocabulary does not match expected")
    for key in ("tokenizer_sha256", "source_snapshot_sha256", "logical_sha256"):
        if not _is_hex64(manifest[key]):
            raise PilotDataError(f"manifest {key} is invalid")
    sources = _validate_sources(manifest["sources"], vocab_size)
    if _sha(_canonical_json(sources)) != manifest["source_snapshot_sha256"]:
        raise PilotDataError("source snapshot SHA-256 mismatch")
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise PilotDataError("manifest shards must be nonempty")
    shard_by_split: dict[str, dict[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != SHARD_KEYS:
            raise PilotDataError("manifest shard schema is invalid")
        split = shard["split"]
        if split not in SPLITS or split in shard_by_split:
            raise PilotDataError("manifest shard split is invalid")
        if shard["path"] != f"{split}.jsonl":
            raise PilotDataError("manifest shard path is invalid")
        if not _is_hex64(shard["sha256"]):
            raise PilotDataError("manifest shard hash is invalid")
        if type(shard["bytes"]) is not int or shard["bytes"] < 0:
            raise PilotDataError("manifest shard bytes is invalid")
        for key in ("blocks", "valid_tokens", "ntp_labels", "offset2_labels"):
            if type(shard[key]) is not int or shard[key] < 0:
                raise PilotDataError(f"manifest shard {key} is invalid")
        shard_by_split[split] = shard
    if "train" not in shard_by_split:
        raise PilotDataError("train shard is required")
    expected_files = {"manifest.json", "tokenizer.json"} | {
        shard["path"] for shard in shards
    }
    actual_files = set()
    for p in root.iterdir():
        if p.is_symlink() or not p.is_file():
            raise PilotDataError("dataset contains a nonregular entry")
        actual_files.add(p.name)
    if actual_files != expected_files:
        raise PilotDataError("dataset file set is invalid")
    tokenizer_path = root / "tokenizer.json"
    tokenizer_raw = tokenizer_path.read_bytes()
    if _sha(tokenizer_raw) != manifest["tokenizer_sha256"]:
        raise PilotDataError("tokenizer SHA-256 mismatch")
    _validate_tokenizer_json(tokenizer_raw, vocab_size)
    logical_parts = []
    blocks: dict[str, list[dict[str, Any]]] = {}
    doc_ids: set[int] = set()
    for split in SPLITS:
        if split not in shard_by_split:
            continue
        shard = shard_by_split[split]
        path = root / shard["path"]
        if path.stat().st_size != shard["bytes"]:
            raise PilotDataError("shard byte count mismatch")
        raw = path.read_bytes()
        if _sha(raw) != shard["sha256"]:
            raise PilotDataError("shard SHA-256 mismatch")
        logical_parts.append(raw)
        if len(raw) > MAX_SHARD_BYTES:
            raise PilotDataError("shard is too large")
        if raw and not raw.endswith(b"\n"):
            raise PilotDataError("shard must end with LF")
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]
        if len(lines) != shard["blocks"]:
            raise PilotDataError("shard block count mismatch")
        parsed = []
        valid_total = 0
        ntp_total = 0
        offset2_total = 0
        for line in lines:
            block = _parse_json(line, "block")
            if _canonical_json(block) != line:
                raise PilotDataError("block is not canonical JSON")
            length, ntp, offset2 = validate_block(block, vocab_size)
            doc_id = block["document_ids"][0]
            if doc_id in doc_ids:
                raise PilotDataError("duplicate document ID across dataset")
            doc_ids.add(doc_id)
            valid_total += length
            ntp_total += ntp
            offset2_total += offset2
            parsed.append(block)
        if (valid_total, ntp_total, offset2_total) != (
            shard["valid_tokens"],
            shard["ntp_labels"],
            shard["offset2_labels"],
        ):
            raise PilotDataError("shard count mismatch")
        blocks[split] = parsed
    if _sha(b"".join(logical_parts)) != manifest["logical_sha256"]:
        raise PilotDataError("dataset logical SHA-256 mismatch")
    if len(blocks.get("train", ())) > MAX_BLOCKS_PER_SPLIT:
        raise PilotDataError("train block count exceeds limit")
    converted = {
        split: [
            {
                **block,
                "labels": {int(key): value for key, value in block["labels"].items()},
            }
            for block in split_blocks
        ]
        for split, split_blocks in blocks.items()
    }
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "manifest_bytes": raw_manifest,
        "blocks": converted,
        "sources": sources,
        "summary": {
            "format": FORMAT,
            "version": VERSION,
            "sequence_length": SEQUENCE_LENGTH,
            "vocab_size": vocab_size,
            "manifest_sha256": manifest_sha256,
            "logical_sha256": manifest["logical_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "splits": {
                split: {
                    "blocks": len(blocks[split]),
                    "valid_tokens": shard_by_split[split]["valid_tokens"],
                    "ntp_labels": shard_by_split[split]["ntp_labels"],
                    "offset2_labels": shard_by_split[split]["offset2_labels"],
                }
                for split in SPLITS
                if split in shard_by_split
            },
        },
    }


def write_dataset(
    output_parent: str | Path,
    *,
    name: str,
    blocks_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    sources: Sequence[Mapping[str, str]],
    tokenizer_json: bytes,
    vocab_size: int,
) -> dict[str, Any]:
    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / name
    if final.exists():
        raise PilotDataError(f"dataset already exists: {final}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise PilotDataError("dataset name is unsafe")
    if set(blocks_by_split) - set(SPLITS) or "train" not in blocks_by_split:
        raise PilotDataError("dataset splits are invalid")
    source_list = _validate_sources(list(sources), vocab_size)
    _validate_tokenizer_json(tokenizer_json, vocab_size)
    for split in blocks_by_split:
        if len(blocks_by_split[split]) > MAX_BLOCKS_PER_SPLIT:
            raise PilotDataError("block count exceeds limit")
    shard_bytes: dict[str, bytes] = {}
    shard_meta = []
    for split in SPLITS:
        if split not in blocks_by_split:
            continue
        blocks = list(blocks_by_split[split])
        if not blocks:
            raise PilotDataError(f"empty split: {split}")
        for block in blocks:
            validate_block(block, vocab_size)
        data = b"".join(_canonical_json(block) + b"\n" for block in blocks)
        shard_bytes[split] = data
        shard_meta.append(
            {
                "split": split,
                "path": f"{split}.jsonl",
                "sha256": _sha(data),
                "bytes": len(data),
                "blocks": len(blocks),
                "valid_tokens": sum(
                    sum(1 for value in block["valid"] if value) for block in blocks
                ),
                "ntp_labels": sum(
                    sum(1 for value in block["labels"]["1"] if value != IGNORE)
                    for block in blocks
                ),
                "offset2_labels": sum(
                    sum(1 for value in block["labels"]["2"] if value != IGNORE)
                    for block in blocks
                ),
            }
        )
    logical = _sha(b"".join(shard_bytes[split] for split in SPLITS if split in shard_bytes))
    tokenizer_sha = _sha(tokenizer_json)
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "vocab_size": vocab_size,
        "tokenizer_sha256": tokenizer_sha,
        "source_snapshot_sha256": _sha(_canonical_json(source_list)),
        "sources": source_list,
        "shards": shard_meta,
        "logical_sha256": logical,
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha = _sha(manifest_bytes)
    temp = parent / f".build-{os.getpid()}-{time.time_ns()}"
    temp.mkdir()
    try:
        (temp / "manifest.json").write_bytes(manifest_bytes)
        (temp / "tokenizer.json").write_bytes(tokenizer_json)
        for split, data in shard_bytes.items():
            (temp / f"{split}.jsonl").write_bytes(data)
        loaded = load_pilot_dataset(temp, manifest_sha, expected_vocab=vocab_size)
        os.rename(temp, final)
    except Exception:
        if temp.exists():
            for p in temp.iterdir():
                if p.is_file():
                    p.unlink()
            temp.rmdir()
        raise
    loaded["dataset_dir"] = str(final)
    loaded["shard_bytes"] = shard_bytes
    loaded["manifest_bytes"] = manifest_bytes
    return loaded


def build_dataset(
    registry_path: str | Path,
    staging_dir: str | Path,
    output_parent: str | Path,
    *,
    name: str = "splot-pilot-v1",
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    extracted, excluded = extract_all(registry, staging_dir)
    training_segments, sampled = select_tokenizer_segments(extracted)
    artifact = train_tokenizer_from_segments(training_segments)
    session = pt.create_session(artifact)
    candidates = build_block_candidates(extracted, session)
    selected = select_block_candidates(candidates)
    vocab_size = artifact["config"]["actual_vocab_size"]
    blocks, mapping = pack_blocks(selected, vocab_size=vocab_size)
    tokenizer_json = _canonical_json(artifact["tokenizer"])
    loaded = write_dataset(
        output_parent,
        name=name,
        blocks_by_split=blocks,
        sources=_source_records(extracted),
        tokenizer_json=tokenizer_json,
        vocab_size=vocab_size,
    )
    parent = Path(output_parent)
    (parent / "tokenizer-artifact.json").write_bytes(
        pt.canonical_artifact_bytes(artifact)
    )
    source_index = {
        "schema_version": 1,
        "dataset_manifest_sha256": loaded["manifest_sha256"],
        "documents": [
            {
                "document_id": doc_id,
                "family_id": candidate.family_id,
                "language": candidate.language,
                "split": candidate.split,
                "text_start": candidate.start,
                "text_end": candidate.end,
                "content_tokens": len(candidate.content_ids),
                "text_bytes": candidate.text_bytes,
                "text_sha256": candidate.text_sha256,
            }
            for doc_id, candidate in sorted(mapping.items())
        ],
    }
    (parent / "source-index.json").write_bytes(_canonical_json(source_index) + b"\n")
    counts = _dataset_counts(loaded, mapping)
    summary = {
        "format": FORMAT,
        "version": VERSION,
        "dataset_name": name,
        "manifest_sha256": loaded["manifest_sha256"],
        "manifest_bytes": len(loaded["manifest_bytes"]),
        "logical_sha256": loaded["manifest"]["logical_sha256"],
        "tokenizer_sha256": loaded["manifest"]["tokenizer_sha256"],
        "tokenizer_artifact_sha256": pt.artifact_sha256(artifact),
        "tokenizer_logical_digest": artifact["logical_digest"],
        "tokenizer_training_digest": artifact["training_digest"],
        "tokenizer_json_bytes": len(tokenizer_json),
        "actual_vocab_size": vocab_size,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "unused_logits": MODEL_VOCAB_SIZE - vocab_size,
        "tokenizer_training": {
            "segments": len(training_segments),
            "bytes": sum(segment.bytes for segment in training_segments),
            "sampled": sampled,
        },
        "sources": [
            {
                "family_id": source.family_id,
                "language": source.language,
                "split": source.split,
                "revision": source.revision,
                "raw_sha256": source.raw_sha256,
                "raw_bytes": source.raw_bytes,
                "content_sha256": source.content_sha256,
                "content_bytes": source.content_bytes,
                "segments": len(source.segments),
                "parts": len(source.parts),
            }
            for source in sorted(extracted, key=lambda item: item.family_id)
        ],
        "excluded": excluded,
        "counts": counts,
        "shards": loaded["summary"]["splits"],
        "source_index_sha256": _sha((parent / "source-index.json").read_bytes()),
        "source_index_documents": len(mapping),
    }
    (parent / "build-summary.json").write_bytes(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    return {
        "dataset": loaded,
        "artifact": artifact,
        "summary": summary,
        "excluded": excluded,
        "mapping": mapping,
        "source_index": source_index,
    }


def _dataset_counts(
    loaded: Mapping[str, Any],
    mapping: Mapping[int, BlockCandidate],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    by_language: dict[str, dict[str, int]] = {}
    by_sld: dict[str, dict[str, int]] = {}
    overall = {
        "blocks": 0,
        "content_tokens": 0,
        "valid_tokens": 0,
        "pad_tokens": 0,
        "ntp_labels": 0,
        "offset2_labels": 0,
    }
    for split, blocks in loaded["blocks"].items():
        for block in blocks:
            doc_id = block["document_ids"][0]
            candidate = mapping[doc_id]
            content = len(candidate.content_ids)
            pad = sum(1 for value in block["valid"] if not value)
            ntp = sum(1 for value in block["labels"][1] if value != IGNORE)
            offset2 = sum(1 for value in block["labels"][2] if value != IGNORE)
            for target, key in (
                (overall, None),
                (by_split.setdefault(split, _empty_counts()), split),
                (by_language.setdefault(candidate.language, _empty_counts()), candidate.language),
                (
                    by_sld.setdefault(
                        f"{split}/{candidate.language}", _empty_counts()
                    ),
                    f"{split}/{candidate.language}",
                ),
            ):
                target["blocks"] += 1
                target["content_tokens"] += content
                target["valid_tokens"] += content + 2
                target["pad_tokens"] += pad
                target["ntp_labels"] += ntp
                target["offset2_labels"] += offset2
    return {
        "overall": overall,
        "by_split": {key: by_split[key] for key in sorted(by_split)},
        "by_language": {key: by_language[key] for key in sorted(by_language)},
        "by_split_language": {key: by_sld[key] for key in sorted(by_sld)},
    }


def _empty_counts() -> dict[str, int]:
    return {
        "blocks": 0,
        "content_tokens": 0,
        "valid_tokens": 0,
        "pad_tokens": 0,
        "ntp_labels": 0,
        "offset2_labels": 0,
    }


def compare_datasets(a: str | Path, b: str | Path) -> dict[str, Any]:
    left = load_pilot_dataset(
        a,
        _sha((Path(a) / "manifest.json").read_bytes()),
    )
    right = load_pilot_dataset(
        b,
        _sha((Path(b) / "manifest.json").read_bytes()),
    )
    checks = {
        "manifest_equal": left["manifest_bytes"] == right["manifest_bytes"],
        "tokenizer_equal": left["manifest"]["tokenizer_sha256"]
        == right["manifest"]["tokenizer_sha256"],
        "logical_equal": left["manifest"]["logical_sha256"]
        == right["manifest"]["logical_sha256"],
        "shards_equal": {
            split: _sha((Path(a) / f"{split}.jsonl").read_bytes())
            for split in left["blocks"]
        }
        == {
            split: _sha((Path(b) / f"{split}.jsonl").read_bytes())
            for split in right["blocks"]
        },
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "manifest_sha256_a": left["manifest_sha256"],
        "manifest_sha256_b": right["manifest_sha256"],
        "logical_sha256_a": left["manifest"]["logical_sha256"],
        "logical_sha256_b": right["manifest"]["logical_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument("--registry", required=True)
    acquire_parser.add_argument("--staging", required=True)
    acquire_parser.add_argument("--family", action="append", default=None)
    acquire_parser.add_argument("--max-parts", type=int, default=None)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--registry", required=True)
    build_parser.add_argument("--staging", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--name", default="splot-pilot-v1")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--dataset", required=True)
    verify_parser.add_argument("--manifest-sha256", required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("dataset_a")
    compare_parser.add_argument("dataset_b")
    args = parser.parse_args(argv)
    try:
        if args.command == "acquire":
            result = acquire(
                args.registry,
                args.staging,
                families=args.family,
                max_parts=args.max_parts,
            )
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
            return 0
        if args.command == "build":
            if not pt.backend_info()["available"]:
                print(
                    f"backend unavailable: tokenizers {pt.TOKENIZERS_VERSION} is required",
                    file=sys.stderr,
                )
                return 2
            result = build_dataset(
                args.registry, args.staging, args.output, name=args.name
            )
            print(
                json.dumps(
                    result["summary"],
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            return 0
        if args.command == "verify":
            result = load_pilot_dataset(args.dataset, args.manifest_sha256)
            print(json.dumps(result["summary"], indent=2, sort_keys=True, ensure_ascii=True))
            return 0
        if args.command == "compare":
            result = compare_datasets(args.dataset_a, args.dataset_b)
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
            return 0 if result["status"] == "passed" else 1
    except PilotDataError as exc:
        print(f"PILOT_DATA_ERROR: {exc}", file=sys.stderr)
        return 1
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
