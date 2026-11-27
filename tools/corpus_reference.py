"""Deterministic synthetic corpus preflight, deduplication and leakage reference.

T0004 reference implementation. Synthetic records only; this is not a source
rights approval, production deduplication pipeline or tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import islice
from typing import Any

MAX_RECORDS = 1000
MAX_RECORD_BYTES = 4096
SHINGLE_SIZE = 5
NEAR_NUMERATOR = 4
NEAR_DENOMINATOR = 5
ALLOWED_LANGUAGES = frozenset({"en", "pl"})
ALLOWED_DOMAINS = frozenset({"prose"})
SYNTHETIC_APPROVAL = "synthetic_approved"
DEFAULT_SEED = "splot-corpus-reference-v1"
DEFAULT_TRAIN_PCT = 98
DEFAULT_VAL_PCT = 1
SPLITS = ("train", "validation", "test")


class CorpusError(ValueError):
    """Invalid input, configuration or corpus state (fail closed)."""


class AuditError(RuntimeError):
    """Leakage audit failed; the report must not be used."""


@dataclass(frozen=True)
class SourceApproval:
    source_id: str
    revision: str
    approval: str


@dataclass(frozen=True)
class Record:
    record_id: str
    source_id: str
    source_revision: str
    group_id: str
    language: str
    domain: str
    text: str


@dataclass(frozen=True)
class ValidatedRecord:
    record: Record
    raw_sha256: str
    normalized_sha256: str
    byte_length: int
    shingles: frozenset[tuple[str, ...]]


@dataclass(frozen=True)
class MemberInfo:
    record_id: str
    source_id: str
    source_revision: str
    group_id: str
    language: str
    domain: str
    byte_length: int
    raw_sha256: str
    normalized_sha256: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ClusterResult:
    cluster_id: str
    members: tuple[MemberInfo, ...]
    representative_id: str
    representative_normalized_sha256: str
    link_reasons: tuple[str, ...]
    split: str | None = None
    quarantine_reasons: tuple[str, ...] = ()


def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{name} must be a nonempty string")
    return value


def _string_tuple(values: Any, name: str) -> tuple[str, ...]:
    if values is None or isinstance(values, str):
        raise CorpusError(f"{name} must be an iterable of strings, not a string")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise CorpusError(f"{name} must be an iterable of strings") from exc
    result = []
    for value in iterator:
        _require_nonempty_str(value, name + " entry")
        result.append(value)
    return tuple(result)


def build_allowlist(entries: Iterable[SourceApproval]) -> dict[str, SourceApproval]:
    """Build a fail-closed synthetic source allowlist."""
    try:
        iterator = iter(entries)
    except TypeError as exc:
        raise CorpusError("allowlist must be an iterable of SourceApproval entries") from exc
    allowlist: dict[str, SourceApproval] = {}
    for entry in iterator:
        if not isinstance(entry, SourceApproval):
            raise CorpusError("allowlist entries must be SourceApproval instances")
        _require_nonempty_str(entry.source_id, "source ID")
        _require_nonempty_str(entry.revision, "source revision")
        if not isinstance(entry.approval, str) or entry.approval != SYNTHETIC_APPROVAL:
            raise CorpusError(
                f"source {entry.source_id!r} approval must be {SYNTHETIC_APPROVAL!r}, "
                f"not {entry.approval!r}"
            )
        if entry.source_id in allowlist:
            raise CorpusError(f"duplicate allowlisted source ID: {entry.source_id!r}")
        allowlist[entry.source_id] = entry
    if not allowlist:
        raise CorpusError("allowlist must not be empty")
    return allowlist


def _allowlist_mapping(allowlist: Any) -> dict[str, SourceApproval]:
    """Normalize and revalidate a synthetic allowlist mapping or iterable."""
    if isinstance(allowlist, Mapping):
        for key, entry in allowlist.items():
            if not isinstance(entry, SourceApproval):
                raise CorpusError("allowlist entries must be SourceApproval instances")
            if key != entry.source_id:
                raise CorpusError("allowlist mapping keys must match source IDs")
        return build_allowlist(tuple(allowlist.values()))
    return build_allowlist(allowlist)


def normalize_text(text: str) -> str:
    """NFC normalize and convert CRLF/CR to LF; original text is not changed."""
    if not isinstance(text, str):
        raise CorpusError("text must be a string")
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def raw_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise CorpusError("text must be a string")
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def decode_text(data: bytes) -> str:
    """Strict UTF-8 ingestion helper for raw bytes."""
    if not isinstance(data, (bytes, bytearray)):
        raise CorpusError("decode_text expects bytes")
    try:
        return bytes(data).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CorpusError(f"invalid UTF-8: {exc}") from exc


def word_shingles(text: str) -> frozenset[tuple[str, ...]]:
    """Unicode whitespace-split 5-word shingles over NFC/LF text."""
    tokens = normalize_text(text).split()
    if len(tokens) < SHINGLE_SIZE:
        return frozenset()
    return frozenset(
        tuple(tokens[index:index + SHINGLE_SIZE])
        for index in range(len(tokens) - SHINGLE_SIZE + 1)
    )


def jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> Fraction:
    if not isinstance(a, (set, frozenset)) or not isinstance(b, (set, frozenset)):
        raise CorpusError("jaccard expects shingle sets")
    if not a and not b:
        return Fraction(0, 1)
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return Fraction(intersection, union)


def near_duplicate(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> bool:
    """Exact integer 0.8 Jaccard test: 5*intersection >= 4*union."""
    if not a or not b:
        return False
    if NEAR_DENOMINATOR * min(len(a), len(b)) < NEAR_NUMERATOR * max(len(a), len(b)):
        return False
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return NEAR_DENOMINATOR * intersection >= NEAR_NUMERATOR * union


def validate_records(
    records: Iterable[Record],
    allowlist: Iterable[SourceApproval] | Mapping[str, SourceApproval],
) -> tuple[ValidatedRecord, ...]:
    """Fail-closed record validation; original text is retained unchanged.

    The allowlist is revalidated at this public entry point, including mapping
    key/source-ID equality and explicit approval. Record consumption is bounded
    to MAX_RECORDS + 1 before materializing.
    """
    allowlist = _allowlist_mapping(allowlist)
    try:
        iterator = iter(records)
    except TypeError as exc:
        raise CorpusError("records must be an iterable of Record instances") from exc
    records = tuple(islice(iterator, MAX_RECORDS + 1))
    if len(records) > MAX_RECORDS:
        raise CorpusError(
            f"record count exceeds {MAX_RECORDS} (at least {MAX_RECORDS + 1})"
        )
    seen: set[str] = set()
    validated: list[ValidatedRecord] = []
    for record in records:
        if not isinstance(record, Record):
            raise CorpusError("records must be Record instances")
        _require_nonempty_str(record.record_id, "record ID")
        if record.record_id in seen:
            raise CorpusError(f"duplicate record ID: {record.record_id!r}")
        seen.add(record.record_id)
        _require_nonempty_str(record.source_id, "source ID")
        _require_nonempty_str(record.source_revision, "source revision")
        _require_nonempty_str(record.group_id, "group ID")
        if not isinstance(record.language, str) or record.language not in ALLOWED_LANGUAGES:
            raise CorpusError(
                f"record {record.record_id!r} has unsupported language: {record.language!r}"
            )
        if not isinstance(record.domain, str) or record.domain not in ALLOWED_DOMAINS:
            raise CorpusError(
                f"record {record.record_id!r} has unsupported domain: {record.domain!r}"
            )
        if record.source_id not in allowlist:
            raise CorpusError(f"record {record.record_id!r} source is not allowlisted")
        approved = allowlist[record.source_id]
        if not isinstance(approved, SourceApproval):
            raise CorpusError(f"allowlist entry for {record.source_id!r} is invalid")
        if record.source_revision != approved.revision:
            raise CorpusError(
                f"record {record.record_id!r} source revision does not match allowlist"
            )
        if not isinstance(record.text, str):
            raise CorpusError(f"record {record.record_id!r} text must be a string")
        if record.text == "":
            raise CorpusError(f"record {record.record_id!r} text must not be empty")
        try:
            data = record.text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise CorpusError(
                f"record {record.record_id!r} contains an invalid surrogate"
            ) from exc
        if len(data) > MAX_RECORD_BYTES:
            raise CorpusError(
                f"record {record.record_id!r} exceeds {MAX_RECORD_BYTES} UTF-8 bytes"
            )
        normalized = normalize_text(record.text)
        validated.append(
            ValidatedRecord(
                record=record,
                raw_sha256=hashlib.sha256(data).hexdigest(),
                normalized_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                byte_length=len(data),
                shingles=word_shingles(record.text),
            )
        )
    return tuple(validated)


def cluster_identifier(member_ids: Iterable[str]) -> str:
    """SHA-256 of canonical JSON of the sorted stable member IDs."""
    ids = sorted(member_ids)
    canonical = json.dumps(ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _DisjointSet:
    def __init__(self, ids: Iterable[str]) -> None:
        self._parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            parent[item], item = root, parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root


def cluster_records(
    validated: Iterable[ValidatedRecord],
) -> tuple[tuple[ClusterResult, ...], tuple[tuple[str, str], ...]]:
    """Union source-group, exact and near-duplicate edges, then components."""
    validated = tuple(sorted(validated, key=lambda item: item.record.record_id))
    if not validated:
        return (), ()
    disjoint = _DisjointSet(item.record.record_id for item in validated)
    groups: dict[str, list[str]] = {}
    raws: dict[str, list[str]] = {}
    normalizeds: dict[str, list[str]] = {}
    for item in validated:
        record_id = item.record.record_id
        for index, key in (
            (groups, item.record.group_id),
            (raws, item.raw_sha256),
            (normalizeds, item.normalized_sha256),
        ):
            index.setdefault(key, []).append(record_id)
    for index in (groups, raws, normalizeds):
        for members in index.values():
            first = members[0]
            for other in members[1:]:
                disjoint.union(first, other)

    near_pairs: list[tuple[str, str]] = []
    near_members: set[str] = set()
    for left_index, left in enumerate(validated):
        for right in validated[left_index + 1:]:
            if not near_duplicate(left.shingles, right.shingles):
                continue
            pair = (left.record.record_id, right.record.record_id)
            near_pairs.append(pair)
            near_members.update(pair)
            disjoint.union(*pair)

    group_counts = Counter(item.record.group_id for item in validated)
    raw_counts = Counter(item.raw_sha256 for item in validated)
    normalized_counts = Counter(item.normalized_sha256 for item in validated)
    by_root: dict[str, list[ValidatedRecord]] = {}
    for item in validated:
        by_root.setdefault(disjoint.find(item.record.record_id), []).append(item)

    clusters: list[ClusterResult] = []
    for members in by_root.values():
        members = sorted(members, key=lambda item: item.record.record_id)
        member_infos: list[MemberInfo] = []
        link_reasons: set[str] = set()
        for item in members:
            reasons: list[str] = []
            if group_counts[item.record.group_id] > 1:
                reasons.append("source_group")
            if raw_counts[item.raw_sha256] > 1:
                reasons.append("exact_raw")
            if normalized_counts[item.normalized_sha256] > 1:
                reasons.append("exact_normalized")
            if item.record.record_id in near_members:
                reasons.append("near_duplicate")
            reasons_tuple = tuple(sorted(reasons))
            link_reasons.update(reasons_tuple)
            member_infos.append(
                MemberInfo(
                    record_id=item.record.record_id,
                    source_id=item.record.source_id,
                    source_revision=item.record.source_revision,
                    group_id=item.record.group_id,
                    language=item.record.language,
                    domain=item.record.domain,
                    byte_length=item.byte_length,
                    raw_sha256=item.raw_sha256,
                    normalized_sha256=item.normalized_sha256,
                    reasons=reasons_tuple,
                )
            )
        member_tuple = tuple(member_infos)
        representative = member_tuple[0]
        clusters.append(
            ClusterResult(
                cluster_id=cluster_identifier(member.record_id for member in member_tuple),
                members=member_tuple,
                representative_id=representative.record_id,
                representative_normalized_sha256=representative.normalized_sha256,
                link_reasons=tuple(sorted(link_reasons)),
            )
        )
    clusters.sort(key=lambda cluster: cluster.cluster_id)
    return tuple(clusters), tuple(sorted(near_pairs))


def _validate_split_config(seed: str, train_pct: int, val_pct: int) -> None:
    _require_nonempty_str(seed, "seed")
    if type(train_pct) is not int or type(val_pct) is not int:
        raise CorpusError("split percentages must be integers")
    if not 0 < train_pct < 100:
        raise CorpusError("train percentage must be between 1 and 99")
    if not 0 < val_pct < 100 - train_pct:
        raise CorpusError(
            "validation percentage must be positive and leave a positive test percentage"
        )


def split_bucket(cluster_id: str, seed: str) -> int:
    """First 8 bytes of SHA-256(seed + NUL + cluster_id), big-endian mod 100."""
    _require_nonempty_str(cluster_id, "cluster ID")
    _require_nonempty_str(seed, "seed")
    payload = (seed + "\0" + cluster_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100


def split_for_cluster(
    cluster_id: str,
    seed: str = DEFAULT_SEED,
    train_pct: int = DEFAULT_TRAIN_PCT,
    val_pct: int = DEFAULT_VAL_PCT,
) -> str:
    _validate_split_config(seed, train_pct, val_pct)
    bucket = split_bucket(cluster_id, seed)
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "validation"
    return "test"


def assign_clusters(
    clusters: Iterable[ClusterResult],
    *,
    benchmark_exclusions: Iterable[str] = (),
    excluded_record_ids: Iterable[str] = (),
    seed: str = DEFAULT_SEED,
    train_pct: int = DEFAULT_TRAIN_PCT,
    val_pct: int = DEFAULT_VAL_PCT,
) -> tuple[ClusterResult, ...]:
    """Whole-cluster quarantine, then deterministic cluster-level split."""
    _validate_split_config(seed, train_pct, val_pct)
    benchmark_hashes = {
        normalized_sha256(value)
        for value in _string_tuple(benchmark_exclusions, "benchmark exclusion")
    }
    excluded = set(_string_tuple(excluded_record_ids, "excluded record ID"))
    assigned: list[ClusterResult] = []
    for cluster in clusters:
        reasons: list[str] = []
        if any(member.normalized_sha256 in benchmark_hashes for member in cluster.members):
            reasons.append("benchmark_exact_normalized")
        if any(member.record_id in excluded for member in cluster.members):
            reasons.append("excluded_record_id")
        if reasons:
            assigned.append(
                replace(
                    cluster,
                    split=None,
                    quarantine_reasons=tuple(sorted(reasons)),
                )
            )
        else:
            assigned.append(
                replace(
                    cluster,
                    split=split_for_cluster(cluster.cluster_id, seed, train_pct, val_pct),
                    quarantine_reasons=(),
                )
            )
    return tuple(assigned)


def audit_splits(
    clusters: Iterable[ClusterResult],
    near_pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Return an explicit leakage audit; callers must fail closed on ok=False."""
    clusters = tuple(clusters)
    near_pairs = tuple(near_pairs)
    by_split: dict[str, list[ClusterResult]] = {name: [] for name in SPLITS}
    for cluster in clusters:
        if cluster.split is None:
            continue
        if cluster.split not in by_split:
            raise CorpusError(f"unknown split: {cluster.split!r}")
        by_split[cluster.split].append(cluster)

    cluster_ids = [cluster.cluster_id for cluster in clusters if cluster.split is not None]
    checks: dict[str, bool] = {
        "cluster_ids_disjoint": len(cluster_ids) == len(set(cluster_ids)),
    }

    def pairwise_disjoint(sets: Mapping[str, set[str]]) -> bool:
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1:]:
                if sets[left] & sets[right]:
                    return False
        return True

    normalized: dict[str, set[str]] = {name: set() for name in SPLITS}
    groups: dict[str, set[str]] = {name: set() for name in SPLITS}
    for cluster in clusters:
        if cluster.split is None:
            continue
        for member in cluster.members:
            normalized[cluster.split].add(member.normalized_sha256)
            groups[cluster.split].add(member.group_id)
    checks["normalized_fingerprints_disjoint"] = pairwise_disjoint(normalized)
    checks["source_groups_disjoint"] = pairwise_disjoint(groups)

    cluster_of = {
        member.record_id: cluster.cluster_id
        for cluster in clusters
        for member in cluster.members
    }
    checks["near_duplicates_within_clusters"] = all(
        cluster_of.get(left) is not None
        and cluster_of.get(left) == cluster_of.get(right)
        for left, right in near_pairs
    )
    checks["quarantine_excluded"] = all(
        (cluster.split is None and bool(cluster.quarantine_reasons))
        or (cluster.split is not None and not cluster.quarantine_reasons)
        for cluster in clusters
    )
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "near_duplicate_edges": len(near_pairs),
        "split_clusters": {name: len(by_split[name]) for name in SPLITS},
        "quarantined_clusters": sum(1 for cluster in clusters if cluster.split is None),
    }


def _stats_for(items: Iterable[tuple[str, str, int]]) -> dict[str, Any]:
    records = 0
    total_bytes = 0
    languages: dict[str, dict[str, int]] = {}
    domains: dict[str, dict[str, int]] = {}
    for language, domain, byte_length in items:
        records += 1
        total_bytes += byte_length
        for key, target in ((language, languages), (domain, domains)):
            entry = target.setdefault(key, {"records": 0, "bytes": 0})
            entry["records"] += 1
            entry["bytes"] += byte_length
    return {
        "records": records,
        "bytes": total_bytes,
        "languages": {key: languages[key] for key in sorted(languages)},
        "domains": {key: domains[key] for key in sorted(domains)},
    }


def _build_stats(
    validated: tuple[ValidatedRecord, ...],
    clusters: tuple[ClusterResult, ...],
    near_edge_count: int,
    excluded_record_ids: tuple[str, ...],
) -> dict[str, Any]:
    retained_members: list[MemberInfo] = []
    quarantined_members: list[MemberInfo] = []
    for cluster in clusters:
        if cluster.split is None:
            quarantined_members.extend(cluster.members)
        else:
            retained_members.append(
                next(
                    member
                    for member in cluster.members
                    if member.record_id == cluster.representative_id
                )
            )
    raw_stats = _stats_for(
        (item.record.language, item.record.domain, item.byte_length)
        for item in validated
    )
    retained_stats = _stats_for(
        (member.language, member.domain, member.byte_length)
        for member in retained_members
    )
    quarantined_stats = _stats_for(
        (member.language, member.domain, member.byte_length)
        for member in quarantined_members
    )
    reason_counts = Counter(
        reason
        for cluster in clusters
        if cluster.split is None
        for reason in cluster.quarantine_reasons
    )
    quarantined_stats["reasons"] = {
        key: reason_counts[key] for key in sorted(reason_counts)
    }
    split_stats: dict[str, dict[str, int]] = {}
    for name in SPLITS:
        selected = [cluster for cluster in clusters if cluster.split == name]
        representatives = [
            next(
                member
                for member in cluster.members
                if member.record_id == cluster.representative_id
            )
            for cluster in selected
        ]
        split_stats[name] = {
            "clusters": len(selected),
            "representatives": len(representatives),
            "member_records": sum(len(cluster.members) for cluster in selected),
            "representative_bytes": sum(member.byte_length for member in representatives),
            "member_bytes": sum(
                member.byte_length
                for cluster in selected
                for member in cluster.members
            ),
        }
    all_member_ids = {
        member.record_id for cluster in clusters for member in cluster.members
    }
    return {
        "raw": raw_stats,
        "retained": retained_stats,
        "quarantined": quarantined_stats,
        "clusters_total": len(clusters),
        "eligible_clusters": sum(1 for cluster in clusters if cluster.split is not None),
        "quarantined_clusters": sum(1 for cluster in clusters if cluster.split is None),
        "duplicate_records_removed": (
            len(validated) - len(retained_members) - len(quarantined_members)
        ),
        "duplicate_records_removed_including_quarantine": (
            len(validated) - len(retained_members)
        ),
        "near_duplicate_edges": near_edge_count,
        "excluded_record_ids_matched": len(set(excluded_record_ids) & all_member_ids),
        "splits": split_stats,
    }


def _member_to_dict(member: MemberInfo) -> dict[str, Any]:
    return {
        "record_id": member.record_id,
        "source_id": member.source_id,
        "source_revision": member.source_revision,
        "group_id": member.group_id,
        "language": member.language,
        "domain": member.domain,
        "bytes": member.byte_length,
        "raw_sha256": member.raw_sha256,
        "normalized_sha256": member.normalized_sha256,
        "reasons": list(member.reasons),
    }


def _cluster_to_dict(cluster: ClusterResult) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "members": [_member_to_dict(member) for member in cluster.members],
        "representative": cluster.representative_id,
        "representative_normalized_sha256": cluster.representative_normalized_sha256,
        "link_reasons": list(cluster.link_reasons),
        "split": cluster.split,
        "quarantine_reasons": list(cluster.quarantine_reasons),
    }


def _record_to_dict(record: Record) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "source_id": record.source_id,
        "source_revision": record.source_revision,
        "group_id": record.group_id,
        "language": record.language,
        "domain": record.domain,
        "bytes": len(record.text.encode("utf-8")),
        "text": record.text,
    }


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 of canonical JSON without the digest field and without timestamps."""
    data = {key: value for key, value in payload.items() if key != "digest"}
    canonical = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_report(
    records: Iterable[Record],
    allowlist: Iterable[SourceApproval] | Mapping[str, SourceApproval],
    *,
    benchmark_exclusions: Iterable[str] = (),
    excluded_record_ids: Iterable[str] = (),
    seed: str = DEFAULT_SEED,
    train_pct: int = DEFAULT_TRAIN_PCT,
    val_pct: int = DEFAULT_VAL_PCT,
    include_inputs: bool = False,
) -> dict[str, Any]:
    """Validate, cluster, quarantine, split, audit and summarize a synthetic corpus."""
    _validate_split_config(seed, train_pct, val_pct)
    benchmark_exclusions = _string_tuple(benchmark_exclusions, "benchmark exclusion")
    excluded_record_ids = _string_tuple(excluded_record_ids, "excluded record ID")
    validated = validate_records(records, allowlist)
    clusters, near_pairs = cluster_records(validated)
    clusters = assign_clusters(
        clusters,
        benchmark_exclusions=benchmark_exclusions,
        excluded_record_ids=excluded_record_ids,
        seed=seed,
        train_pct=train_pct,
        val_pct=val_pct,
    )
    audit = audit_splits(clusters, near_pairs)
    if not audit["ok"]:
        failed = sorted(name for name, value in audit["checks"].items() if not value)
        raise AuditError("split leakage audit failed: " + ", ".join(failed))
    stats = _build_stats(validated, clusters, len(near_pairs), excluded_record_ids)
    report: dict[str, Any] = {
        "report_version": 1,
        "task_id": "T0004",
        "seed": seed,
        "split_percentages": {
            "train": train_pct,
            "validation": val_pct,
            "test": 100 - train_pct - val_pct,
        },
        "limits": {
            "max_records": MAX_RECORDS,
            "max_record_bytes": MAX_RECORD_BYTES,
            "shingle_size": SHINGLE_SIZE,
            "near_duplicate_threshold": "4/5",
            "allowed_languages": sorted(ALLOWED_LANGUAGES),
            "allowed_domains": sorted(ALLOWED_DOMAINS),
            "approval": SYNTHETIC_APPROVAL,
        },
        "clusters": [_cluster_to_dict(cluster) for cluster in clusters],
        "splits": {
            name: [
                cluster.cluster_id for cluster in clusters if cluster.split == name
            ]
            for name in SPLITS
        },
        "quarantined": [
            cluster.cluster_id for cluster in clusters if cluster.split is None
        ],
        "near_duplicate_pairs": [list(pair) for pair in near_pairs],
        "audit": audit,
        "stats": stats,
    }
    if include_inputs:
        report["inputs"] = [
            _record_to_dict(item.record)
            for item in sorted(validated, key=lambda item: item.record.record_id)
        ]
    report["digest"] = canonical_digest(report)
    return report


_DEMO_ALLOWLIST = (
    SourceApproval("synthetic_en", "rev-en-001", SYNTHETIC_APPROVAL),
    SourceApproval("synthetic_pl", "rev-pl-001", SYNTHETIC_APPROVAL),
)

_DEMO_RECORDS = (
    Record(
        "demo-en-nfc-a",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/nfc",
        "en",
        "prose",
        "Café society\r\nmeets in Kraków.",
    ),
    Record(
        "demo-en-nfc-b",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/nfc-copy",
        "en",
        "prose",
        "Cafe\u0301 society\nmeets in Kraków.",
    ),
    Record(
        "demo-pl-family-a",
        "synthetic_pl",
        "rev-pl-001",
        "synthetic_pl/book-01",
        "pl",
        "prose",
        "Pierwszy rozdział opisuje górski krajobraz.",
    ),
    Record(
        "demo-pl-family-b",
        "synthetic_pl",
        "rev-pl-001",
        "synthetic_pl/book-01",
        "pl",
        "prose",
        "Zupełnie inny tekst o mieście i rzece.",
    ),
    Record(
        "demo-en-near-a",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/near",
        "en",
        "prose",
        "the quick brown fox jumps over the lazy",
    ),
    Record(
        "demo-en-near-b",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/near-copy",
        "en",
        "prose",
        "the quick brown fox jumps over the lazy dog",
    ),
    Record(
        "demo-en-below-a",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/below",
        "en",
        "prose",
        "a small green bird sings in the",
    ),
    Record(
        "demo-en-below-b",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/below-copy",
        "en",
        "prose",
        "a small green bird sings in the garden",
    ),
    Record(
        "demo-en-trans-a",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/trans",
        "en",
        "prose",
        "red orange yellow green blue indigo violet white",
    ),
    Record(
        "demo-en-trans-b",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/trans-copy",
        "en",
        "prose",
        "red orange yellow green blue indigo violet white black",
    ),
    Record(
        "demo-en-trans-c",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/trans-copy-2",
        "en",
        "prose",
        "red orange yellow green blue indigo violet white black silver",
    ),
    Record(
        "demo-en-short-a",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/short",
        "en",
        "prose",
        "tiny short text",
    ),
    Record(
        "demo-en-short-b",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/short-copy",
        "en",
        "prose",
        "tiny short words",
    ),
    Record(
        "demo-en-bench-a",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/benchmark",
        "en",
        "prose",
        "Café benchmark question\r\nwith two lines.",
    ),
    Record(
        "demo-en-bench-b",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/benchmark",
        "en",
        "prose",
        "Another member of the benchmark family.",
    ),
    Record(
        "demo-en-excluded-a",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/excluded",
        "en",
        "prose",
        "This document is explicitly excluded by identifier.",
    ),
    Record(
        "demo-en-excluded-b",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/excluded",
        "en",
        "prose",
        "Another member of the same synthetic family.",
    ),
    Record(
        "demo-pl-diacritic-a",
        "synthetic_pl",
        "rev-pl-001",
        "synthetic_pl/diacritic",
        "pl",
        "prose",
        "Żółć gęślą jaźń.",
    ),
    Record(
        "demo-pl-diacritic-b",
        "synthetic_pl",
        "rev-pl-001",
        "synthetic_pl/diacritic-copy",
        "pl",
        "prose",
        "Zolc gesla jazn.",
    ),
    Record(
        "demo-en-near-bench",
        "synthetic_en",
        "rev-en-001",
        "synthetic_en/near-bench",
        "en",
        "prose",
        "Café benchmark question\nwith two lines!",
    ),
)

_DEMO_BENCHMARK = ("Cafe\u0301 benchmark question\nwith two lines.",)
_DEMO_EXCLUDED = ("demo-en-excluded-a",)


def demo_report() -> dict[str, Any]:
    """Deterministic synthetic demo covering all reference scenarios."""
    return build_report(
        _DEMO_RECORDS,
        _DEMO_ALLOWLIST,
        benchmark_exclusions=_DEMO_BENCHMARK,
        excluded_record_ids=_DEMO_EXCLUDED,
        seed="splot-t0004-demo-v1",
        train_pct=50,
        val_pct=25,
        include_inputs=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print the deterministic synthetic demo report as JSON",
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
