# Corpus preflight, deduplication and leakage reference
2026-11-27 | Synthetic reference only, #38 / #8 | T0004

**Status:** executable synthetic oracle for the [data contract](README.md). It is
not a production deduplication pipeline, source-rights approval, benchmark
clearance, acquisition tool or tokenizer. No real corpus, external content or
private identifier is included. The module and its tests use the Python standard
library only and run on CPU.

## 1. Scope

`tools/corpus_reference.py` implements the small-scale, deterministic reference
required by T0004:

- fail-closed source allowlisting and record validation;
- raw and conservative normalized exact fingerprints;
- exhaustive 5-word-shingle near-duplicate detection;
- source-family, exact and near-duplicate union before any split;
- stable cluster identity, sorted members and deterministic representatives;
- whole-cluster quarantine for supplied benchmark strings or excluded IDs;
- deterministic cluster-level 98/1/1 (configurable) split assignment;
- a fail-closed leakage audit and a stable JSON report with digest.

`tools/test_corpus_reference.py` checks independently calculated tiny shingle
pairs, threshold edges, malformed input, quarantine, audit failure and
determinism. This document is the API reference.

## 2. Data model and fail-closed validation

```python
SourceApproval(source_id: str, revision: str, approval: str)
Record(record_id, source_id, source_revision, group_id, language, domain, text)
```

- `source_id` must appear exactly once in the explicit synthetic allowlist, with
  a nonempty immutable `revision` and an explicitly supplied
  `approval == "synthetic_approved"`. There is no permissive default: missing,
  unknown, pending, denied, empty or mismatched-revision approvals are rejected.
  Mapping keys must equal the corresponding source IDs, and the public
  `validate_records` entry point revalidates this mapping. The test-local
  allowlist is never the real registry and grants no rights.
- `record_id` is a stable unique ID; duplicates are rejected.
- `group_id` is the globally namespaced **source-family** identity used for
  union (same article/revision chain, book/edition, translated pair or code
  fork family). `source_id` and `source_revision` are provenance and
  eligibility fields, not the family key.
- `language` is exactly `en` or `pl`. `domain` is exactly `prose` in this first
  reference; `code`, `math`, missing or unknown domains are rejected, so code is
  never silently eligible prose.
- `text` must be nonempty valid UTF-8 without lone surrogates, at most 4096
  UTF-8 bytes. A corpus may contain at most 1000 records; record iterables are
  consumed at most 1001 entries before rejection and are never materialized
  unbounded. The original string is preserved byte-for-byte and is not
  normalized in place. Metadata strings are validated as nonempty strings;
  separate metadata surrogate hardening remains an optional follow-up.
- An empty corpus is valid. An empty allowlist is rejected.

## 3. Fingerprints

- Raw SHA-256 is computed over the original UTF-8 bytes.
- The normalized fingerprint is SHA-256 over NFC-normalized text after
  converting CRLF and lone CR to LF. Nothing else is changed.
- No casefold, lowercasing, diacritic stripping, whitespace collapse or
  tokenizer normalization is applied to exact fingerprints.
- `decode_text` performs strict UTF-8 ingestion for callers holding raw bytes.

NFC + line-ending normalization makes canonically equivalent combining
sequences and line endings exact duplicates while retaining the original text
and its raw hash for lineage.

## 4. Near-duplicate oracle

`word_shingles` splits NFC/LF text on Unicode whitespace (no stemming, no
casefold) and returns the set of 5-word tuples. Documents with fewer than five
tokens have an empty shingle set and therefore no near-duplicate edges; they can
still join a cluster through source-family or exact relations.

For two nonempty sets, `near_duplicate` returns true exactly when

```text
5 * |intersection| >= 4 * |union|
```

using integer arithmetic, i.e. Jaccard similarity >= 0.8 inclusive. A necessary
size-ratio check rejects impossible pairs early. All eligible pairs are
compared exhaustively (quadratic); this is a correctness oracle, not a scalable
MinHash implementation.

## 5. Clustering and representatives

Records are unioned when they share any of:

- the same `group_id`;
- the same raw SHA-256;
- the same normalized SHA-256;
- a near-duplicate edge.

Connected components are transitive across source IDs and languages. Every
member and its non-exclusive reasons (`source_group`, `exact_raw`,
`exact_normalized`, `near_duplicate`) remain in the report; duplicate lineage is
not discarded.

Members are sorted by `record_id`. The cluster ID is the SHA-256 of canonical
JSON (UTF-8, sorted IDs, compact separators) of the sorted member IDs. The
representative is the lexicographically smallest member ID. This is deterministic
identity selection, **not** a quality, length or language preference.

## 6. Quarantine and split

A supplied benchmark exclusion string is normalized with the same NFC/LF rule.
If any member of a cluster has that exact normalized fingerprint, the whole
cluster is quarantined. A cluster is also quarantined when any member's
`record_id` appears in the explicit exclusion list. Reasons are
`benchmark_exact_normalized` and `excluded_record_id`; unknown excluded IDs are
ignored. This is exact-match quarantine only and makes no fuzzy, semantic,
translated or paraphrase contamination claim.

Non-quarantined clusters are assigned as whole units:

```text
bucket = int.from_bytes(SHA-256(seed + "\0" + cluster_id)[:8], "big") % 100
train if bucket < train_pct
validation if bucket < train_pct + val_pct
test otherwise
```

Defaults are 98/1/1; tiny tests may pass explicit percentages. There is no
forced balancing, post-hoc reassignment or record-level split. The tiny demo
must not be read as evidence of real 98/1/1 proportions.

## 7. Audit, report and digest

`audit_splits` checks that non-quarantined cluster IDs are disjoint, normalized
exact fingerprints and source groups are not shared across splits, every
near-duplicate edge lies inside one cluster, and quarantined clusters have no
split. `build_report` raises `AuditError` if any check fails; a failed report
must not be used.

The report contains the validated clusters, member IDs/reasons/fingerprints,
representatives, per-split cluster IDs, quarantined cluster IDs, near-duplicate
pairs, audit result and aggregate statistics: raw, retained-representative and
quarantined record/byte/language/domain totals, duplicate counts, cluster counts,
split counts and quarantine reason counts. It contains no timestamps.

`canonical_digest` is SHA-256 over canonical JSON of the report excluding the
`digest` field (sorted keys, compact separators, ASCII escapes). Output is
independent of input record order.

## 8. Importable API

From `PYTHONPATH=tools`, `import corpus_reference`. Public signatures:

```python
build_allowlist(entries: Iterable[SourceApproval]) -> dict[str, SourceApproval]
validate_records(records: Iterable[Record], allowlist: Iterable[SourceApproval] | Mapping[str, SourceApproval]) -> tuple[ValidatedRecord, ...]
normalize_text(text: str) -> str
raw_sha256(text: str) -> str
normalized_sha256(text: str) -> str
decode_text(data: bytes) -> str
word_shingles(text: str) -> frozenset[tuple[str, ...]]
jaccard(a, b) -> Fraction
near_duplicate(a, b) -> bool
cluster_identifier(member_ids: Iterable[str]) -> str
cluster_records(validated: Iterable[ValidatedRecord]) -> tuple[tuple[ClusterResult, ...], tuple[tuple[str, str], ...]]
split_bucket(cluster_id: str, seed: str) -> int
split_for_cluster(cluster_id: str, seed: str = DEFAULT_SEED,
                  train_pct: int = DEFAULT_TRAIN_PCT,
                  val_pct: int = DEFAULT_VAL_PCT) -> str
assign_clusters(clusters, *, benchmark_exclusions=(), excluded_record_ids=(),
                seed=DEFAULT_SEED, train_pct=DEFAULT_TRAIN_PCT,
                val_pct=DEFAULT_VAL_PCT) -> tuple[ClusterResult, ...]
audit_splits(clusters, near_pairs) -> dict[str, Any]
build_report(records, allowlist, *, benchmark_exclusions=(), excluded_record_ids=(),
             seed=DEFAULT_SEED, train_pct=DEFAULT_TRAIN_PCT,
             val_pct=DEFAULT_VAL_PCT, include_inputs=False) -> dict[str, Any]
demo_report() -> dict[str, Any]
canonical_digest(payload: Mapping[str, Any]) -> str
main(argv: list[str] | None = None) -> int
```

| Function | Purpose |
|---|---|
| `build_allowlist(entries)` | Validate synthetic `SourceApproval` entries; return a source-ID mapping. |
| `validate_records(records, allowlist)` | Revalidate allowlist mapping/approval, bound record consumption, and run fail-closed record/metadata/capacity checks; retain original text. |
| `normalize_text(text)` | NFC plus CRLF/CR to LF for fingerprinting only. |
| `raw_sha256(text)`, `normalized_sha256(text)` | Exact fingerprints. |
| `decode_text(data)` | Strict UTF-8 bytes-to-string ingestion. |
| `word_shingles(text)` | Set of 5-word tuples over NFC/LF whitespace tokens. |
| `jaccard(a, b)`, `near_duplicate(a, b)` | Exact set similarity and inclusive 0.8 test. |
| `cluster_identifier(member_ids)` | Stable SHA-256 cluster ID. |
| `cluster_records(validated)` | Components, sorted members/reasons and near-duplicate pairs. |
| `split_bucket(cluster_id, seed)`, `split_for_cluster(...)` | Deterministic bucket and split label. |
| `assign_clusters(clusters, ...)` | Whole-cluster quarantine and split assignment. |
| `audit_splits(clusters, near_pairs)` | Leakage checks; `ok` is false on any leak. |
| `build_report(records, allowlist, ...)` | End-to-end report; raises on invalid input or audit failure. |
| `demo_report()` | Fixed synthetic demonstration report. |
| `canonical_digest(report)` | Stable report digest. |
| `main(["--demo"])` | CLI that prints the demo report as JSON. |

`build_report` accepts benchmark exclusion strings, excluded record IDs, seed,
train/validation percentages and an optional `include_inputs` flag that embeds
the synthetic input texts for evidence.

## 9. Demo

`python3 tools/corpus_reference.py --demo` prints a deterministic JSON report
over 20 original synthetic records covering EN/PL prose, combining marks,
diacritics, CRLF, source families, an exact normalized duplicate, the inclusive
0.8 threshold edge, a below-threshold pair, short documents, a transitive
near-duplicate bridge, a benchmark-quarantined family, an ID-excluded family and
a near-but-not-exact benchmark string that remains eligible. The demo uses
50/25/25 split percentages only to exercise all three labels; it makes no claim
about production proportions.

## 10. Complexity and limitations

- All-pairs near-duplicate comparison is O(n²) in records and shingle-set
  intersection cost; bounded by the 1000-record synthetic limit.
- No MinHash/LSH, production-scale streaming, compression or shard manifests.
- No source acquisition, rights clearance, PII/secrets filtering, malware
  handling or real registry edits.
- No semantic, paraphrased, translated or code-specific deduplication; exact
  normalized matches and 5-word shingles only.
- No benchmark corpus, benchmark-version tracking or false-positive estimate;
  exact normalized quarantine cannot prove absence of contamination.
- No tokenizer, BPE training, token counts or model input normalization.
- Record text has explicit UTF-8/surrogate rejection; separate metadata surrogate validation remains an optional follow-up.
- `group_id` correctness is a caller/registry obligation; a wrong family key can
  over- or under-merge clusters.
- The report is a synthetic engineering oracle, not a measured corpus result.

## 11. Verification

From the repository root, with `python3` because this macOS environment has no
`python` shim:

```bash
PYTHONPATH=tools python3 tools/test_corpus_reference.py
PYTHONPATH=tools python3 tools/test_data_fixture.py
PYTHONPATH=tools python3 tools/corpus_reference.py --demo
```

Run the demo twice and compare bytes; the evidence attempt records the tested
implementation SHA, commands, exit codes, test counts, elapsed durations and
demo digest under `experiments/results/T0004/attempt-001/`.
