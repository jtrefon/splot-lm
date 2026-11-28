# Bounded streaming pilot corpus format (splot-pilot-stream-v1)

2026-11-28 | T0032 / #120 | Production-oriented synthetic prototype | **Not accepted training input**

This document defines the standalone streaming corpus format implemented by `tools/pilot_stream.py`, the deterministic producer `tools/pilot_stream_producer.py` and the independent auditor `tools/pilot_stream_audit.py`. v1 and v2 formats remain immutable and unchanged.

## 1. Scope and status

- New version `splot-pilot-stream-v1`, separate from `splot-pilot-blocks-v1`/`v2`.
- Up to 1.1M valid positions (train ≤1M, validation/test ≤50k each), padding allowance up to 1.6M positions, T≤512, V≤16,384.
- Separate source raw/content files, canonical manifest, document metadata JSONL and block JSONL shards.
- Explicit bounded caps, streaming reads, bounded metadata/index only, test integrity without test exposure.
- Synthetic fixtures and a near-1M local scale run; no acquisition, tokenizer fit, training, CLI/config/workflow change or shared state edit.
- The format is a production-oriented prototype, not an accepted training input or a corpus-quality/BPE claim.

## 2. Root layout

```text
manifest.json
tokenizer-bytes.json
sources.json
documents.jsonl
train-000.jsonl, train-001.jsonl, ...
validation-000.jsonl, ...
test-000.jsonl, ...
sources/<source_id>.raw
sources/<source_id>.content
```

All files are regular non-symlink files; `sources/` is a real directory. No extra files, directories, path traversal or absolute paths are accepted. Source IDs match `[A-Za-z0-9][A-Za-z0-9._-]*`; shard paths are root-level `split-NNN.jsonl` names. The root is a trusted static private directory for one verification call; reads are bounded, but no concurrent-mutation/race-proof guarantee is claimed.

## 3. Caps

| Cap | Value |
|---|---:|
| Total generated files | 128 MiB |
| Total positions (blocks × T) | 1,600,000 |
| Train valid tokens | 1,000,000 |
| Validation valid tokens | 50,000 |
| Test valid tokens | 50,000 |
| Total valid tokens | 1,100,000 |
| Sequence length | 8–512 |
| Vocabulary | 264–16,384 |
| Sources | 32 |
| Document segments | 20,000 |
| Metadata line | 128 KiB |
| Shard line | 256 KiB |
| Single shard | 32 MiB |
| Manifest | 2 MiB |
| Tokenizer artifact | 4 MiB |
| Single source raw/content | 4 MiB |

Caps are checked from the manifest/shard declarations before allocation and again while streaming/reading/writing. A larger corpus requires a reviewed schema/cap amendment.

## 4. Canonical JSON and hashes

- Manifest, tokenizer, sources and every metadata/block JSONL line are canonical JSON: sorted keys, ASCII escapes, compact separators, no duplicate keys/NaN/Infinity.
- `tokenizer_sha256` hashes `tokenizer-bytes.json`.
- `source_registry_sha256` hashes `sources.json`.
- `document_metadata_sha256` hashes `documents.jsonl`.
- `logical_sha256` hashes the concatenated raw shard bytes in manifest shard order.
- Each source file hash/length is declared in `sources.json` and independently recomputed.
- Each document declares full-document and selected-segment SHA-256; the verifier compares both to source slices.
- The independent auditor recomputes all hashes/counts without importing the primary verifier or producer.

## 5. Manifest

Exact keys:

```json
{
  "format": "splot-pilot-stream-v1",
  "version": 1,
  "sequence_length": 512,
  "vocab_size": 264,
  "tokenizer_sha256": "<hex>",
  "tokenizer_special_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3,
                            "reserved": [4, 5, 6, 7]},
  "source_registry_sha256": "<hex>",
  "document_metadata_sha256": "<hex>",
  "shards": [],
  "counts": {},
  "logical_sha256": "<hex>"
}
```

Shard records are ordered train shards, validation shards, test shards; paths must be split-scoped and unique. Each has `split`, `path`, `sha256`, `bytes`, `blocks`, `valid_tokens`, `content_tokens`, `ntp_labels`, `content_labels`, `eos_labels`, `offset2_labels`, `content_target_bytes`, `selected_bytes`, `family_content_tokens`, `family_content_bytes`, `language_content_tokens`, `language_content_bytes`.

`counts` has `blocks`, the eight count fields (`valid_tokens`, `content_tokens`, `ntp_labels`, `content_labels`, `eos_labels`, `offset2_labels`, `content_target_bytes`, `selected_bytes`), and `by_split`, `by_language`, `by_family`; split entities include `blocks`. The verifier requires exact equality with recomputed block/document/shard totals.

## 6. Source registry and files

`sources.json`:

```json
{"format": "splot-source-registry-stream-v1", "version": 1, "sources": [
  {"source_id": "...", "family_id": "...", "language": "en|pl", "split": "train|validation|test",
   "source_url": "...", "revision": "...", "rights_evidence_url": "...",
   "raw_path": "sources/<id>.raw", "raw_sha256": "<hex>", "raw_bytes": 0,
   "content_path": "sources/<id>.content", "content_sha256": "<hex>", "content_bytes": 0,
   "selected_bytes": 0}
]}
```

- Raw bytes are uninterpreted; content bytes must be valid UTF-8.
- `selected_bytes` is the sum of selected document-segment bytes and must be ≤ `content_bytes`.
- Family IDs map to exactly one split/language; all three splits require a source.
- Hashes are integrity checks, not source-authenticity or rights proof.

## 7. Document metadata

`documents.jsonl` lines are canonical and sorted by strictly ascending `doc_id`:

```json
{"doc_id": 0, "segment_id": "...", "source_id": "...", "family_id": "...",
 "language": "en", "split": "train", "source_byte_start": 0, "source_byte_end": 0,
 "segment_byte_start": 0, "segment_byte_end": 0, "full_document_sha256": "<hex>",
 "segment_sha256": "<hex>", "segment_index": 0, "segment_count": 1,
 "truncated": false, "selected_bytes": 0, "token_count": 0}
```

- `source_byte_start/end` are the full document interval in the source content; `segment_byte_start/end` are the selected segment.
- `truncated` is true iff the segment is shorter than the full document.
- Full-document groups within a source are disjoint; selected intervals within a source are globally disjoint (touching half-open intervals allowed).
- Complete groups require indices `0..segment_count-1` in byte order and exact partition; partial groups require `truncated=true`; a single full segment requires index 0/count 1.
- Full document and selected segment slices must hash to the declared values and decode as strict UTF-8.
- Every declared document must appear in a block; orphan declarations fail.

## 8. Tokenizer table

`tokenizer-bytes.json`:

```json
{"format": "splot-byte-tokenizer-stream-v1", "version": 1,
 "actual_vocab_size": 264,
 "special_token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3,
                       "reserved": [4, 5, 6, 7]},
 "token_bytes": ["", "", "", "", "", "", "", "", "00", "01", "..."]}
```

- Fixed special IDs; special byte entries empty; content entries nonempty lowercase hex.
- Duplicate content byte strings are rejected.
- Production-oriented mode requires the complete 256 distinct single-byte alphabet; byte-split Polish diacritics are represented as consecutive single-byte tokens. This is an explicit synthetic table, not a fitted tokenizer artifact.
- A real byte-level BPE must separately qualify its 256-byte alphabet, train-only fit, special IDs and tokenizer hash.

## 9. Block shards

Each JSONL line is one block with exactly:

```json
{"input_ids": [], "document_ids": [], "position_ids": [], "valid": [],
 "control_mask": [], "token_byte_lengths": [], "token_byte_offsets": [],
 "labels": {"1": [], "2": []}}
```

- Arrays length T; valid is a true prefix; padding uses input 0, doc -1, position 0, valid false, control true, byte length 0, byte offset -1, labels -100.
- Each document run starts BOS, ends EOS, has contiguous positions and at least one content token; segments never cross blocks or splits.
- Content IDs are `8..V-1`; BOS/EOS only at boundaries; control mask/byte lengths/offsets must match token bytes exactly.
- Content bytes concatenate to the selected source segment, hash to `segment_sha256`, and decode as UTF-8.
- Labels offset 1/2 are the same-document future token or -100; no cross-segment/padding labels.

## 10. Content-only BPB target alignment

For an offset-1 label at position `i`, the target is `i+1`:

- content target: byte length `token_byte_lengths[i+1]` enters the content numerator/denominator;
- EOS target: NLL reported separately, zero bytes, excluded from BPB;
- BOS/control/padding targets are invalid.

`iter_targets` yields verified target records for train/validation only; `verify_targets` rejects forged/duplicate/misordered identities; `aggregate_bpb` accepts synthetic per-target NLLs, rejects finite/negative/overflow/zero-byte cases and returns content BPB plus separate EOS/control and ordinary NTP totals. No model evaluation is performed.

## 11. Streaming API and non-exposure

- `load_stream_dataset(root, manifest_sha)` verifies metadata and all shards; returns counts/summary without blocks.
- `iter_blocks(root, manifest_sha, split)` revalidates the dataset and yields train/validation blocks only; test is refused.
- `iter_targets(...)` yields train/validation targets only.
- `verify_targets(...)` checks caller records against validated targets.
- Test shards are integrity-checked but never returned as blocks or scoring targets.
- The producer writes to a new private staging directory, verifies, then atomically renames; existing output/staging paths are refused.

## 12. Deterministic producer and independent audit

- `pilot_stream_producer.py` accepts explicit `SourceSpec`/`SegmentSpec` records and token IDs; validates caps/provenance/UTF-8/hashes/token-byte equality/segment groups; packs multi-document blocks with padding tails; writes separate source files, metadata and multiple shards; refuses existing output; same inputs produce byte-identical files.
- `pilot_stream_audit.py` independently re-parses the canonical files and recomputes source/document/selected bytes, labels, per-family/language/split counts, hashes and test integrity without importing the primary verifier or producer. Audit output never contains test blocks.
- `make_scale_spec()` builds a deterministic synthetic near-1M plan (8/4/4 families, full/partial segments, multiple shards, T512, complete alphabet). It is not a learning corpus.

## 13. Format migration and compatibility

- v1 and v2 are immutable and remain the only accepted training formats; stream-v1 is standalone and not connected to the CLI.
- This is a new version, not a cap inflation of v2. A future production loader/training adapter requires a separate reviewed task, real rights registry, tokenizer qualification, independent audit and explicit owner authorization.
- `research/data/source-registry-template.md` is the machine-readable registry template and review checklist for a future 8/4/4 acquisition. It is not an actual source list.
- `research/data/pilot-v2-scale-plan.md` remains the prior planning document; this stream format implements the bounded separation of source/metadata/shards but does not acquire data.

## 14. Limitations

- Synthetic only: no actual corpus, tokenizer fit, source authenticity, rights clearance, model evaluation or BPB/quality claim.
- Scale fixture is intentionally synthetic and must not be described as a learning corpus.
- Static trusted root/no race-proof assumption; file hashes are integrity checks, not authenticity.
- The producer holds the explicit segment/token plan in memory; the reader streams shards, but this is not a production streaming-service claim.
- No merge, training, closure or next-task launch; no shared roadmap/state edit.
