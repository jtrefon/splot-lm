# Byte-auditable pilot corpus v2 schema and verifier

2026-11-28 | T0028 / #112 | Standalone synthetic prototype | **Not accepted training input**

This document specifies `splot-pilot-blocks-v2`, a canonical corpus format with exact token-byte coverage, and the bounded stdlib verifier in `tools/pilot_v2.py`. It is a prototype for synthetic fixtures only. It does **not** acquire data, fit a tokenizer, train a model, integrate with the training CLI, or establish real-source authenticity. The v1 format and `tools/pilot_ingestion.py` are unchanged.

## 1. Scope and non-goals

In scope:

- versioned canonical JSON manifest and deterministic shard order;
- registered source/family/language/split provenance for every document segment;
- full-source and selected-segment byte intervals/hashes;
- a synthetic tokenizer table mapping token IDs to arbitrary bytes;
- per-token byte lengths/offsets and exact content-target byte aggregation;
- a content-only NLL/BPB helper with EOS reported separately;
- bounded streaming verification and fail-closed synthetic tamper tests.

Out of scope:

- actual corpus acquisition, tokenization, tokenizer fitting, training or evaluation;
- changes to v1, `tools/experiment.py`, configs, workflows or shared state documents;
- a real tokenizer artifact, real source bytes, test-data exposure or legal/rights clearance;
- production streaming throughput, memory or kernel claims.

## 2. Root layout

A dataset directory contains exactly:

```text
manifest.json
tokenizer-bytes.json      # synthetic token-to-byte table
source-bytes.json         # synthetic raw/content source bytes
train.jsonl
validation.jsonl
test.jsonl
```

All six files must be regular non-symlink files. Extra files, directories, symlinks, missing splits or alternate shard names fail closed. Shards are always `train`, `validation`, `test` in that order; no implicit v1 limits are reused.

## 3. Caps

Caps are explicit v2 constants, checked from the manifest before allocation and again while streaming:

| Cap | Value |
|---|---:|
| total shard bytes | 4 MiB |
| total blocks | 128 |
| train blocks | 96 |
| validation blocks | 16 |
| test blocks | 16 |
| manifest bytes | 1 MiB |
| tokenizer artifact bytes | 2 MiB |
| source-bytes artifact bytes | 4 MiB |
| shard line bytes | 128 KiB |
| sources | 32 |
| document segments | 512 |
| vocabulary | 8–16,384 |
| sequence length | 8–512 |

These are not a rescaling of the v1 128/16/16 limits; a larger v2 fixture needs a reviewed cap change.

## 4. Canonical JSON and hashing

- Manifest, tokenizer artifact, source-bytes artifact and every JSONL block are canonical JSON: sorted keys, ASCII escapes, `(",", ":")` separators, no NaN/Infinity, no duplicate keys.
- `manifest_sha256` is the SHA-256 of the canonical manifest bytes supplied by the caller.
- `tokenizer_sha256` is the SHA-256 of `tokenizer-bytes.json`.
- `source_bytes_sha256` is the SHA-256 of `source-bytes.json`.
- `source_snapshot_sha256` is the SHA-256 of canonical source records sorted by `source_id`.
- `logical_sha256` is the SHA-256 of the concatenated raw shard bytes in train, validation, test order.
- `byte_coverage_sha256` is the SHA-256 of the canonical coverage-record list sorted by `doc_id`; the record keys are the document keys listed in §7.
- `segment_sha256` is the SHA-256 of the selected segment bytes; `full_document_sha256` is the SHA-256 of the full extracted document interval.
- Hash checks establish byte integrity for the supplied synthetic artifacts. They do **not** prove that a real source was fetched, licensed, unmodified upstream or independently authentic.

## 5. Manifest

Exact top-level keys:

```json
{
  "format": "splot-pilot-blocks-v2",
  "version": 2,
  "sequence_length": 32,
  "vocab_size": 32,
  "tokenizer_sha256": "<64 hex>",
  "tokenizer_special_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3,
                            "reserved": [4, 5, 6, 7]},
  "source_snapshot_sha256": "<64 hex>",
  "source_bytes_sha256": "<64 hex>",
  "sources": [],
  "documents": [],
  "shards": [],
  "counts": {},
  "logical_sha256": "<64 hex>",
  "byte_coverage_sha256": "<64 hex>"
}
```

`format` and `version` are exact. `sequence_length` and `vocab_size` are strict integers within the caps. No additional top-level keys are allowed.

## 6. Source records

Each source has exactly:

| Field | Meaning |
|---|---|
| `source_id` | unique nonempty registry ID |
| `family_id` | split unit; one split and language per family |
| `language` | `en` or `pl` |
| `split` | `train`, `validation` or `test` |
| `source_url` | official source location |
| `revision` | immutable commit/tag or frozen local SHA |
| `rights_evidence_url` | rights/license evidence |
| `raw_sha256` | SHA-256 of the full raw source bytes |
| `content_sha256` | SHA-256 of full extracted UTF-8 source bytes |
| `raw_bytes` | full raw byte count |
| `content_bytes` | full extracted byte count |
| `selected_bytes` | sum of selected document-segment bytes |

All three splits must have at least one source. A family may have multiple sources but must not cross split or language. `selected_bytes <= content_bytes`; the sum of document `selected_bytes` for the source must equal this field exactly. `source-bytes.json` supplies synthetic raw/content bytes and must match all four byte/hash fields.

`source-bytes.json`:

```json
{"format": "splot-source-bytes-v2", "version": 1,
 "sources": {"src-id": {"raw": "<hex>", "content": "<hex>"}}}
```

Raw and content are distinct fields even when a producer stores identical bytes. The verifier recomputes both hashes and lengths. Real corpora should not publish source bytes in Git; this synthetic artifact exists only to prove the verifier can separate full-source bytes from selected document bytes.

## 7. Document and segment records

Each selected segment is one document record with exactly:

| Field | Meaning |
|---|---|
| `doc_id` | globally unique nonnegative integer |
| `segment_id` | globally unique nonempty string |
| `source_id`, `family_id`, `language`, `split` | provenance; must match the source and family split |
| `source_byte_start`, `source_byte_end` | full document interval in `content` bytes |
| `segment_byte_start`, `segment_byte_end` | selected segment interval inside the full document |
| `full_document_sha256` | hash of `[source_byte_start, source_byte_end)` |
| `segment_sha256` | hash of `[segment_byte_start, segment_byte_end)` |
| `segment_index`, `segment_count` | 0-based index and total segments for this full document |
| `truncated` | `true` iff the selected segment is shorter than the full document |
| `selected_bytes` | `segment_byte_end - segment_byte_start` |
| `token_count` | number of content tokens in the segment |

Boundary policy:

- A complete single segment has `segment_count=1`, `segment_index=0`, `truncated=false`, and covers the full document; its full and segment hashes must match.
- A truncated document has one or more selected segments with the same full-document interval/hash, distinct `segment_index` values, consistent `segment_count`, and non-overlapping segment intervals. If all `segment_count` segments are present, they must exactly partition the full document; partial selection is allowed only with `truncated=true`, and only selected bytes count.
- Every segment is independently tokenized with BOS/content/EOS and its own position reset. No label, byte span or document ID crosses a segment, block or split.
- Source bytes are sliced by these intervals, decoded as strict UTF-8, and hashed. This verifies the full document and selected segment, not real-world source authenticity.

## 8. Tokenizer artifact

`tokenizer-bytes.json` is a synthetic, local/private-style table:

```json
{"format": "splot-byte-tokenizer-v2", "version": 1,
 "actual_vocab_size": 32,
 "special_token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3,
                       "reserved": [4, 5, 6, 7]},
 "token_bytes": ["", "", "", "", "", "", "", "",
                 "c4", "85", "c5", "bc", "61", "..."]}
```

- `token_bytes` has exactly `actual_vocab_size` entries.
- Special entries (`0–7`) are empty; content entries are nonempty lowercase hex.
- Content token bytes may be arbitrary and need not decode to UTF-8 individually. Concatenating all content-token bytes in a segment must decode to valid UTF-8 and equal the selected source segment.
- Token bytes are not published for a real tokenizer; the artifact is a synthetic table with its own SHA. Special IDs are duplicated in the manifest and must match exactly.
- The verifier does not import `tokenizers`, `torch` or any optional package.

## 9. Block schema

Each JSONL line is one block with exactly:

```json
{
  "input_ids": [], "document_ids": [], "position_ids": [], "valid": [],
  "control_mask": [], "token_byte_lengths": [], "token_byte_offsets": [],
  "labels": {"1": [], "2": []}
}
```

All arrays have exactly `sequence_length` entries. Types are strict: integers are `type(x) is int`, masks are `type(x) is bool`.

- `valid` is a true prefix followed by padding; at least one segment with BOS, one content token and EOS.
- Each contiguous `document_ids` run is one segment: `doc_id >= 0`, BOS at the start, EOS at the end, `position_ids` reset to `0..len-1`, at least one content token.
- `control_mask` is true for BOS/EOS and padding, false for content. Content IDs are in `[8, vocab_size)`; BOS=1 and EOS=2 only at boundaries; UNK/reserved IDs are not allowed in blocks.
- `token_byte_lengths`: BOS/EOS = 0; content = exact length of that token's byte entry; padding = 0.
- `token_byte_offsets`: BOS = 0; content starts at the cumulative byte count; EOS = segment byte length; padding = -1. Content offsets must be contiguous with no gaps or overlaps.
- Reconstructed content bytes must have `len == segment.selected_bytes`, hash to `segment_sha256`, decode as strict UTF-8 and equal the source segment slice.
- `labels["1"]`/`labels["2"]` follow v1 semantics: for a valid position with the same document at offset 1/2, the value is the future token ID; otherwise `-100`. Labels never cross a segment, padding or split.
- Padding: `input_ids=0`, `document_ids=-1`, `position_ids=0`, `valid=false`, `control_mask=true`, byte length 0, byte offset -1, labels `-100`.

### Content-target byte rule

For every offset-1 label at position `i` with target `i+1`:

- if the target is a content token, its target byte length is `token_byte_lengths[i+1]` and it enters the content numerator and denominator;
- if the target is EOS, its NLL is reported separately with zero bytes and is excluded from BPB;
- padding/control targets cannot occur.

Because every content token is targeted exactly once (BOS→first content, content→next content, last content→EOS), the v1-style schema has these invariants per segment: `content_labels = content_tokens`, `eos_labels = 1`, `ntp_labels = content_tokens + 1`, `offset2_labels = content_tokens`, and `content_target_bytes = selected_bytes`. The verifier checks them; a different label policy would require a new schema version.

## 10. Counts and reconciliation

Manifest `counts` has exactly `blocks`, the eight count fields (`valid_tokens`, `content_tokens`, `ntp_labels`, `content_labels`, `eos_labels`, `offset2_labels`, `content_target_bytes`, `selected_bytes`), and `by_split`, `by_language`, `by_family`. `by_split` entities include `blocks`; language/family entities do not. Shards repeat the split entity plus `family_content_tokens`, `family_content_bytes`, `language_content_tokens`, `language_content_bytes`.

The verifier computes every count from validated blocks and document records and requires exact equality with the shard and manifest declarations. `selected_bytes` is the sum of document selected bytes; `content_target_bytes` is the sum of exact target-token byte lengths. For this label policy they must be equal.

## 11. Verifier and API

`tools/pilot_v2.py` exposes:

- `load_v2_dataset(root, expected_manifest_sha256)` → verified train/validation blocks and a summary. Test shards are fully integrity-checked but never returned; the summary contains only test counts, not test blocks or token arrays.
- `scoring_targets(dataset, split="validation")` → offset-1 target records (`target_token_id`, `byte_length`, `kind`, block/position, family/language). The `test` split is refused.
- `verify_targets(dataset, split, targets)` → compares caller-supplied targets to validated block labels/bytes, rejects wrong fields, duplicate `(block_index, position)` identities and test access.
- `aggregate_content_bpb(targets)` → arithmetic-only content-only NLL/BPB plus separate EOS/control and ordinary NTP totals. It rejects nonfinite, negative or overflow NLLs, zero-byte content denominators, EOS/control claiming bytes, unknown kinds and empty targets. It does **not** verify target identity.
- `aggregate_verified_bpb(dataset, split, nlls, targets=None)` → dataset-bound aggregation; verifies target identity/order and returns per-family/per-language totals plus EOS/control separately.
- `TOKENIZER_SCOPE = "synthetic-explicit-table"`; the table is not a production tokenizer. A real byte-level BPE must separately qualify a complete 256-byte alphabet, special IDs, duplicate-free content tokens and train-only fit.

T0030 hardening semantics:

- post-scan `seen_doc_ids` must equal the exact declared document-ID set; orphan or unreferenced declarations are rejected;
- verified shard selected bytes must equal document selected bytes and per-source manifest selected bytes;
- full-document groups and selected intervals must be disjoint within each source (touching half-open intervals are allowed);
- complete groups require indices `0..segment_count-1` in byte order and exact partition; partial groups require explicit `truncated=true`; a single full segment requires index 0 / count 1;
- full extracted source and full document bytes must be valid UTF-8, not only selected segments; selected bytes and token-byte concatenation are checked independently;
- strict non-bool integer schema applies to special IDs, padding labels, masks, offsets, counts and segment indices;
- duplicate content token byte strings are rejected; token reconstruction is bounded incrementally by declared selected bytes and the source cap before allocation;
- NLLs must be finite, nonnegative and non-overflowing at per-target and aggregate level; the verified API prevents target/label identity mismatch and duplicate target records;
- the dataset root is a trusted static root for one call; unsafe paths/symlinks are rejected and reads are bounded, but there is no concurrent-mutation or race-proof guarantee.

Verification order:

1. exact file set, regular files, no symlinks;
2. manifest hash/canonical/schema/caps;
3. tokenizer hash/canonical/table/special IDs/duplicate content bytes;
4. source records, source snapshot, source-bytes hash/raw/content hashes/lengths/full UTF-8;
5. document IDs/intervals/full and segment hashes/source slicing/full and selected UTF-8/segment groups/global interval disjointness;
6. byte-coverage digest;
7. shard declarations and aggregate caps before allocation;
8. streaming canonical blocks; strict label/mask/type checks and incrementally bounded byte reconstruction; test blocks validated but not stored;
9. shard hash/byte/line/block counts and family/language maps;
10. exact declared-vs-seen document IDs, selected-byte reconciliation, logical digest and aggregate counts.

Any failure raises `PilotV2Error`. The verifier never imports the model, never evaluates a model, and never exposes test blocks. `TOKENIZER_SCOPE` is synthetic-only; no claim is made that the explicit table is a production BPE.

## 12. Synthetic fixture coverage

The test suite in `tools/test_pilot_v2.py` uses a deterministic 32-token vocabulary and 32-position blocks:

- seven synthetic sources/families, all three splits, EN and PL text;
- eight document segments, including a two-segment truncated document with a shared full-document hash;
- multibyte diacritics and tokens that split UTF-8 codepoints (`C5 BC`, `C4 85`);
- padding, multiple documents per block and a segment tail;
- expected totals: 3 blocks, 36 valid tokens, 20 content tokens, 28 NTP labels, 20 content labels, 8 EOS labels, 20 offset-2 labels, 21 content-target bytes, 21 selected bytes;
- by-language content bytes: EN 13, PL 8;
- hand-computed validation BPB check: content NLL 2.3 over 7 bytes, EOS NLL 1.3, ordinary NTP NLL 3.6;
- T0030 negative cases with recomputed hashes/counts: orphan declared document, cross-group full-document overlap, invalid full-source/full-document/selected-segment UTF-8, duplicate content token bytes, incremental reconstruction overflow, segment index/order/partial-selection violations, strict padding-label type, negative/overflow NLL, forged/duplicate/misordered verified targets, test access;
- positive identity/repeat, hand-computed BPB, test-split refusal, caps, block/label/padding tamper, provenance/split/segment tamper, tokenizer/source/full-document hash tamper, invalid UTF-8 concatenation, noncanonical manifest, extra file, symlink and aggregate digest tamper.

The original valid fixture hashes/counts/BPB are unchanged; T0030 tightens validation only. The fixture text and token arrays are deliberately tiny and synthetic; no real corpus, tokenizer, checkpoint or test data is present.

The producer test suite in `tools/test_pilot_v2_producer.py` adds six tests: byte-identical reproduction of the T0028 fixture, deterministic repeat, expected-coverage/loader reconciliation, existing-output refusal, input validation, and multi-document/tail packing. Producer totals match: 3 blocks, 36 valid, 20 content, 28 NTP, 8 EOS, 21 selected bytes.

## 13. Deterministic synthetic producer

`tools/pilot_v2_producer.py` is a standalone synthetic producer for the exact v2 schema:

- inputs are explicit `SourceSpec` raw/content bytes, `SegmentSpec` source/document/segment intervals, explicit content `token_ids` and a supplied token-byte table; no acquisition, tokenizer fit, source fetch or implicit tokenization;
- it validates provenance, UTF-8, hashes, token-byte equality, segment groups/indices/overlaps, caps and block fit before writing;
- it packs segment-local BOS/content/EOS blocks greedily in split order, preserves no cross-segment labels/bytes, supports multiple documents and a padded tail, and refuses an existing output path;
- `expected_coverage(spec)` computes per-source/segment/family/language/label counts and hashes before the verifier; `write_dataset(spec, out)` writes the canonical six files only after the in-memory build validates;
- the same inputs produce byte-identical files; the producer test reproduces the T0028 fixture manifest/tokenizer/source/shard files byte-for-byte;
- no real text/token arrays are published; producer evidence contains only tiny synthetic fixtures and derived counts/hashes.

The producer is synthetic-only and is not a production acquisition/producer pipeline. It does not enlarge caps or adopt v2 for training.

## 14. Compatibility and migration prerequisites

- **v1 is unchanged.** `splot-pilot-blocks-v1`, `tools/pilot_ingestion.py`, the training CLI and configs still use the accepted v1 path. v2 is a standalone prototype and is **not** accepted training input.
- v2 schema fields/limits are unchanged; T0030 tightens semantic validation. Valid T0028 fixture hashes/counts/BPB remain unchanged; previously accepted malformed cases may now fail closed.
- The synthetic producer is deterministic but does not provide source acquisition, rights review, tokenizer fitting or a real source store. A future production producer must keep the tokenizer table local/private, use streaming/binary source artifacts, verify independent hashes and publish only approved derived evidence.
- A future loader/training integration requires a separate reviewed IMP task, explicit owner authorization, v1 regression preservation, a real source registry with rights/revision evidence, and an independent byte-coverage audit.
- Larger-source registry prerequisites: family-level split and dedup, selected-vs-full byte accounting, rights evidence, frozen tokenizer, test freeze, complete 256-byte tokenizer alphabet qualification, and no reliance on v1 block caps.
- The bounded migration, hypothetical 1M-token sizing, binary/JSONL shard options and separately reviewed acquisition/producer/CLI tasks are in [`pilot-v2-scale-plan.md`](pilot-v2-scale-plan.md).

## 15. Limitations

- Synthetic only: no actual dataset, tokenizer fit, source authenticity, legal clearance, training or evaluation.
- `source-bytes.json` is synthetic; the verifier proves hash/coverage mechanics, not that a real source is correct or licensed.
- The fixture is deliberately tiny and does not establish production streaming throughput, memory or scale.
- The root is a trusted static root for one verification call; symlinks/paths are rejected and reads bounded, but no concurrent-mutation/race-proof guarantee is claimed.
- The explicit tokenizer table is not a production BPE; full 256-byte alphabet and train-only fit require separate qualification.
- No BPB, quality, architecture, cost or generalization claim is made from this prototype.
- v2 is not integrated with training, configs or v1; no shared roadmap/state document is changed by T0028/T0030.
