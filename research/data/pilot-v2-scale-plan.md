# v2 synthetic-to-production scale migration plan

2026-11-28 | T0030 / #116 | Planning only | **No acquisition, download, tokenizer fit, dependency install or training**

This document proposes a bounded migration from the synthetic `splot-pilot-blocks-v2` prototype to a future production corpus. It does **not** enlarge the current v2 caps, change the accepted v1 training path, acquire sources, fit a tokenizer or implement a production loader. The first target is **1,000,000 unique valid training tokens**; larger targets require separate reviewed tasks.

## 1. Principles

- Keep the v2 semantic contract: family-level split, exact source/document/segment byte intervals, full and selected hashes, explicit token-byte mapping, content-only BPB and EOS separation.
- Do not extrapolate production sizes from T0016's full extracted bytes. The ranges below are hypothetical planning assumptions, not measured corpus statistics.
- Source bytes, tokenizer artifacts and token arrays stay out of Git unless explicitly approved; only hashes, counts and synthetic fixtures are published.
- Validation/test families are frozen and never used for tokenizer fitting, tuning or iteration. Test data is integrity-checked only and never returned by the public loader.
- Every production input gets an independent hash and an independent coverage audit before training.

## 2. First 1M-token target

Planning target: **1,000,000 unique valid training tokens**, 5% validation (50,000) and 5% test (50,000), total 1,100,000 tokens.

Hypothetical bytes/token ranges (explicit assumptions, **not** derived from T0016):

| Measure | Hypothetical range | 1.1M-token total (est.) |
|---|---:|---:|
| Original/raw UTF-8 bytes per token | 3.5–6.0 | 3.85–6.6 MB raw |
| Extracted/selected UTF-8 bytes per token | 2.5–4.5 | 2.75–4.95 MB extracted |
| Compact binary shard bytes per token | 12–20 | 13.2–22.0 MB |
| Canonical JSONL bytes per token | 35–60 | 38.5–66.0 MB |
| Tokenizer artifact | 1–2 MB | 1–2 MB |
| Manifest + registry + sidecars | <2 MB | <2 MB |

These ranges only bound storage planning. Actual bytes/token depends on language, source markup, tokenizer fertility and block packing; measure them on the approved corpus before setting a hard budget. No storage/compute extrapolation from T0016 is made.

Family/split planning:

| Split | Minimum families | Preferred families | Approx. tokens/family at 1M |
|---|---:|---:|---:|
| train | 8 (4 EN / 4 PL) | 12 (6/6) | 83k–125k (8 families) |
| validation | 4 (2/2) | 6 (3/3) | 8k–13k |
| test | 4 (2/2) | 6 (3/3) | 8k–13k |

A family is the split unit. More families improve family-level descriptive stability but do not by themselves establish population power. Per-language and per-family token/byte counts are mandatory manifest fields.

## 3. Source artifacts and registry

Production sources should be streamed, not hex-duplicated in manifest JSON:

```text
private/sources/<source_id>/raw.bin
private/sources/<source_id>/content.utf8
private/sources/<source_id>/metadata.json
```

- `metadata.json` records source/family/language/split, official URL, immutable revision, rights evidence, retrieval timestamp, raw/content SHA-256, byte counts, selected bytes, exclusions, dedup status and document intervals.
- The loader streams raw/content artifacts in bounded chunks, hashes incrementally and slices only declared document/segment intervals. It never loads the full corpus into memory.
- The repository stores the registry, hashes, counts and approved synthetic fixtures; restricted text remains in private storage.
- `source-bytes.json` remains a synthetic test artifact. Production must not publish it.
- Rights review and source approval happen before acquisition; a rights registry entry is not legal clearance or redistribution approval.

## 4. Shard strategy

Two interchangeable representations should share the same manifest and coverage records:

1. **Compact binary** for production streaming:
   - block header/version, sequence length, counts and block checksum;
   - `input_ids` uint16 (vocab ≤16,384 fits uint16);
   - `document_ids` uint32, `position_ids` uint16;
   - `valid` and `control_mask` bitsets;
   - `token_byte_lengths` uint16, `token_byte_offsets` uint32;
   - `labels_1`, `labels_2` int32 (or uint16 plus an explicit IGNORE bitmap) with a documented sentinel;
   - family/language IDs as uint16/uint8 side tables;
   - per-block content-byte totals and label counts.
2. **Canonical JSONL** for audit/debug and tiny fixtures:
   - same fields, canonical JSON, streamed line by line;
   - useful for independent inspection and re-hashing, not for the full 1M-token production store.

The manifest pins the chosen representation, shard SHA-256, byte counts, block counts, per-split/family/language token and byte counts, logical concatenation hash and byte-coverage hash. A format change requires a reviewed schema version.

## 5. Tokenizer binding

- Fit one byte-level BPE tokenizer deterministically on a train-only sample; never use validation/test text.
- Target actual vocabulary 16,384; special IDs fixed separately; no normalization or `add_prefix_space` surprises.
- The tokenizer artifact must expose a complete 256-byte alphabet, duplicate-free content token byte strings and a token-to-byte table. The synthetic v2 table is not sufficient evidence for production.
- Publish tokenizer SHA-256, training-input digest, sample bytes/segments, special IDs and complete-alphabet audit; keep the tokenizer JSON local/private.
- Freeze one tokenizer across 1M/3M/5M stages so scale is not confounded by tokenizer changes.
- Independent qualification must encode/decode every selected document, verify token-byte concatenation equals the selected UTF-8 bytes and check all 256 byte values appear.

## 6. Dedup, leakage and rights

- Exact SHA-256 dedup at document level.
- Normalized dedup (NFC, whitespace, case where appropriate) and near-duplicate MinHash/SimHash on 5-gram shingles.
- Remove repeated editions/translations across families unless the design explicitly studies translation.
- Family-level split; no document, segment or family crosses train/validation/test.
- 13-gram cross-split leakage audit; predeclare the threshold (planning default <0.1%) and report observed overlap.
- Probe contamination audit: probe templates/entities/rules disjoint from all splits.
- Rights/revision registry review before acquisition; record exclusions and mutable-source pins.

## 7. Independent hashes and coverage audit

Before training, two independent implementations must agree on:

- manifest SHA, source snapshot SHA, source-bytes SHA (if used), tokenizer SHA;
- every shard SHA/byte count and logical concatenation SHA;
- byte-coverage record SHA and every document/segment full/selected hash;
- per-split/family/language content-token, content-label, EOS, offset-2 and content-target-byte counts;
- exact selected-byte reconciliation across documents, shards and sources;
- test-integrity status without exposing test blocks.

The production verifier may reuse validation ideas but must not be the only implementation that produces the expected counts. A second audit script or reviewer recomputation is required.

## 8. Split/family budgets and byte counts

At 1M training tokens, planning budgets are:

| Item | Target | Acceptance evidence |
|---|---:|---|
| Train unique valid tokens | 1,000,000 | exact token count and per-family map |
| Validation tokens | 50,000 | exact count, frozen families |
| Test tokens | 50,000 | integrity only, no evaluation |
| Train content bytes | measured | per-language/family exact bytes |
| Validation content bytes | measured | per-language/family exact bytes |
| Test content bytes | measured | manifest only, no public blocks |
| Train families | ≥8 | family map and split disjointness |
| Validation families | ≥4 | family map |
| Test families | ≥4 | family map, frozen |

Counts are targets, not permission to acquire. If the approved corpus cannot reach 1M unique valid tokens, stop and report the shortfall rather than padding with duplicates or changing split semantics.

## 9. Validation and test non-exposure

- The loader returns train and validation blocks only; test blocks are hashed/count-checked but never returned.
- `scoring_targets` and verified BPB APIs refuse `test`.
- Test families are frozen before tokenizer fit and never used for tuning, early stopping, probe design or margin selection.
- A held test evaluation requires a separate reviewed final protocol and explicit owner authorization.

## 10. Migration stages and separate tasks

This plan authorizes no execution beyond T0030's synthetic prototype. Proposed follow-ups, each separately reviewed and owner-authorized:

1. **Acquisition/rights registry task (VAL/IMP):** approve source list, rights evidence, family split and 1M-token budget; no downloads until approved.
2. **Tokenizer qualification task (IMP/EXP):** train-only fit, complete 256-byte alphabet, duplicate-free table, encode/decode roundtrip and artifact hash.
3. **Production binary producer/loader task (IMP):** streaming source artifacts, binary shards, independent coverage audit, v1 regression preservation and no CLI change until accepted.
4. **CLI/training integration task (IMP/VAL):** only after the producer/loader and tokenizer are accepted; preserve v1 and explicit family semantics.
5. **Scale review:** 3M/5M targets only after the 1M stage is measured, with a new cap/schema amendment if needed.

Each stage keeps the previous stage's evidence immutable. No automatic progression, retry or acquisition.

## 11. Open decisions

- Exact source list, rights evidence and family assignments.
- Whether 8/4/4 families are sufficient at 1M or 12/6/6 is required.
- Binary layout details and whether JSONL remains the audit representation.
- Tokenizer sample size and one-tokenizer-across-scales freeze.
- Dedup thresholds and cross-split n-gram limit.
- Whether source bytes are stored as files, object store or another private content-addressed store.
- Multi-window compute/storage authorization for 3M/5M.
- Final independent audit ownership and acceptance criteria.

## 12. Non-claims

No acquisition, download, tokenizer fit, package install, training, measured BPB, quality, scaling, cost or architecture claim. The 1M sizing ranges are hypothetical planning assumptions, not T0016 extrapolations. No production loader or CLI integration exists in T0030.
