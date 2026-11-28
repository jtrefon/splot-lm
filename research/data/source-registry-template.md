# Source registry template and review checklist

2026-11-28 | T0032 / #120 | **Template only; no sources acquired or approved**

This is a machine-readable starting point for a future lawful bilingual corpus registry. It is not an actual source list, does not imply that any source is available, and does not infer rights from hashes. Acquisition requires a separately reviewed task and explicit owner authorization.

## 1. Machine-readable template

```json
{
  "schema": "splot-source-registry-template-v1",
  "status": "template-only",
  "review": {
    "owner_authorized": false,
    "rights_reviewed": false,
    "family_split_reviewed": false,
    "tokenizer_sample_reviewed": false,
    "contamination_reviewed": false
  },
  "targets": {
    "train_unique_valid_tokens": 1000000,
    "validation_tokens": 50000,
    "test_tokens": 50000,
    "family_minimum": {"train": 8, "validation": 4, "test": 4},
    "family_preferred": {"train": 12, "validation": 6, "test": 6},
    "language_balance": {"train": {"en": 4, "pl": 4},
                         "validation": {"en": 2, "pl": 2},
                         "test": {"en": 2, "pl": 2}},
    "sequence_length": 512,
    "tokenizer_vocab": 16384
  },
  "tokenizer": {
    "fit_scope": "train-only",
    "sample_bytes_max": 8388608,
    "complete_byte_alphabet_required": true,
    "normalizer": null,
    "add_prefix_space": false,
    "artifact_private": true,
    "artifact_sha256": null,
    "special_token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3,
                          "reserved": [4, 5, 6, 7]}
  },
  "sources": [
    {
      "source_id": "placeholder-source-id",
      "family_id": "placeholder-family-id",
      "language": "en",
      "split": "train",
      "title": "placeholder",
      "source_url": "https://example.invalid/source",
      "revision": "immutable-revision-placeholder",
      "retrieved_at_utc": null,
      "license": "public-domain-or-reviewed-license-placeholder",
      "rights_evidence_url": "https://example.invalid/rights",
      "raw_sha256": null,
      "content_sha256": null,
      "raw_bytes": null,
      "content_bytes": null,
      "selected_byte_intervals": [],
      "extraction_policy": "placeholder; no annotations/paratext/PII",
      "exclusions": [],
      "dedup_status": "not_run",
      "notes": "placeholder only"
    }
  ],
  "review_checklist": [
    "owner explicitly authorizes this acquisition task",
    "8/4/4 family minimum (12/6/6 preferred) with family-level split",
    "each source has immutable URL/revision or frozen local hash",
    "rights evidence and exclusions reviewed; no hash-based rights inference",
    "extraction policy excludes annotations, paratext, PII and unrelated material",
    "exact raw/content hashes and byte counts recorded",
    "full-document and selected-segment byte intervals recorded",
    "dedup exact/normalized/near-duplicate and cross-split n-gram audit",
    "tokenizer sample is train-only, frozen and independently hashed",
    "complete 256-byte alphabet and duplicate-free content token table qualified",
    "per-split/family token and byte budgets within stream-v1 caps",
    "validation/test non-exposure and test freeze verified",
    "independent auditor reconciles all counts/hashes before training",
    "no acquisition, install, training or CLI change before approval"
  ],
  "next_tasks": {
    "acquisition": "separately reviewed VAL/IMP task; actual source list unresolved",
    "producer": "separately reviewed IMP task using stream-v1 producer/validator",
    "tokenizer": "separately reviewed IMP/EXP qualification task",
    "cli_adapter": "separately reviewed IMP/VAL task after producer/tokenizer acceptance"
  }
}
```

## 2. Family and split requirements

- Minimum 8 train families (4 EN / 4 PL), 4 validation (2/2), 4 test (2/2); preferred 12/6/6.
- A family belongs to exactly one split and language. No document, segment, translation or edition crosses splits.
- Record per-family token/byte budgets and actual counts. More families improve descriptive stability but do not guarantee population power.
- Test families stay frozen and are integrity-checked only; no test blocks or scoring targets are exposed.

## 3. Rights and extraction review

- Record official URL, immutable revision/frozen hash, retrieval time, license/rights evidence and exclusion policy.
- No copyrighted text without explicit permission; no PII, private correspondence, leaked/paywalled material or unclear provenance.
- Exclude annotations, introductions, illustrations, editorial notes and other separately licensed matter unless explicitly approved.
- A hash proves byte identity only; it does not establish rights, authorship, source authenticity or redistribution approval.
- Keep raw/content files private; publish only approved hashes/counts and tiny synthetic fixtures.

## 4. Dedup and contamination

- Exact SHA-256 dedup at document level.
- Normalized dedup and near-duplicate MinHash/SimHash on 5-gram shingles.
- Remove repeated editions/translations across families unless explicitly studied.
- 13-gram cross-split leakage audit with a predeclared threshold (planning default <0.1%).
- Probe/template contamination audit separate from corpus split audit.

## 5. Tokenizer binding

- Fit one deterministic byte-level BPE on a train-only sample; never fit on validation/test.
- Freeze the tokenizer across scales; publish artifact SHA, training-input digest, sample bytes/segments, special IDs and complete-alphabet audit.
- Require all 256 byte values, duplicate-free content token bytes and exact encode/decode roundtrip.
- The stream-v1 synthetic tokenizer table is not a production tokenizer artifact.

## 6. Independent audit and next tasks

Before training, a second implementation must reconcile:

- source raw/content hashes and selected bytes;
- full/selected document hashes, intervals and UTF-8;
- labels/masks/content-target bytes and per-split/family/language counts;
- shard/logical/manifest hashes and test non-exposure;
- cap usage and private storage inventory.

Acquisition, production producer/loader, tokenizer qualification and CLI/training adapter are separate reviewed tasks. This template authorizes none of them.
