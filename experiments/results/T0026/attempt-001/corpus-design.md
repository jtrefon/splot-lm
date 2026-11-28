# T0026 — lawful bilingual corpus and scaling design

Task T0026 (ANA), issue #108. Source/instruction `003cf8db6126f0aaefdd599ac159f2503457f709`. This document plans a future corpus; **no acquisition, download, tokenizer fit, text inspection, test access or data publication occurred in T0026**. All sizes are planning estimates with labelled assumptions.

## 1. Why the T0016 snapshot is insufficient

T0016 established a lawful, reproducible eight-family bilingual pilot:

- 8 families (4 English, 4 Polish), 8,987,423 extracted UTF-8 bytes, 2,360 segments.
- One family per split: 4 train, 2 validation, 2 test.
- Materialized `splot-pilot-blocks-v1`: 160 blocks, 56,201 content tokens, 56,521 valid tokens, 56,361 NTP labels; train only 46,204 valid / 46,076 NTP.
- Actual V16384 ByteLevel BPE, no normalization, train-only 4.19 MB sample; tokenizer artifact is local-only.
- Source-level rights/revision/hash registry; family-level split; independent audit.

This is a feasibility instrument, not a learning corpus. Two validation families cannot estimate family variance, 4 train families cannot support family-level generalization, and 46k train tokens cannot separate architecture effects from optimization noise. T0026 therefore plans 1–5M unique valid training tokens, more families, and a frozen byte-auditable schema.

## 2. Rights and source registry

Every future source must enter a committed registry **before** acquisition, with:

| Field | Requirement |
|---|---|
| `source_id` | stable local identifier |
| `family_id` | work/source family; one split only |
| `language` | `en` or `pl` |
| `split` | `train`, `validation` or `test` |
| `source_url` | official URL or institutional repository |
| `revision` | immutable commit/tag or frozen local SHA-256 |
| `retrieved_at_utc` | acquisition timestamp |
| `license` | public domain / CC0 / CC-BY / other reviewed terms |
| `rights_evidence_url` | license or rights statement |
| `raw_sha256`, `content_sha256` | bytes as retrieved and extracted text |
| `raw_bytes`, `content_bytes` | exact counts |
| `segments` | extracted segment count |
| `exclusions` | front matter, notes, illustrations, annotations, PII |
| `dedup_status` | exact/near-duplicate disposition |
| `notes` | provenance or rights caveats |

Rights rules:

- Public-domain works with documented evidence; CC0/CC-BY only with attribution terms recorded; no copyrighted text without explicit permission.
- No personal data, private correspondence, leaked material, paywalled text or unclear provenance.
- Exclude editorial apparatus, annotations, introductions and illustrations unless separately licensed and explicitly in scope.
- Record a documented rights assessment, not a legal opinion or redistribution approval.
- Do not publish raw text, token arrays or tokenizer JSON without separate approval; publish hashes/counts only.

Candidate source classes (not acquired): public-domain literature from Standard Ebooks/Wolne Lektury, CC0/CC-BY institutional text, public-domain government/legal prose with stable revisions. The final list is an unresolved review decision.

## 3. Family and split design

- **Family-level disjointness:** every source family belongs to exactly one split. No document, chapter or derived segment crosses splits.
- **Minimum:** 8 train families (4 EN / 4 PL), 4 validation (2 EN / 2 PL), 4 test (2 EN / 2 PL).
- **Preferred:** 12 train (6/6), 6 validation (3/3), 6 test (3/3) to make family bootstrap and per-language reporting less fragile.
- **Balance:** target roughly equal token mass by language; do not silently let one long family dominate. Report both token-weighted and family-macro aggregates.
- **Test frozen:** test families are integrity-checked but never evaluated during design, tuning or pilot. A test result requires a separately reviewed final protocol.
- **Leakage:** family split plus 13-gram cross-split audit; test/validation n-grams must not appear in train above the predeclared threshold (planning default <0.1%, to be fixed before acquisition).

## 4. Deduplication and contamination

1. Exact SHA-256 dedup at document level.
2. Normalized dedup (Unicode NFC, whitespace collapse, case-fold) to catch near-identical copies.
3. Near-duplicate detection with MinHash/SimHash on 5-gram shingles; planning Jaccard threshold 0.8 at document level.
4. Family-level dedup: remove repeated editions/translations of the same work from different families unless the design explicitly studies translation.
5. Cross-split n-gram audit (13-grams) between train and validation/test.
6. Probe contamination audit: probe templates/entities/rules disjoint from all corpus splits; 13-gram overlap <0.1%.
7. Record every removal with source/document hash; do not silently delete.

## 5. Tokenizer

- One frozen byte-level BPE, actual vocabulary 16,384, no normalization, special tokens PAD0/BOS1/EOS2/UNK3/reserved4–7.
- Fit deterministically on a train-only sample, never on validation/test. Planning sample: up to 8 MB of train text or the full train split if smaller; record exact sample hash and byte count.
- Reuse the same tokenizer artifact across 1M/3M/5M scales to avoid confounding scale with tokenizer changes.
- Require complete 256-byte alphabet and exact encode/decode round-trip on every materialized document.
- Publish tokenizer SHA-256 and a token-to-byte table in the v2 evidence schema; do not publish the tokenizer JSON itself without approval.
- Reject normalization, prefix-space or language-specific preprocessing changes that break byte alignment.

## 6. Token and byte budgets

T0016 ratios used for planning: 3.905057 UTF-8 bytes per full-corpus content token; 4.868481 raw bytes per content token; 48.258 JSONL bytes per valid token. These are observed pilot ratios, not a representativeness claim.

| Train tokens | Validation | Test | Total | Extracted UTF-8 (est.) | Raw bytes (est.) | Compact 16 B/token (est.) | Current JSONL ratio (est.) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 50,000 | 50,000 | 1,100,000 | 4.30 MB | 5.36 MB | 17.6 MB | 53.1 MB |
| 3,000,000 | 150,000 | 150,000 | 3,300,000 | 12.89 MB | 16.07 MB | 52.8 MB | 159.3 MB |
| 5,000,000 | 250,000 | 250,000 | 5,500,000 | 21.48 MB | 26.78 MB | 88.0 MB | 265.4 MB |

Compact range: 13–19 bytes/token depending on whether byte offsets, family/language IDs and labels are stored per position; 16 B/token is the planning midpoint. These sizes are modest for the Mac/Linux scratch budget; the binding constraint is acquisition/rights review and tokenizer/materialization audit, not disk.

Staging:

- **1M:** first descriptive/paired-SD corpus; 8 train families; 1M unique tokens is ~21.6× the T0016 train token count and supports the S0/S1 budgets.
- **3M:** intermediate stability and family-macro check; 10–12 train families.
- **5M:** confirmation scale if the 1M paired SD and family spread justify it; 12 train families preferred. This is still a small research corpus, not a production pretraining set.

Justification: 1M permits repeated matched runs without exhausting the corpus; 5M gives five times the unique-token diversity and more family clusters, but does not by itself guarantee power. Final scale is unresolved until observed variance and acquisition feasibility are reviewed.

## 7. Approximate compute

Analytical prefill MACs per token at T512 (from accepted accounting): S 33,947,904; P 37,088,635; C 33,950,720. Training estimates assume 3× forward MACs and 2 FLOPs/MAC; they exclude embeddings, norms, scalar ops, padding, memory traffic and kernel inefficiency.

| Arm | 5M forward MACs | 5M training MACs (3×) | 5M training FLOPs (6× forward MACs) | GPU-s at 10 TFLOP/s |
|---|---:|---:|---:|---:|
| S | 169,739,520,000,000 | 509,218,560,000,000 | 1,018,437,120,000,000 | 101.8 |
| P | 185,482,240,000,000 | 556,446,720,000,000 | 1,112,893,440,000,000 | 111.3 |
| C | 169,753,600,000,000 | 509,260,800,000,000 | 1,018,521,600,000,000 | 101.9 |

These compute-only numbers are much smaller than the wall-time anchor from T0017 (1.266 s/update; 128 updates/run ≈ 162 s/run). Small-model batch-1 overhead, validation, checkpointing and data I/O dominate. Use the wall-time table in `design.md` for window planning; do not present analytical FLOPs as measured training cost.

## 8. Alternatives if 1–5M is too limited

- **Fixed-token descriptive pilot:** 1M tokens, two arms (S vs C), five paired seeds, 65k–196k positions/run, family-macro and per-language reporting; explicitly descriptive, no noninferiority claim.
- **Mechanism-first synthetic study:** controlled retrieval/composition tasks with known generators to localize a mechanism; not a language-model quality result.
- **Reduce scope rather than inflate claims:** if only 1M tokens and four validation families are feasible, report paired seed ranges and family descriptives; do not compute a population CI or claim scaling.
- **Defer 5M confirmation** until the v2 schema, optimized dense comparator and family registry are accepted; do not start acquisition just to fill a budget.

No power claim is made without observed paired SD and family variance.

## 9. Conditional memory vs matched expert capacity (#86)

Future design only, no architecture adoption:

| Factor | Level A | Level B |
|---|---|---|
| Conditional memory | no external/retrieval memory | frozen or trained conditional memory with declared capacity |
| Expert capacity | matched active expert parameters/MACs | same active capacity, different routing/placement |

Match active parameters and forward MACs across cells; report physical parameters, memory bytes, cache/retrieval hit rate, training cost and inference latency. Audit retrieval sources for train/test contamination and rights. Predeclare whether memory is frozen or trained, and give both cells the same tuning budget. The first useful question is whether a conditional-memory effect survives an active-capacity match, not whether a larger model wins.

## 10. Machine-readable protocol and dependencies

`proposed-protocol.json` contains the corpus range, family minimums, tokenizer rule, split/dedup/rights requirements, byte schema, budgets and dependency graph. `analysis.json` contains the exact byte/compute/precision tables generated by `protocol_analysis.py`.

Dependency graph:

```text
D1 T0026 review
├── D2 v2 byte schema + audit
├── D3 optimized dense comparator
├── D4 rights registry + acquisition
│   └── D5 frozen tokenizer
├── D6 1M matched feasibility + paired SD
│   ├── D7 5M confirmation
│   └── D8 GDN localization
└── D9 #86 conditional-memory vs expert-capacity design
```

## 11. Unresolved decisions

- Final source list, rights evidence and family assignments.
- Whether 8/4/4 families are sufficient at 1M or 12/6/6 is required.
- Tokenizer sample size and one-tokenizer-across-scales freeze.
- Dedup thresholds and cross-split n-gram limit.
- Exact v2 schema/migration and independent byte audit.
- Multi-window compute authorization for 3M/5M.
- Final paired-seed count and margins after observed variance.

## 12. Non-claims

No acquisition, download, tokenizer fit, text inspection, test access, training, storage measurement or quality/scaling claim. The byte and compute tables are planning estimates. Rights assessment is not legal clearance. No corpus or tokenizer artifact is published here.
