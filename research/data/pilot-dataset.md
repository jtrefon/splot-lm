# Bounded real bilingual pilot dataset

2026-11-27 | T0016 / #76 | Real corpus and 16k tokenizer | Local-only artifacts

**Status:** a small real EN/PL prose pilot was acquired from official sources, fitted
to a 16,384-token ByteLevel BPE tokenizer and materialized as
`splot-pilot-blocks-v1`. Raw text, token arrays, `tokenizer.json` and the dataset
shards are local-only and are **not** in Git. This is a feasibility/architecture
pilot, not a production corpus, quality claim, legal opinion or redistribution
approval.

## 1. Authorized scope and caps

| Limit | Authorized | Observed |
|---|---:|---:|
| Foreground time | 120 min | well under |
| Extracted UTF-8 text | 10 MiB | 8,987,423 B |
| Per work | 2.5 MiB | 2,106,903 B max |
| Aggregate network | 50 MiB | 11,204,726 B |
| Local output | 512 MiB | < 20 MiB for raw + two builds |
| Segments | 10,000 | 2,360 |
| Tokenizer train input | 4 MiB | 4,194,289 B (sampled) |
| Blocks | 128 train / 16 validation / 16 test | 160 total |

No model was trained on the Mac. No new dependency, system change, gated access,
paid service, cloud job or large crawl was used. `tokenizers==0.23.3` was recreated
in an isolated venv from the previously verified official macOS arm64 wheel.

## 2. Sources, rights and split

Eight prose families, four per language, with whole-family assignment to one split:

| Split | EN | PL |
|---|---|---|
| train | Pride and Prejudice; Frankenstein | Lalka; Chłopi |
| validation | Alice’s Adventures in Wonderland | Quo vadis |
| test | Moby Dick | Faraon |

*Pan Tadeusz* is an **Epos** (verse epic), recorded as `verse_epic` and explicitly
excluded from this prose pilot. No family, translation or edition is split across
train/validation/test.

English sources are Standard Ebooks XHTML files pinned by first item-history Git
commit (`a77a2db…`, `abfa474…`, `9e5eb58…`, `8972513…`). The item pages provide the
CC0 1.0 production license and a U.S. public-domain warning. Polish sources are
official Wolne Lektury XML volumes. Their public-domain assessment uses author life
dates from the official item pages/Wikidata authority records; annotations and
motifs are excluded. The registry records rights URLs and authority URLs. This is a
documented public-domain assessment, **not legal clearance**.

Standard Ebooks paratext was not selected: titlepage, imprint, colophon,
uncopyright, endnotes, introductions/prefaces and illustrations are outside the
selected chapter/letter files. Wolne Lektury extraction includes only narrative
block tags and explicitly skips `<abstrakt>`, `<nota_red>`, `<nota>`, `<motyw>`,
`<pr>`, `<uwaga>`, `<slowo_obce>`, `<ilustracja>`, `<begin>` and `<end>`. Faraon’s
original `Wstęp` is retained; its modern editorial note and change list are not.

`configs/data/pilot-sources-v0.2.json` records the source decisions, official URLs,
revisions, part hashes, extracted content hashes, byte counts and aggregate
materialization hashes. `revision` is `git:<commit>` for Standard Ebooks and
`local-sha256:<raw_sha256>` for mutable Wolne Lektury files; the latter is a frozen
local identity, not an invented upstream checksum.

## 3. Tokenizer

- Target/actual vocabulary: **16,384 / 16,384**; no padding.
- ByteLevel BPE, no normalization, `add_prefix_space=False`, regex pretokenizer,
  complete 256-byte initial alphabet.
- Special IDs: PAD 0, BOS 1, EOS 2, UNK 3, reserved 4–7.
- `min_frequency=2`, `max_token_length=128`, deterministic sorted train-only input.
- Training input: 1,094 train segments / 4,194,289 bytes, deterministically sampled
  from 1,303 train segments / 5,005,698 bytes because the full train text exceeds
  the 4 MiB cap. No validation or test text was used for fitting.
- All 2,360 eligible extracted segments were encoded and decoded strictly; every
  round-trip passed. Maximum segment length was 1,443 tokens before block-aware
  segmentation.
- The first Nano model vocabulary remains fixed at 32,768: 16,384 unused logits are
  reported; no vocabulary remapping.

Local tokenizer identities:

| Item | Bytes | SHA-256 |
|---|---:|---|
| `tokenizer.json` | 708,869 | `104d3927bb55cdd2a911dbe21c8e755264756cedb5663bd39365c3d4cba3864b` |
| tokenizer artifact | — | `95c5d111cacd3fb668d15d6d7a6cc501a043027b876953b96cc185f974275011` |
| logical artifact digest | — | `4ab9dc91bdaf2a2d34dd09fc7fb53ce92aaccf09a633f174f6020956c4a98e9f` |
| training input digest | — | `f35a114e77f07958a49c3faf5875e6b17c2410c9c5a86143de9ebf770c805a31` |

## 4. Dataset contract

Local layout:

```text
<output>/splot-pilot-v1/
  manifest.json
  tokenizer.json
  train.jsonl
  validation.jsonl
  test.jsonl
<output>/tokenizer-artifact.json
<output>/source-index.json
<output>/build-summary.json
```

`manifest.json` has exactly: `format`, `version=1`, `sequence_length=512`,
`vocab_size`, `tokenizer_sha256`, `source_snapshot_sha256`, `sources`, `shards`,
`logical_sha256`. Each source record has exactly `family_id`, `language`, `split`,
`content_sha256`, `raw_sha256`, `revision`, `rights_evidence_url`, `source_url`.
Each shard record has exactly `split`, `path`, `sha256`, `bytes`, `blocks`,
`valid_tokens`, `ntp_labels`, `offset2_labels`.

Each JSONL line is one block with exactly `input_ids`, `document_ids`,
`position_ids`, `valid`, `labels`. Arrays are length 512; `labels` keys are `"1"` and
`"2"`. Each block is one segment: BOS, ≤510 content tokens, EOS, pad 0; positions
reset at the segment; document IDs are unique nonnegative global segment numbers
with pad `-1`; labels are the future token only within the same segment, else -100.
No control IDs occur inside content, and no block is pad-only.

Manifest identity: `source_snapshot_sha256` is the SHA-256 of canonical sorted
source records; `logical_sha256` is the SHA-256 of the concatenated shard bytes in
train, validation, test order; `manifest_sha256` is the SHA-256 of the canonical
manifest bytes.

Measured materialization:

| Item | Value |
|---|---|
| Manifest SHA-256 | `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55` |
| Manifest bytes | 4,674 |
| Logical SHA-256 | `70ba8d35d2ae334a8601f6389dc39208f3636937cb9dbfda639ead7ab30c052c` |
| Source snapshot SHA-256 | `c84bd8a6e94c5d0f4d429e44a43c7bffd9a6dfb9a263b9a3440126e23b4aa087` |
| Source index SHA-256 | `f7d11bcdccc7a0df9e4b775ad5b6136a76d4fcfd0e87429a373d4d034dde3953` |

Shards:

| Split | Blocks | Content tokens | Valid tokens | NTP labels | Offset-2 labels | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 128 | 45,948 | 46,204 | 46,076 | 45,948 | 1,608,729 | `7f1156eeed4c0de8f802d5db41df0884791429d3f91bff1b2348d0a14ca92a39` |
| validation | 16 | 5,234 | 5,266 | 5,250 | 5,234 | 202,943 | `e5d858a12c975f416c393889d1915110c1854003c922cc26f498af807a7c214f` |
| test | 16 | 5,019 | 5,051 | 5,035 | 5,019 | 202,373 | `00968bee8f71dccf1fa5e52dc6ca8fa948e3ffb89577e45bb6bb00c74172b06d` |

Overall: 160 blocks, 56,201 content tokens, 56,521 valid tokens, 56,361 NTP
labels, 56,201 offset-2 labels, 25,399 pad tokens. By language: EN 80 blocks /
26,941 content tokens; PL 80 / 29,260. By split/language: train 64 EN + 64 PL,
validation 8 + 8, test 8 + 8. No oversampling.

Two independent builds from the same frozen acquisition and the committed registry
were byte-identical in manifest, logical hash, tokenizer and all three shards; a
third build from the updated registry also matched.

## 5. Regeneration

All data stays outside the checkout. A new empty staging directory is required:

```bash
python3 tools/pilot_dataset.py acquire \
  --registry configs/data/pilot-sources-v0.2.json \
  --staging /path/outside/checkout/staging

# The same command is resumable and can be split by family:
python3 tools/pilot_dataset.py acquire ... --family se-pride-and-prejudice
python3 tools/pilot_dataset.py acquire ... --family wl-lalka
```

`acquire` accepts only official HTTPS hosts (`raw.githubusercontent.com`,
`wolnelektury.pl`), enforces the cumulative 50 MiB cap, uses one documented
transient retry per request and records part hashes. `build` requires the isolated
`tokenizers==0.23.3` environment:

```bash
PYTHONPATH=tools TOKENIZERS_PARALLELISM=false <venv>/bin/python \
  tools/pilot_dataset.py build \
  --registry configs/data/pilot-sources-v0.2.json \
  --staging /path/outside/checkout/staging \
  --output /path/outside/checkout/output

PYTHONPATH=tools <venv>/bin/python tools/pilot_dataset.py verify \
  --dataset /path/outside/checkout/output/splot-pilot-v1 \
  --manifest-sha256 dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55

PYTHONPATH=tools <venv>/bin/python tools/pilot_dataset.py compare \
  /path/outside/checkout/output/splot-pilot-v1 \
  /path/outside/checkout/output-repeat/splot-pilot-v1
```

The loader performs no network access. It verifies canonical manifest bytes, exact
schema, source snapshot, tokenizer hash/vocabulary/special IDs, every shard hash and
byte count, every block shape/mask/position/label, global document-ID uniqueness,
logical digest and exact file set. The optional `source-index.json` maps document
IDs to family, language, split and text offsets for per-language evaluation; it is
local-only and not part of the harness contract.

Linux may regenerate from the official sources with the same registry, or use a
small verified owner-authorized local transfer if a mutable Polish source hash no
longer matches. The manifest SHA remains the dataset identity; no silent
substitution is allowed.

## 6. Tests and limitations

`tools/test_pilot_dataset.py` covers XHTML/XML extraction and annotation exclusion,
lossless UTF-8 segmentation, family split, train-only selection, block geometry,
masking/offsets, canonical manifest, malformed paths/counts, tamper/symlink/extra
file rejection, repeat materialization, and conditional pinned-backend tokenizer
tests. Default stdlib tests do not import `tokenizers`; hosted CI runs them without
the backend.

Limitations: this is a small feasibility pilot (56,201 content tokens), not a
representative sample or scaling result. Only prose is included; no code, math or
technical quota is claimed. The tokenizer is a 16k candidate; the first model
vocabulary remains 32,768 with unused logits. Rights are a documented public-domain
assessment, not a legal opinion; Standard Ebooks provides U.S. wording and Wolne
Lektury annotations remain separately licensed and were excluded. Upstream mutable
Polish files may change; the pinned local hashes are the reproducible snapshot.
No model training, loss evaluation, redistribution approval or weight-publication
approval is included.
