# T0016 attempt-001 — bounded real bilingual corpus and tokenizer
2026-11-27 | IMP | Issue #76 | Parents #8/#11 | PR #80

**Outcome:** passed locally. Hosted CPU CI is required on PR #80 before merge.
**Instructions:** `4c4e23e84c2a662d8519d4a2a852116c711f7ee2` (reviewed main, PR #78).
**Approved minimum base:** `8537c533372fb58a5d5ad5c58436f90e58c030ea`.
**Tested implementation:** `0261ca0421abf4aa26c0b69626f06b5d9779f2a4`, clean before and after acceptance.
**Scope:** actual bounded acquisition of eight public-domain prose families, a 16,384-token ByteLevel BPE fit, and local-only `splot-pilot-blocks-v1` materialization. No model training, raw corpus, token arrays or tokenizer JSON in Git.

## 1. Acquisition and rights

The owner authorized actual acquisition of accessible lawfully usable sources within T0016; no further per-item approval was requested. All downloads used official HTTPS routes:

- Standard Ebooks raw XHTML at pinned item-history commits.
- Wolne Lektury official XML volume files.

| Family | Lang | Split | Raw bytes | Extracted UTF-8 | Segments | Revision |
|---|---|---|---:|---:|---:|---|
| Pride and Prejudice | en | train | 805,814 | 695,770 | 186 | `git:a77a2db…` |
| Frankenstein | en | train | 454,863 | 423,191 | 118 | `git:abfa474…` |
| Alice’s Adventures | en | validation | 186,803 | 150,059 | 41 | `git:9e5eb58…` |
| Moby Dick | en | test | 1,376,728 | 1,217,955 | 332 | `git:8972513…` |
| Lalka | pl | train | 2,412,914 | 1,779,834 | 457 | `local-sha256:f19edde…` |
| Chłopi | pl | train | 2,734,087 | 2,106,903 | 542 | `local-sha256:d1429fb…` |
| Quo vadis | pl | validation | 1,491,373 | 1,177,767 | 318 | `local-sha256:cffa632…` |
| Faraon | pl | test | 1,742,144 | 1,435,944 | 366 | `local-sha256:ec595ff…` |

Totals: **11,204,726** network bytes (249 files), **8,987,423** extracted UTF-8 bytes, **2,360** segments. Largest work: 2,106,903 bytes. All caps held: 50 MiB network, 10 MiB extracted, 2.5 MiB/work, 10,000 segments. No item was excluded after acquisition.

Rights evidence is recorded per source in `acquisition.json` and the committed registry. Standard Ebooks provides CC0 1.0 production licensing and a U.S. public-domain warning; Wolne Lektury text is public domain with separately licensed annotations/motifs. Author life dates and authority URLs are recorded. This is a documented public-domain assessment, not a legal opinion or redistribution approval.

English extraction selects only pinned main-text chapter/letter files; titlepage, imprint, colophon, uncopyright, endnotes, introductions/prefaces and illustrations are excluded. Polish extraction includes narrative block tags and excludes `abstrakt`, `nota_red`, `nota`, `motyw`, `pr`, `uwaga`, `slowo_obce`, `ilustracja`, `begin` and `end`; Faraon’s original Wstęp is retained while its modern editorial note is not. *Pan Tadeusz* remains verse (`verse_epic`) and is excluded from the prose pilot. No family appears in more than one split.

## 2. Tokenizer and dataset

The recreated `tokenizers==0.23.3` isolated venv used the verified official macOS arm64 wheel (3,101,381 B, SHA-256 `68649e97…`). No system change or new dependency was introduced.

Tokenizer: ByteLevel BPE, no normalization, no prefix space, regex pretokenizer, complete 256-byte alphabet, PAD0/BOS1/EOS2/UNK3/reserved4–7, `min_frequency=2`, `max_token_length=128`, deterministic sorted train-only input. Target/actual vocabulary is **16,384 / 16,384**, with no padding.

- Training input: **4,194,289 B / 1,094 segments**, deterministically sampled from **5,005,698 B / 1,303 train segments** under the 4 MiB cap. Validation/test text was not used for fitting.
- All **2,360** eligible extracted segments encoded and decoded strictly; zero failures. Maximum pre-block segment length was 1,443 tokens.
- Model vocabulary remains 32,768; unused logits: 16,384. No remapping.
- Local tokenizer JSON: 708,869 B, SHA-256 `104d3927bb55cdd2a911dbe21c8e755264756cedb5663bd39365c3d4cba3864b`.

Dataset `splot-pilot-blocks-v1`: 512-token blocks, one segment per block, BOS/EOS, positions reset, unique global document IDs, pad -1, labels offset 1/2 only within the segment. Materialized:

| Item | Value |
|---|---|
| Manifest SHA-256 | `dea9335d4d7e5b1a7c0065baa8ff0a61b4fd8cd727db322cf76319441b697f55` |
| Manifest bytes | 4,674 |
| Logical SHA-256 | `70ba8d35d2ae334a8601f6389dc39208f3636937cb9dbfda639ead7ab30c052c` |
| Source snapshot SHA-256 | `c84bd8a6e94c5d0f4d429e44a43c7bffd9a6dfb9a263b9a3440126e23b4aa087` |
| Source-index SHA-256 | `f7d11bcdccc7a0df9e4b775ad5b6136a76d4fcfd0e87429a373d4d034dde3953` |
| Dataset bytes | 2,727,588 |

| Split | Blocks | Content tokens | Valid | NTP | Offset-2 | Bytes | Shard SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 128 | 45,948 | 46,204 | 46,076 | 45,948 | 1,608,729 | `7f1156ee…92a39` |
| validation | 16 | 5,234 | 5,266 | 5,250 | 5,234 | 202,943 | `e5d858a1…c214f` |
| test | 16 | 5,019 | 5,051 | 5,035 | 5,019 | 202,373 | `00968bee…7b06d` |

Overall 160 blocks / 56,201 content tokens / 56,521 valid tokens / 56,361 NTP / 56,201 offset-2 / 25,399 pad. By language: EN 80 blocks / 26,941 content; PL 80 / 29,260. Each split has 8 EN and 8 PL blocks; train has 64 + 64. No oversampling.

Two fresh builds from the same frozen acquisition and committed registry were byte-identical in manifest, logical digest, tokenizer, artifact, source-index and all shards. A third build from the updated registry also matched. The independent audit re-mapped all 160 blocks to exact source text offsets, re-tokenized them, verified strict round-trip and family isolation.

## 3. Tests

All acceptance commands exited 0 with no timeout or retry. **145 tests**:

| Suite | Tests | Duration (s) |
|---|---:|---:|
| `tools/test_corpus_reference.py` | 30 | 0.065556 |
| `tools/test_bpe_reference.py` | 28 | 0.108328 |
| `tools/test_data_fixture.py` | 6 | 0.042092 |
| `tools/test_tokenized_shards.py` | 27 | 0.201646 |
| `tools/test_production_tokenizer.py` (stdlib) | 4 | 0.047852 |
| `tools/test_pilot_dataset.py` (stdlib) | 17 | 0.301265 |
| `tools/test_production_tokenizer.py` (backend) | 14 | 0.328070 |
| `tools/test_pilot_dataset.py` (backend) | 19 | 0.346158 |

Build/verify timings: final-a 6.188739 s, final-b 6.207156 s, compare 0.188725 s, loader verify 0.122770 s, independent audit 3.054077 s. Hosted stdlib CI does not import `tokenizers`.

## 4. Development failures before the tested SHA

These were corrected before `0261ca0`; no acceptance command failed.

1. The first block validator treated BOS as a content token: `PilotDataError: invalid content token: 1`. Fixed by scanning content from index 1 and requiring a terminal EOS. Reproduction was the materialization check via `pd.pack_blocks(...)`.
2. `compare_datasets` raised `KeyError: 'manifest_bytes'` because the loader did not return raw manifest bytes. Fixed by returning the verified manifest bytes.
3. The test fixture initially used fewer than the required 32 train blocks and hit the real gate. Fixed by materializing 32 train blocks in the fixture, not by weakening the gate.
4. Initial Polish extraction included editorial `abstrakt`/`nota_red` front matter for Chłopi, Quo vadis and Faraon. Fixed by skipping those containers and all block elements inside skipped ancestors; the final extraction starts at original work text.

## 5. Limitations

- This is a small feasibility pilot: 160 blocks / 56,201 content tokens. It is not a representative sample, scaling result, quality claim or evaluation benchmark.
- Only prose is included; no code, math or technical quota is claimed.
- The 16k tokenizer is a candidate; the first model vocabulary remains 32,768 with 16,384 unused logits. No vocabulary remapping was performed.
- Rights are a documented public-domain assessment, not legal clearance. Standard Ebooks provides U.S. wording; Wolne Lektury annotations remain separately licensed and were excluded.
- Polish upstream XML is mutable; `local-sha256` pins the frozen snapshot, not an upstream revision.
- No model training, loss evaluation, raw-text redistribution or weight-publication approval is included.
- Raw text, token arrays, `tokenizer.json`, model weights and dataset shards remain local-only and are not in Git.
- Hosted CPU CI is required on PR #80 and is not claimed as passed in this summary.

## 6. Handover

Evidence is flat under `experiments/results/T0016/attempt-001/`: summary, result, environment, metrics, sanitized execution log, manifest, aggregate acquisition pins, tokenizer summary, dataset summary, repeat comparison, independent verification and segment round-trip. No raw text, token arrays or tokenizer JSON is present.

Proposed next action: maintainer reviews PR #80 and the manifest. T0017 may use the exact manifest SHA `dea9335d…` on Linux only after this PR is reviewed/merged and the manifest is locally verified. No merge, closure or redistribution is performed here.
