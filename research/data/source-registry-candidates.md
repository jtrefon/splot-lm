# Candidate bilingual source registry (T0034 / #124)

2026-11-28 | **Candidate only; no source acquired, no rights approval, no acquisition authorization**

This document and `source-registry-candidates.json` propose concrete rights-reviewable bilingual prose families for the stream-v1 pilot. They are metadata-level candidates only. Official Standard Ebooks and Wolne Lektury bibliographic/license pages were consulted; **no corpus text was downloaded**, no source was scraped or bypassed, and no rights are inferred from a URL or hash. T0016 acquisition evidence is reused only where explicitly labelled.

## 1. Status and review

- Owner acquisition authorization: **no**
- Rights review: **not performed**
- Family/split freeze: **not performed**
- Tokenizer sample review: **not performed**
- Contamination audit: **not performed**
- Registry freeze: **not performed**

A separately reviewed acquisition task and explicit owner authorization are required before any download. Acquisition must archive the rights statement, freeze the revision/snapshot and record exact hashes.

## 2. Split and language balance

Preferred set: **24 families** — train 12 (6 EN / 6 PL), validation 6 (3/3), test 6 (3/3).

| Split | EN | PL | Total |
|---|---:|---:|---:|
| train | 6 | 6 | 12 |
| validation | 3 | 3 | 6 |
| test | 3 | 3 | 6 |

The minimum 8/4/4 balanced subset is listed in `minimum_subset` (4 EN + 4 PL train, 2+2 validation, 2+2 test). The preferred set adds two more families per split/language.

## 3. Candidate families

Token estimates are planning ranges only (`content bytes / 5..3`); Polish byte-split diacritics may lower bytes per token. Existing T0016 rows use observed content bytes.

| Candidate | Lang/split | Work | Author (d.) | Provider | Status | Est. content tokens |
|---|---|---|---:|---|---|---:|
| `en-train-pride-and-prejudice` | en/train | Pride and Prejudice | Jane Austen (1817) | Standard Ebooks | existing_t0016_evidence | 139,154–231,923 |
| `en-train-frankenstein` | en/train | Frankenstein | Mary Shelley (1851) | Standard Ebooks | existing_t0016_evidence | 84,638–141,063 |
| `en-validation-alices-adventures` | en/validation | Alice's Adventures in Wonderland | Lewis Carroll (1898) | Standard Ebooks | existing_t0016_evidence | 30,011–50,019 |
| `en-test-moby-dick` | en/test | Moby Dick | Herman Melville (1891) | Standard Ebooks | existing_t0016_evidence | 243,591–405,985 |
| `pl-train-lalka` | pl/train | Lalka | Bolesław Prus (1912) | Wolne Lektury | existing_t0016_evidence | 355,966–593,278 |
| `pl-train-chlopi` | pl/train | Chłopi | Władysław Reymont (1925) | Wolne Lektury | existing_t0016_evidence | 421,380–702,301 |
| `pl-validation-quo-vadis` | pl/validation | Quo vadis | Henryk Sienkiewicz (1916) | Wolne Lektury | existing_t0016_evidence | 235,553–392,589 |
| `pl-test-faraon` | pl/test | Faraon | Bolesław Prus (1912) | Wolne Lektury | existing_t0016_evidence | 287,188–478,648 |
| `en-train-sherlock-holmes` | en/train | The Adventures of Sherlock Holmes | Arthur Conan Doyle (1930) | Standard Ebooks | metadata_verified_candidate | 140,000–300,000 |
| `en-train-time-machine` | en/train | The Time Machine | H. G. Wells (1946) | Standard Ebooks | metadata_verified_candidate | 32,000–73,333 |
| `en-train-dorian-gray` | en/train | The Picture of Dorian Gray | Oscar Wilde (1900) | Standard Ebooks | metadata_verified_candidate | 90,000–183,333 |
| `en-train-dracula` | en/train | Dracula | Bram Stoker (1912) | Standard Ebooks | metadata_verified_candidate | 180,000–366,666 |
| `en-validation-jane-eyre` | en/validation | Jane Eyre | Charlotte Brontë (1855) | Standard Ebooks | metadata_verified_candidate | 200,000–400,000 |
| `en-validation-treasure-island` | en/validation | Treasure Island | Robert Louis Stevenson (1894) | Standard Ebooks | metadata_verified_candidate | 80,000–166,666 |
| `en-test-scarlet-letter` | en/test | The Scarlet Letter | Nathaniel Hawthorne (1864) | Standard Ebooks | metadata_verified_candidate | 90,000–183,333 |
| `en-test-call-of-the-wild` | en/test | The Call of the Wild | Jack London (1916) | Standard Ebooks | metadata_verified_candidate | 36,000–80,000 |
| `pl-train-ziemia-obiecana` | pl/train | Ziemia obiecana | Władysław Reymont (1925) | Wolne Lektury | metadata_verified_candidate | 280,000–566,666 |
| `pl-train-ludzie-bezdomni` | pl/train | Ludzie bezdomni | Stefan Żeromski (1925) | Wolne Lektury | metadata_verified_candidate | 200,000–433,333 |
| `pl-train-przedwiosnie` | pl/train | Przedwiośnie | Stefan Żeromski (1925) | Wolne Lektury | metadata_verified_candidate | 160,000–333,333 |
| `pl-train-nad-niemnem` | pl/train | Nad Niemnem | Eliza Orzeszkowa (1910) | Wolne Lektury | metadata_verified_candidate | 240,000–500,000 |
| `pl-validation-rodzina-polanieckich` | pl/validation | Rodzina Połanieckich | Henryk Sienkiewicz (1916) | Wolne Lektury | metadata_verified_candidate | 200,000–433,333 |
| `pl-validation-krzyzacy` | pl/validation | Krzyżacy | Henryk Sienkiewicz (1916) | Wolne Lektury | metadata_verified_candidate | 360,000–733,333 |
| `pl-test-stara-basn` | pl/test | Stara baśń | Józef Ignacy Kraszewski (1887) | Wolne Lektury | metadata_verified_candidate | 180,000–400,000 |
| `pl-test-chata-za-wsia` | pl/test | Chata za wsią | Józef Ignacy Kraszewski (1887) | Wolne Lektury | metadata_verified_candidate | 160,000–333,333 |

## 4. Rights and territory

**Standard Ebooks (EN):** production is dedicated to the public domain via CC0 1.0; the underlying work is public domain; the item page carries a U.S. public-domain warning. This is a US/EU assessment, not a legal opinion.

**Wolne Lektury (PL):** the underlying text is public domain; annotations and motifs are separately licensed (CC BY-SA 4.0 observed on 2026-11-28, with LAL 1.3 also referenced by prior evidence) and are excluded. Audio, covers and metadata are not part of the text candidate. This is an EU/Polish/US assessment, not a legal opinion.

No rights are inferred from a URL, hash or provider name. The acquisition task must archive the exact item-page rights statement and record the jurisdiction caveat.

## 5. Extraction and exclusions

Standard Ebooks: use only pinned main-text chapter/letter XHTML parts; exclude titlepage, imprint, colophon, uncopyright, endnotes, introductions/prefaces, illustrations, editorial notes, cover and metadata.

Wolne Lektury: download the official XML volume at acquisition; select narrative block text; exclude `abstrakt`, `nota_red`, `nota`, `motyw`, `pr`, `uwaga`, `slowo_obce`, `ilustracja`, `begin`, `end`, footnotes, editorial notes, cover/illustrations, audio, metadata and translations.

## 6. Revisions and freeze plan

- **Standard Ebooks:** resolve and pin the item-history git commit; record the 40-hex commit, per-part SHA-256/byte counts, retrieval UTC and the exact extraction allowlist. Never follow a moving branch.
- **Wolne Lektury:** upstream XML is mutable. Download the official XML volume, freeze it in private storage, and record local SHA-256, byte count, retrieval UTC, URL and extraction policy. A URL is not a revision.
- Existing T0016 families record actual `git:` or `local-sha256:` pins and observed raw/content hashes; reuse still requires re-verification.

## 7. Dedup and split policy

- Exact SHA-256 document dedup; Unicode/whitespace-normalized dedup; MinHash/SimHash over 5-gram shingles.
- 13-gram cross-split leakage audit with planning threshold <0.1%.
- Original-language works only; no translated edition crosses splits; one provider edition per work.
- **Known author overlap:** Bolesław Prus is in PL train (Lalka) and PL test (Faraon). This is inherited from T0016 and requires author-aware review or replacement of Faraon before freeze.
- All candidates are prose; drama, verse, essays and reference works are out of scope.

## 8. Budget

Preferred per-family target: ~83,333 train content tokens, ~8,333 validation/test content tokens. Candidate work sizes exceed these targets, so acquisition must select deterministic bounded segments; it must not use all text. The test split remains frozen and integrity-only. Actual token counts require the separately reviewed tokenizer fit and producer run.

## 9. Unresolved decisions

- Coordinator review and freeze of this candidate list before acquisition.
- Prus train/test author overlap and whether to replace Faraon in test.
- Exact per-family segment sampling and byte/token allocation within stream-v1 caps.
- Actual rights/territory review and archival of item-page statements.
- Whether the preferred 12/6/6 set fits the available private storage and acquisition time budget.

## 10. Next tasks

Acquisition, train-only tokenizer fit, bounded stream-v1 producer/loader and CLI/training adapter are separate reviewed tasks. This registry authorizes none of them and contains no corpus text.
