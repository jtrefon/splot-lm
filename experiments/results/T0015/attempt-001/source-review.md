# T0015 source-review summary
2026-11-27 | Metadata-only owner decision | No acquisition

Official metadata was rechecked on 2026-11-27 for every candidate. The
machine-readable packet is `source-review.json` and the reviewed document is
`research/data/pilot-approval-packet.md`. No book text, corpus shard, source
snapshot, gated resource or tokenizer model was acquired.

## Proposed prose pool

Eight prose families, four per language, with two train families per language
and one validation plus one test family per language:

| Split | EN | PL |
|---|---|---|
| train | `se-pride-and-prejudice`, `se-frankenstein` | `wl-lalka`, `wl-chlopi` |
| validation | `se-alices-adventures` | `wl-quo-vadis` |
| test | `se-moby-dick` | `wl-faraon` |

No work, translation or edition family is split across train/validation/test.
Any document segmentation must happen after this family split with preserved
parent/offset provenance.

## Standard Ebooks evidence

All four item pages returned the expected title, CC0 1.0 production license,
U.S. public-domain warning, word count and first item-history commit:

- Pride and Prejudice: <https://standardebooks.org/ebooks/jane-austen/pride-and-prejudice>,
  modified 2026-11-15, 121,970 words, revision
  `a77a2dbbc0d11a17d6c68c581f17e26a65bd8168`.
- Frankenstein: <https://standardebooks.org/ebooks/mary-shelley/frankenstein>,
  modified 2026-11-24, 77,898 words, revision
  `abfa474c5e1df00ae8458ed8495b83e7f8b795cc`.
- Alice’s Adventures in Wonderland: <https://standardebooks.org/ebooks/lewis-carroll/alices-adventures-in-wonderland/john-tenniel>,
  modified 2026-11-24, 26,738 words, revision
  `9e5eb58995d9fd2f609363d14786d8b0166bbbb9`; John Tenniel illustrations are
  excluded unless separately cleared.
- Moby Dick: <https://standardebooks.org/ebooks/herman-melville/moby-dick>,
  modified 2026-11-16, 211,202 words, revision
  `89725132954ec3f297e0d86f4a5ab636675d7f13`.

The U.S. warning is not EU/Polish/Spain clearance. No release-artifact SHA-256 is
published; it must be computed on the first authorized staging download.

## Wolne Lektury evidence

All five official API records returned the expected title, Polish language and
genre, and the official item pages supplied author life dates:

- Lalka: <https://wolnelektury.pl/katalog/lektura/lalka/>; Bolesław Prus
  (1847–1912), prose, two volumes.
- Chłopi: <https://wolnelektury.pl/katalog/lektura/chlopi/>; Władysław
  Stanisław Reymont (1867–1925), prose, four volumes.
- Quo vadis: <https://wolnelektury.pl/katalog/lektura/quo-vadis/>; Henryk
  Sienkiewicz (1846–1916), prose.
- Faraon: <https://wolnelektury.pl/katalog/lektura/faraon/>; Bolesław Prus
  (1847–1912), prose, three volumes.
- Pan Tadeusz: <https://wolnelektury.pl/katalog/lektura/pan-tadeusz/>; Adam
  Mickiewicz (1798–1855), official genre **Epos**; classified `verse_epic` and
  **excluded from the prose pilot** as `excluded_nonprose`.

Terms checked: <https://wolnelektury.pl/info/regulamin/> and
<https://wolnelektury.pl/info/zasady-wykorzystania/>. Annotations and motifs are
editorial material under CC BY-SA 3.0 or LAL 1.3 and must be excluded or
separately attributed. The metadata does not publish an immutable revision or
artifact checksum; the first authorized download SHA-256 becomes the snapshot
identity, without retroactively proving upstream immutability.

## Caps and approval state

- Total raw UTF-8 cap: 10,485,760 bytes (10 MiB).
- Per-item cap: 2,621,440 bytes (2.5 MiB).
- Target 50/50 EN/PL content tokens, prose only; no code/math/technical quota
  claim.
- If the selected families exceed 10 MiB, remove a family before acquisition;
  never truncate a work.
- First authorized download goes to staging, SHA-256 is pinned, then a new
  immutable local snapshot may be promoted only after metadata, license and
  duplicate/benchmark checks.
- Access approval, training use, raw redistribution and weight redistribution
  are separate decisions. No candidate is acquisition-ready.

## Open blockers

- EU/Polish/Spain rights review for every family, including edition obligations.
- Four Standard Ebooks author life dates are unresolved in item metadata and
  require authority-record verification.
- Standard Ebooks artifact checksums; Wolne Lektury immutable revisions.
- Annotation/motif extraction and licensing; Alice illustration exclusion.
- Benchmark and near-duplicate overlap checks before training.
- Owner acceptance of candidate IDs, family split, access routes and staging
  location.
