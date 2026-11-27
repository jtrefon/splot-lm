# Bilingual pilot acquisition approval packet
2026-11-27 | Metadata-only owner decision, #72 / #8 / #11 | T0015

**Status:** approval packet only. No book text, corpus shard, source snapshot,
gated resource or tokenizer model was acquired. No rights approval is claimed.
Machine-readable version:
[`pilot-approval-packet-v0.1.json`](pilot-approval-packet-v0.1.json).

## 1. Decision requested

Approve or reject the proposed eight-family prose pool and the specific
work-family split for a feasibility-only pilot:

- at most **10 MiB raw UTF-8** total and 2.5 MiB per item;
- target **50/50 EN/PL content tokens**, prose only;
- no code/math/technical quota claim;
- 16k/32k ByteLevel BPE candidate, train-only, under the separately pinned
  `tokenizers==0.23.3` lock;
- first authorized download to staging, SHA-256 pinned before any immutable local
  snapshot; no truncation of a work.

No download is authorized by this packet. Training, raw redistribution and
weights redistribution are separate decisions.

## 2. Corrected genre decision

*Pan Tadeusz* is a **verse epic**, not prose. It is retained in the metadata
review as `wl-pan-tadeusz` but is **excluded from the prose pilot**; no silent
prose/verse mixture is proposed. A future explicit mixed literary track may
revisit it under a new reviewed decision. The eight prose families below do not
include it.

## 3. Proposed prose families and work-family split

| Candidate ID | Lang | Work / family | Source ID | Pilot role | Genre |
|---|---|---|---|---|---|
| `se-pride-and-prejudice` | en | Pride and Prejudice | Standard Ebooks `jane-austen/pride-and-prejudice` | train | prose |
| `se-frankenstein` | en | Frankenstein | Standard Ebooks `mary-shelley/frankenstein` | train | prose |
| `se-alices-adventures` | en | Alice’s Adventures in Wonderland | Standard Ebooks `lewis-carroll/alices-adventures-in-wonderland/john-tenniel` | validation | prose |
| `se-moby-dick` | en | Moby Dick | Standard Ebooks `herman-melville/moby-dick` | test | prose |
| `wl-lalka` | pl | Lalka | Wolne Lektury `lalka` | train | prose |
| `wl-chlopi` | pl | Chłopi | Wolne Lektury `chlopi` | train | prose |
| `wl-quo-vadis` | pl | Quo vadis | Wolne Lektury `quo-vadis` | validation | prose |
| `wl-faraon` | pl | Faraon | Wolne Lektury `faraon` | test | prose |
| `wl-pan-tadeusz` | pl | Pan Tadeusz | Wolne Lektury `pan-tadeusz` | excluded_nonprose | verse epic |

The split gives two train families per language, one validation and one test
family per language. No work, translation or edition family is split across
train/validation/test. Document segmentation, if needed for the model context,
must occur only after this family split and preserve parent/offset provenance.

## 4. Source snapshot and access details

### Standard Ebooks candidates

All four are item-level Standard Ebooks editions with CC0 1.0 production
licensing and a U.S. public-domain warning. The warning is not EU/Polish/Spain
clearance. Item pages and edition Git revisions:

- `se-pride-and-prejudice`: <https://standardebooks.org/ebooks/jane-austen/pride-and-prejudice>;
  modified 2026-11-15; revision `a77a2dbbc0d11a17d6c68c581f17e26a65bd8168`;
  121,970 words.
- `se-frankenstein`: <https://standardebooks.org/ebooks/mary-shelley/frankenstein>;
  modified 2026-11-24; revision `abfa474c5e1df00ae8458ed8495b83e7f8b795cc`;
  77,898 words.
- `se-alices-adventures`: <https://standardebooks.org/ebooks/lewis-carroll/alices-adventures-in-wonderland/john-tenniel>;
  modified 2026-11-24; revision `9e5eb58995d9fd2f609363d14786d8b0166bbbb9`;
  26,738 words; illustrations excluded unless separately cleared.
- `se-moby-dick`: <https://standardebooks.org/ebooks/herman-melville/moby-dick>;
  modified 2026-11-16; revision `89725132954ec3f297e0d86f4a5ab636675d7f13`;
  211,202 words.

Access route: official item page or edition GitHub repository after owner
approval. No release-artifact SHA-256 is published; compute SHA-256 on first
authorized staging download. Preserve edition revision, source and commit.

### Wolne Lektury candidates

All five are catalog items presented as public domain; annotations/motifs are
editorial material under CC BY-SA 3.0 or LAL 1.3 and must be excluded or
separately attributed. Metadata does not publish immutable revisions/checksums:

- `wl-lalka`: <https://wolnelektury.pl/katalog/lektura/lalka/>; two volumes,
  Bolesław Prus (1847–1912), first published 1890.
- `wl-chlopi`: <https://wolnelektury.pl/katalog/lektura/chlopi/>; four volumes,
  Władysław Stanisław Reymont (1867–1925), published 1904–1909; life dates from
  the official item page.
- `wl-quo-vadis`: <https://wolnelektury.pl/katalog/lektura/quo-vadis/>; Henryk
  Sienkiewicz (1846–1916), 1895–1896; life dates from the official item page.
- `wl-faraon`: <https://wolnelektury.pl/katalog/lektura/faraon/>; three volumes,
  Bolesław Prus (1847–1912), 1895–1896.
- `wl-pan-tadeusz`: <https://wolnelektury.pl/katalog/lektura/pan-tadeusz/>;
  Adam Mickiewicz (1798–1855), 1834; **verse epic, excluded from prose pilot**.

Terms: <https://wolnelektury.pl/info/regulamin/> and
<https://wolnelektury.pl/info/zasady-wykorzystania/>. Access route: official
item/API or EPUB/XML/txt after owner approval; compute SHA-256 on first staging
download; no gated access.

## 5. Rights and provenance checklist

- Original-work status, author life dates, edition/editor/translator/
  illustration status and attribution must be reviewed under EU, Polish and
  Spain jurisdiction requirements. The five Polish author dates are taken from
  official Wolne Lektury item pages; the four Standard Ebooks author dates are
  unresolved and require authority-record verification. No legal opinion or
  automatic clearance is provided.
- Standard Ebooks' U.S.-only warning does not establish EU/Polish/Spain rights.
- Wolne Lektury public-domain text does not automatically clear annotations,
  motifs, introductions, translations or audio; exclude or license them.
- Preserve author/title, source item/edition, URL, retrieval date, terms/license
  evidence, source revision if any, and computed SHA-256.
- If a source lacks an immutable revision, the checksum pinned on first
  authorized download becomes the snapshot identity; it does not retroactively
  prove upstream immutability.
- Training use, raw-text redistribution and processed-data/weight redistribution
  are separate decisions. This packet does not authorize any of them.

## 6. Acquisition and snapshot protocol (only after approval)

1. Owner accepts/rejects candidate IDs and family assignments.
2. Qualified rights review resolves author dates, edition/editor/translator/
   illustration exclusions and jurisdiction/terms.
3. Owner authorizes exact official HTTPS access routes and local Linux staging.
4. Download only the selected text payloads, excluding covers/audio/annotations;
   enforce the 10 MiB total and 2.5 MiB per-item caps. If over cap, remove a
   family before acquisition; never truncate a work.
5. Record exact bytes and SHA-256 in staging; re-run the bounded metadata and
   license checks; then promote to a new immutable local snapshot only if all
   checks and approvals pass.
6. Run exact/near-duplicate, family and benchmark-overlap checks before any
   family split.
7. Qualify the pinned 16k/32k tokenizer on train-only text; do not download a
   remote tokenizer/model.

## 7. Approval checklist

- [ ] Owner accepts the eight prose families and the two-train/one-validation/
  one-test split per language.
- [ ] Owner accepts that *Pan Tadeusz* is verse and explicitly excluded from the
  prose pilot.
- [ ] Qualified rights review documents EU/Polish/Spain positions, author dates
  and edition/annotation/illustration exclusions.
- [ ] Owner authorizes the official access routes and staging location.
- [ ] Owner accepts the 10 MiB total / 2.5 MiB per-item caps and no-truncation
  rule.
- [ ] Owner accepts that first-download SHA-256 is the snapshot identity where no
  upstream revision exists.
- [ ] Tokenizer package/version and train-only fitting are separately approved.
- [ ] Training, raw redistribution and weight redistribution are decided
  separately.

## 8. Residual risks and recommendation

No candidate is acquisition-ready until author/edition rights, annotations,
illustrations, jurisdiction and access are reviewed. The 10 MiB cap may require
removing a family; this is preferable to truncating a work. A four-family or
two-family subset would break the requested split/language coverage and should
be labeled feasibility-only, not evaluation-ready.

**Recommendation:** approve the packet only conditionally after qualified rights
review; do not acquire in T0015. The strongest initial prose set is the eight
families above, with *Pan Tadeusz* held out of the prose track.
