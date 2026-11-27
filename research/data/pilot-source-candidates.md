# Bounded bilingual pilot source candidates
2026-11-27 | Metadata-only review, #67 / #8 / #11 | T0013

**Status:** candidate research only. No corpus text, dataset shard, gated resource,
artifact checksum or rights approval was acquired. The four candidates below are
marked **pending**; none is acquisition-ready. This is not a legal opinion or an
automatic rights clearance.

## 1. Scope and method

The review used official catalog, item, API, license and terms pages retrieved
2026-11-27. Only metadata and short factual excerpts were recorded; no book text,
dataset shard or full file was downloaded. At least two English and two Polish
individually identifiable prose works are included. Audio, covers, editorial
motifs and other third-party material are not proposed as training text.

The pilot is intentionally a feasibility corpus, not balanced production data or
an evaluation benchmark. It is prose only; code, math and technical material are
deferred.

## 2. Candidate table

| ID | Lang | Work | Source / item | Status | Main blocker |
|---|---|---|---|---|---|
| `se-pride-and-prejudice` | en | *Pride and Prejudice* (1813), Jane Austen | Standard Ebooks `jane-austen/pride-and-prejudice` | pending | EU/edition review; artifact checksum absent |
| `se-alices-adventures` | en | *Alice’s Adventures in Wonderland* (1865), Lewis Carroll; illustrations John Tenniel | Standard Ebooks `lewis-carroll/alices-adventures-in-wonderland/john-tenniel` | pending | EU/author/illustrator review; checksum absent |
| `wl-lalka` | pl | *Lalka* (1890), Bolesław Prus (1847–1912) | Wolne Lektury `lalka`, volumes `lalka-tom-pierwszy` and `lalka-tom-drugi` | pending | revision/checksum absent; annotation extraction/license |
| `wl-pan-tadeusz` | pl | *Pan Tadeusz* (1834), Adam Mickiewicz (1798–1855) | Wolne Lektury `pan-tadeusz` | pending | revision/checksum absent; annotation extraction/license |

The machine-readable record is
[`pilot-source-candidates-v0.1.json`](pilot-source-candidates-v0.1.json). Raw
rights and terms must be rechecked at acquisition time; catalog pages can change.

## 3. English candidates

### Standard Ebooks: *Pride and Prejudice*

- Item/edition: `jane-austen/pride-and-prejudice`; edition history shows
  `dateModified 2026-11-15` and Git commit
  `a77a2dbbc0d11a17d6c68c581f17e26a65bd8168`.
- Source pages: <https://standardebooks.org/ebooks/jane-austen/pride-and-prejudice>
  and <https://standardebooks.org/about>.
- Rights metadata: the item declares `schema:license` CC0 1.0; the about page
  says Standard Ebooks dedicates its own work to the public domain and that its
  text/cover art are believed to be in the U.S. public domain.
- Warning: the item also says the ebook may still be under copyright outside the
  United States. This is not an EU/Polish clearance. Author-life and edition
  review remain required.
- Access route: official item page or edition GitHub repository after owner
  approval. Word count 121,970; no release-artifact SHA-256 is published in the
  metadata. No download was performed.

### Standard Ebooks: *Alice’s Adventures in Wonderland*

- Item/edition: `lewis-carroll/alices-adventures-in-wonderland/john-tenniel`;
  `dateModified 2026-11-24`; Git commit
  `9e5eb58995d9fd2f609363d14786d8b0166bbbb9`.
- Source page: <https://standardebooks.org/ebooks/lewis-carroll/alices-adventures-in-wonderland/john-tenniel>.
- Rights metadata: CC0 1.0 production license, U.S. public-domain warning, and
  illustrations by John Tenniel. The illustrations and any annotations must be
  reviewed separately and excluded from text-only training unless cleared.
- Access route: official item page or GitHub edition source after owner approval.
  Word count 26,738; no release-artifact SHA-256 is published in the metadata.
  No download was performed.

## 4. Polish candidates

### Wolne Lektury: *Lalka*

- Item: `lalka`; two volumes `lalka-tom-pierwszy`, `lalka-tom-drugi`.
  Metadata: <https://wolnelektury.pl/api/books/lalka/>; item page:
  <https://wolnelektury.pl/katalog/lektura/lalka/>.
- Author dates on the item page: Bolesław Prus, 20 August 1847–19 May 1912;
  first published 1890. ISBNs in metadata: 978-83-288-0737-2 (PDF),
  978-83-288-3733-1 (EPUB), 978-83-288-4819-1 (MOBI).
- Rights: the work is presented as public domain. Wolne Lektury states that
  annotations and literary motifs are additional editorial material under a free
  license (CC BY-SA 3.0 or LAL 1.3), and that attribution/share-alike conditions
  apply. The training text must exclude or separately attribute editorial
  material.
- Terms: <https://wolnelektury.pl/info/regulamin/> and
  <https://wolnelektury.pl/info/zasady-wykorzystania/>. No immutable revision or
  artifact checksum is published in the catalog/API metadata. Access route would
  be EPUB/XML after owner approval; no download was performed.

### Wolne Lektury: *Pan Tadeusz*

- Item: `pan-tadeusz`; metadata:
  <https://wolnelektury.pl/api/books/pan-tadeusz/>; item page:
  <https://wolnelektury.pl/katalog/lektura/pan-tadeusz/>.
- Author dates on the item page: Adam Mickiewicz, 24 December 1798–26 November
  1855; published 1834. Metadata lists txt, EPUB, FB2 and XML routes and ISBNs
  978-83-288-0542-2 (PDF), 978-83-288-3538-2 (EPUB), 978-83-288-4624-1 (MOBI).
- Rights: public-domain text; annotations/motifs are editorial material under a
  free license, with attribution/share-alike conditions. The text-only extraction
  must exclude or separately attribute them.
- Terms as above. No immutable revision/checksum in metadata. Access route would
  be the official txt/XML/EPUB after owner approval; no download was performed.

## 5. Jurisdiction, rights and contamination notes

- Standard Ebooks explicitly warns that its public-domain assessment is U.S.
  based; it is not an EU or Polish clearance. EU TDM reservations/exceptions and
  Polish implementation require qualified review.
- Wolne Lektury states its works are public domain or free-licensed, but its
  editorial layer (annotations, motifs) carries free-license obligations. Audio
  and reader voices are expressly separate and are not proposed here.
- Underlying-work status, edition/translation/illustration status, source terms
  and attribution must be reviewed separately. Database or catalog licensing
  does not automatically clear individual content.
- The works are canonical literature and may overlap evaluation benchmarks. The
  pilot must run exact/near-duplicate and benchmark-overlap checks before
  training; no contamination clearance is claimed.
- Texts are literary/historical; no private personal data is collected in this
  review. Any future acquisition still needs a PII/secrets/malware check.
- No rights approval, redistribution permission or weights/tokenizer licensing
  decision is made here. Raw corpus redistribution, processed data and model
  weights are separate decisions under #11.

## 6. Bounded pilot proposal

If the owner approves acquisition after rights review, the proposed initial
feasibility pilot is:

- **Size:** at most 10 MiB raw UTF-8 text total; per-item cap 2.5 MiB, with the
  total cap taking precedence.
- **Composition:** prose only; target 50/50 English/Polish content tokens, not
  raw bytes. Code, math, technical, audio, covers and editorial motifs are
  excluded.
- **Candidate pool:** the four items above. A conservative first pair would be
  *Pride and Prejudice* (EN) and *Pan Tadeusz* (PL); *Alice* and *Lalka* are
  alternates. Final selection requires rights review and token-count checks.
- **Access:** owner-approved direct HTTPS retrieval from the official source,
  no gated access, no scraping, no bulk downloads and no automated mirroring.
  Materialization occurs on Linux in a local data root outside Git.
- **Snapshots:** retain the exact retrieved bytes, URL, retrieval date, source
  item/edition ID, source revision if available, SHA-256 and license/terms
  evidence. If no immutable revision exists, record null and the blocker; do not
  invent a pin.
- **Text extraction:** remove covers, audio, navigation, editorial annotations,
  motifs and other third-party additions unless separately cleared; preserve
  original whitespace/diacritics and record the transformation.
- **Dedup/family policy:** one family per work/edition/volume; translated or
  derivative editions remain in the same family; exact/near-duplicate and
  benchmark checks run before splitting. No family spans train/validation/test.
- **Holdout policy:** hold out by work/edition, not random documents. With only
  four works, a 50/50 train target and independent validation/test per language
  is not possible without adding more works; the owner must accept a documented
  imbalance or add candidates. This is a feasibility pilot, not a balanced
  production corpus.
- **Tokenizer:** 16k and 32k byte-level BPE candidates only; train on train-only
  text. Package/version qualification, deterministic training and artifact
  hashing are a separate reviewed step.
- **Budget:** content-token budget and training budget are not set by this
  proposal; no model training is authorized here.
- **Removal/rebuild:** maintain a manifest mapping source/item/edition, license
  evidence, retrieval hashes, extraction version and split; removal of a source
  invalidates and rebuilds every dependent dataset/tokenizer/model artifact.
- **Approvals still needed:** maintainer rights decision for each item, owner
  acquisition authorization, local Linux materialization approval, benchmark/PII
  review, and separate tokenizer package/version qualification.

## 7. Recommendation

**Do not acquire yet.** No candidate is acquisition-ready: Standard Ebooks gives
a U.S.-only warning and no artifact checksum; Wolne Lektury gives public-domain
text but no immutable revision/checksum and requires annotation/license handling.
The strongest first pair for review is *Pride and Prejudice* + *Pan Tadeusz*,
with *Alice* + *Lalka* as alternates. The owner should explicitly accept or
reject each item, the EU/Polish rights route, the checksum mechanism and the
holdout imbalance before any download. If those conditions cannot be met, the
pilot should stay blocked rather than substitute unverified web text.

## 8. Evidence boundary

This document and its JSON companion are metadata research only. No book text,
dataset shard, tokenizer artifact or real-data materialization is included.
Evidence URLs are official catalog/item/license/terms pages; short excerpts are
factual metadata, not corpus content.
