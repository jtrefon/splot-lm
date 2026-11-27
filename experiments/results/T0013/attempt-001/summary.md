# T0013 attempt-001 — passed (corpus qualification and trusted tokenized-shard boundary)

Task T0013 (IMP), parents #8/#11, issue #67. Reviewed specification merged at `685df410e28a260d45c7cfa2685c6614d64d77e0` (PR #68); approved minimum base `34506006f24a18a0222fb45a61eeaf2f51fd6289`; tested implementation commit `9ac54fc40d6524ad9d75fb9d33295270e3e4c8ae`. Branch `feat/T0013-corpus-attempt-001`. Observed UTC window 2026-11-27T19:22:29Z to 2026-11-27T19:23:15Z. Evidence PR #69.

## Authorization and scope
The owner explicitly authorized T0013 on this Mac M4 after the spec merged. Scope was metadata-only primary-source research plus targeted trusted-boundary fixes; no full text, dataset shard, gated access, package install, training, MPS/CUDA, source-rights approval, system change or acquisition. The 90-minute foreground limit and 120 s per-command limit were respected.

Changed files were exactly the allowed set:

- `tools/tokenized_shards.py`
- `tools/test_tokenized_shards.py`
- `research/data/tokenized-shards.md`
- new `research/data/pilot-source-candidates.md`
- new `research/data/pilot-source-candidates-v0.1.json`

No current source-registry approval, model, harness, BPE/corpus oracle, CI or config was edited. T0010 and earlier evidence were not rewritten.

## Source review
Four individually identifiable prose candidates were reviewed from official catalog/item/API/terms pages; no book text was downloaded. Two English candidates use Standard Ebooks editions of *Pride and Prejudice* and *Alice’s Adventures in Wonderland* (CC0 production, U.S. public-domain warnings, Git edition revisions). Two Polish candidates use Wolne Lektury items *Lalka* and *Pan Tadeusz* (public-domain text; annotations/motifs under CC BY-SA 3.0 or LAL 1.3; no immutable revision/checksum in catalog metadata).

All four are **pending**, not acquisition-ready. The main blockers are EU/Polish rights review, item/edition and annotation/illustration handling, missing artifact checksums for all four, and absent immutable revisions for the Wolne Lektury items. The bounded pilot proposal caps raw UTF-8 at 10 MiB, targets 50/50 EN/PL content tokens, is prose-only, defers code/math/technical, and proposes 16k/32k tokenizer candidates only after a separate package/version qualification. No source is recommended for immediate acquisition; the strongest first review pair is *Pride and Prejudice* + *Pan Tadeusz*.

## Implementation
The tokenized-shard format is now `splot-tokenized-shards-v2` / version 2. This is an intentional schema change, not a silent rewrite: the source identity now carries the complete cluster-split map (including `null` quarantine), cluster representatives and quarantine identity, and the producer requires an explicit approved synthetic source/revision mapping.

The producer now:
- recomputes the accepted corpus report canonical digest and requires unique inputs, clusters, members and representatives;
- requires exactly one representative per cluster, valid quarantine reasons and no quarantine/member overlap;
- verifies every member/representative source/revision against the explicit approved mapping;
- verifies each selected raw text hash, encodes once, rejects control/out-of-vocabulary IDs and checks exact round-trip;
- verifies the train-only BPE input digest against the frozen artifact.

The verified loader now additionally:
- decodes each document's token content with the frozen artifact and rechecks its content SHA-256;
- recomputes the train-only input digest from decoded train documents and compares it with both the artifact and source metadata;
- verifies the exact/complete cluster-split and representative mappings, one representative per cluster, quarantine cluster and member exclusion;
- rejects symlinks and nonregular files, and documents the trusted local single-writer and independently pinned manifest SHA assumptions.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| source_review | passed | `source-review.md`, `source-review.json`, `summary.md` |
| implementation | passed | `summary.md`, `environment.json`, `demo.json`, `manifest.json`, `execution.log` |
| unit_tests | passed | `summary.md`, `metrics.jsonl`, `demo.json`, `manifest.json`, `execution.log` |
| handover | passed | `summary.md`, `environment.json`, `execution.log` |

## Observed results
- `PYTHONPATH=tools python3 tools/test_corpus_reference.py`: exit 0, 30 tests, 0.070036 s.
- `PYTHONPATH=tools python3 tools/test_bpe_reference.py`: exit 0, 28 tests, 0.107701 s.
- `PYTHONPATH=tools python3 tools/test_data_fixture.py`: exit 0, 6 tests, 0.040775 s.
- `PYTHONPATH=tools python3 tools/test_tokenized_shards.py`: exit 0, 27 tests, 0.200657 s. New tests cover report-digest tampering, unapproved source mapping, duplicate cluster/member/input, promoted quarantine, rehashed content/train metadata, incomplete cluster/quarantine mapping, nonregular file rejection and candidate metadata shape.
- Candidate JSON parse: exit 0; 4 candidates (2 EN, 2 PL), all pending; pilot cap 10 MiB.
- Demo twice: exit 0, 0.058125 s and 0.058770 s, 21,209 bytes, byte-identical SHA-256 `1c899be572c22105cf2d37766b9806a5db6daebd9f0a768ce2f793e369511396`.
- Manifest: 10,786 bytes, SHA-256 `1e2b8fa70aafff32331018488992c40ec82521ad26eed5b58532fdf58442bff0`; logical `5d8d5898c41c36ac21cc7ba4785c8747c0a6ec8b20830419a0a082dfd931b5ba`; packed `7585194a799c1bc839591362929bf4e041bb18b4acaf9d15d677ff951743635f`; physical `3229cf587492afd31465fb3731f796f612dbaa279f05bebcedaabf1278813df6`.
- Dataset: 11 documents, 4 blocks, 3 shards; 203 content/22 special/225 valid/159 pad tokens; 214 NTP/203 offset-2 labels. Independent materializations are byte-identical; a different shard partition preserves logical/packed digests, changes the physical manifest and increases shards from 3 to 4.
- The logical and packed digests are unchanged from T0010; the v2 manifest/physical digest and demo hash change because the hardened source identity carries complete cluster/representative/quarantine metadata. T0010 evidence and old T0004/BPE demo digests remain historical and untouched.
- Environment: macOS 27.2.1 (build 26B131), arm64 Apple M4, 10 cores, 17,179,869,184 bytes RAM, CPython 3.15.2, Unicode data 17.0.0, CPU only. Memory availability 46%; available disk 81,709,780 KiB at capture.
- `python3` note: the task names `python`, but this host has no `python` executable. Acceptance commands were run as `python3`; this is an environment note, not a methodological deviation.
- First failure: none. No retry was run.

## Scope diff
Implementation commit `9ac54fc` changes exactly the three allowed existing files and adds the two allowed metadata files. The evidence commit adds `experiments/results/T0013/attempt-001/` and one INDEX append. No source change was made after acceptance.

## Deviations
None. The v2 format/schema bump is the required trusted-boundary change and is documented rather than applied silently.

## Limitations
- Metadata-only review; no full text, corpus shard, tokenizer artifact or real-data materialization was acquired.
- No candidate is acquisition-ready; EU/Polish rights, edition/annotation/illustration handling and checksums/revisions remain open.
- Standard Ebooks provides U.S.-public-domain warnings, not EU clearance. Wolne Lektury metadata lacks immutable revisions/checksums.
- The four-work pilot cannot guarantee 50/50 train/validation/test balance without additional works; the proposal states this imbalance.
- The shard loader provides integrity checks for a trusted local single-writer dataset; it is not a rights gate, multi-writer lock or adversarial filesystem guarantee.
- The synthetic shard demo is not a production tokenizer, quality, throughput, fertility or downstream model claim.
- Cross-platform shard equality is not claimed until a separate Linux comparison verifies the v2 canonical digests.

## Proposed next action
Maintainer review of the evidence PR and hosted CPU check. T0013 (#67) remains open; no merge or issue closure is performed here. The owner must next decide candidate acceptance/rejection and authorize any local Linux acquisition, checksum verification and tokenizer qualification.
