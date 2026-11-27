# T0004 attempt-001 — passed (synthetic corpus preflight, deduplication and leakage reference)

Task T0004 (IMP), parent #8, issue #38. Instruction commit `0fcd9c6b5b9667af47603a63013b89d83e5f565f`; starting base `0fcd9c6b5b9667af47603a63013b89d83e5f565f`; approved minimum base `371aa551a6d553dc38245f9821dfc43490e871cb`; tested implementation commit `126c046720e7cacfc52d9c64f6e7fc6e6afcdb74`. Observed UTC window 2026-11-27T16:12:38Z to 2026-11-27T16:21:13Z, one foreground session well under the 90-minute cap. Preflight was posted in #38; the draft evidence PR is #45.

## Authorization and scope
The owner explicitly authorized T0004 attempt-001 in the current local session after the reviewed task specification merged at `0fcd9c6`. Dependency review found no overlapping implementation branch or PR for the three allowed files; T0005 (#40 / PR #44) is disjoint. Scope was exactly three new implementation files:

- `tools/corpus_reference.py`
- `tools/test_corpus_reference.py`
- `research/data/corpus-reference.md`

No edits were made to `tools/data_fixture.py`, `tools/experiment.py`, `src/`, the source registry, configs, global instructions or other tasks. The run was CPU only, synthetic only, standard library only: no downloads, packages, external corpora, training, MPS, CUDA, sudo, services, rights clearance or real data. Each command was capped at 120 s and no automatic retry was run.

## Outcome
**Passed.** All four required checks passed. This is a deterministic synthetic engineering oracle for the data contract; it is not production deduplication, source-rights approval, benchmark clearance, corpus acquisition or tokenizer work.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `summary.md`, `environment.json`, `demo.json`, `execution.log` |
| unit_tests | passed | `summary.md`, `metrics.jsonl`, `demo.json`, `execution.log` |
| handover | passed | `summary.md`, `environment.json`, `execution.log` |

## Observed results
- Tested implementation HEAD was exactly `126c046720e7cacfc52d9c64f6e7fc6e6afcdb74`; the worktree was clean before acceptance and after the commands. The starting base and instruction commit were `0fcd9c6b5b9667af47603a63013b89d83e5f565f`; the approved minimum base was `371aa551a6d553dc38245f9821dfc43490e871cb`.
- Implementation commit `126c046` adds exactly three files and changes no existing file:
  - `tools/corpus_reference.py` — 991 lines, SHA-256 `9d2f6e80d74b337429bf610f0d2d9c560b42350d91668326c4263c1ceec1ee14`.
  - `tools/test_corpus_reference.py` — 673 lines, SHA-256 `a91cf4196d19762f162b828905e4eb77212051f040f91ef3567726553472ad41`.
  - `research/data/corpus-reference.md` — 239 lines, SHA-256 `41334674404f8fe647593773b7b3d518de67cd21b286051692fa938e3c30cc05`.
- `PYTHONPATH=tools python3 tools/test_corpus_reference.py`: exit 0, 26 tests, 0 failures, 0 errors, 0.074083 s; stdout 0 bytes, stderr 3,110 bytes, SHA-256 `dc1e40ea3d433483e03425e73af3bf15855c3c566b70a3c0af5f821596676561`.
- `PYTHONPATH=tools python3 tools/test_data_fixture.py`: exit 0, 6 tests, 0 failures, 0 errors, 0.043552 s; stdout 0 bytes, stderr 638 bytes, SHA-256 `001ae2eacc11c1ba7723b367a8cd862346db78a79bfcb210ef2be321c7d1ca63`. The existing fixture was unchanged and its tests still pass.
- `PYTHONPATH=tools python3 tools/corpus_reference.py --demo`: exit 0 twice, 0.033421 s and 0.032597 s, 25,423 stdout bytes each, 0 stderr bytes, no timeout. The two outputs were byte-identical with SHA-256 `befc46f9aa4cd25b4d2e223e80d6f7db13a417b3cc50e36e05acd5f34cb1bd66`; the report digest was `b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f`.
- Demo corpus: 20 synthetic records (16 EN, 4 PL; all prose; 750 UTF-8 bytes), 13 clusters, 11 eligible, 2 quarantined clusters containing 4 records and 175 bytes, 5 duplicate records removed from eligible clusters (9 including quarantine), and 4 near-duplicate edges. The demo 50/25/25 split produced 8 train, 1 validation and 2 test clusters. The audit passed all five checks: disjoint cluster IDs, disjoint normalized fingerprints, disjoint source groups, near-duplicate edges within clusters, and no quarantined cluster in a split.
- Demo scenario coverage includes a combining-mark/CRLF exact normalized duplicate, Polish diacritics that are not stripped, source-family union, an inclusive 0.8 threshold pair, a below-threshold pair, short documents without near edges, a transitive near-duplicate bridge, a benchmark exact-normalized quarantine, an excluded-ID quarantine and a near-but-not-exact benchmark string that remains eligible. Quarantine reasons were one `benchmark_exact_normalized` cluster and one `excluded_record_id` cluster.
- Environment: macOS 27.2.1 (build 26B131), arm64 Apple M4, 10 cores, 17,179,869,184 bytes RAM, CPython 3.15.2 via `python3`. Memory availability was 46% before and 47% after; available disk was 81,381,068 KiB before and 81,385,824 KiB after. No CUDA or MPS was used.
- `python3` note: the task text names `python`, but this host has no `python` executable. The acceptance commands were run as `python3`; this is recorded as an environment note, not a methodological deviation.
- First failure: none. No retry was run.

## Scope diff
The implementation commit `126c046` changes exactly the three allowed new files and no existing file. The evidence commit adds `experiments/results/T0004/attempt-001/` and one append to `experiments/results/INDEX.md`; it changes no source, model, harness, fixture, registry, config or other task file. No source change was made after acceptance.

## Deviations
None. The `python3` substitution is an environment note, not a deviation from the method or acceptance criteria.

## Limitations
- Synthetic records only; no real corpus, source rights decision, clearance, PII/secrets filtering, malware handling or benchmark corpus.
- Exhaustive all-pairs near-duplicate comparison is quadratic and bounded by the 1,000-record synthetic limit; no MinHash/LSH, streaming, compression or production shard manifests.
- Exact normalized matching plus 5-word shingles only; no semantic, paraphrase, translated or code-specific deduplication.
- Benchmark quarantine is exact normalized match only; it makes no fuzzy contamination claim and provides no coverage or false-positive estimate.
- `group_id` correctness is a caller/registry obligation; a wrong family key can over- or under-merge clusters.
- The demo's 50/25/25 split exercises all three labels; it is not evidence of real 98/1/1 proportions.
- No tokenizer, BPE training, token counts or model-input normalization.
- One machine and one Python version; no cross-platform determinism, quality, generalization or performance claim.

## Proposed next action
Maintainer review of PR #45. T0004 (#38) remains open; no merge or issue closure is performed here. Production-scale deduplication, source rights/pins, benchmark acquisition and tokenizer work remain separate reviewed tasks.
