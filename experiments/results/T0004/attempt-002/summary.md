# T0004 attempt-002 — passed (correction: explicit approval, allowlist revalidation, bounded ingestion)

Task T0004 (IMP), parent #8, issue #38. Instruction/task commit `0fcd9c6b5b9667af47603a63013b89d83e5f565f` (reviewed specification unchanged); approved minimum base `371aa551a6d553dc38245f9821dfc43490e871cb`. Attempt-001 implementation `126c046720e7cacfc52d9c64f6e7fc6e6afcdb74` and evidence `65bd4f1620f2d7235ee076c491902e2b788d6601` are preserved. Attempt-002 synced current main `2503a4ed2b219e2f9ca0d941f8d1e2ed675e5e70` at merge commit `d2910d38039e7b2f5dc6289ecc387a68d73f9f41`, then tested implementation commit `07a86f0e3a192276ad9a9954a3673dc7719e5544`. Observed UTC window 2026-11-27T16:36:00Z to 2026-11-27T16:48:19Z. Evidence PR #45.

## Review instruction and scope
The PR #45 research review required a correction before merge: `SourceApproval` must not default to `synthetic_approved`; the public `validate_records` entry point must revalidate approval and mapping key/source-ID equality; record iterables must be bounded before materialization; and direct tests must cover denied/pending/mismatched mappings and missing approval. The follow-up owner comment required synchronizing current main, preserving both agents' INDEX rows and passing the hosted CPU check as well as updated REP evidence.

Scope remained the same three implementation files plus this attempt's evidence and one INDEX append:

- `tools/corpus_reference.py`
- `tools/test_corpus_reference.py`
- `research/data/corpus-reference.md`

No edits were made to attempt-001 evidence, the model, harness, existing fixture, registry, configs, CI workflow or other tasks. The run remained CPU-only, synthetic-only and standard-library-only: no real corpus, downloads, packages, training, MPS, CUDA, acquisition or rights clearance. No merge or issue closure was performed.

## Precision-direction check
A 2026-11-27T16:43:45Z clarification in #7 recommends FP8-first candidate work after in-flight T0006 and explicitly states that no existing approved task or runtime changed. T0004 is a synthetic CPU corpus preflight, independent of precision choice; its task file hash is unchanged. No FP8, quantization or BF16 work was started under T0004.

## Corrections
1. `SourceApproval.approval` is now a required field with no permissive default. `SourceApproval("src", "rev")` raises `TypeError`; explicit `None`, empty, pending and denied values are rejected by `build_allowlist`.
2. `validate_records` now calls `_allowlist_mapping` at the public entry point, revalidating the allowlist mapping/source-ID equality, approval value and revision before any record is accepted. Denied, pending, key-mismatched and non-`SourceApproval` mappings are rejected.
3. Record iterables are consumed with `itertools.islice(iterator, MAX_RECORDS + 1)` and rejected when the limit is exceeded, so an arbitrary iterable is not materialized unbounded.
4. Three direct tests were added: explicit/missing approval, denied/pending/mismatched/non-allowlist mappings, and bounded generator consumption. The suite grew from 26 to 29 tests.
5. The demo and test constructors now pass `SYNTHETIC_APPROVAL` explicitly. The canonical demo output is byte-identical to attempt-001, so no output diff needs explanation.
6. Separate metadata surrogate validation was left as the review's optional follow-up; record text still has explicit UTF-8/surrogate rejection.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `summary.md`, `environment.json`, `demo.json`, `execution.log` |
| unit_tests | passed | `summary.md`, `metrics.jsonl`, `demo.json`, `execution.log` |
| handover | passed | `summary.md`, `environment.json`, `execution.log` |

## Observed results
- Tested implementation HEAD was exactly `07a86f0e3a192276ad9a9954a3673dc7719e5544`; the worktree was clean before acceptance and after the commands. The correction starting base was merge commit `d2910d3` after syncing main `2503a4e`; the instruction/task commit remained `0fcd9c6`.
- Implementation commit `07a86f0` changes exactly three files and no other file:
  - `tools/corpus_reference.py` — 999 lines, SHA-256 `65c5fd4d2bfc58b00985f989060bce8c4adead20fe270d0190f9253e71f37cc1`.
  - `tools/test_corpus_reference.py` — 727 lines, SHA-256 `45d5d728f1fc509d3656be0bc2a64a0f7d90709d86d8cccb6ab7aedb47b8b12a`.
  - `research/data/corpus-reference.md` — 245 lines, SHA-256 `4c769e613ee1f2b4d855ad3dabb2836d9db12c7090e8ff2b6bb7c834686ef9dd`.
- `PYTHONPATH=tools python3 tools/test_corpus_reference.py`: exit 0, 29 tests (3 added), 0 failures, 0 errors, 0.067817 s; stdout 0 bytes, stderr 3,462 bytes, SHA-256 `2cb4389f56a675d5c50a13c70b3443f9b0f6a60e7f994ca8323930545c46ed11`.
- `PYTHONPATH=tools python3 tools/test_data_fixture.py`: exit 0, 6 tests, 0 failures, 0 errors, 0.042010 s; existing fixture unchanged.
- `PYTHONPATH=tools python3 tools/corpus_reference.py --demo`: exit 0 twice, 0.033227 s and 0.034378 s, 25,423 stdout bytes each, no stderr or timeout. The two runs were byte-identical and also byte-identical to attempt-001 `demo.json`: SHA-256 `befc46f9aa4cd25b4d2e223e80d6f7db13a417b3cc50e36e05acd5f34cb1bd66`; report digest `b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f`.
- Demo statistics are unchanged: 20 synthetic records, 13 clusters, 11 eligible, 2 quarantined clusters containing 4 records, 4 near-duplicate edges, 8 train / 1 validation / 2 test clusters at the demo 50/25/25 percentages, and audit all five checks true.
- Environment: macOS 27.2.1 (build 26B131), arm64 Apple M4, 10 cores, 17,179,869,184 bytes RAM, CPython 3.15.2 via `python3`; memory availability 48% and available disk 81,388,564 KiB at capture. CPU only; no CUDA or MPS.
- The correction was also exercised under the CI-equivalent environment (`PYTHONPATH=src:tools:tools/rep`, `CUDA_VISIBLE_DEVICES=''`, single-threaded BLAS): reference tests 29/29, fixture tests 6/6 and REP validator tests 9/9 passed.
- Hosted CPU regression is required by the review and is tracked in PR #45; this summary does not fabricate a hosted result.
- First failure: none. No retry was run.

## Scope diff
The correction implementation commit `07a86f0` modifies exactly the three allowed files and no other file. Merge commit `d2910d3` only synchronizes `origin/main` (`2503a4e`) and preserves both the T0004 attempt-001 and T0005 INDEX rows. The attempt-002 evidence commit adds `experiments/results/T0004/attempt-002/` and one INDEX row. Attempt-001 evidence and the implementation commit `126c046` are untouched. No source change was made after acceptance.

## Deviations
None. The `python3` substitution is an environment note, not a deviation from the method or acceptance criteria.

## Limitations
- Synthetic records only; no real corpus, source-rights decision, clearance, PII/secrets filtering, malware handling or benchmark corpus.
- Exhaustive all-pairs near-duplicate comparison is quadratic and bounded by the 1,000-record synthetic limit; no MinHash/LSH, streaming, compression or production shard manifests.
- Exact normalized matching plus 5-word shingles only; no semantic, paraphrase, translated or code-specific deduplication.
- Benchmark quarantine is exact normalized match only; no fuzzy contamination claim and no coverage or false-positive estimate.
- `group_id` correctness is a caller/registry obligation; a wrong family key can over- or under-merge clusters.
- The demo's 50/25/25 split exercises all three labels; it is not evidence of real 98/1/1 proportions.
- No tokenizer, BPE training, token counts or model-input normalization.
- Record text has explicit UTF-8/surrogate rejection; separate metadata surrogate validation remains an optional follow-up.
- One machine and one Python version; no cross-platform determinism, quality, generalization or performance claim.

## Proposed next action
Maintainer review of the corrected PR #45 and the hosted CPU check. T0004 (#38) remains open; no merge or issue closure is performed here. Production-scale deduplication, source rights/pins, benchmark acquisition and tokenizer work remain separate reviewed tasks.
