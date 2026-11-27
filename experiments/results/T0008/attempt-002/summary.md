# T0008 attempt-002 — passed (bounded BPE expansion and canonical reload)

Task T0008 (IMP), parent #8, issue #52. Reviewed specification remains `d2dfd903d252c1ba48641bf7de79fba42c385af8`; approved minimum base `f60bd5622733f3619e761681a20472d8c40f478c`; correction starting base `441a9ec57ed502d04cff55f59eed417cb5aa1f72`; tested implementation commit `58c094f36b468c515d30ae1902a5057c7ff11b18`. Branch `feat/T0008-bpe-attempt-002`. Observed UTC window 2026-11-27T17:57:01Z to 2026-11-27T17:59:48Z. Attempt-001 implementation `c7ac8fb` and evidence `d600ccd` are preserved. Evidence PR #58.

## Publication note
The original evidence PR #55 is already merged at `441a9ec`, so it cannot accept new commits. Attempt-001 remains untouched; attempt-002 is published on a fresh branch and follow-up PR #58 referencing issue #52 and PR #55. This changes only the publication path, not the authorized synthetic CPU scope or methodology.

## Review correction and scope
The research review required a targeted resource-bounds correction: reject adversarial token expansions before allocating bytes, cap artifact and input/output sizes, bound token iterable consumption, and require canonical serialized-byte equality on reload. No corpus, model, fixture, harness, registry, config or CI change was permitted.

Implemented in exactly three files:

- `tools/bpe_reference.py` — integer expansion-length validation, 4096-byte per-token cap, 64 KiB artifact input cap, 4096-byte encode input cap, 4097-token consumption bound, 4096-byte decoded-output cap before extension, canonical reload equality.
- `tools/test_bpe_reference.py` — adversarial doubling-chain, exact 4096 boundary, overlong JSON, generator consumption, decoded-output, encode-input and noncanonical reload tests; learned demo-hash lock.
- `research/data/bpe-reference.md` — documented resource limits and canonical reload behavior.

No external corpus or real data was used.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `summary.md`, `environment.json`, `demo.json`, `execution.log` |
| unit_tests | passed | `summary.md`, `metrics.jsonl`, `demo.json`, `execution.log` |
| handover | passed | `summary.md`, `environment.json`, `execution.log` |

## Observed results
- Tested implementation HEAD was exactly `58c094f36b468c515d30ae1902a5057c7ff11b18`; the worktree was clean before acceptance and after the commands.
- `PYTHONPATH=tools python3 tools/test_corpus_reference.py`: exit 0, 30 tests, 0 failures/errors, 0.072546 s.
- `PYTHONPATH=tools python3 tools/test_data_fixture.py`: exit 0, 6 tests, 0 failures/errors, 0.043723 s.
- `PYTHONPATH=tools python3 tools/test_bpe_reference.py`: exit 0, 28 tests (5 added), 0 failures/errors, 0.107550 s.
- `PYTHONPATH=tools python3 tools/bpe_reference.py --demo`: exit 0 twice, 0.037137 s and 0.036300 s, 4,343 stdout bytes each, no stderr or timeout. The outputs were byte-identical with SHA-256 `bd743d70d6d0c96d9c6bc7780b8fe004e1a9520d3b47e7a43c3b10945543215e`.
- Genuine learned hashes are unchanged from attempt-001: artifact SHA-256 `f67ef4ab5c343e3c9f81570b0d7426a5668e578aa84806058265d0aaf633ce7e`; corpus digest `6fa4bbf3cc680d4058d067059545b787edf87eba3c87a71cd1013684ee99e8cd`. Training remains 6 documents/612 bytes, 1 quarantined document, 56 merges and actual vocabulary 320; all 8 held-out groups round-trip with zero control IDs. The original T0004 demo digest remains `b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f`.
- Adversarial regressions passed: a 2^56 doubling chain is rejected at integer length validation without expansion; depth 12 (exactly 4096 bytes) remains accepted; serialized artifacts over 64 KiB, noncanonical JSON with a matching hash, 4097-byte encode input, 4097-token generators and 4097-byte decoded output are rejected. Exact 4096 boundaries are accepted.
- Environment: macOS 27.2.1 (build 26B131), arm64 Apple M4, 10 cores, 17,179,869,184 bytes RAM, CPython 3.15.2, Unicode data 17.0.0, CPU only. Memory availability was 49% and available disk 81,378,352 KiB at capture.
- `python3` note: the task names `python`, but this host has no `python` executable. Acceptance commands were run as `python3`; this is an environment note, not a methodological deviation.
- First failure: none. No retry was run.

## Scope diff
Attempt-002 implementation commit `58c094f` changes exactly the three allowed files listed above and no others. The correction branch starts from main `441a9ec`, which already contains attempt-001. The attempt-002 evidence commit adds `experiments/results/T0008/attempt-002/` and one INDEX append. Attempt-001 evidence and all other task rows remain untouched.

## Deviations
None in implementation, method or acceptance criteria. PR #55 was already merged, so attempt-002 evidence is in follow-up PR #58; this is a publication-path constraint, not a methodological deviation.

## Limitations
- Synthetic documents only; no external corpus, source-rights decision, clearance, acquisition, PII filtering or benchmark corpus.
- Small correctness oracle, not a production BPE trainer or HF ByteLevel-equivalent tokenizer; it merges across whitespace and performs no regex pretokenization by design.
- The 4096-byte expansion, encode-input and decoded-output caps are deliberate for this oracle; valid training documents cannot produce a longer token.
- No cross-platform artifact equality claim until a separate Linux comparison verifies the canonical digest.
- No quality, compression, fertility, throughput or downstream model claim.
- No production tokenizer backend, special-token training, sentencepiece/HF compatibility, sharding, streaming or serving work.
- Document text has strict UTF-8/surrogate rejection and IDs are nonempty and unique; separate metadata-surrogate hardening is not claimed.

## Proposed next action
Maintainer review of PR #58 and the hosted CPU check. T0008 (#52) remains open; no merge or issue closure is performed here. Production tokenizer backend, source approvals and corpus materialization remain separate reviewed work.
