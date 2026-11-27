# T0008 attempt-001 — passed (byte BPE oracle and source mapping hardening)

Task T0008 (IMP), parent #8, issue #52. Reviewed specification merged at `d2dfd903d252c1ba48641bf7de79fba42c385af8` (PR #53); approved minimum base `f60bd5622733f3619e761681a20472d8c40f478c`; tested implementation commit `c7ac8fbd8cd3beb8641ad27f1e39cc027613ca9a`. Branch `feat/T0008-bpe-attempt-001`. Observed UTC window 2026-11-27T17:35:24Z to 2026-11-27T17:41:47Z. Evidence PR #55.

## Authorization and scope
The owner explicitly authorized T0008 on this Mac M4 in the current session after the spec merged. Scope was bound by the task: Mac CPU only, one foreground session under 90 minutes, existing Python standard library, at most 1000 synthetic documents of at most 4096 UTF-8 bytes each, total train bytes at most 256 KiB, vocabulary targets 264/288/320, at most 56 merges, no downloads, packages, external corpora, model training, MPS, system changes or external data.

Allowed files were exactly the remaining T0004 mapping fix plus the new oracle and documentation. No model, fixture, harness, registry, config or CI file changed. Historical T0004 evidence was not modified. Every document and phrase used here is original synthetic content; no corpus was acquired.

## Implementation
- `_allowlist_mapping` now requires every mapping key to equal its entry's `source_id` before reconstruction. Swapped-key two-source mappings raise `CorpusError`; a direct regression covers the previous set-equality bypass. The original T0004 demo digest remains `b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f`.
- `tools/bpe_reference.py` implements the deterministic byte BPE oracle: IDs 0 PAD, 1 BOS, 2 EOS, 3 UNK, 4–7 reserved, 8–263 all bytes, 264+ learned merges; strict raw UTF-8; no normalization or regex pretokenization; document boundaries never crossed; train-only non-quarantined pair counts; greatest-frequency plus lexicographic tie-break; nonoverlapping left-to-right merges; sequential IDs; rank-order encoding; strict byte-concatenation UTF-8 decoding; explicit control-token handling; canonical strict artifact schema, SHA-256 reload verification and no pickle.
- `tools/test_bpe_reference.py` adds independent toy pair-count/tie/overlap tests, no-boundary and no-normalization checks, all-256-byte codec round-trip, Unicode/whitespace/control round-trip, special-spelling safety, malformed-artifact rejection, tiny-corpus early stop, train-only artifact equality, deterministic serialization/reload and the stable original-corpus digest.
- `research/data/bpe-reference.md` documents the API, token layout, artifact schema, training/encoding rules, demo, complexity and the explicit divergence from production HF ByteLevel regex behavior.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `summary.md`, `environment.json`, `demo.json`, `execution.log` |
| unit_tests | passed | `summary.md`, `metrics.jsonl`, `demo.json`, `execution.log` |
| handover | passed | `summary.md`, `environment.json`, `execution.log` |

## Observed results
- Tested implementation HEAD was exactly `c7ac8fbd8cd3beb8641ad27f1e39cc027613ca9a`; the worktree was clean before acceptance and after the commands. The starting base and instruction commit were `d2dfd90`; the approved minimum base was `f60bd56`.
- `PYTHONPATH=tools python3 tools/test_corpus_reference.py`: exit 0, 30 tests, 0 failures/errors, 0.070484 s. This includes the swapped-key regression and the unchanged T0004 demo digest check.
- `PYTHONPATH=tools python3 tools/test_data_fixture.py`: exit 0, 6 tests, 0 failures/errors, 0.044478 s. The existing fixture was unchanged.
- `PYTHONPATH=tools python3 tools/test_bpe_reference.py`: exit 0, 23 tests, 0 failures/errors, 0.094902 s.
- `PYTHONPATH=tools python3 tools/bpe_reference.py --demo`: exit 0 twice, 0.036288 s and 0.036002 s, 4,343 stdout bytes each, no stderr or timeout. The runs were byte-identical with SHA-256 `bd743d70d6d0c96d9c6bc7780b8fe004e1a9520d3b47e7a43c3b10945543215e`.
- Demo artifact: canonical SHA-256 `f67ef4ab5c343e3c9f81570b0d7426a5668e578aa84806058265d0aaf633ce7e`; 6 training documents, 612 training bytes, 1 quarantined training document, 56 merges, actual vocabulary 320 (requested 320). Corpus digest `6fa4bbf3cc680d4058d067059545b787edf87eba3c87a71cd1013684ee99e8cd`.
- Held-out validation/test coverage includes all eight `split/language/domain` groups (EN/PL × prose/code), each with nonzero byte/token counts and exact round-trip true. Special spelling checks emitted zero control IDs.
- Environment: macOS 27.2.1 (build 26B131), arm64 Apple M4, 10 cores, 17,179,869,184 bytes RAM, CPython 3.15.2, Unicode data 17.0.0, CPU only. Memory availability was 48% and available disk 81,341,920 KiB at capture. No CUDA or MPS was used.
- `python3` note: the task names `python`, but this host has no `python` executable. Acceptance commands were run as `python3`; this is an environment note, not a methodological deviation.
- Hosted CPU regression is required after PR publication and is tracked in the evidence PR; this summary does not fabricate a hosted result.
- First failure: none. No retry was run.

## Scope diff
Implementation commit `c7ac8fb` changes exactly five allowed files and no others: the mapping fix in `tools/corpus_reference.py` and its regression in `tools/test_corpus_reference.py`, plus new `tools/bpe_reference.py`, `tools/test_bpe_reference.py` and `research/data/bpe-reference.md`. The evidence commit adds `experiments/results/T0008/attempt-001/` and one INDEX append. No source change was made after acceptance.

## Deviations
None. The `python3` substitution is an environment note, not a deviation from the method or acceptance criteria.

## Limitations
- Synthetic documents only; no external corpus, source-rights decision, clearance, acquisition, PII filtering or benchmark corpus.
- Small correctness oracle, not a production BPE trainer or an HF ByteLevel-equivalent tokenizer. It merges across whitespace and does no regex pretokenization by design.
- No cross-platform artifact equality claim until a separate Linux comparison verifies the canonical digest.
- No quality, compression, fertility, throughput or downstream model claim.
- Training cost grows with merge count and token count; it is bounded for synthetic fixtures rather than production-scale corpora.
- No production tokenizer backend, special-token training, sentencepiece/HF compatibility, sharding, streaming or serving work.
- Document text has strict UTF-8/surrogate rejection and IDs are nonempty and unique; separate metadata-surrogate hardening is not claimed.

## Proposed next action
Maintainer review of the evidence PR and the hosted CPU check. T0008 (#52) remains open; no merge or issue closure is performed here. Production tokenizer backend selection, source approvals and corpus materialization remain separate reviewed work.
