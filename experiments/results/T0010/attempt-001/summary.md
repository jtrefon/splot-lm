# T0010 attempt-001 — passed (portable tokenized shards and document-safe packing)

Task T0010 (IMP), parent #8/#10, issue #60. Reviewed specification merged at `9aabfa9dae9d1d7b5713cee09a73e0e4d50b5d88` (PR #61); approved minimum base `9fc3b48c10d2e477df971a1a133562da9d699c7b`; tested implementation commit `4edff6d4d9de0cf9d0da24845e751ec8f7eea23f`. Branch `feat/T0010-shards-attempt-001`. Observed UTC window 2026-11-27T18:54:18Z to 2026-11-27T18:54:47Z. Evidence PR #64.

## Authorization and scope
The owner explicitly authorized T0010 on this Mac M4 after the spec merged. Scope was Mac CPU only, standard library only, one foreground session under 90 minutes, at most 1000 synthetic documents of at most 4096 UTF-8 bytes, at most 256 KiB total training text, pack lengths 96/128/256, at most 8 blocks per shard, 1024 blocks, 1 MiB per shard and 16 MiB total. No external data, downloads, packages, model training, MPS/CUDA, network storage, source-rights approval, system changes, retries or additional tasks.

Allowed new files were exactly:

- `tools/tokenized_shards.py`
- `tools/test_tokenized_shards.py`
- `research/data/tokenized-shards.md`

No existing code, fixture, harness, CI, config or registry file changed. All demo text is original synthetic EN/PL prose from the accepted corpus oracle.

## Implementation
- The bridge starts from `corpus_reference.demo_report()` (digest `b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f`) and selects retained eligible cluster representatives. Training, validation and test clusters are disjoint; quarantined and excluded members are never packed.
- BPE 320 is trained only on non-quarantined train representatives. The artifact SHA-256 is `851742b8ae6fbe9cde68b9db6e09cb77c6f7f978e96b64da162841a311bd2b84`, actual vocabulary 320; the training digest is `ac0ddc51a214f38ea9779af63a7a0adddd94f4f087628f9de20e20e3da29e3cf`.
- Eleven eligible representatives are encoded once, checked for control IDs and vocabulary range, and verified by exact UTF-8 round trip. The packer adds BOS/EOS at document boundaries only, uses PAD=0 and IGNORE=-100, resets positions per document, and emits offset-1/offset-2 labels that never cross a document or padding boundary. Overlong documents are rejected rather than truncated.
- Hand-calculated tiny blocks match `tools/data_fixture.py` semantics on a no-merges artifact.
- The format is versioned canonical JSONL shards plus canonical `manifest.json` and `tokenizer.json`. The manifest records document/cluster/source/split identity, numeric document mapping, token records, pack config, tokenizer hash, per-shard paths/bytes/hashes/counts, source identity and three distinct digests: logical, packed logical and physical manifest.
- `load_dataset` verifies the complete file set, canonical bytes, duplicate JSON keys, UTF-8, hashes, schema/types, token ranges, BOS/EOS boundaries, masks, positions, labels, shard order/coverage, counts and independently repacked blocks before returning data. It rejects symlinks, extra/missing files, unsafe paths and corrupted manifests/shards.
- Materialization stages in a temporary sibling and atomically promotes to a new nonexistent destination; overwrite is refused. The demo materializes twice byte-identically and verifies that a different shard partition preserves logical/packed digests while changing the physical manifest.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| implementation | passed | `summary.md`, `environment.json`, `demo.json`, `manifest.json`, `execution.log` |
| unit_tests | passed | `summary.md`, `metrics.jsonl`, `demo.json`, `manifest.json`, `execution.log` |
| handover | passed | `summary.md`, `environment.json`, `execution.log` |

## Observed results
- Tested implementation HEAD was exactly `4edff6d4d9de0cf9d0da24845e751ec8f7eea23f`; the worktree was clean before acceptance and after the commands.
- `PYTHONPATH=tools python3 tools/test_corpus_reference.py`: exit 0, 30 tests, 0 failures/errors, 0.078373 s.
- `PYTHONPATH=tools python3 tools/test_bpe_reference.py`: exit 0, 28 tests, 0 failures/errors, 0.109476 s.
- `PYTHONPATH=tools python3 tools/test_data_fixture.py`: exit 0, 6 tests, 0 failures/errors, 0.041408 s.
- `PYTHONPATH=tools python3 tools/test_tokenized_shards.py`: exit 0, 21 tests, 0 failures/errors, 0.168155 s.
- `PYTHONPATH=tools python3 tools/tokenized_shards.py --demo`: exit 0 twice, 0.055080 s and 0.054925 s, 19,754 stdout bytes each, no stderr or timeout. Outputs were byte-identical with SHA-256 `e329c8fbb2fb34e4f7de66ec7c0ad661786e70725db620bfb6d85fb4743e2dcf`.
- Demo manifest: 9,497 bytes, SHA-256 `d44629937047fc543f3ce028c7aabf01214f3e9e93547b81262d4f04dd3d47e9`; logical digest `5d8d5898c41c36ac21cc7ba4785c8747c0a6ec8b20830419a0a082dfd931b5ba`; packed logical digest `7585194a799c1bc839591362929bf4e041bb18b4acaf9d15d677ff951743635f`; physical manifest digest `a413fbfd1c34e953b6f105cf70101cd16937440989775d215fd711cfc32abc7a`.
- Demo dataset: 11 documents, 4 blocks, 3 shards. Counts: 203 content tokens, 22 special tokens, 225 valid tokens, 159 pad tokens, 214 NTP labels and 203 offset-2 labels. Training used 8 documents; validation/test and quarantine counts are recorded in the manifest by split/language/domain.
- Two independent materializations were byte-identical. A one-block-per-shard partition increased shards from 3 to 4 while preserving logical and packed digests and changing the physical manifest digest.
- Environment: macOS 27.2.1 (build 26B131), arm64 Apple M4, 10 cores, 17,179,869,184 bytes RAM, CPython 3.15.2, Unicode data 17.0.0, CPU only. Memory availability 42% and available disk 81,695,732 KiB at capture.
- `python3` note: the task names `python`, but this host has no `python` executable. Acceptance commands were run as `python3`; this is an environment note, not a methodological deviation.
- First failure: none. No retry was run.

## Scope diff
Implementation commit `4edff6d` adds exactly the three allowed new files and changes no existing file. The evidence commit adds `experiments/results/T0010/attempt-001/` and one INDEX append. No source change was made after acceptance.

## Deviations
None. The `python3` substitution is an environment note, not a deviation from the method or acceptance criteria.

## Limitations
- Synthetic documents only; no external corpus, source-rights decision, clearance, acquisition, PII filtering or benchmark corpus.
- Small reference format, not production-scale sharding, streaming, compression, memory mapping, concurrent writers or distributed storage.
- Packed shards intentionally use whole documents and homogeneous blocks; no automatic document segmentation for overlong documents.
- No production tokenizer backend, special-token training, SentencePiece/HF compatibility or serving work.
- No quality, throughput, tokenizer fertility or downstream model claim.
- Cross-platform shard equality is not claimed until a separate Linux comparison verifies the canonical demo digests.
- Production tokenizer/source provenance and materialization remain separate reviewed gates.

## Proposed next action
Maintainer review of PR #64 and the hosted CPU check. T0010 (#60) remains open; no merge or issue closure is performed here.
