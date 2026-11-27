# T0015 attempt-001 — production tokenizer backend and bilingual pilot approval packet
2026-11-27 | IMP | Issue #72 | Parents #8/#11 | PR #75

**Outcome:** passed locally. Hosted CPU CI is required on PR #75 before merge.
**Instructions:** `cb97b045c965beb614083b54dd7be26393bbe7b8` (reviewed main, PR #73).
**Approved minimum base:** `3fb9388e782a9da8ca68e1458d9caaeb296c2312`.
**Tested implementation:** `34ba392e16df1273dcd54f68a25563f2c0597f7c`, clean before and after acceptance.
**Scope:** synthetic tokenizer qualification and metadata-only source approval packet. No real text, corpus snapshot, training run, gated access, paid service or system change.

## 1. Environment and dependency qualification

The spec was merged at `cb97b04`; the previous T0013 PR was merged at `3fb9388`. The execution environment was macOS 27.2.1 on arm64 Apple M4, 10 cores / 16 GiB RAM, CPU only.

- Existing default interpreter: CPython 3.15.2, Unicode 17.0.0.
- Existing compatible backend interpreter: CPython 3.12.16, Unicode 15.0.0.
- Candidate: `tokenizers==0.23.3`, `requires_python >=3.10`.
- Official wheel: `tokenizers-0.23.3-cp310-abi3-macosx_11_0_arm64.whl`.
- Wheel size: 3,101,381 bytes.
- Wheel SHA-256: `68649e97d5b43c44c031d8d848874a6eecae8f8fe40ea989aa777a5a83aca716`.
- Installation: session-local venv outside the repository with `--no-index --no-deps --only-binary=:all:`.
- No interpreter, system package, compiler build, sudo or other package change.
- `tokenizers` declares `huggingface-hub>=0.16.4,<3.0`; the Hub package was deliberately not installed because no Hub/model download or Hub API is used. Local import and ByteLevel BPE training were verified without it. The lock records the declared specifier and the reason.
- The default adapter path probes availability with `importlib.util.find_spec` and distribution metadata; it does not import `tokenizers`. Hosted stdlib CI therefore runs without the backend.

## 2. Implementation

The tested commit adds exactly the six allowed files:

| File | SHA-256 |
|---|---|
| `tools/production_tokenizer.py` | `830a300fe6a6030e2b46ffd3af00a910a95f01fbcd8abc357f7c1743c48c7b4f` |
| `tools/test_production_tokenizer.py` | `f1454ec721b388cc42e837fbe9a644c6c4f8fb929850f4b79b84246d3cb5ce4d` |
| `research/data/production-tokenizer.md` | `f64ad2d04fae1c0c96372871af55476249e03e7b418f36e8d399d6a99f2d65dd` |
| `research/data/pilot-approval-packet.md` | `3b2ef9e9faf745125ae0b41b4b94e408502eedfd21ed5de63feda339f40a0ad8` |
| `research/data/pilot-approval-packet-v0.1.json` | `b2a54b7d27437ca5415a061e2bcfd5d52b54a58c7de72522d00cdb3368331cee` |
| `configs/data/tokenizers-0.23.3-lock.json` | `2c69b4720829e4b6a650c1a06d44b5fdb232997119cb2898bcad4b4436069838` |

The adapter uses HF ByteLevel BPE with no normalization, `add_prefix_space=False`, `use_regex=True`, all 256 byte-alphabet symbols, fixed special IDs (PAD 0, BOS 1, EOS 2, UNK 3, reserved 4–7), `min_frequency=2`, `max_token_length=128`, `TOKENIZERS_PARALLELISM=false`, and deterministic sorted train-only input. Target sizes are 16k and 32k; the achieved vocabulary is reported without padding.

Raw text is fail-closed: any literal `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` or reserved spelling is rejected before encoding; encoded control/UNK/out-of-vocabulary IDs are rejected; exact UTF-8 round-trip is required. BOS/EOS are inserted only by the explicit boundary operation. Encode is capped at 4096 UTF-8 bytes and decode iterates at most 4098 tokens before failing closed. Artifact JSON is canonical, hash-pinned, bounded to 8 MiB, reloaded exactly, and independently pinned for the qualified demo. No nondeterministic tie ordering was observed.

## 3. Source approval packet

The packet proposes eight prose families, four per language, with two train families per language and one validation plus one test family per language:

- EN train: Pride and Prejudice, Frankenstein.
- PL train: Lalka, Chłopi.
- EN validation: Alice’s Adventures in Wonderland.
- PL validation: Quo vadis.
- EN test: Moby Dick.
- PL test: Faraon.

All four Standard Ebooks item pages were rechecked for title, CC0 1.0 production license, U.S. public-domain warning, word count and first item-history commit. The four proposed revisions match the official pages. All five Wolne Lektury API records were rechecked for title, language and genre; Polish author dates are taken from the official item pages.

*Pan Tadeusz* is officially classified **Epos** and is recorded as `verse_epic` / `excluded_nonprose`; it is explicitly excluded from the prose pilot. No silent prose/verse mixture is proposed.

No source is acquisition-ready. Open blockers include EU/Polish/Spain rights review, four unresolved Standard Ebooks author dates, artifact checksums or immutable Wolne Lektury revisions, annotation/motif licensing, illustration exclusion, benchmark/duplicate checks and owner approval. The pilot is capped at 10 MiB raw UTF-8 total and 2.5 MiB per item; if over cap, a family is removed rather than truncating a work. No download, immutable snapshot or rights clearance occurred.

## 4. Unit tests and demo

All acceptance commands exited 0 with no timeout or retry. Total: **109 tests**.

| Check | Tests | Duration (s) |
|---|---:|---:|
| `tools/test_corpus_reference.py` | 30 | 0.063404 |
| `tools/test_bpe_reference.py` | 28 | 0.109828 |
| `tools/test_data_fixture.py` | 6 | 0.041888 |
| `tools/test_tokenized_shards.py` | 27 | 0.203057 |
| `tools/test_production_tokenizer.py` (stdlib) | 4 | 0.047825 |
| `tools/test_production_tokenizer.py` (venv backend) | 14 | 0.335114 |

The stdlib run printed `backend unavailable: tokenizers module not installed; stdlib tests only` and did not claim a backend pass. The backend run used the qualified venv and covered special IDs, all 256 bytes, Polish/English/code/whitespace/combining/emoji, train-only isolation and input order, literal-special rejection, boundaries, artifact reload/digest mismatch, malformed artifacts, pinned digests, decode bounds and demo/CLI determinism.

Two demo runs were byte-identical: **4,103 bytes**, SHA-256 `769a2d2b36bb625b444e493c43bfb4e1598903b89f1e32db8c53e78600527d98`.

| Target | Actual vocab | Artifact bytes | Artifact SHA-256 | Logical digest |
|---:|---:|---:|---|---|
| 16,384 | 431 | 9,711 | `2f148ac98771660e8479411f90fa8c1ff0e80907896a1d823bf0a322542e6fff` | `b6474d4cff55fdd526c4bfcba342dada259353864fce33e1863d614b02972cf0` |
| 32,768 | 431 | 9,711 | `8026f3058d154ce86105ebf4b66ee4ccab941a9d569d39267edd5d84c8827b40` | `1a49d99f5ffaa2be78aa9722dfcc8adf78bf41af3ea7faa9910dedb8bcdf0c58` |

Held-out synthetic data: 510 bytes / 385 tokens, 1.324675 bytes/token descriptively, zero control/UNK tokens, eight groups with exact round-trip. This is not a production fertility, quality or downstream performance claim.

## 5. First failure and reproduction

During intermediate development, the new decode-bound test initially omitted the artifact argument:

```text
TypeError: decode() missing 1 required positional argument: 'artifact'
```

The test was corrected to pass `self.artifact_16`; the final tested SHA `34ba392` passes all 14 backend tests. Reproduction command:

```bash
PYTHONPATH=tools TOKENIZERS_PARALLELISM=false <SESSION_VENV>/bin/python tools/test_production_tokenizer.py
```

No acceptance command failed, timed out or was retried.

## 6. Limitations and differences

- The synthetic corpus is six train documents / 612 bytes; actual vocabularies are 431 for both targets. This is not a production tokenizer release, a 16k/32k trained vocabulary, or evidence of fertility or model quality.
- Physical artifact hashes pin the qualified macOS arm64 wheel; logical digests are the canonical identity. Cross-platform physical equality is not claimed. Any differing physical hash must be reported as build nondeterminism.
- `tokenizers` was installed without `huggingface-hub` because no Hub access is used. The declared dependency is recorded, but no installed transitive version is pinned; no other package was installed.
- The approval packet is metadata-only and pending. It does not clear EU/Polish/Spain rights, edition/annotation/illustration obligations, benchmark overlap or access.
- Standard Ebooks only provide a U.S. public-domain warning; Wolne Lektury metadata lacks immutable revisions/checksums. First-download SHA-256 is proposed as the snapshot identity, not proof of upstream immutability.
- The 10 MiB cap may require removing a family; truncating a work is prohibited.
- No legal opinion, source registry change, corpus acquisition or training is included.
- Hosted CPU CI is not claimed as passed in this summary; it is required and tracked on PR #75.

## 7. Handover

Evidence is flat under `experiments/results/T0015/attempt-001/`: summary, result, environment, metrics, sanitized execution log, deterministic demo, backend test report, wheel lock, source-review summary and machine-readable source packet. No raw corpus or binary wheel is committed. `tools/rep/validate_result.py` was run after hashing; the validator output is recorded in PR #75 and the result handover detail.

Proposed next action: maintainer reviews PR #75 and the eight-family packet; any actual snapshot acquisition requires a separate explicit owner approval after qualified rights review. No merge, issue closure or source acquisition is performed here.
