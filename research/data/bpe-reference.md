# Byte-level BPE correctness oracle
2026-11-27 | Synthetic reference only, #52 / #8 | T0008

**Status:** deterministic, dependency-free byte BPE mathematical oracle for
small synthetic EN/PL prose/code fixtures. It is not a production tokenizer, not
an HF `ByteLevel` implementation, not a source-rights decision, and not a
quality or fertility benchmark. No external corpus or model is used.

## 1. Scope and token layout

The oracle trains on explicit synthetic documents and produces a canonical JSON
artifact. It deliberately merges across whitespace inside each document and does
no regex pretokenization; this is a documented divergence from the proposed HF
ByteLevel regex pipeline.

| Token IDs | Meaning |
|---|---|
| 0 | PAD |
| 1 | BOS |
| 2 | EOS |
| 3 | UNK |
| 4–7 | reserved |
| 8–263 | all 256 raw byte values, in order |
| 264+ | learned merge tokens, sequential by rank |

The base vocabulary is 264. Allowed requested sizes are exactly 264, 288 and
320; the maximum merge count is 56. The real artifact size is
`264 + len(merges)`; vocabulary is never padded. Tiny corpora may stop early
with an actual size below the request.

Raw text is strict UTF-8. No Unicode normalization, casefolding, whitespace
collapse, control interpretation or regex splitting is applied. Diacritics,
combining marks, emoji, tabs, indentation and CRLF are preserved exactly.
Encoding never emits control IDs for raw text, even when the text spells
`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` or a reserved spelling.

## 2. Document model and validation

```python
Document(doc_id, split, language, domain, text, quarantined=False)
```

- `doc_id` is a nonempty unique stable ID across all splits.
- `split` is exactly `train`, `validation` or `test`.
- `language` is exactly `en` or `pl`.
- `domain` is exactly `prose` or `code`.
- `text` is nonempty strict UTF-8 without lone surrogates and at most 4096 UTF-8
  bytes. It is retained unchanged.
- `quarantined` is a boolean; quarantined training documents are excluded from
  training, pair counts and the train-input digest.
- At most 1000 documents are accepted. Total bytes across all `train`-split
  documents, including quarantined train documents, must not exceed 256 KiB.
  Document iterables are bounded to 1001 entries before rejection.

Documents are sorted by `doc_id` for stable processing. Validation/test content
is never used for pair counts or training. The train-input digest is SHA-256 over
canonical JSON of the sorted non-quarantined train `(doc_id, content SHA-256)`
pairs, where content hashes are over exact original UTF-8 bytes.

## 3. Training algorithm

1. Validate documents and select only non-quarantined `train` documents.
2. Convert each document's raw UTF-8 bytes to base byte IDs.
3. Count adjacent token pairs **within each document**; document boundaries are
   never crossed.
4. Choose the greatest frequency; ties are broken by lexicographically smallest
   pair ID tuple.
5. Merge nonoverlapping occurrences left-to-right in every train document and
   assign the next sequential ID.
6. Recompute counts and repeat until no pair remains, the requested vocabulary
   size is reached, or the maximum merge count is reached.

The artifact records the ordered merge pairs, so encoding applies learned merges
in rank order. This is a correctness oracle, not a claim of equivalence to a
particular production trainer's tie-breaking or regex pipeline.

## 4. Artifact and canonical serialization

The artifact is a strict JSON object with exactly these keys:

| Key | Meaning |
|---|---|
| `version` | schema version, currently `1` |
| `algorithm` | `byte_bpe_reference_v1` |
| `special_tokens` | exact ID-to-name map for IDs 0–7 |
| `byte_offset` | `8` |
| `byte_count` | `256` |
| `merges` | ordered `[left_id, right_id]` pairs |
| `training_digest` | SHA-256 of sorted train IDs/content hashes |
| `requested_vocab_size` | 264, 288 or 320 |
| `actual_vocab_size` | `264 + len(merges)` |

Validation rejects missing or extra keys, wrong version/algorithm, altered
special-token maps, invalid byte bounds, unsupported requested sizes, malformed
digests, duplicate merge pairs, control references, future/unknown references,
too many merges, and inconsistent actual sizes. Reload requires the SHA-256 of
the exact serialized bytes; no pickle is used.

Canonical artifact bytes are compact JSON with sorted keys, ASCII escapes,
compact separators and UTF-8 encoding. `artifact_sha256` is the SHA-256 of those
canonical bytes. `serialize_artifact` returns the canonical text without a
trailing newline.

## 5. Encode and decode safety

`encode(text, artifact)` maps strict UTF-8 bytes to base IDs and applies learned
merges in rank order. It never produces IDs 0–7.

`decode(tokens, artifact)` expands byte and merge tokens, concatenates the exact
bytes and performs strict UTF-8 decoding. Any control ID raises `BpeError` by
default. A separate explicit `allow_control=True` caller operation skips control
IDs; this is not used by normal text decoding. Unknown, negative, boolean or
out-of-range token IDs are rejected. Invalid UTF-8 byte sequences are rejected.

`bytes_to_token_ids` and `token_ids_to_bytes` expose the fixed byte codec and
round-trip all 256 byte values exactly.

## 6. Importable API

From `PYTHONPATH=tools`, `import bpe_reference`:

```python
validate_documents(documents) -> tuple[Document, ...]
document_content_sha256(text) -> str
train_input_digest(documents) -> str
corpus_digest(documents) -> str
train_bpe(documents, *, requested_vocab_size=320, max_merges=56) -> dict
bytes_to_token_ids(data) -> list[int]
token_ids_to_bytes(token_ids, artifact, *, allow_control=False) -> bytes
encode(text, artifact) -> list[int]
decode(token_ids, artifact, *, allow_control=False) -> str
count_adjacent_pairs(sequences) -> Counter
merge_sequence(tokens, pair, new_id) -> list[int]
best_pair(counts) -> tuple[int, int] | None
validate_artifact(artifact) -> None
canonical_artifact_bytes(artifact) -> bytes
artifact_sha256(artifact) -> str
serialize_artifact(artifact) -> str
load_artifact(data, expected_sha256) -> dict
demo_documents() -> tuple[Document, ...]
demo_report() -> dict
main(["--demo"]) -> int
```

`train_bpe` accepts only the allowed vocabulary targets and merge bounds. It
records the actual size rather than padding the vocabulary.

## 7. Demo

`python3 tools/bpe_reference.py --demo` emits a deterministic JSON report with:

- the full artifact and its canonical SHA-256;
- the synthetic corpus digest;
- training document/byte/quarantine/merge counts and requested/actual size;
- held-out validation/test counts by language and domain, including bytes,
  tokens and exact round-trip status;
- a control-safety check showing zero control IDs emitted for special spellings.

The demo corpus contains original synthetic EN/PL prose and code in disjoint
train, validation and test IDs, held-out unique phrases, a quarantined train
document, combining marks, diacritics, emoji, tabs and indentation. It makes no
quality, compression, fertility or production-tokenizer claim.

## 8. Source-mapping hardening

The same implementation PR fixes the T0004 allowlist mapping edge case:
`_allowlist_mapping` requires every mapping key to equal its entry's
`source_id` before reconstruction. Swapped-key two-source mappings now raise
`CorpusError` instead of passing a set-equality check. The original T0004 demo
digest remains unchanged and is locked by a regression test.

## 9. Complexity and limitations

- Training repeatedly counts pairs and merges; cost grows with merge count and
  total train tokens. It is intended for bounded synthetic fixtures, not
  production-scale corpora.
- No MinHash, sharding, streaming, multiprocessing, memory-mapped training or
  production artifact serving.
- No HF `ByteLevel` regex pretokenization, special-token interpretation,
  normalization, casefolding or semantic tokenization.
- No external corpora, source clearance, quality evaluation, model training,
  MPS, CUDA or tokenizer-fertility claim.
- Do not claim cross-platform equality until a separate Linux comparison checks
  the canonical artifact digest.
- The oracle is a mathematical reference; production backend choice, source
  approvals and corpus materialization remain separate reviewed work.

## 10. Verification

From the repository root, with `python3` because this macOS environment has no
`python` shim:

```bash
PYTHONPATH=tools python3 tools/test_corpus_reference.py
PYTHONPATH=tools python3 tools/test_data_fixture.py
PYTHONPATH=tools python3 tools/test_bpe_reference.py
PYTHONPATH=tools python3 tools/bpe_reference.py --demo
```

Run the demo twice and compare bytes. Evidence under
`experiments/results/T0008/attempt-001/` records the tested implementation SHA,
commands, exit codes, test counts, elapsed durations, artifact hash and corpus
digest.
