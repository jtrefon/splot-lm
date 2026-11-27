# Production tokenizer backend qualification
2026-11-27 | Synthetic qualification, #72 / #8 / #11 | T0015

**Status:** bounded local qualification of Hugging Face Tokenizers 0.23.3 as a
ByteLevel BPE backend candidate. This is not a production tokenizer release, a
trained model, a corpus-rights decision or a remote tokenizer download. The
backend is optional and imported lazily; default standard-library CI does not
require it.

## 1. Isolated dependency qualification

- Package: `tokenizers==0.23.3`, `requires_python >=3.10`.
- Interpreter: existing CPython 3.12.16; no interpreter installed or changed.
- Platform: macOS arm64.
- Wheel: `tokenizers-0.23.3-cp310-abi3-macosx_11_0_arm64.whl`, official PyPI
  wheel, 3,101,381 bytes, SHA-256
  `68649e97d5b43c44c031d8d848874a6eecae8f8fe40ea989aa777a5a83aca716`.
- Installation: session-local venv outside the repository, `--no-index
  --no-deps --only-binary=:all:` from the verified wheel. No system install,
  sudo, compiler build, other packages or interpreter change.
- `tokenizers` declares `huggingface-hub>=0.16.4,<3.0`; the Hub package was not
  installed because this task performs no remote model/tokenizer download and
  local ByteLevel BPE import/training was verified without it.
- Full lock evidence: `configs/data/tokenizers-0.23.3-lock.json`.

## 2. Adapter contract

The adapter uses Hugging Face `Tokenizer(models.BPE(unk_token="<UNK>"))` with:

- `pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)`;
- `decoders.ByteLevel()`;
- fixed special IDs: 0 PAD, 1 BOS, 2 EOS, 3 UNK, 4–7 reserved;
- all 256 ByteLevel alphabet symbols in the initial alphabet;
- `BpeTrainer(vocab_size=16384 or 32768, min_frequency=2,
  max_token_length=128, show_progress=False)`;
- deterministic train-only input sorted by stable document ID;
- `TOKENIZERS_PARALLELISM=false`.

The actual vocabulary is reported; it is not padded to the target. The demo
corpus is small, so both 16k/32k requests currently achieve 431 tokens. No
production-size or fertility claim follows.

The adapter's byte-alphabet token IDs need not equal the T0008 reference oracle's
fixed byte offsets. It deliberately uses the HF ByteLevel regex pretokenizer;
therefore merge sequences are not expected to equal the no-regex reference
oracle. Tests compare exact UTF-8 semantics and round-trip, not merge equality.

## 3. Raw-text and special-token safety

`add_special_tokens=False` is not sufficient because literal special strings can
still be recognized as control tokens by the tokenizer vocabulary. The adapter:

1. rejects raw text containing `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` or any reserved
   spelling before encoding (fail closed; content is not deleted or normalized);
2. encodes with `add_special_tokens=False`;
3. rejects any emitted control ID 0–7, UNK, or out-of-vocabulary ID;
4. verifies exact byte-level UTF-8 decode equals the original text.

BOS/EOS are inserted only by the explicit `encode(..., add_boundaries=True)`
operation; normal encoding and decoding do not interpret or emit controls.
`decode(..., allow_control=True)` is the separate explicit operation that strips
boundary/control tokens.

## 4. Artifact and canonical identity

The artifact is canonical JSON with:

- format/version (`splot-production-tokenizer-v1`, version 1);
- package name/version;
- target/actual vocabulary size, min frequency, max token length,
  `add_prefix_space=False`, `use_regex=True`, complete special map, byte alphabet
  count and tokenizer JSON version;
- train-only input digest and sorted train document IDs;
- the HF tokenizer JSON (vocab/merges/model/pre-tokenizer/decoder);
- logical digest over the canonical artifact excluding `logical_digest`.

`artifact_sha256` is the SHA-256 of the exact canonical artifact bytes. Reload is
canonical-byte and hash checked, bounded to 8 MiB, and reconstructs the tokenizer
to verify special IDs, complete byte alphabet, vocabulary bounds and logical
identity. The package's internal physical serialization may differ across builds;
logical identity is compared separately and nondeterministic tie ordering must be
reported rather than hidden.

## 5. API

From `PYTHONPATH=tools`, `import production_tokenizer`:

```python
backend_info() -> dict
validate_documents(documents) -> tuple[TrainingDocument, ...]
training_digest(documents) -> str
train_tokenizer(documents, target_vocab_size=16384) -> dict
validate_artifact(artifact) -> None
canonical_artifact_bytes(artifact) -> bytes
artifact_sha256(artifact) -> str
load_artifact(data, expected_sha256) -> dict
encode(text, artifact, *, add_boundaries=False) -> list[int]
decode(token_ids, artifact, *, allow_control=False) -> str
demo_documents() -> tuple[TrainingDocument, ...]
demo_report() -> dict
main(["--demo"]) -> int
```

Training input is bounded to 1000 original synthetic documents, 4096 UTF-8 bytes
per document and 256 KiB total train text. Encode and decode inputs are bounded
to 4096 UTF-8 bytes and 4098 tokens respectively; decode iterates at most 4099
tokens before failing closed. Validation/test/quarantined documents do not
affect training, and input order does not change the artifact. The backend tests
independently pin the qualified demo's physical artifact hashes
`2f148ac9…e6fff` (16k) and `8026f305…827b40` (32k) and logical digests
`b6474d4c…972cf0` (16k) and `1a49d99f…d0c58` (32k). A different physical hash
must be reported as build nondeterminism; logical identity is the canonical
comparison.

## 6. Demo

With the qualified venv, `python3 tools/production_tokenizer.py --demo` trains
16k and 32k ByteLevel BPE artifacts on the six non-quarantined synthetic train
documents (612 bytes) and reports held-out EN/PL prose/code counts. Observed
again in the local qualification:

- 16k target: actual 431 tokens; artifact 9,711 bytes, SHA-256
  `2f148ac98771660e8479411f90fa8c1ff0e80907896a1d823bf0a322542e6fff`,
  logical digest
  `b6474d4cff55fdd526c4bfcba342dada259353864fce33e1863d614b02972cf0`.
- 32k target: actual 431 tokens; artifact 9,711 bytes, SHA-256
  `8026f3058d154ce86105ebf4b66ee4ccab941a9d569d39267edd5d84c8827b40`,
  logical digest
  `1a49d99f5ffaa2be78aa9722dfcc8adf78bf41af3ea7faa9910dedb8bcdf0c58`.
- Both repeat runs were byte/logical identical. Held-out: 510 bytes, 385 tokens,
  zero control/UNK, exact round-trip; descriptive bytes/token 1.3247 only.

No quality, production fertility or downstream performance claim is made.

## 7. Default CI and limitations

Default hosted CI has no `tokenizers` package. The adapter probes availability
through `importlib.util.find_spec` and distribution metadata, so importing the
adapter or its default tests never imports `tokenizers`. `tools/test_production_tokenizer.py`
therefore runs stdlib boundary tests and prints an explicit “backend
unavailable; stdlib tests only” message; it does not claim a backend pass and
does not use a hidden unittest skip. The optional backend test command must be
run in the qualified venv:

```bash
PYTHONPATH=tools TOKENIZERS_PARALLELISM=false <venv>/bin/python tools/test_production_tokenizer.py
PYTHONPATH=tools TOKENIZERS_PARALLELISM=false <venv>/bin/python tools/production_tokenizer.py --demo
```

Limitations: no remote tokenizer/model download; no production-size corpus; no
claim of HF equivalence to the no-regex oracle; no byte-for-byte cross-platform
artifact guarantee; no corpus acquisition or rights approval. The accompanying
pilot approval packet is metadata-only and requires owner and qualified rights
review before any download.

## 8. Verification

```bash
PYTHONPATH=tools python3 tools/test_corpus_reference.py
PYTHONPATH=tools python3 tools/test_bpe_reference.py
PYTHONPATH=tools python3 tools/test_data_fixture.py
PYTHONPATH=tools python3 tools/test_tokenized_shards.py
PYTHONPATH=tools python3 tools/test_production_tokenizer.py  # stdlib fallback
```

Evidence is under `experiments/results/T0015/attempt-001/`, including the wheel
lock, backend test report and deterministic demo JSON; no raw corpus is included.
