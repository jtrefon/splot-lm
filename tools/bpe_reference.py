"""Deterministic byte-level BPE correctness oracle (synthetic data only).

This is a mathematical reference, not a production tokenizer and not an HF
ByteLevel equivalent. It deliberately merges across whitespace inside each
document and performs no regex pretokenization. Control IDs are never produced
by encoding raw text and must be handled explicitly by a caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
FIRST_RESERVED_ID = 4
LAST_RESERVED_ID = 7
BYTE_OFFSET = 8
BYTE_COUNT = 256
BASE_VOCAB_SIZE = BYTE_OFFSET + BYTE_COUNT
MAX_MERGES = 56
ALLOWED_VOCAB_SIZES = frozenset({264, 288, 320})
MAX_DOCUMENTS = 1000
MAX_DOCUMENT_BYTES = 4096
MAX_TRAIN_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024
SPLITS = ("train", "validation", "test")
ALLOWED_LANGUAGES = frozenset({"en", "pl"})
ALLOWED_DOMAINS = frozenset({"prose", "code"})
ARTIFACT_VERSION = 1
ALGORITHM = "byte_bpe_reference_v1"
SPECIAL_TOKENS = {
    "0": "PAD",
    "1": "BOS",
    "2": "EOS",
    "3": "UNK",
    "4": "RESERVED_4",
    "5": "RESERVED_5",
    "6": "RESERVED_6",
    "7": "RESERVED_7",
}
ARTIFACT_KEYS = frozenset(
    {
        "version",
        "algorithm",
        "special_tokens",
        "byte_offset",
        "byte_count",
        "merges",
        "training_digest",
        "requested_vocab_size",
        "actual_vocab_size",
    }
)


class BpeError(ValueError):
    """Invalid document, artifact, token sequence or configuration."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    split: str
    language: str
    domain: str
    text: str
    quarantined: bool = False


def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BpeError(f"{name} must be a nonempty string")
    return value


def _utf8(text: str, name: str = "text") -> bytes:
    if not isinstance(text, str):
        raise BpeError(f"{name} must be a string")
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise BpeError(f"{name} contains an invalid surrogate") from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def validate_documents(documents: Iterable[Document]) -> tuple[Document, ...]:
    """Validate immutable synthetic split-tagged documents in bounded memory."""
    try:
        iterator = iter(documents)
    except TypeError as exc:
        raise BpeError("documents must be an iterable of Document instances") from exc
    docs = tuple(islice(iterator, MAX_DOCUMENTS + 1))
    if len(docs) > MAX_DOCUMENTS:
        raise BpeError(f"document count exceeds {MAX_DOCUMENTS}")
    seen: set[str] = set()
    train_bytes = 0
    checked: list[Document] = []
    for doc in docs:
        if not isinstance(doc, Document):
            raise BpeError("documents must be Document instances")
        _require_nonempty_str(doc.doc_id, "document ID")
        if doc.doc_id in seen:
            raise BpeError(f"duplicate document ID: {doc.doc_id!r}")
        seen.add(doc.doc_id)
        if not isinstance(doc.split, str) or doc.split not in SPLITS:
            raise BpeError(f"document {doc.doc_id!r} has invalid split: {doc.split!r}")
        if not isinstance(doc.language, str) or doc.language not in ALLOWED_LANGUAGES:
            raise BpeError(
                f"document {doc.doc_id!r} has unsupported language: {doc.language!r}"
            )
        if not isinstance(doc.domain, str) or doc.domain not in ALLOWED_DOMAINS:
            raise BpeError(
                f"document {doc.doc_id!r} has unsupported domain: {doc.domain!r}"
            )
        if type(doc.quarantined) is not bool:
            raise BpeError(f"document {doc.doc_id!r} quarantined flag must be bool")
        if not isinstance(doc.text, str):
            raise BpeError(f"document {doc.doc_id!r} text must be a string")
        if doc.text == "":
            raise BpeError(f"document {doc.doc_id!r} text must not be empty")
        data = _utf8(doc.text, f"document {doc.doc_id!r} text")
        if len(data) > MAX_DOCUMENT_BYTES:
            raise BpeError(
                f"document {doc.doc_id!r} exceeds {MAX_DOCUMENT_BYTES} UTF-8 bytes"
            )
        if doc.split == "train":
            train_bytes += len(data)
            if train_bytes > MAX_TRAIN_BYTES:
                raise BpeError(f"train bytes exceed {MAX_TRAIN_BYTES}")
        checked.append(doc)
    checked.sort(key=lambda item: item.doc_id)
    return tuple(checked)


def document_content_sha256(text: str) -> str:
    """SHA-256 over the exact original UTF-8 bytes, with no normalization."""
    return hashlib.sha256(_utf8(text)).hexdigest()


def _sorted_train_documents(
    documents: Sequence[Document],
) -> tuple[Document, ...]:
    return tuple(
        sorted(
            (
                doc
                for doc in documents
                if doc.split == "train" and not doc.quarantined
            ),
            key=lambda item: item.doc_id,
        )
    )


def _train_input_digest(train_docs: Sequence[Document]) -> str:
    payload = [
        {
            "doc_id": doc.doc_id,
            "sha256": document_content_sha256(doc.text),
        }
        for doc in sorted(train_docs, key=lambda item: item.doc_id)
    ]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def train_input_digest(documents: Iterable[Document]) -> str:
    """Digest of sorted non-quarantined train IDs and exact content hashes."""
    docs = validate_documents(documents)
    return _train_input_digest(_sorted_train_documents(docs))


def corpus_digest(documents: Iterable[Document]) -> str:
    """Stable digest of all synthetic document metadata and content hashes."""
    docs = validate_documents(documents)
    payload = [
        {
            "doc_id": doc.doc_id,
            "split": doc.split,
            "language": doc.language,
            "domain": doc.domain,
            "quarantined": doc.quarantined,
            "sha256": document_content_sha256(doc.text),
        }
        for doc in docs
    ]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def bytes_to_token_ids(data: bytes | bytearray) -> list[int]:
    """Map every raw byte to its fixed base-vocabulary token ID."""
    if not isinstance(data, (bytes, bytearray)):
        raise BpeError("bytes_to_token_ids expects bytes")
    return [BYTE_OFFSET + value for value in bytes(data)]


def _build_expansions(artifact: Mapping[str, Any]) -> list[bytes]:
    actual = artifact["actual_vocab_size"]
    expansions: list[bytes | None] = [None] * actual
    for value in range(BYTE_COUNT):
        expansions[BYTE_OFFSET + value] = bytes([value])
    for index, pair in enumerate(artifact["merges"]):
        left, right = pair
        new_id = BASE_VOCAB_SIZE + index
        left_bytes = expansions[left]
        right_bytes = expansions[right]
        if left_bytes is None or right_bytes is None:
            raise BpeError("artifact merge references an unavailable expansion")
        if len(left_bytes) + len(right_bytes) > MAX_DOCUMENT_BYTES:
            raise BpeError(
                f"artifact token expansion exceeds {MAX_DOCUMENT_BYTES} bytes"
            )
        expansions[new_id] = left_bytes + right_bytes
    if any(expansion is None for expansion in expansions[BYTE_OFFSET:]):
        raise BpeError("artifact token expansion is incomplete")
    return [expansion if expansion is not None else b"" for expansion in expansions]


def token_ids_to_bytes(
    token_ids: Iterable[int],
    artifact: Mapping[str, Any],
    *,
    allow_control: bool = False,
) -> bytes:
    """Expand tokens to bytes; control IDs require explicit caller opt-in."""
    validate_artifact(artifact)
    try:
        iterator = iter(token_ids)
    except TypeError as exc:
        raise BpeError("token_ids must be an iterable of integers") from exc
    tokens = tuple(islice(iterator, MAX_DOCUMENT_BYTES + 1))
    if len(tokens) > MAX_DOCUMENT_BYTES:
        raise BpeError(f"token count exceeds {MAX_DOCUMENT_BYTES}")
    expansions = _build_expansions(artifact)
    output = bytearray()
    for token in tokens:
        if type(token) is not int:
            raise BpeError("token IDs must be integers")
        if token < 0 or token >= artifact["actual_vocab_size"]:
            raise BpeError(f"token ID out of range: {token}")
        if token < BYTE_OFFSET:
            if not allow_control:
                raise BpeError(f"control token ID requires explicit handling: {token}")
            continue
        expansion = expansions[token]
        if len(output) + len(expansion) > MAX_DOCUMENT_BYTES:
            raise BpeError(f"decoded output exceeds {MAX_DOCUMENT_BYTES} bytes")
        output.extend(expansion)
    return bytes(output)


def decode(
    token_ids: Iterable[int],
    artifact: Mapping[str, Any],
    *,
    allow_control: bool = False,
) -> str:
    """Decode tokens by byte concatenation and strict UTF-8 validation."""
    data = token_ids_to_bytes(token_ids, artifact, allow_control=allow_control)
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BpeError(f"token bytes are not valid UTF-8: {exc}") from exc


def count_adjacent_pairs(
    sequences: Iterable[Sequence[int]],
) -> Counter[tuple[int, int]]:
    """Count adjacent token pairs within each sequence; never across sequences."""
    counts: Counter[tuple[int, int]] = Counter()
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            counts[(left, right)] += 1
    return counts


def merge_sequence(
    tokens: Sequence[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    """Replace nonoverlapping left-to-right occurrences of a pair."""
    left, right = pair
    merged: list[int] = []
    index = 0
    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == left
            and tokens[index + 1] == right
        ):
            merged.append(new_id)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


def best_pair(counts: Mapping[tuple[int, int], int]) -> tuple[int, int] | None:
    """Choose greatest frequency, then lexicographically smallest pair."""
    if not counts:
        return None
    return min(counts, key=lambda pair: (-counts[pair], pair))


def train_bpe(
    documents: Iterable[Document],
    *,
    requested_vocab_size: int = 320,
    max_merges: int = MAX_MERGES,
) -> dict[str, Any]:
    """Train a deterministic byte BPE artifact on non-quarantined train docs."""
    if type(requested_vocab_size) is not int or requested_vocab_size not in ALLOWED_VOCAB_SIZES:
        raise BpeError(
            f"requested_vocab_size must be one of {sorted(ALLOWED_VOCAB_SIZES)}"
        )
    if type(max_merges) is not int or not 0 <= max_merges <= MAX_MERGES:
        raise BpeError(f"max_merges must be between 0 and {MAX_MERGES}")
    docs = validate_documents(documents)
    train_docs = _sorted_train_documents(docs)
    training_digest = _train_input_digest(train_docs)
    effective_max = min(max_merges, requested_vocab_size - BASE_VOCAB_SIZE)
    sequences = [bytes_to_token_ids(_utf8(doc.text)) for doc in train_docs]
    merges: list[list[int]] = []
    while len(merges) < effective_max:
        pair = best_pair(count_adjacent_pairs(sequences))
        if pair is None:
            break
        new_id = BASE_VOCAB_SIZE + len(merges)
        sequences = [merge_sequence(sequence, pair, new_id) for sequence in sequences]
        merges.append([pair[0], pair[1]])
    artifact: dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "algorithm": ALGORITHM,
        "special_tokens": dict(SPECIAL_TOKENS),
        "byte_offset": BYTE_OFFSET,
        "byte_count": BYTE_COUNT,
        "merges": merges,
        "training_digest": training_digest,
        "requested_vocab_size": requested_vocab_size,
        "actual_vocab_size": BASE_VOCAB_SIZE + len(merges),
    }
    validate_artifact(artifact)
    return artifact


def encode(text: str, artifact: Mapping[str, Any]) -> list[int]:
    """Apply learned merges in rank order; never emits control IDs."""
    validate_artifact(artifact)
    data = _utf8(text)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise BpeError(f"encode input exceeds {MAX_DOCUMENT_BYTES} UTF-8 bytes")
    tokens = bytes_to_token_ids(data)
    for index, pair in enumerate(artifact["merges"]):
        tokens = merge_sequence(tokens, (pair[0], pair[1]), BASE_VOCAB_SIZE + index)
    return tokens


def validate_artifact(artifact: Any) -> None:
    """Strictly validate a byte BPE artifact schema and merge references."""
    if not isinstance(artifact, dict):
        raise BpeError("artifact must be an object")
    if set(artifact) != ARTIFACT_KEYS:
        raise BpeError("artifact has missing or unexpected keys")
    if type(artifact["version"]) is not int or artifact["version"] != ARTIFACT_VERSION:
        raise BpeError("artifact version is unsupported")
    if artifact["algorithm"] != ALGORITHM:
        raise BpeError("artifact algorithm is unsupported")
    if artifact["special_tokens"] != SPECIAL_TOKENS:
        raise BpeError("artifact special-token map is invalid")
    if type(artifact["byte_offset"]) is not int or artifact["byte_offset"] != BYTE_OFFSET:
        raise BpeError("artifact byte offset is invalid")
    if type(artifact["byte_count"]) is not int or artifact["byte_count"] != BYTE_COUNT:
        raise BpeError("artifact byte count is invalid")
    requested = artifact["requested_vocab_size"]
    if type(requested) is not int or requested not in ALLOWED_VOCAB_SIZES:
        raise BpeError("artifact requested vocabulary size is invalid")
    digest = artifact["training_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise BpeError("artifact training digest is invalid")
    merges = artifact["merges"]
    if not isinstance(merges, list):
        raise BpeError("artifact merges must be a list")
    if len(merges) > MAX_MERGES or len(merges) > requested - BASE_VOCAB_SIZE:
        raise BpeError("artifact has too many merges for its vocabulary size")
    seen_pairs: set[tuple[int, int]] = set()
    expansion_lengths: dict[int, int] = {
        BYTE_OFFSET + value: 1 for value in range(BYTE_COUNT)
    }
    for index, merge in enumerate(merges):
        if not isinstance(merge, list) or len(merge) != 2:
            raise BpeError("artifact merge must be a two-element list")
        left, right = merge
        if type(left) is not int or type(right) is not int:
            raise BpeError("artifact merge IDs must be integers")
        if left < BYTE_OFFSET or right < BYTE_OFFSET:
            raise BpeError("artifact merge references a control or reserved ID")
        if left >= BASE_VOCAB_SIZE + index or right >= BASE_VOCAB_SIZE + index:
            raise BpeError("artifact merge references a future or unknown ID")
        pair = (left, right)
        if pair in seen_pairs:
            raise BpeError("artifact contains a duplicate merge pair")
        seen_pairs.add(pair)
        left_length = expansion_lengths.get(left)
        right_length = expansion_lengths.get(right)
        if left_length is None or right_length is None:
            raise BpeError("artifact merge references an unavailable expansion")
        new_length = left_length + right_length
        if new_length > MAX_DOCUMENT_BYTES:
            raise BpeError(
                f"artifact token expansion exceeds {MAX_DOCUMENT_BYTES} bytes"
            )
        expansion_lengths[BASE_VOCAB_SIZE + index] = new_length
    actual = artifact["actual_vocab_size"]
    if type(actual) is not int or actual != BASE_VOCAB_SIZE + len(merges):
        raise BpeError("artifact actual vocabulary size is inconsistent")
    if actual > requested or actual > BASE_VOCAB_SIZE + MAX_MERGES:
        raise BpeError("artifact actual vocabulary size exceeds its bounds")


def canonical_artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    """Canonical compact JSON bytes used for the artifact SHA-256."""
    validate_artifact(artifact)
    return _canonical_json(artifact)


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_artifact_bytes(artifact)).hexdigest()


def serialize_artifact(artifact: Mapping[str, Any]) -> str:
    """Return canonical artifact JSON text (no trailing newline)."""
    return canonical_artifact_bytes(artifact).decode("utf-8")


def _reject_constant(value: str) -> None:
    raise BpeError(f"nonfinite JSON constant: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BpeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_artifact(
    data: bytes | bytearray | str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Parse, strictly validate, hash-check and canonical-check an artifact."""
    if isinstance(data, str):
        try:
            raw = data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BpeError("artifact input contains an invalid surrogate") from exc
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        raise BpeError("artifact input must be bytes or text")
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise BpeError(f"artifact input exceeds {MAX_ARTIFACT_BYTES} bytes")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise BpeError("expected artifact SHA-256 is invalid")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise BpeError("artifact SHA-256 mismatch")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_pairs,
        )
    except UnicodeDecodeError as exc:
        raise BpeError("artifact is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise BpeError(f"artifact is not valid JSON: {exc}") from exc
    validate_artifact(parsed)
    if canonical_artifact_bytes(parsed) != raw:
        raise BpeError("artifact is not canonical serialized bytes")
    return parsed


_DEMO_DOCUMENTS = (
    Document(
        "demo-train-en-prose-1",
        "train",
        "en",
        "prose",
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog.\n",
    ),
    Document(
        "demo-train-en-prose-2",
        "train",
        "en",
        "prose",
        "Deterministic data preparation keeps the corpus reproducible. "
        "Deterministic data preparation keeps the corpus reproducible.\n",
    ),
    Document(
        "demo-train-pl-prose-1",
        "train",
        "pl",
        "prose",
        "Szybki brązowy lis przeskakuje nad leniwym psem. "
        "Szybki brązowy lis przeskakuje nad leniwym psem.\n",
    ),
    Document(
        "demo-train-pl-prose-2",
        "train",
        "pl",
        "prose",
        "Przygotowanie danych jest deterministyczne i powtarzalne. "
        "Przygotowanie danych jest deterministyczne i powtarzalne.\n",
    ),
    Document(
        "demo-train-en-code",
        "train",
        "en",
        "code",
        "def add(a, b):\n    return a + b\n\n"
        "def add(a, b):\n    return a + b\n",
    ),
    Document(
        "demo-train-pl-code",
        "train",
        "pl",
        "code",
        "# Dodawanie dwóch liczb\n"
        "def suma(a, b):\n    return a + b\n\n"
        "# Dodawanie dwóch liczb\n"
        "def suma(a, b):\n    return a + b\n",
    ),
    Document(
        "demo-train-quarantine",
        "train",
        "en",
        "prose",
        "QUARANTINE: this synthetic document must never affect pair counts. "
        "QUARANTINE: this synthetic document must never affect pair counts.\n",
        quarantined=True,
    ),
    Document(
        "demo-val-en-prose",
        "validation",
        "en",
        "prose",
        "Held-out English prose mentions a silver heron beside a quiet lake.\n",
    ),
    Document(
        "demo-val-pl-prose",
        "validation",
        "pl",
        "prose",
        "Wyjątkowy polski tekst wspomina srebrną czaplę nad spokojnym jeziorem.\n",
    ),
    Document(
        "demo-val-en-code",
        "validation",
        "en",
        "code",
        "def multiply(x, y):\n    return x * y\n",
    ),
    Document(
        "demo-val-pl-code",
        "validation",
        "pl",
        "code",
        "# Unikalny komentarz walidacyjny\n"
        "def iloczyn(a, b):\n    return a * b\n",
    ),
    Document(
        "demo-test-en-prose",
        "test",
        "en",
        "prose",
        "Test-only Cafe\u0301\temoji 🦊 and indentation\n    preserved exactly.\n",
    ),
    Document(
        "demo-test-pl-prose",
        "test",
        "pl",
        "prose",
        "Tylko test: żółta łódź i gęśl.\n",
    ),
    Document(
        "demo-test-en-code",
        "test",
        "en",
        "code",
        "def divide(x, y):\n    if y == 0:\n        raise ValueError('zero')\n"
        "    return x / y\n",
    ),
    Document(
        "demo-test-pl-code",
        "test",
        "pl",
        "code",
        "# Test: zażółć gęślą jaźń\n"
        "def różnica(a, b):\n    return a - b\n",
    ),
)


def demo_documents() -> tuple[Document, ...]:
    return _DEMO_DOCUMENTS


def demo_report() -> dict[str, Any]:
    """Deterministic synthetic demo with held-out per-language/domain counts."""
    docs = demo_documents()
    artifact = train_bpe(docs, requested_vocab_size=320, max_merges=MAX_MERGES)
    heldout: dict[str, dict[str, int | bool]] = {}
    for split in ("validation", "test"):
        for language in ("en", "pl"):
            for domain in ("prose", "code"):
                selected = [
                    doc
                    for doc in docs
                    if doc.split == split
                    and doc.language == language
                    and doc.domain == domain
                ]
                byte_count = sum(len(_utf8(doc.text)) for doc in selected)
                token_count = sum(len(encode(doc.text, artifact)) for doc in selected)
                roundtrip = all(
                    decode(encode(doc.text, artifact), artifact) == doc.text
                    for doc in selected
                )
                heldout[f"{split}/{language}/{domain}"] = {
                    "documents": len(selected),
                    "bytes": byte_count,
                    "tokens": token_count,
                    "roundtrip": roundtrip,
                }
    train_docs = _sorted_train_documents(docs)
    quarantined = [doc for doc in docs if doc.split == "train" and doc.quarantined]
    special_spellings = (
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "<RESERVED_4>",
        "<|endoftext|>",
    )
    control_ids = [
        token
        for text in special_spellings
        for token in encode(text, artifact)
        if token < BYTE_OFFSET
    ]
    return {
        "artifact": artifact,
        "artifact_sha256": artifact_sha256(artifact),
        "corpus_digest": corpus_digest(docs),
        "training": {
            "documents": len(train_docs),
            "bytes": sum(len(_utf8(doc.text)) for doc in train_docs),
            "quarantined_documents": len(quarantined),
            "merges": len(artifact["merges"]),
            "requested_vocab_size": artifact["requested_vocab_size"],
            "actual_vocab_size": artifact["actual_vocab_size"],
        },
        "heldout": heldout,
        "control_safety": {
            "special_spellings_checked": list(special_spellings),
            "control_ids_emitted": len(control_ids),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print the deterministic synthetic BPE demo report as JSON",
    )
    args = parser.parse_args(argv)
    if not args.demo:
        parser.print_help(sys.stderr)
        return 2
    print(
        json.dumps(
            demo_report(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
