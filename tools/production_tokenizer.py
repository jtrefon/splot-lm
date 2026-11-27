"""Production tokenizer qualification adapter (HF Tokenizers 0.23.3).

This is a bounded synthetic reference, not a production tokenizer release. The
backend is imported lazily so the default standard-library tests and CI do not
require the optional package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import bpe_reference as bpe

FORMAT = "splot-production-tokenizer-v1"
VERSION = 1
TOKENIZERS_VERSION = "0.23.3"
TARGET_VOCAB_SIZES = (16384, 32768)
SPECIAL_TOKENS = {
    0: "<PAD>",
    1: "<BOS>",
    2: "<EOS>",
    3: "<UNK>",
    4: "<RES4>",
    5: "<RES5>",
    6: "<RES6>",
    7: "<RES7>",
}
SPECIAL_STRINGS = tuple(SPECIAL_TOKENS.values())
MAX_DOCUMENTS = 1000
MAX_DOCUMENT_BYTES = 4096
MAX_TRAIN_BYTES = 256 * 1024
REAL_PILOT_MAX_DOCUMENTS = 10000
REAL_PILOT_MAX_TRAIN_BYTES = 4 * 1024 * 1024
MAX_TEXT_BYTES = 4096
MAX_TOKEN_COUNT = MAX_TEXT_BYTES + 2
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_TOKEN_LENGTH = 128
MIN_FREQUENCY = 2
USE_REGEX = True
ADD_PREFIX_SPACE = False
SPLITS = ("train", "validation", "test")
ARTIFACT_KEYS = frozenset(
    {
        "format",
        "version",
        "package",
        "config",
        "training_digest",
        "train_doc_ids",
        "tokenizer",
        "logical_digest",
    }
)


class TokenizerBackendError(RuntimeError):
    """Optional backend unavailable or incompatible."""


class TokenizerError(ValueError):
    """Invalid synthetic document, text, token sequence or artifact."""


@dataclass(frozen=True)
class TrainingDocument:
    doc_id: str
    text: str
    split: str = "train"
    language: str = "en"
    domain: str = "prose"
    quarantined: bool = False


def _utf8(text: str, name: str = "text") -> bytes:
    if not isinstance(text, str):
        raise TokenizerError(f"{name} must be a string")
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TokenizerError(f"{name} contains an invalid surrogate") from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def backend_info() -> dict[str, Any]:
    """Return optional backend availability without importing the package."""
    try:
        spec = importlib.util.find_spec("tokenizers")
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {
            "available": False,
            "version": None,
            "expected": TOKENIZERS_VERSION,
            "reason": f"tokenizers module lookup failed: {exc}",
        }
    if spec is None:
        return {
            "available": False,
            "version": None,
            "expected": TOKENIZERS_VERSION,
            "reason": "tokenizers module not installed",
        }
    try:
        version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        return {
            "available": False,
            "version": None,
            "expected": TOKENIZERS_VERSION,
            "reason": "tokenizers distribution metadata not installed",
        }
    return {
        "available": version == TOKENIZERS_VERSION,
        "version": version,
        "expected": TOKENIZERS_VERSION,
        "reason": None if version == TOKENIZERS_VERSION else "version mismatch",
    }


def _backend():
    info = backend_info()
    if not info["available"]:
        raise TokenizerBackendError(
            info.get("reason") or "tokenizers backend unavailable"
        )
    import tokenizers  # type: ignore

    if getattr(tokenizers, "__version__", None) != TOKENIZERS_VERSION:
        raise TokenizerBackendError("tokenizers version changed after probe")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    return Tokenizer, decoders, models, pre_tokenizers, trainers


def validate_documents(
    documents: Iterable[TrainingDocument],
) -> tuple[TrainingDocument, ...]:
    try:
        iterator = iter(documents)
    except TypeError as exc:
        raise TokenizerError("documents must be iterable") from exc
    docs = tuple(islice(iterator, MAX_DOCUMENTS + 1))
    if len(docs) > MAX_DOCUMENTS:
        raise TokenizerError(f"document count exceeds {MAX_DOCUMENTS}")
    seen: set[str] = set()
    train_bytes = 0
    for doc in docs:
        if not isinstance(doc, TrainingDocument):
            raise TokenizerError("documents must be TrainingDocument instances")
        if not isinstance(doc.doc_id, str) or not doc.doc_id.strip():
            raise TokenizerError("document ID must be nonempty")
        if doc.doc_id in seen:
            raise TokenizerError(f"duplicate document ID: {doc.doc_id!r}")
        seen.add(doc.doc_id)
        if doc.split not in SPLITS:
            raise TokenizerError(f"invalid split: {doc.split!r}")
        if doc.language not in ("en", "pl"):
            raise TokenizerError(f"unsupported language: {doc.language!r}")
        if doc.domain not in ("prose", "code"):
            raise TokenizerError(f"unsupported domain: {doc.domain!r}")
        if type(doc.quarantined) is not bool:
            raise TokenizerError("quarantined must be bool")
        if not isinstance(doc.text, str) or not doc.text:
            raise TokenizerError(f"document {doc.doc_id!r} text must be nonempty")
        data = _utf8(doc.text, f"document {doc.doc_id!r}")
        if len(data) > MAX_DOCUMENT_BYTES:
            raise TokenizerError(f"document {doc.doc_id!r} is too large")
        if doc.split == "train":
            train_bytes += len(data)
            if train_bytes > MAX_TRAIN_BYTES:
                raise TokenizerError(f"train bytes exceed {MAX_TRAIN_BYTES}")
    return tuple(sorted(docs, key=lambda item: item.doc_id))


def _train_documents(documents: tuple[TrainingDocument, ...]) -> tuple[TrainingDocument, ...]:
    return tuple(
        doc
        for doc in documents
        if doc.split == "train" and not doc.quarantined
    )


def _training_digest(train: tuple[TrainingDocument, ...]) -> str:
    payload = [
        {
            "doc_id": doc.doc_id,
            "sha256": _sha(_utf8(doc.text)),
            "language": doc.language,
            "domain": doc.domain,
        }
        for doc in train
    ]
    return _sha(_canonical_json(payload))


def training_digest(documents: Iterable[TrainingDocument]) -> str:
    docs = validate_documents(documents)
    return _training_digest(_train_documents(docs))


def validate_real_pilot_documents(
    documents: Iterable[TrainingDocument],
) -> tuple[TrainingDocument, ...]:
    """Validate the larger T0016 train-only bounds without changing synthetic limits."""
    try:
        iterator = iter(documents)
    except TypeError as exc:
        raise TokenizerError("documents must be iterable") from exc
    docs = tuple(islice(iterator, REAL_PILOT_MAX_DOCUMENTS + 1))
    if len(docs) > REAL_PILOT_MAX_DOCUMENTS:
        raise TokenizerError(
            f"real pilot document count exceeds {REAL_PILOT_MAX_DOCUMENTS}"
        )
    seen: set[str] = set()
    train_bytes = 0
    for doc in docs:
        if not isinstance(doc, TrainingDocument):
            raise TokenizerError("documents must be TrainingDocument instances")
        if not isinstance(doc.doc_id, str) or not doc.doc_id.strip():
            raise TokenizerError("document ID must be nonempty")
        if doc.doc_id in seen:
            raise TokenizerError(f"duplicate document ID: {doc.doc_id!r}")
        seen.add(doc.doc_id)
        if doc.split != "train":
            raise TokenizerError("real pilot tokenizer accepts train-only documents")
        if doc.language not in ("en", "pl"):
            raise TokenizerError(f"unsupported language: {doc.language!r}")
        if doc.domain != "prose":
            raise TokenizerError("real pilot tokenizer accepts prose only")
        if doc.quarantined:
            raise TokenizerError("real pilot tokenizer rejects quarantined documents")
        if not isinstance(doc.text, str) or not doc.text:
            raise TokenizerError(f"document {doc.doc_id!r} text must be nonempty")
        data = _utf8(doc.text, f"document {doc.doc_id!r}")
        if len(data) > MAX_DOCUMENT_BYTES:
            raise TokenizerError(f"document {doc.doc_id!r} is too large")
        _text_is_safe(doc.text)
        train_bytes += len(data)
        if train_bytes > REAL_PILOT_MAX_TRAIN_BYTES:
            raise TokenizerError(
                f"real pilot train bytes exceed {REAL_PILOT_MAX_TRAIN_BYTES}"
            )
    return tuple(sorted(docs, key=lambda item: item.doc_id))


def _config(
    target_vocab_size: int,
    actual_vocab_size: int,
    tokenizer_json: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target_vocab_size": target_vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "min_frequency": MIN_FREQUENCY,
        "max_token_length": MAX_TOKEN_LENGTH,
        "add_prefix_space": ADD_PREFIX_SPACE,
        "use_regex": USE_REGEX,
        "special_tokens": {str(key): value for key, value in SPECIAL_TOKENS.items()},
        "byte_alphabet_count": 256,
        "tokenizers_version": TOKENIZERS_VERSION,
        "tokenizer_json_version": tokenizer_json.get("version"),
    }


def _tokenizer_from_json(tokenizer_json: Mapping[str, Any]):
    Tokenizer, _, _, _, _ = _backend()
    return Tokenizer.from_str(json.dumps(tokenizer_json, ensure_ascii=False))


def _verify_tokenizer(tokenizer, artifact_config: Mapping[str, Any]) -> None:
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != artifact_config["actual_vocab_size"]:
        raise TokenizerError("tokenizer vocabulary size mismatch")
    if vocab_size > artifact_config["target_vocab_size"]:
        raise TokenizerError("tokenizer exceeds target vocabulary size")
    for token_id, token in SPECIAL_TOKENS.items():
        if tokenizer.token_to_id(token) != token_id:
            raise TokenizerError(f"special token ID mismatch for {token}")
    alphabet = _backend()[3].ByteLevel.alphabet()
    if len(alphabet) != 256:
        raise TokenizerError("ByteLevel alphabet is not complete")
    vocab = tokenizer.get_vocab()
    missing = [token for token in alphabet if vocab.get(token) is None]
    if missing:
        raise TokenizerError("byte alphabet is incomplete")


def _text_is_safe(text: str) -> None:
    if not isinstance(text, str):
        raise TokenizerError("text must be a string")
    data = _utf8(text)
    if len(data) > MAX_TEXT_BYTES:
        raise TokenizerError(f"text exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
    for special in SPECIAL_STRINGS:
        if special in text:
            raise TokenizerError(
                f"raw text contains reserved special spelling {special!r}; fail closed"
            )


def _train_artifact(
    train: tuple[TrainingDocument, ...],
    target_vocab_size: int,
    digest: str,
) -> dict[str, Any]:
    if not train:
        raise TokenizerError("no non-quarantined train documents")
    Tokenizer, decoders, models, pre_tokenizers, trainers = _backend()
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=ADD_PREFIX_SPACE,
        use_regex=USE_REGEX,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=target_vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=list(SPECIAL_STRINGS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=MAX_TOKEN_LENGTH,
        show_progress=False,
    )
    tokenizer.train_from_iterator((doc.text for doc in train), trainer=trainer)
    tokenizer_json = json.loads(tokenizer.to_str())
    config = _config(target_vocab_size, tokenizer.get_vocab_size(), tokenizer_json)
    _verify_tokenizer(tokenizer, config)
    artifact: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "package": {"name": "tokenizers", "version": TOKENIZERS_VERSION},
        "config": config,
        "training_digest": digest,
        "train_doc_ids": [doc.doc_id for doc in train],
        "tokenizer": tokenizer_json,
    }
    artifact["logical_digest"] = _sha(
        _canonical_json({key: value for key, value in artifact.items() if key != "logical_digest"})
    )
    validate_artifact(artifact)
    return artifact


def train_tokenizer(
    documents: Iterable[TrainingDocument],
    target_vocab_size: int = 16384,
) -> dict[str, Any]:
    """Train a deterministic ByteLevel BPE artifact on non-quarantined train docs."""
    if target_vocab_size not in TARGET_VOCAB_SIZES:
        raise TokenizerError(f"target vocabulary must be one of {TARGET_VOCAB_SIZES}")
    docs = validate_documents(documents)
    train = _train_documents(docs)
    return _train_artifact(train, target_vocab_size, training_digest(docs))


def train_real_pilot_tokenizer(
    documents: Iterable[TrainingDocument],
    target_vocab_size: int = 16384,
) -> dict[str, Any]:
    """Train the T0016 16k candidate on bounded train-only real-pilot text."""
    if target_vocab_size != 16384:
        raise TokenizerError("real pilot tokenizer target must be 16384")
    docs = validate_real_pilot_documents(documents)
    return _train_artifact(docs, target_vocab_size, _training_digest(docs))


def validate_artifact(artifact: Any) -> None:
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
        raise TokenizerError("artifact schema is invalid")
    if artifact["format"] != FORMAT or artifact["version"] != VERSION:
        raise TokenizerError("artifact format/version is unsupported")
    package = artifact["package"]
    if not isinstance(package, dict) or set(package) != {"name", "version"}:
        raise TokenizerError("artifact package metadata is invalid")
    if package["name"] != "tokenizers" or package["version"] != TOKENIZERS_VERSION:
        raise TokenizerError("artifact package version is unsupported")
    config = artifact["config"]
    if not isinstance(config, dict):
        raise TokenizerError("artifact config is invalid")
    required = {
        "target_vocab_size",
        "actual_vocab_size",
        "min_frequency",
        "max_token_length",
        "add_prefix_space",
        "use_regex",
        "special_tokens",
        "byte_alphabet_count",
        "tokenizers_version",
        "tokenizer_json_version",
    }
    if set(config) != required:
        raise TokenizerError("artifact config keys are invalid")
    if config["target_vocab_size"] not in TARGET_VOCAB_SIZES:
        raise TokenizerError("artifact target vocabulary is invalid")
    if type(config["actual_vocab_size"]) is not int:
        raise TokenizerError("artifact actual vocabulary is invalid")
    if config["actual_vocab_size"] > config["target_vocab_size"]:
        raise TokenizerError("artifact vocabulary exceeds target")
    if config["min_frequency"] != MIN_FREQUENCY or config["max_token_length"] != MAX_TOKEN_LENGTH:
        raise TokenizerError("artifact training constants are invalid")
    if config["add_prefix_space"] is not False or config["use_regex"] is not True:
        raise TokenizerError("artifact ByteLevel config is invalid")
    expected_specials = {str(key): value for key, value in SPECIAL_TOKENS.items()}
    if config["special_tokens"] != expected_specials:
        raise TokenizerError("artifact special-token mapping is invalid")
    if config["byte_alphabet_count"] != 256:
        raise TokenizerError("artifact byte alphabet count is invalid")
    if config["tokenizers_version"] != TOKENIZERS_VERSION:
        raise TokenizerError("artifact package config version is invalid")
    if type(config["tokenizer_json_version"]) not in (str, int, type(None)):
        raise TokenizerError("artifact tokenizer JSON version is invalid")
    if not _is_hex64(artifact["training_digest"]):
        raise TokenizerError("artifact training digest is invalid")
    if not isinstance(artifact["train_doc_ids"], list) or not all(
        isinstance(item, str) and item for item in artifact["train_doc_ids"]
    ):
        raise TokenizerError("artifact train document IDs are invalid")
    if artifact["train_doc_ids"] != sorted(set(artifact["train_doc_ids"])):
        raise TokenizerError("artifact train document IDs are not sorted/unique")
    if not isinstance(artifact["tokenizer"], dict):
        raise TokenizerError("artifact tokenizer JSON is invalid")
    if not _is_hex64(artifact["logical_digest"]):
        raise TokenizerError("artifact logical digest is invalid")
    expected_logical = _sha(
        _canonical_json({key: value for key, value in artifact.items() if key != "logical_digest"})
    )
    if artifact["logical_digest"] != expected_logical:
        raise TokenizerError("artifact logical digest mismatch")
    tokenizer = _tokenizer_from_json(artifact["tokenizer"])
    _verify_tokenizer(tokenizer, config)


def canonical_artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    validate_artifact(artifact)
    return _canonical_json(artifact)


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    return _sha(canonical_artifact_bytes(artifact))


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TokenizerError(f"nonfinite JSON constant: {value}")


def load_artifact(data: bytes | bytearray | str, expected_sha256: str) -> dict[str, Any]:
    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        raise TokenizerError("artifact input must be bytes or text")
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise TokenizerError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    if not _is_hex64(expected_sha256) or _sha(raw) != expected_sha256:
        raise TokenizerError("artifact SHA-256 mismatch")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_pairs,
        )
    except UnicodeDecodeError as exc:
        raise TokenizerError("artifact is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise TokenizerError(f"artifact JSON is invalid: {exc}") from exc
    validate_artifact(parsed)
    if _canonical_json(parsed) != raw:
        raise TokenizerError("artifact is not canonical JSON bytes")
    return parsed


def _tokenizer(artifact: Mapping[str, Any]):
    validate_artifact(artifact)
    return _tokenizer_from_json(artifact["tokenizer"])


class TokenizerSession:
    """A single validated tokenizer instance reused for bounded pilot work."""

    def __init__(self, artifact: Mapping[str, Any]):
        validate_artifact(artifact)
        self.artifact = artifact
        self.actual_vocab_size = artifact["config"]["actual_vocab_size"]
        self._tokenizer = _tokenizer_from_json(artifact["tokenizer"])

    def encode(self, text: str, *, add_boundaries: bool = False) -> list[int]:
        _text_is_safe(text)
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = list(encoding.ids)
        if len(ids) > MAX_TOKEN_COUNT:
            raise TokenizerError(f"encoded token count exceeds {MAX_TOKEN_COUNT}")
        if any(token < 8 for token in ids):
            raise TokenizerError("tokenizer emitted a control/special ID for raw text")
        if 3 in ids:
            raise TokenizerError("tokenizer emitted UNK for raw text")
        if any(token >= self.actual_vocab_size for token in ids):
            raise TokenizerError("tokenizer emitted an out-of-vocabulary ID")
        if self._tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise TokenizerError("tokenizer round-trip failed")
        if add_boundaries:
            return [1, *ids, 2]
        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        allow_control: bool = False,
    ) -> str:
        try:
            iterator = iter(token_ids)
        except TypeError as exc:
            raise TokenizerError("token_ids must be iterable") from exc
        ids = list(islice(iterator, MAX_TOKEN_COUNT + 1))
        if len(ids) > MAX_TOKEN_COUNT:
            raise TokenizerError(f"token count exceeds {MAX_TOKEN_COUNT}")
        if any(
            type(token) is not int or not 0 <= token < self.actual_vocab_size
            for token in ids
        ):
            raise TokenizerError("token ID is invalid")
        if not allow_control and any(token < 8 for token in ids):
            raise TokenizerError("control/special token requires explicit handling")
        content = [token for token in ids if token >= 8]
        try:
            return self._tokenizer.decode(content, skip_special_tokens=False)
        except Exception as exc:
            raise TokenizerError(f"token decode failed: {exc}") from exc


def create_session(artifact: Mapping[str, Any]) -> TokenizerSession:
    return TokenizerSession(artifact)


def encode(
    text: str,
    artifact: Mapping[str, Any],
    *,
    add_boundaries: bool = False,
) -> list[int]:
    _text_is_safe(text)
    tokenizer = _tokenizer(artifact)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = list(encoding.ids)
    if len(ids) > MAX_TOKEN_COUNT:
        raise TokenizerError(f"encoded token count exceeds {MAX_TOKEN_COUNT}")
    if any(token < 8 for token in ids):
        raise TokenizerError("tokenizer emitted a control/special ID for raw text")
    if 3 in ids:
        raise TokenizerError("tokenizer emitted UNK for raw text")
    if any(token >= artifact["config"]["actual_vocab_size"] for token in ids):
        raise TokenizerError("tokenizer emitted an out-of-vocabulary ID")
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise TokenizerError("tokenizer round-trip failed")
    if add_boundaries:
        return [1, *ids, 2]
    return ids


def decode(
    token_ids: Iterable[int],
    artifact: Mapping[str, Any],
    *,
    allow_control: bool = False,
) -> str:
    tokenizer = _tokenizer(artifact)
    try:
        iterator = iter(token_ids)
    except TypeError as exc:
        raise TokenizerError("token_ids must be iterable") from exc
    ids = list(islice(iterator, MAX_TOKEN_COUNT + 1))
    if len(ids) > MAX_TOKEN_COUNT:
        raise TokenizerError(f"token count exceeds {MAX_TOKEN_COUNT}")
    if any(type(token) is not int or not 0 <= token < artifact["config"]["actual_vocab_size"] for token in ids):
        raise TokenizerError("token ID is invalid")
    if not allow_control and any(token < 8 for token in ids):
        raise TokenizerError("control/special token requires explicit handling")
    content = [token for token in ids if token >= 8]
    try:
        return tokenizer.decode(content, skip_special_tokens=False)
    except Exception as exc:
        raise TokenizerError(f"token decode failed: {exc}") from exc


def demo_documents() -> tuple[TrainingDocument, ...]:
    documents = []
    for doc in bpe.demo_documents():
        documents.append(
            TrainingDocument(
                doc.doc_id,
                doc.text,
                doc.split,
                doc.language,
                doc.domain,
                doc.quarantined,
            )
        )
    return tuple(documents)


def _token_stats(artifact, docs):
    rows = {}
    for doc in docs:
        try:
            ids = encode(doc.text, artifact)
            decoded = decode(ids, artifact)
            key = f"{doc.split}/{doc.language}/{doc.domain}"
            entry = rows.setdefault(
                key,
                {"documents": 0, "bytes": 0, "tokens": 0, "roundtrip": True, "control_or_unk": 0},
            )
            entry["documents"] += 1
            entry["bytes"] += len(_utf8(doc.text))
            entry["tokens"] += len(ids)
            entry["roundtrip"] = entry["roundtrip"] and decoded == doc.text
            entry["control_or_unk"] += sum(1 for token in ids if token < 8 or token == 3)
        except TokenizerError:
            raise
    return rows


def demo_report() -> dict[str, Any]:
    docs = demo_documents()
    train = _train_documents(validate_documents(docs))
    heldout = tuple(doc for doc in docs if doc.split != "train" and not doc.quarantined)
    targets: dict[str, Any] = {}
    for target in TARGET_VOCAB_SIZES:
        first = train_tokenizer(docs, target)
        second = train_tokenizer(docs, target)
        first_bytes = canonical_artifact_bytes(first)
        second_bytes = canonical_artifact_bytes(second)
        stats = _token_stats(first, heldout)
        total_bytes = sum(row["bytes"] for row in stats.values())
        total_tokens = sum(row["tokens"] for row in stats.values())
        targets[str(target)] = {
            "requested_vocab_size": target,
            "actual_vocab_size": first["config"]["actual_vocab_size"],
            "artifact_bytes": len(first_bytes),
            "artifact_sha256": _sha(first_bytes),
            "logical_digest": first["logical_digest"],
            "repeat_artifact_identical": first_bytes == second_bytes,
            "repeat_logical_equal": first["logical_digest"] == second["logical_digest"],
            "heldout": stats,
            "heldout_bytes": total_bytes,
            "heldout_tokens": total_tokens,
            "bytes_per_token": (total_bytes / total_tokens) if total_tokens else None,
            "tokens_per_byte": (total_tokens / total_bytes) if total_bytes else None,
        }
    return {
        "format": FORMAT,
        "version": VERSION,
        "package": {"name": "tokenizers", "version": TOKENIZERS_VERSION},
        "training_documents": len(train),
        "training_bytes": sum(len(_utf8(doc.text)) for doc in train),
        "targets": targets,
        "quality_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    if not args.demo:
        parser.print_help(sys.stderr)
        return 2
    info = backend_info()
    if not info["available"]:
        print(
            f"backend unavailable: tokenizers {TOKENIZERS_VERSION} not importable "
            f"({info.get('reason')}); default stdlib tests only",
            file=sys.stderr,
        )
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
