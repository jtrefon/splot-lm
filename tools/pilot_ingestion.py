"""Bounded loader for the T0016/T0017 splot-pilot-blocks-v1 contract."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FORMAT = 'splot-pilot-blocks-v1'
VERSION = 1
SEQUENCE_LENGTH = 512
MODEL_VOCAB = 32768
MIN_VOCAB = 264
MAX_VOCAB = 16384
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_BLOCKS = 160
MAX_TRAIN_BLOCKS = 128
MAX_VALIDATION_BLOCKS = 16
MAX_TEST_BLOCKS = 16
MIN_TRAIN_BLOCKS = 32
MIN_VALIDATION_BLOCKS = 2
PAD = 0
BOS = 1
EOS = 2
CONTENT_MIN = 8
IGNORE = -100
SPLIT_ORDER = ('train', 'validation', 'test')


class PilotError(ValueError):
    """Invalid pilot dataset, manifest, source metadata or block."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _hex64(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _require_regular(path, label, max_bytes):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PilotError(f'{label} must be a regular non-symlink file')
    size = path.stat().st_size
    if size > max_bytes:
        raise PilotError(f'{label} exceeds byte bound')
    return path.read_bytes()


def _require_object(value, name, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise PilotError(f'{name} schema is invalid')
    return value


def _require_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise PilotError(f'{name} must be a nonempty string')
    return value


def _source_snapshot(sources):
    ordered = sorted(sources, key=lambda item: (
        item['family_id'], item['split'], item['language'], item['source_url'],
        item['revision'], item['content_sha256'], item['raw_sha256']))
    return _sha256(_canonical(ordered))


def _validate_sources(sources):
    if not isinstance(sources, list) or not sources:
        raise PilotError('sources must be a nonempty list')
    families = {}
    for index, source in enumerate(sources):
        _require_object(source, f'sources[{index}]', (
            'family_id', 'language', 'split', 'content_sha256', 'raw_sha256',
            'revision', 'rights_evidence_url', 'source_url'))
        family = _require_str(source['family_id'], 'family_id')
        language = _require_str(source['language'], 'language')
        split = source['split']
        if language not in ('en', 'pl'):
            raise PilotError('source language must be en or pl')
        if split not in SPLIT_ORDER:
            raise PilotError('source split is invalid')
        if not _hex64(source['content_sha256']) or not _hex64(source['raw_sha256']):
            raise PilotError('source content/raw digest is invalid')
        _require_str(source['revision'], 'source revision')
        _require_str(source['rights_evidence_url'], 'rights evidence URL')
        _require_str(source['source_url'], 'source URL')
        families.setdefault(family, set()).add(split)
    if any(len(splits) != 1 for splits in families.values()):
        raise PilotError('a source family crosses splits')
    if not any(source['split'] == 'train' for source in sources):
        raise PilotError('sources have no train family')
    if not any(source['split'] == 'validation' for source in sources):
        raise PilotError('sources have no validation family')
    return families


def _validate_block(block, vocab_size, global_doc_ids):
    _require_object(block, 'block', ('input_ids', 'document_ids', 'position_ids',
                                     'valid', 'labels'))
    for key in ('input_ids', 'document_ids', 'position_ids', 'valid'):
        if not isinstance(block[key], list) or len(block[key]) != SEQUENCE_LENGTH:
            raise PilotError(f'block {key} length is invalid')
    labels = block['labels']
    if not isinstance(labels, dict) or set(labels) != {'1', '2'}:
        raise PilotError('block labels schema is invalid')
    for key in ('1', '2'):
        if not isinstance(labels[key], list) or len(labels[key]) != SEQUENCE_LENGTH:
            raise PilotError('block labels length is invalid')
    if not all(type(value) is bool for value in block['valid']):
        raise PilotError('block valid mask must contain booleans')
    valid_count = sum(block['valid'])
    if valid_count == 0:
        raise PilotError('block has no valid tokens')
    if block['valid'] != [True] * valid_count + [False] * (SEQUENCE_LENGTH - valid_count):
        raise PilotError('block valid mask must be a true prefix followed by padding')
    pads = range(valid_count, SEQUENCE_LENGTH)
    for index in pads:
        if (block['input_ids'][index] != PAD or block['document_ids'][index] != -1
                or block['position_ids'][index] != 0
                or labels['1'][index] != IGNORE or labels['2'][index] != IGNORE):
            raise PilotError('block padding is inconsistent')
    segments = []
    start = 0
    for index in range(1, valid_count + 1):
        if index == valid_count or block['document_ids'][index] != block['document_ids'][index - 1]:
            segments.append((start, index))
            start = index
    seen = set()
    counts = {'valid_tokens': valid_count, 'ntp_labels': 0, 'offset2_labels': 0}
    for start, end in segments:
        doc_id = block['document_ids'][start]
        if type(doc_id) is not int or doc_id < 0:
            raise PilotError('document ID must be a nonnegative integer')
        if doc_id in seen or doc_id in global_doc_ids:
            raise PilotError('document ID is duplicated')
        seen.add(doc_id)
        global_doc_ids.add(doc_id)
        if end - start < 2 or end - start > SEQUENCE_LENGTH:
            raise PilotError('document segment length is invalid')
        if block['input_ids'][start] != BOS or block['input_ids'][end - 1] != EOS:
            raise PilotError('document segment lacks BOS/EOS boundaries')
        if block['position_ids'][start:end] != list(range(end - start)):
            raise PilotError('document positions are not contiguous')
        for index in range(start + 1, end - 1):
            token = block['input_ids'][index]
            if type(token) is not int or not CONTENT_MIN <= token < vocab_size:
                raise PilotError('document content token is invalid')
        for offset in (1, 2):
            values = labels[str(offset)]
            expected_count = 0
            for index in range(start, end):
                value = values[index]
                if index + offset < end and block['document_ids'][index] == block['document_ids'][index + offset]:
                    expected = block['input_ids'][index + offset]
                    if value != expected:
                        raise PilotError('label does not match same-document future token')
                    expected_count += 1
                elif value != IGNORE:
                    raise PilotError('label crosses a document or padding boundary')
            counts[f'ntp_labels' if offset == 1 else 'offset2_labels'] += expected_count
    for index in range(valid_count):
        token = block['input_ids'][index]
        if type(token) is not int or not 0 <= token < vocab_size:
            raise PilotError('block token is out of vocabulary range')
    return counts


def load_pilot_dataset(root, expected_manifest_sha256, *, expected_source_snapshot_sha256=None,
                       expected_model_vocab=MODEL_VOCAB):
    """Verify a bounded splot-pilot-blocks-v1 dataset and return train/validation."""
    if not _hex64(expected_manifest_sha256):
        raise PilotError('expected manifest SHA-256 is invalid')
    if expected_source_snapshot_sha256 is not None and not _hex64(expected_source_snapshot_sha256):
        raise PilotError('expected source snapshot SHA-256 is invalid')
    if (type(expected_model_vocab) is not int or isinstance(expected_model_vocab, bool)
            or not MIN_VOCAB <= expected_model_vocab <= MODEL_VOCAB):
        raise PilotError('pilot model vocabulary must be an integer in [264, 32768]')
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise PilotError('dataset root must be a regular directory')
    expected_files = {'manifest.json', 'tokenizer.json', 'train.jsonl',
                      'validation.jsonl', 'test.jsonl'}
    for entry in root.iterdir():
        if entry.name not in expected_files or entry.is_symlink() or not entry.is_file():
            raise PilotError('dataset contains an unexpected or nonregular file')
    manifest_bytes = _require_regular(root / 'manifest.json', 'manifest.json',
                                      1024 * 1024)
    if _sha256(manifest_bytes) != expected_manifest_sha256:
        raise PilotError('manifest SHA-256 mismatch')
    try:
        manifest = json.loads(manifest_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotError(f'invalid manifest JSON: {exc}') from exc
    if _canonical(manifest) != manifest_bytes:
        raise PilotError('manifest is not canonical JSON')
    _require_object(manifest, 'manifest', (
        'format', 'version', 'sequence_length', 'vocab_size', 'tokenizer_sha256',
        'source_snapshot_sha256', 'sources', 'shards', 'logical_sha256'))
    if manifest['format'] != FORMAT or manifest['version'] != VERSION:
        raise PilotError('pilot format/version is unsupported')
    if manifest['sequence_length'] != SEQUENCE_LENGTH:
        raise PilotError('pilot sequence length must be 512')
    vocab_size = manifest['vocab_size']
    if type(vocab_size) is not int or not MIN_VOCAB <= vocab_size <= MAX_VOCAB:
        raise PilotError('pilot vocabulary is out of bounds')
    if expected_model_vocab not in (MODEL_VOCAB, vocab_size):
        raise PilotError('pilot model vocabulary must be 32768 or exactly the manifest vocabulary')
    if expected_model_vocab < vocab_size:
        raise PilotError('pilot model vocabulary is smaller than the manifest vocabulary')
    if not _hex64(manifest['tokenizer_sha256']) or not _hex64(manifest['source_snapshot_sha256']):
        raise PilotError('pilot tokenizer/source digest is invalid')
    if not _hex64(manifest['logical_sha256']):
        raise PilotError('pilot logical digest is invalid')
    families = _validate_sources(manifest['sources'])
    if _source_snapshot(manifest['sources']) != manifest['source_snapshot_sha256']:
        raise PilotError('source snapshot digest mismatch')
    if (expected_source_snapshot_sha256 is not None
            and manifest['source_snapshot_sha256'] != expected_source_snapshot_sha256):
        raise PilotError('source snapshot SHA-256 mismatch')
    tokenizer_bytes = _require_regular(root / 'tokenizer.json', 'tokenizer.json',
                                       8 * 1024 * 1024)
    if _sha256(tokenizer_bytes) != manifest['tokenizer_sha256']:
        raise PilotError('tokenizer SHA-256 mismatch')
    shards = manifest['shards']
    if not isinstance(shards, list) or not shards:
        raise PilotError('manifest shards must be a nonempty list')
    by_split = {}
    total_bytes = len(manifest_bytes) + len(tokenizer_bytes)
    total_blocks = 0
    for index, shard in enumerate(shards):
        _require_object(shard, f'shards[{index}]', (
            'split', 'path', 'sha256', 'bytes', 'blocks', 'valid_tokens',
            'ntp_labels', 'offset2_labels'))
        split = shard['split']
        if split not in SPLIT_ORDER or split in by_split:
            raise PilotError('shard split is invalid or duplicated')
        if shard['path'] != f'{split}.jsonl':
            raise PilotError('shard path is not the fixed split path')
        if not _hex64(shard['sha256']):
            raise PilotError('shard SHA-256 is invalid')
        if type(shard['bytes']) is not int or shard['bytes'] < 0:
            raise PilotError('shard byte count is invalid')
        if type(shard['blocks']) is not int or shard['blocks'] < 0:
            raise PilotError('shard block count is invalid')
        total_bytes += shard['bytes']
        total_blocks += shard['blocks']
        by_split[split] = shard
    if total_bytes > MAX_TOTAL_BYTES or total_blocks > MAX_BLOCKS:
        raise PilotError('pilot dataset exceeds aggregate bounds')
    if 'train' not in by_split or 'validation' not in by_split:
        raise PilotError('pilot dataset requires train and validation shards')
    if not MIN_TRAIN_BLOCKS <= by_split['train']['blocks'] <= MAX_TRAIN_BLOCKS:
        raise PilotError('train block count is out of bounds')
    if not MIN_VALIDATION_BLOCKS <= by_split['validation']['blocks'] <= MAX_VALIDATION_BLOCKS:
        raise PilotError('validation block count is out of bounds')
    if 'test' in by_split and by_split['test']['blocks'] > MAX_TEST_BLOCKS:
        raise PilotError('test block count is out of bounds')
    blocks = {}
    shard_bytes = {}
    global_doc_ids = set()
    for split in SPLIT_ORDER:
        if split not in by_split:
            continue
        shard = by_split[split]
        data = _require_regular(root / shard['path'], f'{split}.jsonl',
                                MAX_TOTAL_BYTES)
        if len(data) != shard['bytes'] or _sha256(data) != shard['sha256']:
            raise PilotError(f'{split} shard hash/size mismatch')
        if not data.endswith(b'\n'):
            raise PilotError('shard lacks LF terminator')
        lines = data[:-1].split(b'\n')
        if len(lines) != shard['blocks']:
            raise PilotError('shard block count mismatch')
        selected = []
        counts = {'valid_tokens': 0, 'ntp_labels': 0, 'offset2_labels': 0}
        for line_index, line in enumerate(lines):
            try:
                block = json.loads(line.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PilotError(f'invalid JSON block: {exc}') from exc
            if _canonical(block) != line:
                raise PilotError('block is not canonical JSON')
            item_counts = _validate_block(block, vocab_size, global_doc_ids)
            for key in counts:
                counts[key] += item_counts[key]
            converted = dict(block)
            converted['labels'] = {int(key): list(value)
                                   for key, value in block['labels'].items()}
            selected.append({'split': split, **converted})
        for key in counts:
            if shard[key] != counts[key]:
                raise PilotError(f'shard {key} count mismatch')
        shard_bytes[split] = data
        blocks[split] = selected
    logical = _sha256(b''.join(shard_bytes[split] for split in SPLIT_ORDER
                               if split in shard_bytes))
    if logical != manifest['logical_sha256']:
        raise PilotError('logical SHA-256 mismatch')
    summary = {
        'format': FORMAT,
        'version': VERSION,
        'manifest_sha256': expected_manifest_sha256,
        'tokenizer_sha256': manifest['tokenizer_sha256'],
        'source_snapshot_sha256': manifest['source_snapshot_sha256'],
        'logical_sha256': manifest['logical_sha256'],
        'sequence_length': SEQUENCE_LENGTH,
        'vocab_size': vocab_size,
        'model_vocab_size': expected_model_vocab,
        'unused_token_id_count': expected_model_vocab - vocab_size,
        'unused_token_id_ranges': ([[vocab_size, expected_model_vocab - 1]]
                                   if expected_model_vocab > vocab_size else []),
        'splits': {split: {
            'blocks': by_split[split]['blocks'],
            'bytes': by_split[split]['bytes'],
            'valid_tokens': by_split[split]['valid_tokens'],
            'ntp_labels': by_split[split]['ntp_labels'],
            'offset2_labels': by_split[split]['offset2_labels'],
        } for split in SPLIT_ORDER if split in by_split},
        'source_families': {family: sorted(splits)[0] for family, splits in families.items()},
        'test_validated_only': 'test' in by_split,
    }
    return {'blocks': {'train': blocks['train'], 'validation': blocks['validation']},
            'manifest': summary}
