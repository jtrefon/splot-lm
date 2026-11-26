"""Synthetic data contract oracle. No external corpus or production BPE."""
import hashlib
import json
import unicodedata
from dataclasses import dataclass

PAD, BOS, EOS = 0, 1, 2
BYTE_BASE = 8
IGNORE = -100


@dataclass(frozen=True)
class Document:
    id: str
    group: str
    language: str
    domain: str
    text: str


def encode(text):
    return [BYTE_BASE + b for b in text.encode('utf-8', errors='strict')]


def decode(ids):
    if any(not BYTE_BASE <= i < BYTE_BASE + 256 for i in ids):
        raise ValueError('not a byte token')
    return bytes(i - BYTE_BASE for i in ids).decode('utf-8', errors='strict')


def digest(text):
    # Conservative dedup fingerprint, never replace training text with this.
    normalized = unicodedata.normalize('NFC', text.replace('\r\n', '\n'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def split_documents(documents, seed='splot-fixture-v1', train_pct=80, val_pct=10):
    """Union source groups and exact fingerprints BEFORE dedup and splitting."""
    if not (0 < train_pct < 100 and 0 < val_pct < 100 - train_pct):
        raise ValueError('invalid split percentages')
    docs = sorted(documents, key=lambda d: d.id)
    if len({d.id for d in docs}) != len(docs):
        raise ValueError('duplicate document ID')
    parents = {d.id: d.id for d in docs}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = sorted((root(a), root(b)))
        parents[b] = a

    groups, fingerprints = {}, {}
    for d in docs:
        if not d.id or not d.group or not d.text:
            raise ValueError('empty ID, group or text')
        for index, key in ((groups, d.group), (fingerprints, digest(d.text))):
            if key in index:
                union(d.id, index[key])
            else:
                index[key] = d.id
    result = {'train': [], 'validation': [], 'test': []}
    seen = set()
    for d in docs:
        fp = digest(d.text)
        if fp in seen:
            continue
        seen.add(fp)
        cluster = root(d.id)
        h = hashlib.sha256((seed + '\0' + cluster).encode()).digest()
        bucket = int.from_bytes(h[:8], 'big') % 100
        split = 'train' if bucket < train_pct else 'validation' if bucket < train_pct + val_pct else 'test'
        result[split].append(d)
    return result


def pack(documents, length, offsets=(1, 2)):
    """Reference whole-document packing; rejects overlong docs, never truncates."""
    if length < 3 or not offsets or any(k < 1 for k in offsets):
        raise ValueError('invalid packing config')
    blocks, ids, segments, positions = [], [], [], []

    def flush():
        if not ids:
            return
        pad = length - len(ids)
        x = ids + [PAD] * pad
        seg = segments + [-1] * pad
        pos = positions + [0] * pad
        labels = {k: [x[t+k] if t+k < length and seg[t] >= 0 and seg[t] == seg[t+k]
                      else IGNORE for t in range(length)] for k in offsets}
        blocks.append({'input_ids': x, 'document_ids': seg, 'position_ids': pos,
                       'valid': [s >= 0 for s in seg], 'labels': labels})
        ids.clear(); segments.clear(); positions.clear()

    for n, d in enumerate(documents):
        tokens = [BOS] + encode(d.text) + [EOS]
        if len(tokens) > length:
            raise ValueError('overlong document; explicit segmentation required')
        if len(ids) + len(tokens) > length:
            flush()
        ids.extend(tokens); segments.extend([n] * len(tokens)); positions.extend(range(len(tokens)))
    flush()
    return blocks


def synthetic_documents():
    topics = [
        ('en', 'education', 'A prism bends light.'),
        ('pl', 'education', 'Żółty pryzmat załamuje światło.'),
        ('en', 'code', 'def twice(x):\n    return 2 * x\n'),
        ('pl', 'math', 'Suma liczb 7 i 5 wynosi 12.'),
        ('en', 'math', 'A square has four equal sides.'),
        ('pl', 'education', 'Łódź leży w Polsce.'),
        ('en', 'education', 'Water freezes at zero Celsius under standard pressure.'),
        ('pl', 'code', 'def suma(a, b):\n    return a + b\n'),
    ]
    return [Document(f'synthetic-{i:02}-{j:02}', f'family-{i:02}-{j:02}', lang, domain,
                     text + f' [fixture {j}]')
            for i, (lang, domain, text) in enumerate(topics) for j in range(10)]


def manifest():
    splits = split_documents(synthetic_documents())
    return {name: {'documents': len(docs), 'ids': [d.id for d in docs],
                   'byte_tokens': sum(len(encode(d.text)) for d in docs),
                   'sha256': hashlib.sha256(json.dumps([(d.id, digest(d.text)) for d in docs],
                                                       separators=(',', ':')).encode()).hexdigest()}
            for name, docs in splits.items()}


if __name__ == '__main__':
    print(json.dumps(manifest(), indent=2, sort_keys=True))
