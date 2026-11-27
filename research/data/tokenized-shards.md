# Portable tokenized shards and document-safe packing
2026-11-27 | Synthetic reference only, #60 / #67 / #8 / #10 | T0010 / T0013

**Status:** deterministic, dependency-free bridge from the accepted synthetic
corpus oracle and byte BPE artifact to portable tokenized shard files. It is not
a production-scale sharding/streaming pipeline, production tokenizer, source
rights decision or data acquisition tool. Only original synthetic text is used;
no external corpus or network storage is involved.

**T0013 boundary update:** the manifest format is now
`splot-tokenized-shards-v2` / version 2. The source identity includes the complete
cluster-split mapping, cluster representatives and quarantine identity, and the
producer requires an explicit approved synthetic source/revision mapping. T0010
evidence remains historical and unchanged; the v2 demo/manifest digests are
expected to differ because the hardened schema carries more identity.

## 1. Source bridge and train-only tokenizer

The demo starts from `corpus_reference.demo_report()`: 20 original synthetic
EN/PL prose records with explicit synthetic source approvals, duplicate
families, benchmark quarantine, an excluded cluster and deterministic
cluster-level 50/25/25 splitting. The report digest is
`b80708762570ce1ba31eb5ed34e0c025dedf7e3728dcf9b53974490e7a942e0f`.

Only the retained eligible cluster representatives are encoded. The BPE
artifact is trained on the non-quarantined `train` representatives only; no
validation, test or quarantined document contributes pair counts. The
train-input identity and BPE training digest are recorded in the manifest.
Encoding is performed once per representative. Every content token must be in
the byte/merge range (8..actual_vocab_size-1), no raw control ID (0–7) may be
emitted, and strict UTF-8 decode must reproduce the original text exactly.

The producer does not trust `audit.ok` alone. It recomputes the corpus report
canonical digest, requires unique input/cluster/member/representative identities,
exactly one representative per cluster, explicit approved source/revision
mapping, valid quarantine reasons and no quarantine/member overlap. It verifies
each selected representative's original text hash and the train-only BPE input
digest before accepting the artifact. The accepted corpus report identity,
cluster splits (including `null` for quarantined clusters), cluster
representatives, quarantine member IDs and BPE training digest are recorded in
the manifest source identity.

## 2. Packing semantics

Whole documents are packed; a document is never split or silently truncated.
Each document contributes exactly:

```text
[BOS=1] + encoded content tokens + [EOS=2]
```

`PAD=0` and `IGNORE=-100` are packer/control values. A document whose
`[BOS] + content + [EOS]` does not fit the configured block length is rejected
with an explicit error; the demo uses a suitable length. Documents are sorted
by stable `doc_id` within each split. Blocks are flushed when the next document
would overflow or when the language/domain changes, so each block is
homogeneous by split/language/domain and pad counts remain attributable.

Each block contains:

| Field | Meaning |
|---|---|
| `split` | train, validation or test |
| `block_index` | zero-based block index within the split |
| `language`, `domain` | block grouping |
| `input_ids` | tokens, right-padded with PAD |
| `document_ids` | stable numeric document ID, `-1` for padding |
| `position_ids` | document-local positions reset at each document, 0 for padding |
| `valid` | true for real tokens, false for padding |
| `labels` | `{"1": [...], "2": [...]}` offset labels |

`labels[k][t]` is `input_ids[t+k]` only when `t+k` is inside the same valid
document and the mask permits it; otherwise it is `IGNORE`. No label crosses a
document boundary or predicts padding. This matches the `tools/data_fixture.py`
offset-1/offset-2 semantics; tests compare a no-merges artifact against that
fixture.

Block lengths in this reference are exactly 96, 128 or 256. Content-token
counts exclude BOS/EOS; special tokens are two per document; valid tokens include
BOS/EOS; pad and NTP/offset-2 labels are derived from the blocks. Counts are
reported overall, by split, by language, by domain and by
split/language/domain.

## 3. Dataset layout

A materialized dataset is a new directory containing exactly:

```text
manifest.json
tokenizer.json
shards/<split>-<index>.jsonl
```

`tokenizer.json` is the canonical compact JSON artifact bytes (no trailing
newline). `manifest.json` is canonical compact JSON (no trailing newline).
Each shard is canonical JSONL: one canonical JSON block per line, compact sorted
keys, ASCII escapes, UTF-8 and exactly one LF terminator per line. There are no
blank lines, timestamps or absolute paths in any digest.

## 4. Manifest and digests

The strict manifest schema records:

- format/version (`splot-tokenized-shards-v2`, version 2);
- pack config: length, offsets `[1,2]`, max blocks per shard, total block bound,
  PAD/BOS/EOS/IGNORE IDs;
- tokenizer artifact SHA-256, requested/actual vocabulary size, algorithm and
  schema version;
- source identity: corpus report digest, seed, split percentages, approved
  synthetic source/revision pairs, train document IDs, BPE training digest,
  complete cluster split mapping (split string or `null` for quarantine),
  cluster representative mapping, document split mapping and quarantine
  metadata;
- ordered document records: stable doc/cluster IDs, split, language, domain,
  source/revision, exact content SHA-256, numeric ID and encoded content tokens;
- ordered shard records: relative path, split, exact byte count, SHA-256, block
  count and first/last block index;
- counts and block/shard totals.

Three distinct digests are used:

| Digest | Covers |
|---|---|
| `logical_digest` | canonical ordered document/token records and identity; independent of shard boundaries |
| `packed_logical_digest` | canonical ordered packed blocks; independent of shard partition |
| `physical_manifest_digest` | canonical manifest excluding this field, including shard paths/bytes/hashes; changes when the physical layout changes |

`manifest_sha256` is the SHA-256 of the exact canonical manifest file bytes and
is required by the loader. The manifest's own physical digest is separately
recomputed.

## 5. Verified loader

`load_dataset(root, expected_manifest_sha256)` refuses symlinks, unsafe or
absolute paths and nonregular files. It verifies the complete file set
(manifest, tokenizer, every declared shard, no extras/missing files), per-file
and total size limits before returning blocks, canonical UTF-8/JSON bytes,
duplicate JSON keys, nonfinite numbers, manifest schema/types, tokenizer
hash/reload, document identity, contiguous numeric IDs, exact cluster/document
split consistency, one representative per cluster, source approvals,
quarantine cluster and member exclusion, shard ordering/ranges/hashes, block
schema, token ranges, BOS/EOS boundaries, position resets, valid/pad
consistency, label offsets and cross-document leakage.

It decodes each document's token content with the frozen artifact, rechecks the
content SHA-256 and recomputes the train-only `train_input_digest`; the result
must match both the artifact training digest and the source identity metadata.
It independently recomputes the logical and packed digests, recomputes counts
from retained records and blocks, and repacks the manifest documents to compare
the resulting canonical blocks with the loaded blocks. Caller-supplied counts
are never trusted. All checks must pass before blocks are returned.

The loader provides integrity checks for a trusted local single-writer dataset.
`expected_manifest_sha256` must be independently pinned/reviewed; the manifest's
self-declared source approvals do not establish real-world rights. Atomic rename
is not a multi-writer locking protocol and there is no concurrent adversarial
filesystem guarantee.

## 6. Atomic materialization

`materialize_dataset(...)` builds and verifies the complete dataset in a
temporary sibling directory and atomically renames it to a new nonexistent
destination. It refuses overwrite, never rewrites an existing dataset, and
removes only its own temporary staging path on failure. There are no background
services, network operations or cleanup of unrelated paths.

## 7. API

From `PYTHONPATH=tools`, `import tokenized_shards`:

```python
build_token_documents(report, approved_sources, *, requested_vocab_size=320, max_merges=56) -> dict
build_demo_bundle() -> dict
pack_documents(documents, artifact, config) -> list[dict]
compute_counts(documents, blocks, config) -> dict
build_manifest(documents, blocks, artifact, config, source_identity, shards) -> dict
manifest_bytes(manifest) -> bytes
manifest_sha256(manifest) -> str
materialize_dataset(root, documents, artifact, config, source_identity) -> dict
load_dataset(root, expected_manifest_sha256) -> dict
demo_report() -> dict
main(["--demo"]) -> int
```

`TokenDocument` carries the stable doc/cluster identity, source revision,
content digest, numeric ID and encoded content tokens. `PackConfig` fixes the
length, offsets and per-shard block bound.

## 8. Demo

`python3 tools/tokenized_shards.py --demo` builds the synthetic corpus/BPE
bridge, materializes the same dataset twice in temporary directories, compares
all bytes, then materializes a different shard partition. The report includes
the logical, packed and physical digests, manifest SHA-256, counts, tokenizer
artifact hash, train document IDs and a full manifest example. It verifies:

- two independent materializations are byte-identical;
- different shard partitions preserve logical and packed digests but change the
  physical manifest digest;
- the verified loader returns the expected blocks.

No real data, quality, fertility or production-tokenizer claim is made.

## 9. Limits and limitations

- At most 1000 documents, 4096 UTF-8 bytes/document and 256 KiB total train
  text. Tokenizer training uses at most 320 target vocabulary / 56 merges.
- Pack lengths are 96/128/256; at most 8 blocks/shard, 1024 blocks total,
  1 MiB/shard and 16 MiB total dataset bytes. Manifest ≤1 MiB; tokenizer
  artifact ≤64 KiB; at most 2048 files.
- No production sharding, streaming, memory mapping, compression, network
  storage, concurrent writers or distributed filesystems.
- No source acquisition, rights clearance, real corpus, model training, MPS,
  CUDA or additional tasks.
- Do not claim cross-platform artifact/shard equality until a separate Linux
  comparison verifies the canonical demo digests.
- Production tokenizer/source provenance and materialization remain separate
  reviewed gates.

## 10. Verification

From the repository root:

```bash
PYTHONPATH=tools python3 tools/test_corpus_reference.py
PYTHONPATH=tools python3 tools/test_bpe_reference.py
PYTHONPATH=tools python3 tools/test_data_fixture.py
PYTHONPATH=tools python3 tools/test_tokenized_shards.py
PYTHONPATH=tools python3 tools/tokenized_shards.py --demo
```

Run the demo twice and compare bytes. REP evidence under
`experiments/results/T0013/attempt-001/` records the tested implementation SHA,
commands, exit codes, test counts, demo hash, logical/physical digests and the
manifest example. T0010 evidence under `experiments/results/T0010/attempt-001/`
remains historical and is not rewritten.
