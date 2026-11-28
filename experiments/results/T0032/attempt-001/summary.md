# T0032 attempt-001 — bounded streaming corpus format and scale qualification

Task T0032 (IMP), issue #120, parents #85/#8/#10/#116. Instruction/base `04b5a8b8962b40cc19f4afdd8216d6270f7f7f29`; tested implementation `42ca0df0cf535abcc63b142a51091597bdb512a9`; branch `feat/T0032-pilot-stream-attempt-001`; draft PR #122. UTC 2026-11-28T19:59:24Z to 2026-11-28T20:02:22Z for the implementation window; evidence assembly followed.

**Outcome:** **passed as a standalone streaming-format/auditor/scale handover**. Added `splot-pilot-stream-v1`, a deterministic synthetic producer, an independent auditor, a near-1M synthetic scale run and a source-registry template. No actual acquisition, tokenizer fit, real corpus/test/checkpoint access, training, package install, GPU/MPS, CLI/config/workflow change or shared state-document edit. v1 and v2 remain immutable.

## 1. Preflight

- `origin/main` and local `HEAD`: `04b5a8b8962b40cc19f4afdd8216d6270f7f7f29`.
- **Zero open PRs** verified before branch creation (start record 19:59:24Z).
- Only this draft PR #122 was open at evidence capture; no new task launched.
- Local macOS 27.2.1 arm64 Apple M4, 10 cores / 17,179,869,184 B RAM, CPU only, CPython 3.15.2; torch absent.
- v1 loader, v2 verifier/producer, training CLI, configs and shared state documents unchanged.

## 2. Streaming format

`splot-pilot-stream-v1` uses:

```text
manifest.json
tokenizer-bytes.json
sources.json
documents.jsonl
train-000.jsonl, train-001.jsonl, ...
validation-000.jsonl, ...
test-000.jsonl, ...
sources/<source_id>.raw
sources/<source_id>.content
```

- Canonical manifest, shard records in train/validation/test order, logical concatenation hash, source-registry and document-metadata hashes.
- Separate raw/content source files with immutable per-file hashes; no hex-duplicated full source blobs in metadata.
- Document metadata JSONL with source/family/language/split, source/document/segment byte intervals, full/segment hashes, truncation and token counts.
- Complete 256 distinct single-byte tokenizer alphabet (V=264), fixed special IDs and duplicate-free content token bytes; byte-split Polish diacritics.
- Block JSONL with v1/v2-compatible BOS/content/EOS, positions, valid/control masks, token byte lengths/offsets and offset-1/offset-2 labels.
- Content-only BPB target alignment at label `i+1`; EOS/control separate; `iter_targets`, `verify_targets` and `aggregate_bpb`.
- Bounded streaming reader, train/validation iterators, test integrity without test blocks/targets; forged/duplicate target rejection.

Caps: 128 MiB total files, 1.6M positions, train/validation/test valid ≤1M/50k/50k, total valid ≤1.1M, T≤512, V≤16,384, 32 sources, 20k documents, 128 KiB metadata line, 256 KiB shard line, 32 MiB shard, 4 MiB tokenizer/source file, 2 MiB manifest.

## 3. Synthetic scale qualification

Deterministic local run: 8/4/4 families (4/2/2 EN and PL), T=512, full and partial document segments, four train shards plus one validation and one test shard.

| Metric | Value |
|---|---:|
| Content tokens | 986,880 |
| Train valid tokens | 919,040 |
| Validation valid tokens | 46,240 |
| Test valid tokens | 46,240 |
| Total valid tokens | 1,011,520 |
| Blocks (train/val/test) | 1,868 / 94 / 94 |
| Total positions | 1,052,672 |
| Generated files | 45,637,698 B (34.0% of 128 MiB) |
| Producer write + primary verify | 1.763 s |
| Independent audit | 0.764 s |
| Total process | 2.528 s |
| Peak RSS (macOS `ru_maxrss`, bytes) | 188,006,400 |
| Independent audit matches | true |
| Test blocks audited / exposed | 94 / 0 |
| Manifest | `24c677ce21e3cebfe7869b42233d8ca17f806f84ec872910678861e94e5a3dba` |

The scale content is intentionally synthetic and is **not** a learning corpus. No quality, corpus, BPB or training claim.

## 4. Independent audit

`tools/pilot_stream_audit.py` does not import the primary verifier or producer. It independently re-parses canonical artifacts and reconciles:

- source raw/content hashes, bytes and selected bytes;
- document full/selected hashes, intervals, disjoint groups and UTF-8;
- shard/block labels, masks, byte coverage and per-family/language/split counts;
- manifest/shard/logical/tokenizer/document/source-registry hashes;
- test integrity without exposing test blocks or scoring targets.

The scale audit matched the producer/loader counts exactly; tiny tests cover extra file, shard/source tamper, symlink, incomplete alphabet, count mismatch, test exposure, forged/duplicate targets, negative/overflow NLL and existing-output refusal.

## 5. Producer and registry plan

`tools/pilot_stream_producer.py` consumes explicit `SourceSpec`/`SegmentSpec` records and token IDs, validates caps/provenance/UTF-8/hashes/token-byte equality/segment groups, packs segment-local multi-document blocks with padding tails, enforces total positions/valid/file caps, writes into a new private staging directory, verifies, and atomically publishes; existing output/staging paths are refused. `make_scale_spec()` is the deterministic near-1M plan.

`research/data/source-registry-template.md` provides a machine-readable template and review checklist for 8/4/4 minimum families, immutable URL/revision, rights evidence, extraction policy, dedup/contamination, train-only tokenizer sample, target budgets and separately authorized acquisition/producer/tokenizer/CLI tasks. No source list is approved or inferred from hashes.

## 6. Tests and hosted CI

- `tools/test_pilot_stream.py`: **8/8 passed**.
- Existing v2 verifier **19/19** and v2 producer **6/6** regressions passed locally.
- Hosted run [46184014190](https://github.com/jtrefon/splot-lm/actions/runs/46184014190) on `42ca0df` **succeeded**: stream 8/8, full suite **33 CPU scripts / 40 REP reports**.
- No local torch; hosted CPU is the only test execution evidence.

## 7. First failures

Development before `42ca0df` had three local fixture/format bring-up failures: a missing `CONTENT_MIN` export, a partial-segment fixture using the full source interval, and full-document intervals counted per segment rather than per document group. All were corrected before the tested commit; no real data, protocol value or hosted gate was involved. They are preserved in `execution-first-failures.txt`; the first hosted run passed.

## 8. Compatibility and limitations

- v1/v2 are immutable; stream-v1 is standalone and not accepted training input or connected to the CLI.
- The scale fixture is synthetic; no real corpus, tokenizer fit, rights/authenticity, quality or measured BPB claim.
- The root is a trusted static private directory for one verification call; paths/symlinks are rejected and reads bounded, but no concurrent-mutation/race-proof guarantee is claimed.
- The tokenizer table is an explicit synthetic table, not a fitted production tokenizer; production BPE must separately qualify the complete alphabet/train-only fit.
- No merge, closure, training or next-task launch; no shared roadmap/state edit.

## 9. Handover

Evidence: stream schema reference/hash, registry template reference/hash, implementation hashes/source refs, scale summary and independent audit, tiny test transcript, environment, metrics, execution log and preserved first failures. One INDEX row; PR #122 and issue #120 carry the handover.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0032/attempt-001 \
  --task T0032 --source 42ca0df0cf535abcc63b142a51091597bdb512a9 \
  --checks environment,schema,synthetic_scale,registry_plan,handover
```

Output: STRUCTURALLY VALID: passed (not scientific acceptance). No merge, closure or next-task launch.
