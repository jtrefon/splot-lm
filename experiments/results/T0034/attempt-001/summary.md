# T0034 attempt-001 — bounded streaming completion and candidate registry

Task T0034 (IMP), issue #124, parents #85/#8/#10/#120. Instruction/base `4f86f62d3fb673483ddf87001331db90e354b985` (spec ancestor `04b5a8b8962b40cc19f4afdd8216d6270f7f7f29`); tested implementation `44df94e2e97f06d4a752097b1129ef70598c2e7c`; branch `feat/T0034-stream-completion-attempt-001`; draft PR #128. Local evidence window 2026-11-28T20:42:20Z onward.

**Outcome:** **passed locally as a bounded-I/O and registry-candidate handover**. Completed the stream-v1 reader/producer/auditor bounded-I/O gaps, preserved the T0032 synthetic fixture identity, and proposed a concrete 12/6/6 bilingual prose candidate registry with a 16-family 8/4/4 minimum subset. No actual corpus acquisition, tokenizer fit, training, real test/checkpoint access, dependency installation, GPU/MPS, CLI/v1/v2 change or shared project-state edit.

## 1. Preflight and queue

- `origin/main` and local `HEAD`: `4f86f62d3fb673483ddf87001331db90e354b985`.
- **Zero open PRs** verified before branch creation.
- Concurrent PR #127 (T0035, created 2026-11-28T20:40:02Z) appeared during execution; it is not part of T0034 and was not modified. Draft PR #128 carries this attempt.
- Local macOS 27.2.1 arm64 Apple M4, 10 cores / 17,179,869,184 B RAM, CPU only, CPython 3.15.2; torch absent.
- v1/v2, training CLI, configs, workflows and shared project-state documents unchanged.

## 2. Bounded I/O completion

- **Preflight:** canonical manifest/tokenizer/source metadata, strict nonbool lengths, exact file set, root/source parent paths and per-file/aggregate caps are checked before source reads. `_check_file_set` also compares declared source/shard byte counts before reading.
- **Reader:** source raw/content hashes and UTF-8 are incremental (64 KiB chunks); the loader retains a file index and document metadata instead of a full `content_map`. Full-document and selected-segment slices are hashed/UTF-8-checked incrementally with a verified-full cache.
- **Producer:** token reconstruction is bounded to the declared segment before join; logical SHA is streamed while writing; document metadata is written line by line; no shard `read_bytes`, payload array or all-document byte join; per-line, per-shard and aggregate caps are enforced during writes. `verify=False` is removed; verification before `os.replace` is mandatory.
- **Staging:** `tempfile.mkdtemp` creates an exclusive private staging directory; cleanup removes only that directory; pre-existing output/staging paths are preserved and existing output is refused.
- **Auditor:** independent stdlib implementation (no primary/producer imports) with preflight caps, incremental source/document/shard/logical hashes, incremental token reconstruction, and test integrity without test blocks.
- **Targets:** strict target record types, manifest-bound identity, streamed lockstep `verify_targets`/`verify_and_aggregate_bpb`, one-pass `aggregate_bpb`; train/validation only. `StreamIdentity` is a frozen verified identity.
- No concurrent-mutation guarantee is claimed; the root is a trusted static private directory for one verification call.

## 3. Scale qualification

Same deterministic T0032 fixture and manifest `24c677ce21e3cebfe7869b42233d8ca17f806f84ec872910678861e94e5a3dba`; manifest match: **True**.

| Metric | Value |
|---|---:|
| Content tokens | 986,880 |
| Train / validation / test valid tokens | 919,040 / 46,240 / 46,240 |
| Total valid tokens | 1,011,520 |
| Blocks / positions | 2,056 / 1,052,672 |
| Generated files | 45,637,698 B (34.0% of 128 MiB) |
| Spec / producer write / loader verify | 0.055 / 2.124 / 1.523 s |
| Target verify / verify+BPB / audit | 3.105 / 3.103 / 1.494 s |
| Total measured | 11.405 s |
| Peak RSS (macOS `ru_maxrss`, bytes) | 143,998,976 |
| Max source / shard file bytes | 112,084 / 8,590,522 |
| Chunk bound | 65,536 B |
| Full source/shard `read_bytes` calls | 0 / 0 |
| Independent audit matches | True |
| Test blocks audited / exposed | 94 / False |
| Verified validation targets | 45,680 (content 45,120, EOS 560, control 0) |

The constant-NLL synthetic BPB is `0.721347520444482` from NLL 0.5 per target; it is arithmetic evidence only, not a model or corpus-quality claim. The synthetic content is not a learning corpus.

## 4. Independent audit

`tools/pilot_stream_audit.py` does not import the primary verifier or producer. It re-parses canonical artifacts, validates the exact file set and caps before source reads, hashes sources/documents/shards/logical bytes incrementally, hashes token reconstruction without a payload join, and reconciles source/document/selected bytes, labels, family/language/split counts, all digests and test integrity. The scale audit matched the producer/loader counts exactly.

## 5. Candidate registry

`research/data/source-registry-candidates.json` and `.md` propose **24 families** (preferred 12/6/6: 6 EN + 6 PL train, 3/3 validation, 3/3 test), with a **16-family 8/4/4 minimum subset**. Existing T0016 evidence supplies 8 families; 16 new metadata-verified candidates come from official Standard Ebooks and Wolne Lektury pages. Candidate token/byte ranges are estimates only. No corpus text was downloaded.

- Rights: Standard Ebooks CC0 production/underlying public domain; Wolne Lektury public-domain text with separately licensed annotations/motifs excluded. US/EU/Polish territory caveats; not legal advice.
- Freeze: Standard Ebooks pinned item-history git commit; Wolne Lektury frozen official XML volume with local SHA-256.
- Dedup: exact/normalized/near-duplicate and 13-gram cross-split audit; original-language works only; one edition per work.
- Unresolved: coordinator freeze, Prus train/test author overlap (Lalka/Faraon), per-family segment allocation, actual rights review and storage budget.
- Acquisition, tokenizer fit, producer/loader adapter and CLI/training adapter remain separate reviewed tasks.

## 6. Tests and hosted CI

- `tools/test_pilot_stream.py`: **11/11 passed** locally.
- Existing v2 verifier **19/19** and v2 producer **6/6** passed locally (25 tests).
- Hosted CPU run [46187833076](https://github.com/jtrefon/splot-lm/actions/runs/46187833076) on `44df94e2e97f06d4a752097b1129ef70598c2e7c`: **success**
- Full local `tools/ci_cpu.py` is not runnable on this Mac because torch is absent; hosted CPU is the complete CPU gate. Local stdlib suites and the scale run are the local evidence.

## 7. Limitations and disclosures

- T0034 is a bounded synthetic prototype and registry proposal, not an accepted training input or corpus acquisition.
- The producer still holds the explicit `StreamSpec` plan and input source bytes in memory; it no longer retains shard payloads or joins all documents.
- The root is a trusted static private directory for one verification call; no concurrent-mutation/race-proof guarantee is claimed.
- A concurrent T0035 PR #127 appeared after the zero-PR preflight; it is unrelated and untouched.
- No merge, closure, training or next-task launch; no shared roadmap/state edit.

## 8. Handover

Evidence: source diff, implementation/source/schema/registry hashes, scale summary and independent audit, bounded-I/O checks, negative-case matrix, tiny test transcript, environment, metrics and execution log.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0034/attempt-001 \
  --task T0034 --source 44df94e2e97f06d4a752097b1129ef70598c2e7c \
  --checks environment,bounded_io,scale,registry,handover
```

Output: STRUCTURALLY VALID: passed (not scientific acceptance). No merge, closure or next-task launch.
