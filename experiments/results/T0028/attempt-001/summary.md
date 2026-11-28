# T0028 attempt-001 — standalone byte-auditable v2 corpus schema and verifier

Task T0028 (IMP), issue #112, parents #85/#8/#10/#108. Instruction/base `edd2036e284a4f7d1d5904b5284971d33c5c0d39`; tested implementation `4681d75668d7d253e9d82fb9f85041774e9a1f04`; branch `feat/T0028-pilot-v2-attempt-001`. UTC 2026-11-28T18:29:09Z to 2026-11-28T18:32:40Z for the main implementation window; evidence assembly followed.

**Outcome:** **passed as a standalone implementation/verifier handover**. New synthetic-only `splot-pilot-blocks-v2` schema, bounded stdlib verifier, exact token-byte/BPB helper and 12-case tamper suite. No actual dataset acquisition, tokenizer fit, real corpus/test/checkpoint access, training, package install, GPU/MPS, v1/CLI/config/workflow change or shared state-document edit. v2 is **not** accepted training input.

## 1. Preflight

- `origin/main` and local `HEAD`: `edd2036e284a4f7d1d5904b5284971d33c5c0d39`.
- **Zero open PRs** verified before branch creation (start record 18:29:09Z).
- By evidence capture, this task's draft PR #114 and a separately launched T0029 PR #115 were open; no new task was launched here.
- Local macOS 27.2.1 arm64 Apple M4, 10 cores / 17,179,869,184 B RAM, CPU only, CPython 3.15.2; torch absent and not installed.
- v1 loader, training CLI, configs, workflow and shared state documents remain unchanged.

## 2. Delivered files

| File | Purpose |
|---|---|
| `tools/pilot_v2.py` | v2 schema constants, canonical JSON/hash checks, bounded streaming verifier, token-byte/UTF-8/coverage checks, `scoring_targets`, `aggregate_content_bpb` |
| `tools/test_pilot_v2.py` | 12 stdlib tests with deterministic synthetic EN/PL fixtures, positive metrics, hand-computed BPB, test non-exposure and tamper cases |
| `research/data/pilot-v2-schema.md` | canonical schema, boundary policy, hashes/caps, verifier order, migration prerequisites |
| `tools/README.md` | one new row marking v2 as a standalone synthetic prototype, not training input |

Implementation identities:

| File | SHA-256 | Bytes |
|---|---|---:|
| `tools/pilot_v2.py` | `2e2e06754422212867806bfdf5863cf8bd61d5e9a5863b96ea812f796111af53` | 44,803 |
| `tools/test_pilot_v2.py` | `70728661df9807746f1c608f04e2c46cba79955f41bf781cfb53b2646130e494` | 24,770 |
| `research/data/pilot-v2-schema.md` | `d458b5c3bd445a50ace55dc88185ed79fb67dfca91eb7e5d8e4d4aefac84e8ba` | 16,296 |
| `tools/README.md` | `92a7725e332c132c35834d329c21eebf21f3651786bc233a32336ebe22d58520` | 2,057 |

## 3. Schema and verifier

- Canonical manifest/shards with exactly six dataset files; unsafe paths, extra files and symlinks fail closed.
- Source/family/language/split registry; full-source raw/content hashes and `selected_bytes`; synthetic `source-bytes.json` verifies full and selected intervals.
- Synthetic tokenizer table with special IDs separate; arbitrary token bytes are concatenated and decoded only at segment level.
- Per-token byte lengths/offsets; BOS/EOS/control/padding/positions/labels follow v1 semantics with no boundary crossing.
- Full-document and segment hashes, truncated multi-segment policy, gaps/overlaps checks and exact selected/source byte reconciliation.
- Bounded streaming reads with explicit v2 caps: 4 MiB shards, 128 blocks (96/16/16), 16,384 vocab, 512 sequence, 1 MiB manifest, 2 MiB tokenizer, 4 MiB source bytes, 128 KiB line.
- Test shards are integrity-validated but never returned; `scoring_targets` refuses `test`.
- `aggregate_content_bpb` uses only content target bytes at label `i+1`, reports EOS/control separately, includes EOS in ordinary NTP, and rejects nonfinite NLL, negative/zero content bytes, mismatched masks and zero content denominators.

## 4. Synthetic fixture metrics

Fixture: `T=32`, `V=32`, 7 sources/families, 8 document segments, 3 blocks, all three splits, EN/PL, multibyte diacritics, UTF-8 codepoint splitting, padding and a two-segment truncated document.

| Split | Blocks | Valid | Content | NTP | Content labels | EOS | Offset-2 | Content bytes | Selected bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1 | 17 | 9 | 13 | 9 | 4 | 9 | 9 | 9 |
| validation | 1 | 10 | 6 | 8 | 6 | 2 | 6 | 7 | 7 |
| test | 1 | 9 | 5 | 7 | 5 | 2 | 5 | 5 | 5 |
| **total** | **3** | **36** | **20** | **28** | **20** | **8** | **20** | **21** | **21** |

By language: EN 13 content target bytes / 13 labels; PL 8 / 7. Content labels equal content tokens and content target bytes equal selected bytes under the v1-style label policy.

Fixture hashes:

| Identity | SHA-256 |
|---|---|
| Manifest | `90931377e0e6ee0b20d82e022ce6062bc38658fbbce01e2518fac2a164d19630` |
| Tokenizer artifact | `09b920696db8709fcb7d1b183147d28f6fa1b0c50d5a0080cdc12a45589739fd` |
| Source bytes artifact | `c58df95abefda42ac678c99670719097e63abd35beaf15de288aeae1c8bb9535` |
| Source snapshot | `852ed2d62a2b675f4f34aac4dbc830e0219dabb230907d3891a81731619fd9e4` |
| Logical shard concatenation | `979e6f907c33a40a12247f19154f3cc6b2d553769c7af1bac724d1bb36883549` |
| Byte-coverage records | `f7e94a862a00408713020c814aa352d7b4c7dae5174918b452c5eb3b0a1e52e9` |

Hand-computed validation BPB: synthetic content NLL **2.3** over exactly **7** bytes; `2.3 / (ln(2) × 7) = 0.4740283705778022` bits/byte. EOS NLL **1.3** over 2 labels is reported separately; ordinary NTP is **3.6** over 8 labels.

## 5. Tests and hosted CI

Local stdlib suite: **12/12 passed** in 0.059 s. Coverage includes positive summary/counts, hand-computed BPB, deterministic repeat, test-split refusal, caps, block/label/padding/position/offset tamper, unknown source/split/duplicate ID, segment gap/overlap/full-document/segment hash, source raw/content/full-document hash, tokenizer/special IDs, invalid UTF-8 concatenation, noncanonical manifest, manifest digests, extra file and symlink.

Hosted CPU run [46176140359](https://github.com/jtrefon/splot-lm/actions/runs/46176140359) on `4681d75` **succeeded**: `tools/test_pilot_v2.py` **12/12**, full suite **31 CPU scripts / 36 REP reports**. No local torch; v1 regressions were exercised by the hosted suite.

## 6. First failures

Development before the tested commit had two local test-harness failures: missing `MAX_SEQUENCE`/`MAX_VOCAB` aliases plus a single-split summarizer `KeyError`, and a duplicate-doc-ID tamper fixture that renumbered a referenced document. Both were corrected before `4681d75`; no schema value, acceptance gate or real data changed. They are preserved in `execution-first-failures.txt`. No hosted run failed.

## 7. Compatibility and migration prerequisites

- v1 remains canonical for training: `splot-pilot-blocks-v1`, `tools/pilot_ingestion.py`, `tools/experiment.py`, configs and workflow are unchanged.
- v2 is a standalone synthetic prototype, not accepted training input and not wired into the CLI.
- A future producer/loader migration requires a separate reviewed IMP task with explicit owner authorization, real rights/revision registry, local/private tokenizer table, actual source hashes (not published restricted text), family-level dedup/split, and an independent byte-coverage audit.
- Larger-source registry prerequisites: selected-vs-full byte accounting, test freeze, frozen tokenizer, and a reviewed cap increase beyond the synthetic fixture.

## 8. Limitations and non-claims

- Synthetic only: no actual corpus, tokenizer fit, source authenticity, legal clearance, training or evaluation.
- `source-bytes.json` is synthetic; hash/coverage mechanics are tested, not real-source correctness or licensing.
- No real BPB, quality, cost, scaling, architecture or production-throughput claim.
- The fixture is deliberately tiny; v2 caps and streaming behavior are prototype bounds, not production limits.
- No merge, closure, training or next-task launch. No shared roadmap/state edit.

## 9. Handover

Evidence: schema reference/hash, implementation file hashes, fixture summary with counts/coverage/BPB, local test transcript, environment, metrics, execution log and preserved development failures. One INDEX row; PR #114 and issue #112 carry the handover.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0028/attempt-001 \
  --task T0028 --source 4681d75668d7d253e9d82fb9f85041774e9a1f04 \
  --checks environment,schema,verifier,handover
```

Output: STRUCTURALLY VALID: passed (not scientific acceptance). No merge, closure or next-task launch.
