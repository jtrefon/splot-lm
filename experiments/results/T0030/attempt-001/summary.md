# T0030 attempt-001 — v2 coverage hardening, deterministic producer and scale plan

Task T0030 (IMP), issue #116, parents #85/#8/#10/#112. Instruction/base `145db440cf60bcd575226b475952f2d7129f09c1`; tested implementation `976445fdb975365823f2ddb88f604bb68ac492f5`; branch `feat/T0030-v2-coverage-attempt-001`; draft PR #118. UTC 2026-11-28T19:24:03Z to 2026-11-28T19:27:35Z for the implementation window; evidence assembly followed.

**Outcome:** **passed as a standalone hardening/producer/scale-plan handover**. The v2 verifier now enforces orphan/interval/UTF-8/duplicate-token/reconstruction/target-identity cases, the deterministic synthetic producer reproduces the T0028 fixture byte-for-byte, and the 1M-token migration plan is documented. No actual dataset acquisition, tokenizer fit, real corpus/test/checkpoint access, training, package install, GPU/MPS, v1/CLI/config/workflow change or shared state-document edit.

## 1. Preflight

- `origin/main` and local `HEAD`: `145db440cf60bcd575226b475952f2d7129f09c1`.
- **Zero open PRs** verified before branch creation (start record 19:24:03Z).
- By evidence capture, draft PR #118 and separately launched T0031 PR #119 were open; no new task was launched here.
- Local macOS 27.2.1 arm64 Apple M4, 10 cores / 17,179,869,184 B RAM, CPU only, CPython 3.15.2; torch absent.
- v1 loader, training CLI, configs, workflow and shared state documents remain unchanged.

## 2. Coverage hardening

The hardened verifier now rejects:

| Case | Observed rejection |
|---|---|
| Orphan declared document | `orphan or missing document IDs after shard scan` |
| Cross-group full-document overlap | `full-document groups overlap within a source` |
| Invalid full source UTF-8 | `source content is not valid UTF-8` |
| Invalid full document UTF-8 | `full document is not valid UTF-8` |
| Invalid selected-segment UTF-8 | `source segment is not valid UTF-8` |
| Duplicate content token bytes | `duplicate content token bytes for IDs 12 and 13` |
| Reconstruction beyond selected bytes | `token reconstruction exceeds declared selected bytes` |
| Segment indices out of byte order | `segment_index values are not ordered by byte start` |
| Partial selection not marked truncated | `partial document selection must be explicitly truncated` |
| Float padding label | `padding labels must be integer IGNORE` |
| Negative NLL | `targets[0].nll must be a finite nonnegative number` |
| Overflow NLL | `targets[0].nll must be a finite nonnegative number` |
| Forged target byte identity | `targets[0].byte_length does not match the dataset` |
| Duplicate target identity | `duplicate target identity: (0, 0)` |
| Test target access | `test targets are not exposed by this API` |

15/15 semantic cases rejected with recomputed hashes/counts. The original valid T0028 fixture remains unchanged:

- manifest `90931377e0e6ee0b20d82e022ce6062bc38658fbbce01e2518fac2a164d19630`;
- tokenizer `09b920696db8709fcb7d1b183147d28f6fa1b0c50d5a0080cdc12a45589739fd`;
- source bytes `c58df95abefda42ac678c99670719097e63abd35beaf15de288aeae1c8bb9535`;
- source snapshot `852ed2d62a2b675f4f34aac4dbc830e0219dabb230907d3891a81731619fd9e4`;
- logical `979e6f907c33a40a12247f19154f3cc6b2d553769c7af1bac724d1bb36883549`;
- byte coverage `f7e94a862a00408713020c814aa352d7b4c7dae5174918b452c5eb3b0a1e52e9`;
- 3 blocks, 36 valid / 20 content / 28 NTP / 20 content labels / 8 EOS / 20 offset-2 / 21 content-target bytes / 21 selected bytes;
- validation BPB `2.3 / (ln(2) × 7) = 0.4740283705778022`.

New APIs: `verify_targets(dataset, split, targets)` and `aggregate_verified_bpb(dataset, split, nlls, targets=None)` verify target identity/order, reject duplicates and return per-family/per-language totals. `aggregate_content_bpb` remains clearly arithmetic-only and now rejects negative/overflow NLLs too. `TOKENIZER_SCOPE = "synthetic-explicit-table"` and the schema document distinguish this table from a production tokenizer requiring a complete 256-byte alphabet and train-only fit.

## 3. Deterministic producer

`tools/pilot_v2_producer.py` consumes explicit `SourceSpec`/`SegmentSpec` records and token IDs, validates caps/provenance/UTF-8/hashes/token-byte equality/segment groups, packs multi-document segment-local blocks with padding tails, refuses an existing output, and writes canonical six-file output only after an in-memory build. `expected_coverage(spec)` computes counts/hashes before the verifier.

Producer evidence:

| Check | Result |
|---|---|
| Same inputs byte-identical | true |
| Expected coverage = loader counts | true |
| Matches T0028 fixture files | true |
| Existing output refused | true |
| Producer manifest | `90931377e0e6ee0b20d82e022ce6062bc38658fbbce01e2518fac2a164d19630` |
| Totals | 3 blocks, 36 valid / 20 content / 28 NTP / 8 EOS / 21 selected bytes |
| Sources / segments | 7 / 8 |

The producer is synthetic-only and does not acquire sources, fit a tokenizer or integrate with training.

## 4. Scale migration plan

`research/data/pilot-v2-scale-plan.md` proposes a bounded first target of **1,000,000 unique valid training tokens** plus 50,000 validation and 50,000 test tokens. The hypothetical ranges are explicitly planning assumptions, **not** T0016 extrapolations:

- 3.5–6.0 raw UTF-8 bytes/token; 2.5–4.5 selected UTF-8 bytes/token;
- 12–20 compact binary bytes/token; 35–60 canonical JSONL bytes/token.

The plan covers streaming source artifacts instead of hex-duplicated blobs, compact binary and JSONL shard strategies, source/rights registry review, complete 256-byte tokenizer qualification and train-only deterministic fit, dedup/cross-split leakage, independent hashes/coverage audit, per-split/family budgets, test non-exposure, private storage/pins, and separately reviewed acquisition/producer/CLI tasks. No acquisition, cap enlargement or production loader is implemented.

## 5. Tests and hosted CI

- `tools/test_pilot_v2.py`: **19/19 passed**.
- `tools/test_pilot_v2_producer.py`: **6/6 passed**.
- Local py_compile and diff checks passed; no local torch.
- Hosted run [46181008581](https://github.com/jtrefon/splot-lm/actions/runs/46181008581) on `976445f` **succeeded**: v2 19/19, producer 6/6, full suite **32 CPU scripts / 38 REP reports**.

Development failures before `976445f` were test-fixture construction errors while adding the new negative cases; they are preserved in `execution-first-failures.txt`. No hosted gate failed and no verifier behavior was relaxed.

## 6. Compatibility and limitations

- v1 remains canonical for training; v2 is a standalone synthetic prototype and not accepted training input.
- Schema fields/limits are unchanged; validation is tighter. The valid T0028 fixture hashes/counts/BPB are unchanged; previously accepted malformed cases now fail closed.
- `source-bytes.json` and the tokenizer table are synthetic; no real source authenticity, licensing, tokenizer fit or BPB claim.
- The root is a trusted static root for one call; paths/symlinks are rejected and reads bounded, but no race-proof guarantee is claimed.
- No merge, closure, training or next-task launch; no shared roadmap/state edit.

## 7. Handover

Evidence: coverage-hardening JSON with actual rejection reasons, producer summary/hashes, scale-plan reference, implementation file hashes, source references, environment, metrics, test transcripts and execution log. One INDEX row; PR #118 and issue #116 carry the handover.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0030/attempt-001 \
  --task T0030 --source 976445fdb975365823f2ddb88f604bb68ac492f5 \
  --checks environment,coverage_hardening,producer,scale_plan,handover
```

Output: STRUCTURALLY VALID: passed (not scientific acceptance). No merge, closure or next-task launch.
