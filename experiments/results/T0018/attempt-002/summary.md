# T0018 attempt-002 — model-call accounting and full decode timing
2026-11-28 | IMP | Issue #81 | Parents #2/#10/#6 | PR #91

**Outcome:** passed locally and on hosted CPU for the tested implementation. Final evidence-head CI is recorded on PR #91. No performance or quality claim.
**Instructions:** `d3752eeb77541866e2332ed238def0c4bac50549`.
**Minimum reviewed base:** `a93141e3160c0e9bb845d5b0a64df69c0e97157b` (ancestor).
**Tested implementation:** `42b8e6ca07ca5d3be1de1241a92f30a8d1b1cf01`.
**Prior evidence:** attempt-001 at `339969cf4964adf20e52e3e329cd2a7f170b61a5`, tested `d32c52780a0e6d565ae20422fc83061596a12110`, review-rejected and preserved unchanged. Attempt-001 is not overwritten or relabelled as accepted.
**Scope:** correct the two reviewed T0018 blockers within the original task: actual model-call accounting and decode timing. No model, topology, training, FP8, data or CI change; no package install.

## 1. Review blockers and corrections

The review requested revision at exact head `339969c`:

1. **Call accounting:** the reducer counted phase occurrences as model calls. `shallow_draft` is one boundary but D1 performs one `shallow.advance` per actual scratch token (0 with EOS/short budget, 1 usually, >1 with `draft_length>2`), while D2 performs zero. Correction: every `phase_end` now carries explicit `model_calls`; the reducer reports `model_calls_by_phase` and `phase_occurrences_by_phase` separately, and role totals use actual calls. Validation enforces `shallow_draft.model_calls == actual_tokens` and one call for every other phase.
2. **Decode timing:** decode previously started at the first commit, excluding the first generation round. Correction: `decode_seconds = run_end − prefill_end`, including the complete first round and all committed work. TTFT remains start-to-first-commit. `post_first_commit_seconds`, `first_round_committed_tokens` and `tokens_after_first_commit` are separate diagnostics. `committed_tokens_per_decode_second` uses all committed tokens over `decode_seconds`; no throughput is computed over post-first-commit time.

The trace schema is now `splot.mtp.trace.v2` because v2 adds the explicit `model_calls` field. Attempt-001 v1 evidence remains historical.

## 2. Tests

| Run | Tests | Duration | Result |
|---|---:|---:|---|
| Local stdlib `tools/test_speculation_trace.py` | 28 | 0.052514 s | passed |
| Hosted `tests/test_speculative_observer.py` | 16 | 0.279 s | passed |
| Hosted `tools/test_speculation_trace.py` | 28 | 0.010 s | passed |

Hosted CPU on tested `42b8e6c`: [run 46154938101](https://github.com/jtrefon/splot-lm/actions/runs/46154938101), **success**; Ubuntu 24.04, CPython 3.12.7, torch 2.10.0+cpu, numpy 1.26.4, CUDA hidden. Full suite: 29 scripts and 22 REP reports structurally valid.

Local Mac: CPython 3.15.2, **no torch** and none installed per task. Local checks cover compile, 28 stdlib tests, nine fixture cases and the unchanged attempt-001 tree; the 16 torch integration tests are hosted-only and no local integration pass is claimed.

New integration coverage counts actual runner invocations with a delegating wrapper and compares them to the reducer for both roles. It covers D1 0/1/2 shallow-draft calls, D2 zero draft calls, EOS, short budget, multiple rounds, D0 and actual-model cache/RNG identity.

## 3. Fixtures and timing gates

Nine deterministic fake-clock fixtures; timings are injected 1 ms event increments, not measurements.

| Case | Rounds | Proposed | Accepted | Rejected | Unexamined | Committed | Target calls | Shallow calls | Decode (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 all accepted + bonus | 1 | 2 | 2 | 0 | 0 | 3 | 3 | 3 | 0.012 |
| D1 first rejection + suffix | 2 | 3 | 1 | 1 | 1 | 2 | 4 | 4 | 0.020 |
| D2 later rejection | 1 | 2 | 1 | 1 | 0 | 2 | 3 | 2 | 0.012 |
| D1 EOS | 1 | 1 | 1 | 0 | 0 | 1 | 2 | 2 | 0.009 |
| D2 short budget | 1 | 1 | 1 | 0 | 0 | 1 | 2 | 2 | 0.009 |
| D1 zero output | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0.001 |
| D0 target only | 2 | 0 | 0 | 0 | 0 | 2 | 3 | 0 | 0.007 |
| D1 draft length 3 | 1 | 3 | 3 | 0 | 0 | 4 | 3 | 4 | 0.013 |
| D2 multiple rounds | 2 | 3 | 2 | 1 | 0 | 3 | 4 | 3 | 0.020 |

Key gates:

- D2 later rejection: one `shallow_draft` phase occurrence, zero draft model calls, two total shallow calls (prefill + committed replay).
- D1 draft length 3: one `shallow_draft` phase occurrence, two actual scratch calls; total shallow calls 4 (prefill + 2 drafts + replay).
- D1 EOS and D2 short budget: one `shallow_draft` occurrence, zero calls.
- D1 all accepted + bonus: TTFT 0.015 s, decode 0.012 s, first round commits all 3 tokens, post-first 0.001 s, tokens after first 0, throughput 250 tokens/s fake-clock.
- D1 first rejection: two rounds, decode 0.020 s, first round commits 1, one token after first commit.
- D0: zero shallow calls; decode 0.007 s and post-first 0.004 s.
- Zero output: TTFT and post-first null, throughput null; decode is the post-prefill trace tail by the exact `end − prefill_end` definition.

## 4. Reproducibility and preservation

`implementation.diff` is the five-file correction diff from prior evidence head `339969c` to tested `42b8e6c`. `implementation.files.json` records all tested-file hashes. `execution.log` records that `git diff 339969c..HEAD -- experiments/results/T0018/attempt-001` is empty.

No raw text, token arrays, tensors, checkpoints, virtualenvs or secrets are present. Fixture timings are fake-clock and are not performance results. Observer overhead, GPU timing and trained-checkpoint throughput remain unmeasured.

## 5. Limitations

- Local Mac has no torch; the 16 integration tests are hosted-CPU evidence only.
- Fixture timings are injected fake-clock values, not measured latency.
- Observer callback overhead is unmeasured; the future benchmark must pair observer-off/on and synchronize outside callbacks.
- D0 stochastic sequences are not expected to match D1/D2 token sequences; only greedy equality is claimed.
- The benchmark document is a protocol, not an executed trained-checkpoint benchmark. No speed, memory, acceptance-rate or quality result is claimed.
- The observer is exact-reference instrumentation, not a production cache manager or fused kernel.

## 6. Handover

Flat evidence under `experiments/results/T0018/attempt-002/`: summary, result, environment, metrics, sanitized execution log, hosted run metadata and excerpts, local stdlib output, nine fake-clock fixtures with summaries, implementation correction diff/file hashes, and the benchmark contract copy.

The exact validator command is:

```bash
python3 tools/rep/validate_result.py experiments/results/T0018/attempt-002 \
  --task T0018 --source 42b8e6ca07ca5d3be1de1241a92f30a8d1b1cf01 \
  --checks environment,implementation,unit_tests,benchmark_contract,handover
```

Its output is recorded in `result.json` and PR #91. One attempt-002 INDEX row is appended; attempt-001 and its row remain. No merge, closure or T0019 launch is performed.
