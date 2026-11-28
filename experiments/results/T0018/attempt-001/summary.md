# T0018 attempt-001 — exact MTP speculation observability and D0 baseline
2026-11-28 | IMP | Issue #81 | Parents #2/#10/#6 | PR #91

**Outcome:** passed locally and on hosted CPU. No performance or model-quality claim.
**Instructions:** `d3752eeb77541866e2332ed238def0c4bac50549` (reviewed main).
**Minimum reviewed base:** `a93141e3160c0e9bb845d5b0a64df69c0e97157b` (ancestor).
**Tested implementation:** `d32c52780a0e6d565ae20422fc83061596a12110`.
**Superseded implementation:** `a199bc96187cff3a83ee7031e22b0f3d1b584083`; its first hosted run failed four D2 integration tests on a trace-count bug and was corrected in the tested commit.
**Scope:** opt-in observer for the exact D1/D2 reference, stdlib trace reducer, target-only D0 baseline, synthetic fixtures, hosted CPU integration tests and a future benchmark contract. No model, topology, training, FP8, CI or data change; no package install.

## 1. Implementation

`src/splot/speculative.py` gains optional keyword-only `observer` and `clock`. With `observer=None` there is no callback, clock call or extra branch in the sampling path; tokens, RNG draw order, target/shallow model calls, `Result` and cache/pending-logit state are unchanged. The observer receives bounded structured events at phase boundaries only; it never receives token IDs, text, logits, caches or tensors.

`tools/speculation_trace.py` is stdlib-only at import time. It provides:

- strict `splot.mtp.trace.v1` validation with a 100,000-event cap, exact field allowlists, contiguous sequence numbers, nondecreasing integer nanosecond timestamps, phase-order/count checks and content rejection;
- per-offset and total proposed/verified/accepted/rejected/unexamined counts; acceptance is `accepted/verified`, coverage is `accepted/proposed`, each null with a zero denominator;
- separate target and shallow calls/tokens, prefill separated from decode, correction/bonus counts, committed reconciliation and `target_decode_tokens/committed` replay amplification;
- TTFT, prefill, decode and total host-clock metrics; fake-clock fixtures are marked synthetic and are not throughput measurements;
- `run_d0`, a target-only baseline with the same prompt, budget, temperature, greedy/EOS rules and target runner, importing torch only when called.

`tests/test_speculative_observer.py` contains the host-CPU integration suite. `research/implementation/mtp-benchmark-contract.md` defines the later trained-checkpoint D0/D1/D2 protocol, exact greedy/cache gates, separate stochastic acceptance, 128/512 contexts with 2048 resource-gated, three paired repetitions, synchronization, memory/precision hashes and no test-set access. It makes no T0018 performance claim.

## 2. Tests

| Run | Tests | Duration | Result |
|---|---:|---:|---|
| Local stdlib `tools/test_speculation_trace.py` | 24 | 0.050771 s | passed |
| Hosted `tests/test_speculative_observer.py` | 12 | 0.272 s | passed |
| Hosted `tools/test_speculation_trace.py` | 24 | 0.007 s | passed |

Hosted CPU: [run 46153049369](https://github.com/jtrefon/splot-lm/actions/runs/46153049369), **success** on `d32c527`, Ubuntu 24.04, CPython 3.12.7, torch 2.10.0+cpu, numpy 1.26.4, CUDA hidden; 29 scripts and 21 existing REP reports validated structurally.

Local Mac: CPython 3.15.2, **no torch module**. Per the reviewed task, torch was not installed on this Mac. Local acceptance therefore covers compile checks, the 24 stdlib tests, fixture generation and structural diff checks; the 12 torch integration tests are executed only on hosted CPU. No local integration pass is claimed.

Integration checks include observer off/on token, RNG, model-call, target/shallow cache and pending-logit identity for D1/D2; actual-model D0 greedy equality with D1/D2; fresh-prefix cache replay; zero budget; EOS; first and later rejection; unexamined suffix; all-accepted bonus; short final draft; and no extra target advance. The reducer rejects malformed schema, sequence, timestamps, phase order, counts, draft status order, commit arithmetic and content fields.

## 3. Synthetic fixtures

`trace-fixtures.json` contains seven deterministic fake-clock traces. Timings are injected 1 ms event increments, not measured latency.

| Case | Rounds | Proposed | Accepted | Rejected | Unexamined | Committed |
|---|---:|---:|---:|---:|---:|---:|
| D1 all accepted + bonus | 1 | 2 | 2 | 0 | 0 | 3 |
| D1 first rejection, unexamined suffix | 2 | 3 | 1 | 1 | 1 | 2 |
| D2 later rejection | 1 | 2 | 1 | 1 | 0 | 2 |
| D1 EOS | 1 | 1 | 1 | 0 | 0 | 1 |
| D2 short budget | 1 | 1 | 1 | 0 | 0 | 1 |
| D1 zero output | 0 | 0 | 0 | 0 | 0 | 0 |
| D0 target only | 2 | 0 | 0 | 0 | 0 | 2 |

## 4. First failure and correction

The first hosted run, [46152872698](https://github.com/jtrefon/splot-lm/actions/runs/46152872698) on `a199bc9`, failed four D2 integration tests with `TraceError: D2 shallow_draft must not advance a scratch cache`. D2 correctly performed zero serial shallow advances, but the observer reported `requested_tokens=1` from the draft length. The tested commit reports `requested_tokens=0` and keeps `actual_tokens=0` for D2. No gate was relaxed; the failure is preserved in `execution.log`.

## 5. Limitations

- Local Mac has no torch; integration is hosted-CPU evidence only. No MPS/CUDA path was tested.
- Synthetic fixture timings are fake-clock and are not performance measurements.
- Observer callback overhead is not measured. The future benchmark must pair observer-off/on and synchronize outside callbacks.
- D0 stochastic output is not expected to match D1/D2 token sequences because RNG consumption differs; only greedy equality is claimed.
- The contract is a protocol, not an executed trained-checkpoint benchmark. T0017's real checkpoint remains unreviewed and no model-quality, acceptance-rate, throughput or memory result is claimed.
- The observer is reference-instrumentation only, not a production cache manager or fused kernel.

## 6. Handover

Flat evidence: summary, result, environment, metrics, sanitized execution log, hosted run metadata and test excerpts, local stdlib test output, seven synthetic trace fixtures with summaries, implementation diff/file hashes and the benchmark contract copy. No raw text, token arrays, model weights or checkpoint tensors are present.

The REP validator command is:

```bash
python3 tools/rep/validate_result.py experiments/results/T0018/attempt-001 \
  --task T0018 --source d32c52780a0e6d565ae20422fc83061596a12110 \
  --checks environment,implementation,unit_tests,benchmark_contract,handover
```

Its observed output is recorded in `result.json` and PR #91. One T0018 INDEX row is appended. Issue #81 receives the handover. No merge, closure, T0019 launch or additional task execution is performed.
