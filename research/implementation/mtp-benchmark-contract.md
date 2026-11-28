# MTP speculation observability and benchmark contract

Status: T0018 implementation contract, 2026-11-28. Task [T0018](../../experiments/tasks/T0018-mtp-observability.md), issue [#81](https://github.com/jtrefon/splot-lm/issues/81). This document defines the trace and the later trained-checkpoint benchmark protocol. It reports **no measured speed, acceptance or quality result**. The T0018 evidence covers the reducer, synthetic fixtures and hosted CPU integration tests only.

## 1. Scope and invariants

- `src/splot/speculative.py` remains the exact D1/D2 reference. The observer is optional; `observer=None` preserves tokens, RNG draw sequence, model calls, `Result`, target/shallow caches and pending logits.
- Observer events contain counts, lengths, role/phase/status names, booleans and integer monotonic-nanosecond timestamps only. Token IDs, text, logits, caches and tensors are forbidden. `tools/speculation_trace.py` rejects unknown fields, so content-bearing traces fail structurally.
- D0 in `run_d0` is target-only and uses the same prompt, budget, temperature, greedy/EOS rules and target runner. It makes no shallow call. Greedy D0 output must equal greedy D1/D2 output for the same target; stochastic D0 is not expected to match a speculative sample sequence because RNG consumption differs.
- No extra model call, sampling draw, tensor mutation or per-token host synchronization is added on the default path. The observer adds phase-boundary callbacks only when requested.
- T0018 makes no trained-checkpoint, throughput, latency, memory or acceptance claim. Synthetic fixture timings are fake-clock and are not performance measurements.

## 2. Trace schema `splot.mtp.trace.v2`

Version 2 adds explicit `model_calls` per phase boundary; attempt-001 v1 evidence is historical and is not rewritten.

The trace is a bounded list (at most 100,000 events), strictly ordered by contiguous `seq` and nondecreasing `timestamp_ns`. The first event is `run_start`; the last is `run_end`.

| Event | Fields | Purpose |
|---|---|---|
| `run_start` | strategy D0/D1/D2, draft_length, max_new_tokens, prompt_length, greedy, temperature, eos | Run identity and fixed settings; no prompt tokens. |
| `phase_start` | role, phase, round, before_length | Opens one measured work boundary. |
| `phase_end` | role, phase, round, before_length, after_length, requested_tokens, actual_tokens, model_calls | Closes the boundary; `actual = after - before`; `model_calls` is the number of actual runner `advance` invocations in that phase. |
| `draft` | round, offset, status accepted/rejected/unexamined, proposed, verified | Per-offset opportunity and examination result. |
| `round_commit` | strategy, round, proposed, verified, accepted, rejected, unexamined, correction, bonus, committed, target/shallow/output before/after, eos_committed | Reconciles one round. |
| `run_end` | rounds, proposed, accepted, rejected, output_length, target_length, shallow_length | Final totals and cache lengths. |

Phases are role-specific:

- target: `target_prefill`, `target_verify`, `target_rejection_replay`, `target_bonus_advance`, `target_step` (D0 only);
- shallow: `shallow_prefill`, `shallow_draft`, `shallow_committed_replay`.

`before_length` and `after_length` are prefix lengths, not output lengths. For a round, `target_after - target_before = shallow_after - shallow_before = committed` and `target_after = shallow_after = prompt_length + output_after`. D0 mirrors target lengths into shallow counters solely for schema reconciliation and emits no shallow phase.

Phase occurrences and model calls are distinct. A phase pair is one instrumentation boundary; `model_calls` counts actual `advance` invocations. `target_prefill`, `target_verify`, `target_rejection_replay`, `target_bonus_advance`, `target_step`, `shallow_prefill`, and `shallow_committed_replay` are one call per occurrence. `shallow_draft` is variable under the current D1 contract: one call per actual scratch token (0 with EOS or a short final draft, 1 for the usual second proposal, greater than 1 for `draft_length>2`). D2 performs zero shallow-draft calls. The reducer exposes `model_calls_by_phase` and `phase_occurrences_by_phase` separately, plus role totals from model calls.

Draft status forms one accepted prefix, at most one rejected position, then an unexamined suffix. A rejection is followed by exactly one correction token; all-accepted rounds may draw at most one bonus token. EOS may commit from a draft, correction or bonus, after which no further round is allowed.

## 3. Metric definitions

`tools/speculation_trace.py` is the canonical reducer.

- `proposed_tokens`: all draft positions proposed, including an unexamined suffix.
- `verified_tokens`: examined positions, `accepted + rejected`.
- `accepted_tokens / rejected_tokens / unexamined_tokens`: per-offset and total status counts.
- `acceptance_per_verified = accepted / verified`; null when no position was examined.
- `accepted_per_proposed = accepted / proposed`; null when nothing was proposed.
- `unexamined_per_proposed`: rejected-suffix and EOS-truncation share, null when nothing was proposed.
- `correction_tokens`: one per rejected round. `bonus_tokens`: at most one per all-accepted round that still has budget.
- `committed_tokens = accepted + correction + bonus = output length`.
- `per_offset`: proposed, verified, accepted, rejected and unexamined counts plus the two ratios. Offset 1 and offset 2 are never collapsed in D2 analysis.
- Target work: actual model calls and advanced tokens per phase, with prefill separated. `target_decode_tokens` excludes target prefill.
- Shallow work: actual model calls and advanced tokens, with prefill, variable serial draft calls and committed-prefix replay separated. `shallow_draft` model calls equal its actual scratch advances; D2 reports zero.
- `target_tokens_per_committed_token = target_decode_tokens / committed`, null for zero output. This is the target replay/work amplification, not a speedup.
- `target_extra_tokens_over_committed = target_decode_tokens - committed`.

## 4. Timing contract

T0018 uses `time.monotonic_ns()` or an injected integer fake clock. Event timestamps are host wall observations, not GPU kernel timings. They include Python dispatch, host/device scheduling, observer callbacks and any CPU logits transfer already present in the exact reference.

- `prefill_seconds`: start of run to the end of the last prefill phase (target plus shallow).
- `ttft_seconds`: start of run to the first `round_commit` (first committed output token); null for zero generation.
- `decode_seconds`: end of prefill to `run_end`. It includes the complete first generation round, including a multi-token first commit, plus all later verification, replay and bonus work. This is the denominator for committed-token throughput.
- `post_first_commit_seconds`: first commit to `run_end`; null for zero generation. This is a diagnostic tail, not a throughput denominator.
- `first_round_committed_tokens` and `tokens_after_first_commit`: expose how much work the first commit carried.
- `committed_tokens_per_decode_second = committed_tokens / decode_seconds`; null when nothing was committed or decode is zero. Never divide all committed tokens by post-first-commit time.
- `total_seconds = prefill_seconds + decode_seconds`; for nonzero output, `ttft_seconds + post_first_commit_seconds = total_seconds`.
- `phase_seconds`: sum of paired start/end durations per role and phase.

A later GPU benchmark must wrap each timed region with explicit device synchronization outside the observer. Synchronization overhead is measured and reported separately. An observer-off/observer-on paired control estimates callback overhead. The reported latency is **observed committed-token advance wall time**, not true kernel latency. No claim may present it as kernel-only time.

## 5. Later trained-checkpoint benchmark protocol

This protocol is not executed by T0018. It becomes eligible only after the T0017 real checkpoint and its exact dataset identity are reviewed, and after a separate benchmark task is merged and explicitly launched by the hardware owner.

### Fixed inputs

- One reviewed trained checkpoint, tokenizer, manifest and source-snapshot identity; record all hashes.
- Same target runner, prompt set, tokenizer, precision, device, batch size one, sampling rule and EOS semantics across D0/D1/D2.
- Fixed committed-token budget per prompt. Greedy and stochastic runs are separate experiments.
- No test-split access and no prompt/token text publication. Prompts are represented by hashes or reviewed local identifiers.
- Context lengths 128 and 512. 2048 is allowed only if the resource gate in the task amendment passes; no silent escalation.

### Systems

- D0: target-only baseline.
- D1: serial shallow drafts, shallow depth 1, target depth 2, draft length 2.
- D2: independent offset-2 drafts from the same committed shallow prefix, draft length 2.

### Correctness gates

- Greedy: D1 and D2 committed tokens must equal D0 tokens exactly for the same checkpoint and prompt; final cache lengths and pending logits must match a fresh target prefill. Any mismatch fails the run, not a timing comparison.
- Stochastic: do not require token equality. Verify empirical acceptance against `accepted/verified`, proposal coverage, per-offset denominators, residual/correction and EOS accounting. Existing distribution tests remain the mathematical gate; the benchmark reports observed acceptance separately.
- Cache identity: target and shallow caches must equal fresh-prefix replay for committed tokens. No stale or mutated prefix.
- Trace validation: `summarize` must pass on every measured run; any malformed event fails the run.

### Measurement protocol

- At least three paired repetitions per system/context/prompt after one warmup. Rotate system order across repetitions to reduce drift; report every pair, not only the best.
- Synchronize before and after each measured prefill, verification, replay, bonus and full generation region. Keep synchronization outside the observer callbacks.
- Record observer-off and observer-on controls. Report callback overhead and exclude it from no claim; do not hide it.
- Report TTFT, prefill seconds, decode seconds, committed tokens/second using all committed tokens over `decode_seconds`, post-first-commit tail separately, target/shallow model calls and tokens, per-offset proposed/verified, rejection replay, correction and bonus counts, peak allocated/reserved memory, actual precision/device, and checkpoint/tokenizer/manifest hashes.
- Use the same process-level thread settings and device clocks policy for all systems. Record thermal/clock caveats rather than normalizing them away.
- No performance claim from untrained weights. No quality or convergence claim from this benchmark.

### Relationship to native MTP and recurrence depth

The benchmark measures the exact reference verifier, not a production fused kernel. Native MTP acceptance and recurrence-depth effects must be reported as separate architecture evidence with their own reviewed task. A speed result for D1/D2 does not by itself justify a native-MTP or adaptive-depth change, and an acceptance-rate result does not establish end-to-end quality. Adoption remains a quality-cost decision under the research charter and matched-control evaluation.

### Proposed follow-up

After T0017 checkpoint review, propose one benchmark task that executes this protocol on the reviewed Linux checkpoint. That task must pin the checkpoint and dataset identity, preregister contexts, repetitions and resource gates, and preserve failures. T0018 does not launch it.

## 6. T0018 validation

- Stdlib: `tools/test_speculation_trace.py` covers all-accepted plus bonus, first rejection with unexamined suffix, later rejection, EOS, short final draft, zero output, D0, ratios with zero denominators and malformed schema/order/count/content cases.
- Hosted CPU: `tests/test_speculative_observer.py` covers observer off/on token, RNG, model-call, cache and pending-logit identity; D0 greedy equality with D1/D2; zero budget, EOS, rejection, all-accepted and short-budget accounting.
- Local Mac has no torch; integration is executed by the hosted CPU workflow and is not claimed locally.

## Polski — podsumowanie

T0018 dodaje opcjonalny obserwator do dokładnego mechanizmu spekulacji D1/D2 bez zmiany wyników, losowania, wywołań modelu ani pamięci podręcznych. Reduktor w `tools/speculation_trace.py` waliduje zdarzenia, rozdziela wystąpienia faz od rzeczywistych wywołań `advance`, liczy akceptację i pracę celu/płytkiego modelu oraz definiuje metryki czasu. Dekodowanie liczone jest od końca prefillu i obejmuje pełną pierwszą rundę; przepustowość używa wszystkich zakomitowanych tokenów. D0 to baseline tylko z modelem docelowym. Kontrakt przyszłego benchmarku na wytrenowanym checkpoincie wymaga tych samych wejść, dokładnych bramek zachłannych, osobnej analizy stochastycznej, synchronizacji urządzenia i raportowania narzutu obserwatora. Ten dokument nie zawiera zmierzonych wyników wydajności.
