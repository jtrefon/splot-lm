# T0019 — matched architecture baseline and capability evaluation design
2026-11-28 | ANA | Issue #89 | Source pin `6e8911662ac56df665147cb79e412f0383cf71d8` | Instructions `f49af5bfb0085cbbc7161d5b3354b8e8fe309696`

**Status:** design and analytical inventory only. No model was trained, no test data or checkpoint was accessed, and no package was installed. All MAC values are ideal contraction estimates, not training FLOPs, latency or memory measurements. Proposed margins and budgets are drafts for review, not preregistered confirmatory gates.

## 1. Purpose and constraints

The [charter](../../../research/CHARTER.md) requires quality–cost evidence against a credible, optimized conventional comparator. This document designs that comparator for SPLOT Nano R2 and a first bounded study under the [evaluation framework](../../../research/evaluation/README.md). It does not adopt an architecture, claim superiority, or authorize training.

Pinned inventory code: `6e89116`. The T0019 specification is read from reviewed instruction main `f49af5b`; the source pin predates the spec by design and contains the accounting/dense contracts audited here. Local Mac has no `torch`; parameter formulas were verified arithmetically and against committed historical inventories, not re-instantiated locally.

## 2. Inventory and compatibility audit

### 2.1 SPLOT Nano actual counts

`python tools/accounting.py --nano ...` from the pinned checkout gives:

| Regime | Context | Physical | Base | MTP aux | Logical blocks | Decode MACs | Prefill all-logit MACs | Cache bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NTP-only (readouts 1, offsets 0) | 512 | 43,055,076 | 42,907,236 | 147,840 | 12 | 41,416,704 | 20,602,552,320 | 2,204,160 |
| A2 (readouts 2, offsets 2) | 512 | 43,055,076 | 42,907,236 | 147,840 | 14 | 83,392,512 | 41,893,232,640 | 2,728,448 |
| NTP-only | 2048 | 43,055,076 | 42,907,236 | 147,840 | 12 | 48,494,592 | 89,657,966,592 | 6,922,752 |
| A2 | 2048 | 43,055,076 | 42,907,236 | 147,840 | 14 | 92,829,696 | 177,236,606,976 | 9,019,904 |

The Nano physical count **43,055,076 at V32768** is confirmed. The existing dense 8-block control is **25,172,352** without an MTP head (25,320,192 with one), overwhelmingly smaller; it is a correctness control, not a matched comparator. Component check: `12,582,912 embedding + 8 × 1,573,632 blocks + 384 final norm = 25,172,352`.

### 2.2 What the current code actually supports

| Capability | Dense reference | SPLOT Nano | Consequence |
|---|---|---|---|
| Forward/cache GQA | `DenseLM`, exact cache/ABI tests | `SPLOTLM` hybrid/MoE | dense forward is usable as a reference |
| Full training CLI | only `tools/smoke_dense.py` overfit smoke | `tools/experiment.py` hardcodes `SPLOTLM` | dense needs a reviewed training/eval adapter |
| Real pilot loader | loader is model-independent but dense CLI does not use it | pilot path exists | data interface can be reused, not the model path |
| MTP/offset-2 | absent | native MTP head/objective | A2 comparison needs a matched dense auxiliary head |
| Optimizer partition | standard DenseLM names likely classify, dense MTP head does not | hybrid Muon/AdamW qualified | add dense MTP classification before optimization |
| Precision | FP32/BF16 smoke only | FP32/BF16/FP8 paths | primary comparison should use BF16 both sides; FP8 is not symmetric |
| Accounting | formula in dense contract only | `tools/accounting.py` | add reviewed dense accounting; current evidence script is analytical |
| Checkpoint/identity/resume | not integrated with experiment CLI | qualified | dense needs identical identity/checkpoint semantics |

The audit therefore rejects the existing `dense-nano`/`dense-mini` configs as matched controls and identifies concrete implementation work in §4.

MTP accounting: SPLOT adds 147,840 parameters and, for A2, two logical readout/coda blocks. At context 512 the A2 prefill MAC difference is 21,290,680,320 and decode difference 41,975,808; at 2048 they are 87,578,640,384 and 44,335,104. These are architecture-level auxiliary costs, not a dense offset head alone.

## 3. Matched candidate shapes

Formulas (stdlib script `candidate-counts.py`, output `candidate-counts.json`):

```text
params = V*d + L*(d*(2q+2kv) + 3*d*m + 2*d) + d + (d*d+d if MTP head else 0)
decode(T) = L*(d*(2q+2kv)+3*d*m) + L*2*q*T + d*V
prefill(T) = T*(L*(d*(2q+2kv)+3*d*m) + d*V) + L*q*T*(T+1)
q = q_heads*head_dim, kv = kv_heads*head_dim
```

The linked script computes all entries; `accounting.json` records the four exact pinned commands. No floating-point tuning or hidden fitting is involved.

### 3.1 Primary candidates at V16384 (actual T0016 vocabulary)

All have `head_dim=64`, tied embeddings, no biases, and include a matched shared offset head (`d*d+d = 262,656` parameters) only for physical parity. NTP-only comparisons do not use that head.

| Candidate | Width | Layers | Q/KV | FFN | Physical (with aux) | vs 43,055,076 | Prefill 512 | vs NTP 512 | Prefill 2048 | vs NTP 2048 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Iso-parameter | 512 | 9 | 8/2 | 1,920 | 41,101,312 | −4.54% | 22,114,729,984 | +7.34% | 102,954,434,560 | +14.83% |
| Iso-forward-MAC | 512 | 6 | 8/2 | 2,816 | 38,542,336 | −10.48% | 20,402,667,520 | −0.97% | 91,274,346,496 | +1.80% |

The iso-parameter candidate meets the 5% parameter target but is 7.34% above NTP forward MAC at 512 and 14.83% above at 2048. The iso-MAC candidate meets the 5% prefill-MAC target at both contexts and decode 512 (+0.005%) and decode 2048 (+4.87%), but is 10.48% below the parameter target. This is the central trade-off: recurrence reuses weights, so parameter parity and compute parity cannot both be exact with a conventional dense shape.

### 3.2 Compatibility candidates at V32768

| Candidate | Width | Layers | Q/KV | FFN | Physical (with aux) | vs target | Prefill 512 | vs NTP 512 | Prefill 2048 | vs NTP 2048 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Iso-parameter | 512 | 6 | 8/2 | 2,176 | 41,032,704 | −4.70% | 21,677,735,936 | +5.22% | 96,374,620,160 | +7.49% |
| Iso-forward-MAC | 512 | 6 | 8/2 | 1,920 | 38,673,408 | −10.18% | 20,469,776,384 | −0.64% | 91,542,781,952 | +2.10% |

The V32768 iso-parameter candidate is a nearest shape with an explicit 0.22 percentage-point prefill-512 gap above the 5% target; it is not presented as exact. V16384 remains the primary design because T0016's actual tokenizer is 16,384 and the charter says later matched training should use actual vocab if feasible.

### 3.3 Enumeration and selection

The search enumerated widths 384/512, 6/2, 8/2 and 8/4 GQA groupings, layers 6–18, and FFN widths 1,024–4,096 with an FFN/width ratio of 2–5.5. Representative near candidates:

| Vocab | Width | Layers | Q/KV | FFN | Parameters | Prefill 512 |
|---:|---:|---:|---|---:|---:|---:|
| 16384 | 384 | 14 | 6/2 | 1,920 | 42,921,216 | 23,306,305,536 |
| 16384 | 512 | 9 | 8/2 | 2,048 | 42,870,784 | 23,020,699,648 |
| 16384 | 512 | 11 | 8/2 | 1,280 | 37,498,880 | 20,538,195,968 |
| 16384 | 512 | 6 | 8/2 | 2,816 | 38,542,336 | 20,402,667,520 |
| 16384 | 512 | 7 | 8/2 | 2,176 | 36,642,816 | 19,564,068,864 |
| 32768 | 384 | 11 | 6/2 | 2,048 | 43,017,216 | 23,054,057,472 |
| 32768 | 512 | 8 | 8/2 | 1,664 | 42,738,688 | 22,819,110,912 |
| 32768 | 512 | 6 | 8/2 | 1,920 | 38,673,408 | 20,469,776,384 |
| 32768 | 512 | 6 | 8/2 | 2,176 | 41,032,704 | 21,677,735,936 |

Selection rules, fixed before outcomes: preserve integer GQA grouping and `width=q_heads*head_dim`; stay within 5% of the declared target where feasible; report the nearest shape and gap otherwise; reject shapes whose FFN/width aspect is outside 2–5.5; do not silently change vocabulary. The selected shapes are not claimed to be uniquely optimal.

## 4. Matched experiment design

### 4.1 Primary NTP-only comparison

Arms, all at actual vocab 16,384:

1. SPLOT Nano R2, NTP readout 1, no offset loss (`loops=2, readouts=1, offsets=0`).
2. Dense iso-parameter candidate (V16384, 512×9, 8/2, FFN 1,920).
3. Dense iso-forward-MAC candidate (V16384, 512×6, 8/2, FFN 2,816).

For physical parity the dense arms include the same shared offset head but do not apply its loss in the primary comparison; report parameter totals both with and without it. Primary quality metric is final validation NTP NLL; primary cost metrics are analytical forward MACs, measured wall time, peak memory and checkpoint bytes. The NTP-only comparison isolates mixer/depth/MoE recurrence from the auxiliary objective.

### 4.2 Secondary A2 auxiliary comparison

Run SPLOT A2 (`readouts=2, offsets=2`) against dense + the same shared offset head and loss. Report final NTP NLL separately from offset-2 NLL. The dense offset head adds 262,656 parameters and, per offset, `T*(d*d+d*V)` MACs: at V16384/d512 this is 4,429,185,024 at 512 and 17,716,740,096 at 2048 for one offset; two offsets double those values. SPLOT's A2 increase also includes an extra readout/coda path, so the full 21.29B/87.58B prefill increase is not attributable to the offset head alone. No architecture claim may conflate final NTP with auxiliary offset loss.

### 4.3 Fixed-token regime and fairness

- One pass over the T0016 training split: 128 blocks, B1, T512, 46,076 NTP labels, 65,536 valid-token presentations per run.
- Same tokenizer artifact, block order per seed, document/position masks, sequence length, batch size, precision and evaluation code.
- Primary precision: BF16 autocast with FP32 master weights and optimizer state on both arms; FP8 is excluded because the dense path is not qualified for symmetric FP8 and would confound architecture with precision.
- Same optimizer family and hyperparameters on both arms: semantic Muon for hidden 2-D matrices, AdamW for embeddings/norms/scalars. A common LR is fixed before the run; if tuning is needed, both arms receive the same grid, trial count and validation-only selection rule.
- Same seed set and paired data order: planning seeds 42, 43, 44. No per-arm early stopping, extra epochs or private tuning.
- `test` split remains untouched until configurations are frozen; final test is a single integrity-checked evaluation, reported with the validation result.
- The comparison is architecture-level. Forward MACs are not training FLOPs; backward, optimizer, padding and kernel work must be measured or reported separately.

## 5. Evaluation protocol

### 5.1 Data and splits

Use the T0016 `splot-pilot-blocks-v1` manifold only after a reviewed local copy and exact manifest/tokenizer/source hashes are verified. Eight prose families are whole-family assigned: train 2 EN + 2 PL, validation 1 EN + 1 PL, test 1 EN + 1 PL. No family crosses splits. The test split is not read during design, tuning or checkpoint selection. The bounded pilot is a feasibility corpus, not a representative sample.

### 5.2 Language-model metrics

- **Held-out NTP NLL** in nats per valid NTP label, token-weighted; per language and aggregate.
- **Bits per byte** `= sum(NLL nats) / (ln(2) × covered original UTF-8 bytes)`. Covered bytes are the UTF-8 bytes of the target text corresponding to valid non-ignore labels, excluding BOS/EOS/pad and masked positions. Coverage ratio and missing-byte count are reported; bits/byte is compared only when coverage is identical or explicitly reconciled.
- **PPL** only when tokenizer and label masks are identical; otherwise omit.
- Report label counts, valid tokens, padding, per-language denominators and byte coverage for every run.

### 5.3 Controlled capability guardrails

Two synthetic, independently generated probes with unseen instances and explicit chance baselines:

1. **Key–value retrieval with distractors.** Query one key after N distractors; exact-match accuracy over held-out keys/values and generator seeds; chance `1/N` (e.g. 1/16). Report floor/ceiling and per-length breakdown.
2. **Compositional pattern completion.** A held-out bracket/induction or modular-arithmetic generator with disjoint operands/templates from training; exact-match or sequence accuracy; chance baseline from uniform outputs. Check that no probe instance appears in the pilot corpus or training stream.

Both probes are noninferiority guardrails, not claims of general capability. If either is at floor for all arms, report that the probe is uninformative rather than a win.

### 5.4 Proposed planning margins (not preregistered)

- Primary NTP NLL: noninferiority margin **+0.05 nats/valid label**; minimum meaningful advantage **−0.03 nats** (lower is better). Bits/byte secondary margin **+0.01 bits/byte**.
- Guardrails: absolute accuracy noninferiority margin **−2 percentage points** per probe, with chance/floor/ceiling reported.
- Cost: report paired wall-time, peak allocated/reserved memory and checkpoint bytes; a provisional **+10%** wall/memory guardrail is descriptive only. No throughput claim from one machine.
- Uncertainty: paired bootstrap over held-out blocks/segments stratified by family, 10,000 resamples per seed, plus seed mean and range. Three seeds are planning evidence, not a powered confirmatory test; no p-value-only equivalence claim. The interval must meet the declared margin in a later confirmatory design.
- Learning curves: validation NLL at 16/32/64/96/128 updates; time-to-target (e.g. a preregistered validation NLL) is censored if unreached.
- Failure rules: nonfinite loss/state, tokenizer/manifest/source mismatch, test access, unequal token or tuning budget, missing checkpoint identity, or a cache/eval mismatch fails the affected run; no gate relaxation or silent retry.

### 5.5 Budget estimate

Base matrix: 3 arms × 3 seeds × 128 updates = **1,152 updates** and **589,824 valid-token presentations** (414,684 NTP label evaluations across runs). Accepted T0012 measured a single BF16 T512 update at 1.2696 s E2E and FP8 at 1.6069 s on the RTX 4090; accepted T0011 measured 35-step synthetic epochs at 47–80 s wall including checkpointing. Applying those descriptive anchors suggests roughly **45–75 minutes** for updates plus validation/guardrails/checkpoints, and up to about **2 hours** if a small equal tuning stage is added. This is a planning range, not a measured matched-pilot budget: the data, model shapes, implementation, checkpoint cadence and hardware differ. Mac M4 CPU is not the training target; no training is authorized by T0019.

## 6. Exact implementation gates before any matched run

1. Reviewed dense model-family selector in `tools/experiment.py` with architecture identity in the run hash; no hardcoded `SPLOTLM` assumptions in reporting.
2. Dense NTP training/evaluation adapter using the verified pilot loader, checkpoint/resume identity and the same label/mask contract.
3. Dense MTP/offset-2 head plus objective for the secondary A2 comparison, or an explicitly reviewed decision to drop A2 from the dense arm.
4. Extend semantic optimizer partitioning/classification to the dense MTP owner; verify DenseLM and DenseMTPLM partitions exhaustively.
5. Reviewed dense parameter/MAC accounting in product tooling, replacing the analytical evidence script; instantiate candidate configs and verify parameter counts on CPU or meta device.
6. Checkpointed configs for the selected V16384 arms; V32768 compatibility only.
7. Per-block UTF-8 byte coverage metadata or local detokenization for bits/byte, with no raw-text publication.
8. Capability probe generators/evaluators with disjoint instance manifests and chance baselines.
9. Paired run matrix, seed/data-order manifest, equal tuning budget and analysis script for the proposed margins.
10. CPU/CI tests for accounting, config identity, dense partition, MTP objective, data path and evaluation masks.

## 7. Risks and limitations

- The pilot corpus is small and not representative; 46,076 training NTP labels cannot support a general quality claim.
- The 3-seed planning design has low power; margins are proposals pending review.
- Dense and SPLOT differ in parameter efficiency and attention scaling; no single shape matches both parameter and forward-MAC targets.
- Forward MACs exclude backward, optimizer, padding, memory traffic and kernel efficiency. They are not training FLOPs or wall time.
- Existing dense reference is a correctness reference, not an optimized baseline; the iso-param/iso-MAC candidates require implementation and validation.
- T0017 results are not used: the real pilot is not accepted. No T0017 checkpoint or test data was accessed.
- No architecture, quality, speed, memory or scaling claim is made.

## 8. Proposed next action

Review this design and the candidate counts. If accepted, create a bounded implementation task for the dense training/accounting/MTP/optimizer gates in §6, then a separate matched-experiment task with the margins and budget above preregistered. Do not implement or train from T0019. T0020 remains separate.

## Polski — podsumowanie

T0019 definiuje porównanie SPLOT Nano R2 z konwencjonalnym gęstym GQA. Przy V16384 kandydat iso-parametrowy (512×9, 8/2, FFN 1920) mieści się w 5% parametrów, ale ma o 7,3% więcej MAC prefill@512; kandydat iso-MAC (512×6, 8/2, FFN 2816) mieści się w 5% MAC przy 512 i 2048, ale ma o 10,5% mniej parametrów. V32768 jest tylko zgodnościowe. Protokół rozdziela NTP od A2, używa actual vocab, family split, NLL/bits-per-byte, dwóch strażników zdolności, trzech seedów i równych budżetów. To projekt, nie wynik treningowy; żadne dane testowe ani checkpoint nie zostały użyte.
