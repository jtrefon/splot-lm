# T0026 — prospective matched learning evaluation design

Task T0026 (ANA), issue #108, parents #85/#8/#10/#86/#87. Source and instruction commit `003cf8db6126f0aaefdd599ac159f2503457f709`; T0023 implementation accepted at `4a4bb5e`; T0025 chunk64 full-model candidate rejected and excluded. This is a **prospective design**, not a preregistration of final margins, not a training protocol execution, and not a claim based on T0016 or T0027.

**Scope:** Mac M4 CPU/stdlib only. No training, dataset acquisition, tokenizer fit, package install, network data, test access, checkpoint access, model/source change, GPU/MPS or shared state-document edit. All numbers below are analytical estimates with labelled assumptions.

## 1. Question and claim boundary

Primary question: at matched valid-token and forward-contraction budgets, does SPLOT A0 scan R2 preserve or improve held-out token-weighted final NTP NLL versus a credible optimized dense GQA comparator, without material capability regression, at comparable or lower training cost?

The design separates:

- **quality** at fixed valid-token budget;
- **training cost** from analytical forward contraction MACs, estimated training MAC/FLOPs and measured wall time;
- **inference cost** from decode MACs/cache and measured latency in a later task;
- **capability guardrails** from frozen, disjoint probes.

It does not establish population generalization from one pilot, does not convert a small bilingual corpus into a representative sample, and does not treat T0016's two validation families as a variance estimate.

## 2. Arms and matching regimes

| Arm | Model | Config | Physical params | NTP base / gradient-present | Active params | Prefill 512 MACs | vs S prefill |
|---|---|---:|---:|---:|---:|---:|---:|
| **S** | SPLOT A0 scan R2, actual V16384, `balance_weight=0.01` | `splot-nano-v16384-a0.json` | 36,763,620 | 36,615,780 | 22,460,004–27,178,596 | 17,381,326,848 | — |
| **P** | Dense GQA, Q8/KV4, FFN 2432, optimized SDPA | `dense-ntp-512x6-q8kv4-ffn2432.json` | 35,527,168 | 35,527,168 | 35,527,168 | 18,993,381,376 | +9.275% |
| **C** | Dense GQA, Q8/KV2, FFN 2176, optimized SDPA | `dense-ntp-512x6-q8kv2-ffn2176.json` | 32,381,440 | 32,381,440 | 32,381,440 | 17,382,768,640 | +0.0083% |

Dense NTP has no dormant MTP head. SPLOT physical includes the 147,840-parameter dormant offset head under A0; the NTP base and gradient-present count exclude it. Active parameters are the unique per-token routed/core count; they are not a substitute for physical or training-compute accounting.

Matching regimes:

1. **Matched token (primary quality regime).** All arms consume the identical token stream, document order, valid-label mask and update count. One pass means one traversal of the frozen training split; the matrix uses fixed sub-epoch budgets before full-pass budgets.
2. **Matched forward MAC (primary cost regime).** C is within 0.01% of S at 512 tokens of analytical prefill MACs/token. At 2048 the same C is about +3.15%, so the achieved context gap is reported rather than assumed away.
3. **Matched parameter (secondary).** P is within 3.5% of S physical parameters and +3.06% of the S NTP base. It is a credibility control, not the primary compute match.
4. **Matched training compute (optional).** If used, adjust update count so estimated 3×-forward MACs match within a declared tolerance; report the achieved gap. Do not claim simultaneous exact iso-parameter, iso-MAC and iso-token matching.

Primary contrast: **S − C**, paired by seed, final validation NLL. Secondary: **S − P**, paired by seed, interpreted with multiplicity caution. Cost is reported separately; a quality win at materially higher measured cost is not an unconditional adoption result.

## 3. Optimized dense comparator and symmetric tuning

The current `src/splot/dense.py` is a slow explicit correctness reference, not an optimized comparator. A future IMP task must deliver an optimized dense path before confirmatory quality or cost claims:

- PyTorch `scaled_dot_product_attention` with an explicit causal mask and document-segmented attention; no Python per-document loop in the training path.
- Same RMSNorm, RoPE, SwiGLU and tied input/output embedding semantics; no biases; same tokenizer and actual vocabulary.
- BF16 autocast with FP32 master weights and optimizer state; deterministic algorithms; same batch size, context, data order, optimizer family and LR schedule.
- Same evaluation path and label masking as SPLOT; no dense-only test-time tricks.
- Equal implementation effort: both paths must have an optimized training path or both must use the reference path. A slow SPLOT scan against SDPA dense is reported as an implementation gap, not an architecture cost.

Symmetric optimization budget:

- Predeclare a small search grid, for example LR ∈ {1e-3, 3e-3, 6e-3} × warmup ∈ {2%, 5%} = 6 trials per arm, with identical trial count, token budget and validation cadence.
- SPLOT may tune `balance_weight` within its own predeclared grid; dense has no equivalent term but receives the same number of trials over its own grid. No arm gets extra trials because it has more hyperparameters.
- Selection is on the frozen validation split only; the test split is untouched. Report the selected configuration, all trial results and the selection rule.
- No early stopping for the primary endpoint: the primary is the final checkpoint at the predeclared budget. Learning-curve points are descriptive.
- If tuning is infeasible, use the accepted fixed hyperparameters for all arms and label the comparison "untuned"; do not describe it as equal tuning.

## 4. Endpoints and aggregation

### 4.1 Primary language-modeling endpoint

Token-weighted final NTP NLL on the frozen validation split:

```text
NLL = sum_over_valid_ntp_labels(-log p(target_token | prefix)) / count_valid_ntp_labels
```

- Labels are the existing packer-shifted offset-1 targets; padding and cross-document positions are excluded with `ignore_index=-100`.
- The denominator is the exact number of counted NTP labels, not positions, blocks, content tokens or model vocabulary.
- Ordinary NTP NLL, content-only BPB and EOS/control NLL are reported separately; they are never mixed into one denominator.

### 4.2 Per-family and language aggregation

- **Work family** = source `family_id`; **language** = `en`/`pl`.
- Primary aggregate is token-weighted over all validation labels.
- Secondary macro-average weights each validation family equally, so a long family cannot dominate the family-level conclusion.
- Report per-language and per-family NLL, label counts and byte denominators. A family with fewer than a predeclared minimum number of labels is descriptive only.
- Cluster uncertainty resamples families, not tokens. With four validation families, a family bootstrap is only a sensitivity analysis; it is not a population confidence interval.

### 4.3 Byte-aligned BPB

Primary BPB is content-only:

```text
BPB = sum_NLL(content-token targets) / (ln(2) * sum_UTF8_bytes(content-token targets))
```

- BOS is never a target; EOS is predicted but excluded from the content numerator and its NLL is reported separately per document.
- Padding, special tokens and control labels have zero byte coverage and never enter the denominator.
- `sum_UTF8_bytes` is the exact byte length of the original UTF-8 text covered by the predicted content tokens, not an estimate from characters or a global bytes/token ratio.
- Reversibility gate: for every document, concatenating the tokenizer's byte strings for content tokens and decoding must equal the original UTF-8 bytes; the tokenizer has no normalizer.
- Report ordinary NTP NLL alongside BPB so a tokenizer/fertility change cannot masquerade as a quality change.

### 4.4 Why current data cannot yet support audited BPB

`splot-pilot-blocks-v1` stores token IDs, document IDs, positions, valid masks and shifted labels. The manifest stores source-level content/raw hashes and token/valid/NTP counts. It does **not** store per-token byte lengths/offsets, per-document byte ranges, per-token family/language IDs, an explicit control mask or the tokenizer token-to-byte table in the evidence package. Aggregate bytes/token can be estimated (T0016: 3.905 UTF-8 bytes per full-corpus content token; 4.868 raw bytes per content token), but an auditable per-label BPB cannot be reconstructed from the committed evidence alone.

Required future schema `splot-pilot-blocks-v2`:

| Layer | Required field | Purpose |
|---|---|---|
| Tokenizer | `token_bytes[vocab]`, `special_token_ids`, `byte_level=true`, `normalizer=null` | exact token-to-byte coverage and reversibility |
| Block | `token_byte_lengths[T]`, `token_byte_offsets[T]` | per-label byte denominator |
| Block | `control_mask[T]` | BOS/EOS/PAD exclusion |
| Block | `source_family_ids[T]`, `language_ids[T]` | per-family/language aggregation |
| Block | `document_byte_start/end`, `document_sha256` | document-level audit |
| Manifest | shard `content_bytes`, family/language token and byte counts, `byte_coverage_sha256` | reconcile totals and detect gaps/overlaps |
| Evaluation | content-only NLL numerator and byte denominator; EOS NLL separate | no zero-byte division, no control-label contamination |

Audit gates: token byte lengths sum to each document's byte range; document ranges sum to shard content bytes; offsets are monotone with no gaps/overlaps; decode(byte strings) equals the original UTF-8 bytes for every document; all counts reconcile to the manifest.

## 5. Data order, seeds and training controls

- **Paired seeds:** planning default 42/43/44; confirmation 5–12 seeds, finalized after observed paired SD. The same seed controls data order and RNG across arms; initialization shapes differ by architecture but the seed pairing is preserved.
- **Data order:** deterministic shuffle by `seed + epoch`, identical across arms; one pass until the frozen training split is exhausted, then reshuffle. Record the exact batch indices and one-pass digest per run.
- **Precision:** BF16 autocast, FP32 master weights/optimizer state, deterministic algorithms, TF32 off on CUDA. FP8 and chunk64 are excluded from the primary quality comparison.
- **Batch/context:** B1, T512 for the primary matched-token pilot; T2048 is a separate context/cost regime, not silently mixed with T512.
- **Optimizer:** same Muon/AdamW partition and hyperparameters across arms; SPLOT's balance term is part of its objective and is declared, not hidden.
- **Checkpoint cadence:** every 16–32 updates, atomic write with identity hash; final checkpoint always saved. Do not overwrite or delete runs.
- **Validation cadence:** fixed token milestones (quarter-budget preferred) and final; the primary endpoint is the final checkpoint. Validation data are never trained on and never used to select the test result.
- **Test:** frozen, integrity-checked only until the final protocol is accepted; no test evaluation in the pilot or tuning.

## 6. Budgets and quality–cost curves

T0016 has only 46,204 valid train tokens (46,076 NTP labels) and two validation families. It is a feasibility snapshot, not a learning corpus. The planning range is 1–5M unique valid training tokens, with 5% validation and 5% test. Corpus byte/storage estimates are in `corpus-design.md`; compute estimates are analytical and labelled.

| Stage | Corpus | Positions/run | Updates/run | Runs | Total valid tokens (est.) | Wall time at T0017 anchor | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| S0 feasibility | 1M | 65,536 | 128 | 9 | 415,836 | 24.3 min | matched sanity; paired-SD estimate |
| S1 learning curve | 1M | 196,608 | 384 | 9 | 1,247,508 | 72.9 min | four milestones; fits ~90 min without overhead margin |
| S2 confirmation | 3–5M | 1,048,576 | 2,048 | 9 | 6,653,376 | 389 min | multi-window; full-pass fraction declared |
| S3 scaling (optional) | 5M | 5,242,880 | 10,240 | 9 | 33,266,880 | 1,945 min | one epoch/run; separate authorization |

The wall-time anchor is T0017's accepted 81.042 s / 64 updates = 1.266 s/update on a 43M FP8 A2 model. It is a conservative overhead anchor, not a BF16 A0 prediction. Analytical compute-only estimates are much smaller; small-model batch-1 overhead dominates. Any stage beyond S1 needs a new owner-authorized window.

Quality–cost curves:

- validation NLL/BPB versus consumed valid tokens;
- validation NLL/BPB versus analytical forward/training MACs;
- validation NLL/BPB versus measured wall time;
- checkpoint bytes and peak memory versus tokens;
- time-to-target: first checkpoint at or below a target NLL chosen from a baseline run; if unreached at the budget, report right-censored, never impute.

## 7. Uncertainty, margins and decision rules

Paired-seed precision planning (assumed paired SD; no observed variance exists):

| Seeds | 95% CI half-width, SD 0.02 | SD 0.05 | SD 0.10 | MDE80 approx., SD 0.05 |
|---:|---:|---:|---:|---:|
| 3 | 0.0497 | 0.1242 | 0.2484 | 0.1787 |
| 5 | 0.0248 | 0.0621 | 0.1241 | 0.0964 |
| 8 | 0.0167 | 0.0418 | 0.0836 | 0.0668 |
| 12 | 0.0127 | 0.0318 | 0.0635 | 0.0515 |
| 20 | 0.0094 | 0.0234 | 0.0468 | 0.0384 |

Consequences:

- n=3 cannot resolve a 0.05-nat margin at plausible SD 0.05; it is descriptive and useful mainly to estimate paired SD.
- n≈12 is the planning minimum for a 0.05-nat margin at SD 0.05; n≈30 is more realistic for a 0.03-nat advantage. These are approximations, not power guarantees.
- If observed paired SD is ≥0.10, the 0.05 margin is not credible within a small-seed pilot; reduce the claim, change the comparison, or use a more controlled synthetic task.

Family dependence:

- Two validation families cannot estimate family variance. Report per-family and per-language NLL, family ranges and a family bootstrap sensitivity analysis; do not report a population CI or equivalence claim.
- Future corpus design targets at least four validation families (preferably six) with family-level disjointness. Even then, family count limits generalization.

Proposed planning margins (not preregistered):

| Quantity | Noninferiority | Meaningful advantage | Notes |
|---|---:|---:|---|
| Final NLL, nats/label | +0.05 | −0.03 | primary S−C; secondary S−P with multiplicity caution |
| Content BPB, bits/byte | +0.01 | −0.005 | requires v2 byte audit |
| Capability probe | −2 pp vs dense | — | per probe; floor/ceiling rules apply |
| Analytical training cost | +10% | — | matched-forward-MAC regime; measured wall time reported separately |

Decision rule: noninferiority only if the two-sided 95% paired CI upper bound is below +0.05 nats; superiority only if it is below −0.03 nats. A point estimate inside the margin without an interval that meets it is inconclusive, not a win. No p-value or equivalence claim is made from T0016's two validation families.

## 8. Frozen capability probes

Probes are frozen before training, generated from held-out templates/entities/rules, never used for tuning, and audited against train/validation n-grams. The test probe set stays hidden until final evaluation.

| Probe | Chance | Instances | Scoring | Floor / ceiling | Contamination audit |
|---|---:|---:|---|---|---|
| Key–value retrieval (16 values, 4 distractors) | 1/16 | 256 | exact match | chance + 2 binomial SE; uninformative if all arms >0.90 | 13-gram overlap <0.1% |
| Compositional rule completion (8-symbol held-out rule) | 1/8 | 256 | exact match | chance + 2 binomial SE; uninformative if all arms >0.90 | held-out symbol mapping/templates |
| Document-order reconstruction (synthetic pairs) | 1/2 | 128 | pairwise order accuracy | chance + 2 binomial SE; uninformative if all arms >0.95 | no corpus text |

Guardrail: no probe may fall more than 2 percentage points below the dense comparator; if all arms are within 2 pp of chance, mark the probe uninformative rather than claiming a win. No universal capability or intelligence claim.

## 9. GDN chunk64 localization follow-up

T0025 rejected full-model chunk64 at T512: six hidden/logit/offset2 max-abs gates failed and top2 routing disagreed (max invocation 1.953%; token-weighted 34/4096 = 0.830%). The scan default remains. A separate diagnostic proposal, no production change:

1. Reproduce exact T0025 initialization, input, partition and scan control.
2. Compare scan vs chunk64 at FP32 recurrence with autocast disabled.
3. Insert BF16 casts one at a time at projection input, residual input and router input.
4. Record per-module max abs/relative L2, route logits, top1/top2 margins and rank flips.
5. Group by depth (prelude/core/coda), pool (0–3), pass (C1/C2) and token position.
6. Run fixed-route teacher forcing to separate routing divergence from contraction-order effects.

Gates: FP32 recurrence state/output within 1e-9 relative L2 and max abs; no T0025 threshold relaxation; report the first nonzero top2 flip and per-group counts, not inferred causes. Teacher-forced fixed-route results are diagnostic only and cannot establish exact equivalence. Any fix requires a separate reviewed IMP task.

## 10. Dependencies and unresolved decisions

Dependency graph (also machine-readable in `proposed-protocol.json`):

```text
D1 T0026 design review
├── D2 v2 byte-aligned schema + independent audit
├── D3 optimized dense SDPA comparator + correctness gates
├── D4 lawful corpus acquisition + rights/family registry
│   └── D5 frozen train-only tokenizer
├── D6 1M-token matched feasibility + paired-SD estimate
│   ├── D7 5M-token confirmation
│   └── D8 GDN chunk64 localization
└── D9 conditional-memory vs expert-capacity design (#86)
```

Unresolved: corpus sources/rights/families; whether 8/4/4 families suffice; tokenizer fit sample and one-tokenizer-across-scales policy; optimized dense implementation acceptance; final seed count; final margins; multi-window compute; v1→v2 migration and byte audit; whether #86 design belongs in the first follow-up.

## 11. Non-claims and limitations

- No model quality, superiority, equivalence, scaling or family-generalization claim.
- No observed variance, power or family-level confidence interval; precision tables are assumption-driven planning aids.
- No training, acquisition, tokenizer fit, test access, checkpoint access or dependency install in T0026.
- No optimized dense implementation or measured cost result; the current dense path is a correctness reference.
- No architecture adoption, final preregistration or task launch.
- T0027 is an independent descriptive pilot on the existing T0016 snapshot; its unreviewed outcomes are not used to tune this protocol.

## 12. Handover

Evidence in `experiments/results/T0026/attempt-001/`: this design, `corpus-design.md`, `proposed-protocol.json`, `analysis.json`, `protocol_analysis.py`, `source-refs.json`, environment, result, metrics and execution log. One INDEX row; PR and issue #108 carry the handover. No merge, closure or next-task launch.
