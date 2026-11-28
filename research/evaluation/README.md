# Quality–cost evaluation and architecture decision framework v0.1
2026-11-28 · [Charter](../CHARTER.md) · #85/#89/#90. Operational protocol; numerical effect sizes, budgets and noninferiority margins must be preregistered per experiment before outcomes are observed. No model superiority or validated scaling claim.

## Comparative success
Compare a candidate with a credible, optimized conventional baseline and with its immediate parent/ablation where applicable. Establish at least one of (1) higher quality at equivalent cost, (2) equivalent quality at lower cost, (3) both. Specify the primary cost and quality metrics, practical effect threshold, equivalence/noninferiority margin, uncertainty method, and guardrails in advance. An absence of statistical significance is not equivalence; require the interval to meet the declared margin. A trade-off outside the guardrails is an inconclusive or negative result, not a win. No weighted universal IQ/resource score.

## Minimum measurements
| Dimension | Measures and constraints |
|---|---|
| Language modeling | Held-out token-weighted NTP NLL (nats/valid label), PPL only with identical tokenizer; bits/byte = summed NLL / (ln(2) × covered original UTF-8 bytes), explicitly reconcile byte coverage, BOS/EOS and label masks; per-language and aggregate |
| Learning cost | Fixed valid NTP token budget and separate matched analytical contraction/estimated training FLOP regime; distinguish forward MAC estimates from actual training work; report updates, tokens, wall time and time to preregistered validation target (unreached = censored) |
| Capability guardrails | Scale-appropriate held-out tasks (e.g. compositional retrieval, multi-step synthetic reasoning with disjoint generators/instances); check chance and floor/ceiling; no universal intelligence claim |
| Inference | Identical prompts/checkpoint/precision/decoding semantics, committed tokens/sec, TTFT and decode time, repeated paired warm/cold measurements as relevant, exact greedy/distribution checks for exact transformations |
| Memory/storage | Peak allocated and reserved GPU bytes separately, host RSS, actual serialized bytes and tier traffic when relevant; energy only when actually measured, with boundary and instrument |
| Validity | Seeds, model/config/data/tokenizer/source hashes, document/work-family split, train-only fitting, validation vs untouched test, equal tuning budgets, failures and uncertainty |

## Comparator and experiment design
Existing dense GQA reference is a correctness control, not a matched or optimized performance comparator. Audit actual code and configs; design dense GQA/RoPE/RMSNorm/SwiGLU tied-vocab baseline and stronger optimized path with equivalent causal masking, data, precision and reasonable equal tuning budgets. Maintain separate iso-parameter, iso-compute and fixed-token regimes; disclose achieved differences rather than pretending all can be matched simultaneously. Compare same tokenizer and actual vocab; existing T0017 V32768 is unchanged, later matched training should use actual vocab if feasible. Report physical/active parameters, logical depth, MTP auxiliary work, optimizer and total memory. Do not conflate FP8 vs BF16, optimizer effects or runtime implementation with architecture. Use multiple independent training seeds (initial planning minimum three; power/precision assessed before confirmation), paired data ordering and validation intervals. Hold test until frozen; preserve all runs. Optimize both paths before performance superiority claims.

## Adoption gate
For each candidate preregister question, comparator, exact intervention, expected benefit, possible regressions, cost/quality margins, data/token budget, seeds, stopping rule, implementation correctness gates and required reproducible evidence. Stage: causal/numerical correctness → resource characterization → matched learning/capability ablation → cross-scale validation. Exact inference optimizations may demonstrate output/distribution-preserving cost reduction independently; architectural changes must earn quality/cost evidence. Positive feasibility permits further investigation, not unconditional adoption. Reject, revise or defer if evidence is inadequate; preserve negative results. Recurrence, MoE, n-gram memory, hybrid attention, precision, MTP and residency are candidates, not the charter.

## Immediate programme
T0017/#77 and T0018/#81 closed; T0017 bounded feasibility accepted with deviations. [T0019 corrected design](T0019-acceptance.md) and [T0020 cost review](T0020-acceptance.md) accepted. T0021 dense NTP harness and T0022 BF16 backward attribution next, no matched training authorization. Then review #85/#86/#87/#6 with evidence and select bounded ablations/optimization tasks.
