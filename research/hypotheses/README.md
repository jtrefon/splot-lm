# Hypothesis register / Rejestr hipotez
All untested. Pre-register effect sizes, budget and metrics before execution.

| ID | Hypothesis | Falsification |
|---|---|---|
| H1 | Hybrid recurrence reduces long-context memory at matched quality | No improvement on memory/quality frontier |
| H2 | Sparse MoE increases capacity per executed FLOP | No improvement vs compute-matched dense model |
| H3 | Shared depth improves quality per resident byte | No gain at matched FLOPs/quality |
| H4 | Intermediate MTP improves draft acceptance and end-to-end throughput | No measured net speedup vs optimized baseline |
| H5 | Locality-aware routing reduces transfers without unacceptable quality loss | Transfer reduction absent or quality/load balance degrades |
| H6 | Loop-aware QAT mitigates depth-dependent quantization error | No robustness gain vs PTQ matched precision |
| H7 | Adaptive halting improves quality per compute | Fixed depth dominates at matched budgets |

Candidate papers: hardware-aware expert routing in recurrent-depth models; quantization error propagation; adaptive sparse computation under consumer constraints. Prior-art review is mandatory.

## Additional proposed hypotheses (architecture review #85)
H8: at matched total capacity and comparable active compute, causal conditional n-gram memory improves held-out quality and/or reduces measured bytes per committed token versus additional expert capacity. Falsified by absent Pareto gain or unacceptable collision/quality cost. H9: recurrence-aware tiered scheduling with deterministic row prefetch reduces bytes and decode latency against equal-budget exact LRU without target change. Falsified by RFC0004 joint transfer/latency gate or numerical/causal failure. H10: bounded residual/injection and selective mixer/router untying improve incremental depth quality and stability at disclosed equal-budget controls. Falsified if full sharing dominates or overhead exceeds gain. These are untested; preregister before observing confirmatory results. See [review](../architecture-review/2026-11-28-en.md).
