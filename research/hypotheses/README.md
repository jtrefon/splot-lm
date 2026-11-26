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
