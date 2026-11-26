# Research roadmap / Plan badań

| Phase | Deliverables | Gate |
|---|---|---|
| 0 Foundation / Podstawy | Bilingual repository, licensing, bibliography, hypotheses | Maintainer review |
| 1 Specification / Specyfikacja | RFCs 0001–0006, math, shapes, exact parameter accounting, MTP selection | Literature review and causal semantics |
| 2 Reference implementation / Implementacja | Dense baseline, fixed recurrence, native MTP, slow oracle | Numerical and cache tests |
| 3 Ablations / Ablacje | GDN, MoE, combined architecture, speculative verification | Matched compute, tokens and parameter comparisons |
| 4 Original research / Badania | Hardware locality, adaptive depth, FP8 and QAT | Measured quality–latency–memory frontier |
| 5 Publication / Publikacja | Reproducible manifests, checkpoints, papers, DOI releases | Independent replication and provenance |

Vision/audio, sparse indexing, latent experts, external n-gram tables and optimized CUDA/Triton kernels are deferred. No delivery dates or performance promises. Start with Nano/Mini on RTX 4090, scale only after evidence.
