# Literature register / Rejestr literatury
As of 2026-11-26. Abstract metadata checked for the first seven records; full texts and implementation reproduction remain outstanding. Results are author-reported and not transferable to SPLOT without testing.

| Reference | Relevance | Status |
|---|---|---|
| [Floodgates](https://arxiv.org/abs/2611.33896) | shallow multi-token masked suffix drafts, up to 7.5x reported | Abstract confirmed |
| [LoopSpec](https://arxiv.org/abs/2609.17184) | shallow self-speculative pipelining, up to 6.83x reported | Abstract confirmed |
| [LoopMTP](https://arxiv.org/abs/2608.03624) | latent future-token supervision, up to 8.1% relative accuracy vs non-looped baseline | Abstract confirmed |
| [Towards Looped Models Done Right](https://arxiv.org/abs/2610.23036) | sandwich topology, injection, MoE recurrence | Abstract confirmed |
| [Qwen3.8-Next architecture](https://arxiv.org/abs/2608.30320) | 3:1 GDN/global attention, sparse experts | Abstract confirmed |
| [DeepSeek V4.1 Flash](https://arxiv.org/abs/2609.19969) | KV compression | Metadata confirmed |
| [LoopFill](https://arxiv.org/abs/2611.19649) | pipelined multi-token drafts | Abstract confirmed 2026-11-26; full HTML unavailable |
| [Huginn](https://arxiv.org/abs/2502.05171) | latent recurrent depth | Candidate, verify |
| [Looped Transformers](https://arxiv.org/abs/2502.17416) | latent reasoning | Candidate, verify |
| [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | recurrent token mixer | Candidate, verify |
| [Fine-grained MoE scaling](https://arxiv.org/abs/2402.07871) | expert scaling | Candidate, verify |
| [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) | attention and speculative decoding | Candidate, verify |
| [Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) | FP4, LatentMoE, MTP | Candidate, verify |
| [Strata](https://github.com/ro99/strata) | host expert placement | Candidate, verify |

Search priorities: Medusa, EAGLE-3, DeepSeek MTP, GLM/Kimi Delta Attention, all-depth training, exact recurrent speculation, expert cache-aware routing, quantization error propagation, FP8 on Ada, data licensing. For each record capture version, authors, method, hardware, baselines, training budget, limitations and reproducibility. Do not claim novelty before systematic prior-art review.

## Focused review v0.2
[English](reviews/native-mtp/en.md) | [Polski](reviews/native-mtp/pl.md). Primary-source review of native MTP and exact speculation; includes Medusa, Gloeckle et al., DeepSeek-V3, EAGLE-3 and Leviathan et al. Full-text coverage is partial, as documented in the review.

## Expert locality review v0.1
[EN](reviews/expert-locality/en.md) | [PL](reviews/expert-locality/pl.md). Cross-round expert reuse is established prior art (DuoMI); LOOM highlights expert selection collapse, and CCAMoE/SP-MoE overlap draft-aware caching. Evidence coverage and open retrievals in the review. No novelty established.
