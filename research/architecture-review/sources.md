# Architecture review source register — accessed 2026-11-28
Primary sources unless labelled engineering report. Coverage is selected sections/abstracts, not a full systematic review. External results are author-reported. See [review](2026-11-28-en.md).

| Source | Date/version | Inspected material / relevance |
|---|---|---|
| Qwen Team, [On the Design of Qwen3.8-Next](https://arxiv.org/html/2608.30320v1), [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 2026-08-31 v1 | HTML abstract, selected §§2.1.1, 2.3 Tables 7–9 and 3.1 excerpts, official model card; hybrid GDN/QSA, n-grams, gated residual, Muon; loss/downstream divergence at fixed budget |
| Cheng et al., [Conditional Memory via Scalable Lookup](https://arxiv.org/html/2601.07372v1), [code](https://github.com/deepseek-ai/Engram) | 2026-01-12 v1 | §§2.2–2.5, 3 excerpt; hashing, gating, prefetch, allocation |
| DeepSeek-AI, [V4.1-Flash](https://arxiv.org/abs/2609.19969), [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 2026-09-17 v1 | abstract metadata and card; no full report claim |
| Z.ai, [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 2026 release cards | architecture descriptions, not full independent evaluation |
| Thinking Machines Lab, [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/), [card](https://thinkingmachines.ai/model-card/inkling-2-mini/) | 2026-11-20 | architecture/training section, loop position, residual message, pruning/quantization |
| He et al., [LOOM](https://arxiv.org/html/2610.01153v1) | 2026-10-01 v1 | abstract, §§3.3/4.3.2 excerpt and method section map; residual, routing and gradient truncation |
| Wang et al., [Foil](https://arxiv.org/html/2609.35751v1) | 2026-09-28 v1 | abstract and §2.2 resource identities; flatten experts, untie attention |
| Qiu et al., [MoRE](https://arxiv.org/abs/2609.18176) | 2026-09-16 | abstract, grouped shared pools |
| Hua et al., [Meta Expert](https://arxiv.org/abs/2610.27598) | 2026-10-20 | abstract, cross-layer sharing |
| Lee et al., [Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165) | 2026-05-09 | abstract, routing divergence and early exit |
| Kang et al., [DuoMI](https://arxiv.org/abs/2611.33952) | 2026-11-25 | abstract, intro and selected theory/related work; no reproduction |
| [Cacheable by Design?](https://arxiv.org/abs/2608.18261) | 2026-08-18 | abstract, negative preregistered quality gate |
| [Floodgates](https://arxiv.org/abs/2611.33896) | 2026-11-25 | abstract and intro excerpt, shallow masked suffix |
| [LoopSpec](https://arxiv.org/abs/2609.17184) | 2026-09-15 | abstract, pipeline |
| Shomali et al., [LoopMTP](https://arxiv.org/abs/2608.03624) | 2026-08-04 | abstract, selected limitations/appendix excerpts |
| NVIDIA, [Nemotron 3.5 Super/Lightning](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Super-and-Lightning/) | 2026-10-14 | release/technical summary, MTP post-training |
| Elango et al., [LatentMoE](https://arxiv.org/abs/2601.18089), [technical explanation](https://research.nvidia.com/labs/nemotron/LatentMoE/) | 2026-01-26 | abstract and projection description |
| [Strata docs](https://strata.amperor.ai/docs/) | live | runtime vendor docs; not peer-reviewed |
| [HF expert streaming report](https://huggingface.co/blog/moe-expert-streaming) | Nov 2026 | engineering report sections on other engines, 32GB M5 Qwen, measured setup and caveats; no independent replicate |
| Liang et al., [SSD-LLaMA](https://arxiv.org/abs/2609.18110) | 2026-09-16 | abstract, three tiers and exact selected expert execution |
| Kim et al., [FlashMoE](https://arxiv.org/abs/2601.17063) | 2026-01-22 | abstract, cache replacement |
| [vLLM Engram](https://docs.vllm.ai/en/stable/features/engram/) | live | host-memory backend documentation excerpt |
| Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | abstract, exact attention IO |
| Wijmans et al., [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) | 2024 | abstract; distinguish exact tiled CE and gradient cutoff |
| [GDN official](https://github.com/NVlabs/GatedDeltaNet), [cuDNN linear attention](https://docs.nvidia.com/cudnn/latest/operations/LinearAttention) | live | kernel support descriptions; Qwen report §2.1.1 also lists FlashQLA fused kernels; no verified local compatibility |
| [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | 2026-05 | abstract, erase/write decoupling |
| [mHC](https://arxiv.org/abs/2512.24880), [Attention Residuals](https://arxiv.org/abs/2603.15031) | 2025-12 / 2026-03 | abstracts, residual alternatives |
| Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | 2025-02-24 | abstract, update scale/decay and efficiency |
