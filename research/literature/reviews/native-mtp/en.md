# Native MTP and recurrent self-speculation — focused literature review
Date: 2026-11-26 | Draft v0.2 | Issue #2 | [Polski](pl.md)

## Research scope and method
Question: which training objectives and inference mechanisms should SPLOT implement from inception, given a shared hybrid GDN/GQA core, sparse experts and a single RTX 4090? Target is exact full-depth decoding plus improved sample/compute efficiency, not maximum reported speedup. This is a targeted narrative review, not a systematic review or independent replication. Search terms included recurrent depth, looped Transformer, self-speculation, MTP, masked suffix drafting, and Medusa/EAGLE/DeepSeek. Primary sources were preferred. Full-text sections consulted for Floodgates v1, LoopSpec v1, LoopMTP v1 and DeepSeek-V3; abstracts/proceedings consulted for Medusa, Gloeckle et al., EAGLE-3, Leviathan et al. and LoopFill. No peer-review status assumed for 2026 preprints. LoopFill full HTML was unavailable; its abstract is now independently located. No code execution or reproduction performed.

## Evidence matrix
| Work | Training mechanism | Inference mechanism | SPLOT assessment |
|---|---|---|---|
| [Leviathan et al. (2023)](https://proceedings.mlr.press/v202/leviathan23a.html) | None required | Exact speculative sampling with conditional proposal | Foundational correctness reference |
| [Gloeckle et al. (2024)](https://proceedings.mlr.press/v235/gloeckle24a.html) | Independent future-token heads on shared trunk | Self-speculation | Native MTP control; reported gains primarily at larger scales, not a guarantee for Mini |
| [Medusa (2024)](https://proceedings.mlr.press/v235/cai24b.html) | Added future-token heads; frozen or joint tuning | Tree proposals | Useful parallel head control; tree verification is difficult with recurrent state |
| [DeepSeek-V3 (2024)](https://arxiv.org/html/2412.19437) §2.2 | Sequential auxiliary modules conditioned on preceding future token | Optional draft use | Conditioned MTP control; added sequential cost |
| [EAGLE-3 (2025)](https://arxiv.org/abs/2503.01840) | Rollout-aligned autoregressive drafter using target features | Draft tree | External comparison; not first implementation |
| [LoopMTP (2026)](https://arxiv.org/html/2608.03624v1) §3 | Cosine alignment to detached future token embeddings, aggregation, ponder loss | Not itself an exact speculative decoder | Low-cost latent objective candidate; changes other components too, isolate alignment |
| [LoopSpec (2026)](https://arxiv.org/html/2609.17184v1) §4–6 | Training-free method applied to trained looped models | Early-depth proposal, pipelining and optional second proposal | Start with serial shallow-loop oracle; pipeline later |
| [Floodgates (2026)](https://arxiv.org/html/2611.33896v1) §3–4, appendices | Low-rank adaptation of a shallow loop on mask slots; all-depth pretraining helps | Multi-token draft conditioned on final-loop prefix cache | Primary advanced candidate; requires adapted draft state, not simple cache substitution |
| [LoopFill (2026)](https://arxiv.org/abs/2611.19649) | Training-free (abstract) | Suffix lookup + shallow-loop draft tree | Later comparator; full method pending review |

## Findings and limitations
The training and inference questions are distinct. Predictive supervision can improve hidden representations without furnishing a calibrated proposal distribution. LoopMTP uses cosine rather than a vocabulary projection; its reported benefit is not evidence of speculative acceleration. Conversely, LoopSpec exploits existing shallow predictions without adding a training objective. Floodgates builds on all-depth supervision and adapts shallow execution to consume deep context. Its experiments include 4-loop and 32-loop models; the depth advantage is not directly comparable to SPLOT's two prelude and two coda blocks or its much smaller parameter budget. Authors report up to 7.5x for Floodgates, up to 6.83x for LoopSpec and LoopFill abstract averages of 5.85x on Ouro-2.6B and 5.13x on Huginn-3.5B. These are different hardware, tasks, target depths and implementations; do not rank methods by headline numbers. Medusa's typical acceptance heuristic must not be conflated with exact sampling. No source here establishes performance on hybrid GDN/MoE.

## Provisional decision
Train an all-depth next-token readout and one additional future-token objective from the first experimental generation; retain final-depth NTP-only and all-depth-only controls. Treat latent alignment as an independent low-cost intervention. Begin exact self-speculation using the shallow fixed-loop model as a serial conditional proposer, then experiment with adapted masked slots. Use a shared vocabulary matrix and initially small horizon K=2; increase after profiling. Medusa/independent CE and sequential conditioned MTP remain useful controls rather than production defaults.

A proposed extension is to train a draft adapter that reads final-pass GQA cache and GDN state without mutating target state, alongside all-depth and future-token supervision. This is not an established novelty claim. Prior-art search must cover deep-prefix conditioning, GDN speculation, state rollback, and cache-aware expert routing. Full target verification must be unchanged by the draft path.

## Explicit SPLOT research questions
- Does intermediate all-depth supervision improve shallow proposal acceptance at fixed full-depth quality?
- Does adding latent alignment or direct future-token CE improve acceptance per unit of training FLOP?
- Is deep-prefix conditioning useful in a hybrid model after adaptation, and does extra GDN state restoration eliminate any speedup?
- Does expert reuse across draft and verify improve wall time under constrained residency without degrading quality?

See [design decision and protocol](../../rfcs/0005-native-mtp/decision-v0.2.md). Review status: partial full-text review completed; LoopFill full text, EAGLE-3 details, Medusa details, Gloeckle memory-efficient loss implementation, broader current literature and exact implementation licenses remain outstanding.
