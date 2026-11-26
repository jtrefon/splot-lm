# Expert locality, recurrent depth and speculative verification
Focused review v0.1 | 2026-11-26 | #6 | [Polski](pl.md)

## Scope and evidence
Search targeted expert offloading, cache-aware routing, training-time locality, speculative prefetch and cross-round expert reuse. Primary proceedings/abstracts and selected HTML sections checked on 2026-11-26; not systematic, no reproduction. This review supersedes the initial suggestion that cross-loop routing locality alone might be original. In particular DuoMI was discovered during this review.

| Source | Coverage | Relevance |
|---|---|---|
| [Liang et al., Not All Models Suit Expert Offloading, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6a7a2238da2b755e6dc1dbe278c60c2a-Abstract-Conference.html) | Proceedings abstract | SRP/SCH quantify locality; local balancing and consistency can conflict; global balance may coexist. Shared experts affect routing space. |
| [Zhu et al., ReMoE, ICML 2026](https://proceedings.mlr.press/v306/zhu26ab.html) | Proceedings abstract | Router fine-tuning for adjacent-token reuse; reports 8.4% throughput improvement with GPU–CPU offload and 1.77–1.99x decode on Jetson in its settings. |
| [Suram, Cacheable by Design?, arXiv:2608.18261](https://arxiv.org/abs/2608.18261) | Abstract and selected sections 4, 6–9 | 137M multi-domain models reduced misses but failed preregistered <=1% relative perplexity gate; 340M follow-up did not remove the tradeoff. Cacheability alone is insufficient. |
| [Wu et al., Cache-Aware Joint Router Adaptation, arXiv:2609.04895](https://arxiv.org/abs/2609.04895) | Abstract and introduction | Joint backbone and auxiliary cache routing, temporal retention and causal prefetch; native top-k retained. Separate traffic accounting matters. |
| [SP-MoE, arXiv:2510.10302](https://arxiv.org/abs/2510.10302) | Abstract and implementation excerpts | Draft/target structural correspondence for predictive expert staging; overlaps any broad claim of speculation-aware prefetch. |
| [CCAMoE, Zhang et al., arXiv:2611.28550](https://arxiv.org/abs/2611.28550) | Primary metadata verified; search excerpt describes cached-expert drafting, full method not reviewed | Very close candidate prior art for GPU-resident expert self-drafting and cache preparation; mandatory full review. |
| [DuoMI, arXiv:2611.33952](https://arxiv.org/abs/2611.33952) | Abstract, selected text including Appendix A | Direct cross-round expert reuse prior art: marginal MI and conditional MI interval penalties; reported +1.74 percentage points on 15-task mean on paired-layer OLMoE. Not evidence of SPLOT speedup. |
| [He et al., LOOM, arXiv:2610.01153](https://arxiv.org/abs/2610.01153) | Abstract and introduction | Identifies repeated expert choices across loops as collapse; per-loop routers, input injection and residual scaling seek diversity. |
| [Towards Looped Models Done Right, arXiv:2610.23036](https://arxiv.org/abs/2610.23036) | Search abstract | Topology and expert-routing balance; revisit full text in #4. |

Candidate leads not verified at primary text: Sticky Routing (arXiv:2607.08780; primary fetch failed), CECAR (arXiv:2611.20119; primary fetch failed), Cache-Conditional Experts, ExpertFlow, DoMoE, MoEUT, Foil, Sparse Layers are Critical to Scaling Looped Language Models. Resolve metadata/full text before using as evidence. Search scope did not cover patents or exhaustive citation chaining.

## Consequence for SPLOT
1. Adjacent-token locality, cross-round expert consistency and draft-directed prefetch are not original in isolation. DuoMI and LOOM make cross-round *diversity* an explicit competing requirement; optimizing overlap to 100% is unjustified.
2. Keep the target model's routing unchanged during initial cache optimization; schedule/retain/prefetch expert weights only. A cache-biased target router changes the target function and must be a separate quality experiment. An approximate cache-restricted drafter may still be corrected by the unchanged target if the actual conditional proposal is available.
3. The narrower question is whether a depth- and draft-aware residency policy, with bounded routing diversity and hybrid recurrent state rollback, improves quality-adjusted wall time vs strong cache/speculation baselines. Novelty is not established. Distinguish expert weight cache (immutable weights) from per-sequence KV/GDN states (transactional). Rejected drafts may warm the weight cache without modifying target activations; measure pollution.
4. Start trace measurement and policy replay before training any locality loss. Report hit rate, bytes transferred, false prefetches, deadline misses, cache occupancy and actual latency separately. Small fully resident models cannot demonstrate PCIe benefits; simulate a constrained expert pool and later measure forced offloading under fixed byte budgets, honestly label each.

See [RFC 0004 proposal](../../rfcs/0004-expert-routing/en.md). Remaining: complete full-text review especially DuoMI/CCAMoE/LOOM, verify Sticky/CECAR, inspect code/licenses, citation chaining, preregister thresholds and run experiments. No SPLOT measurements yet.
