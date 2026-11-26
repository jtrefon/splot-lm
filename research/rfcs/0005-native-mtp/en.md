# RFC 0005 — Native multi-token prediction and self-speculation
Draft v0.1 | [Polski](pl.md)

**Update:** [v0.2 decision and protocol](decision-v0.2.md) proposes revised sequencing; see [literature review](../../literature/reviews/native-mtp/en.md). LoopFill abstract now verified; full text pending.

MTP is a first-generation requirement, not a bolt-on afterthought. Distinguish auxiliary future-token training, latent supervision, draft generation and exact target verification. A latent MTP objective does not automatically produce a useful multi-token drafter.

| Candidate | Mechanism | Role |
|---|---|---|
| Medusa | Parallel future-token heads/tree proposals | Simple comparator |
| DeepSeek MTP | Sequential conditioned future-token prediction | Strong training baseline |
| EAGLE-3 | Feature/token autoregressive drafter | External comparator |
| LoopMTP | Future-embedding alignment across loop depths | Latent training candidate |
| LoopSpec | Training-free shallow-depth draft and pipelining | Inference baseline |
| Floodgates | Masked suffix parallel drafting at shallow depth via adaptation | Primary inference candidate |
| LoopFill | Pipelined multi-token loop drafts | Candidate; reference unverified |

## Proposed implementation sequence
M0: final-depth NTP only, train shared core at fixed R. M1: tied LM head applied to intermediate normalized representations with causal future-token CE, offsets k=1..K (K=2 initially); compare per-depth next-token supervision separately. Do not give future tokens to the backbone; mask out padding, document boundaries and unavailable offsets. M2: sequential conditioned MTP baseline with teacher-forced previous future token only within auxiliary head; measure exposure bias and separate train/test paths. M3: latent alignment to future embedding (compare stop-gradient and learned projections); test against M1 at matched compute. M4: shallow-core draft of multiple tokens in a masked suffix inspired by Floodgates; train with a distinct mask policy, fixed R target, and no access to ground-truth suffix. M5: exact full-depth verifier; benchmark one-token shallow self-speculation, sequential and parallel drafts. Read full papers before selecting topology; candidates above are not implementation specifications.

Loss example: L_NTP + lambda * mean_(r,k,t valid) CE(H_(r,k)(h_r,t), x_(t+k)). Weights, head sharing and horizon are ablation axes; avoid accidental future leakage through packing. For sampled lossless verification with proposal q and target p, accept proposed token y with probability min(1,p(y)/q(y)); on rejection sample normalized positive part of p-q. For a parallel draft, define valid conditional proposal distributions or use a proven alternative verifier. Greedy accept only if candidate equals target argmax with consistent tie rules. Stop at first rejection and restore all logical-pass GDN/KV states. Exactness refers to the fixed target distribution, not necessarily identical sampled sequences.

## Evaluation
Acceptance per offset, accepted tokens per verification, draft latency, verification latency, cache snapshot/rollback cost, total tok/s, TTFT, energy/token, VRAM, quality, training overhead, and full-depth distribution equivalence. Compare equal decoding settings, output length, context and batch; greedy exact match and statistical sampling tests. Report speedup against optimized non-speculative reference, not Python-only strawman. Test 1–4 loops, draft depth and horizon, MoE cache residency, quantized and BF16 paths. Reject designs that improve nominal acceptance but reduce end-to-end throughput.

## Evidence status
LoopMTP, LoopSpec and Floodgates abstract metadata confirmed 2026-11-26. Floodgates reports up to 7.5x in its own experiments; LoopSpec up to 6.83x; LoopMTP reports up to 8.1% relative average accuracy improvement against a non-looped baseline. These are not SPLOT results. LoopFill abstract now verified; full text pending. Full-text review and reproducibility comparison are outstanding.
