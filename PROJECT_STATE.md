# Project state / Stan projektu
Snapshot: 2026-11-27. **Live GitHub issues and PRs are authoritative.** This file is an orientation map, not a duplicate issue tracker.

## Read first
[AGENTS.md](AGENTS.md) · [Roadmap](ROADMAP.md) · [Issue index](research/BACKLOG.md) · [Decision log](research/decisions/README.md) · [Evidence rules](research/literature/EVIDENCE.md)

## Established
- PRs #1/#12/#14/#15/#16/#17/#18 merged: foundational RFCs, native MTP and locality reviews, continuity, recurrence and hybrid equation tests, architecture contract and working whitepaper. Research issues remain open where acceptance is incomplete.
- PR #19 merged: bilingual data contract, candidate source registry and synthetic data fixture. Actual source rights/pins, near-dedup, corpus and BPE remain gated (#8).
- PR #20 merged: trainable dense GQA reference, six CPU tests and FP32 overfit smoke, GPU protocol outstanding.
- PR #21 merged: fixed-depth all-GQA shared recurrent reference, seven CPU tests, six dense regression tests, 40-step synthetic A1 smoke. See [implementation](research/implementation/recurrent-reference.md) and [evidence](experiments/results/recurrent-reference-v0.1.md). No full hybrid/MoE model, real corpus training, trained release or measured hardware improvement.

- PR #22 merged: native MTP offset-2 head and A0/A1/A2 training; seven CPU tests and three synthetic smoke runs. [Contract](research/implementation/native-mtp.md), [evidence](experiments/results/native-mtp-v0.1.md).

- PR #23 merged: exact D1/D2 speculative reference: six CPU tests and seven MTP regressions passed; [contract](research/implementation/exact-speculation.md), [evidence](experiments/results/exact-speculation-v0.1.md). No speed claims.

- PR #24 merged: learned GDN/GQA hybrid dense reference: six CPU tests, 13 regressions, 40-step A2 smoke, short Nano forward/backward. [Contract](research/implementation/hybrid-gdn.md), [evidence](experiments/results/hybrid-gdn-v0.1.md). Dense FFN control.

- PR #25 merged: complete sparse hybrid SPLOTLM: exact Nano/Mini model inventories, six CPU MoE tests and 19 regressions, synthetic A2+balance training and short full Nano forward/backward. [Contract](research/implementation/sparse-moe.md), [evidence](experiments/results/sparse-moe-v0.1.md). Mini meta only.

- PR #26 merged: reproducible synthetic FP32 experiment harness; full actual SPLOT integration and CUDA smoke validated in T0001.
- PRs #29/#30 merged: REP v0.2, issue dialogue and structured evidence handoff. PRs #31/#32 merged: T0001 #27 accepted/closed, blocked setup attempt preserved and successful deterministic CUDA smoke (67 CPU tests including integration, 7/7 gates, 16 evidence hashes verified). See [index](experiments/results/INDEX.md). #3/#4/#5/#9 closed for original reference specification/implementation acceptance. GPU benchmarks and precision remain separate.

## Next work
- #10: T0002 (#28) Nano one-step FP32 CUDA resource validation specification in review; ready only after merge and explicit owner launch. Then review evidence, precision #7, benchmark/matched controls #10. #8 approved corpus/tokenizer gate before real training.
- #6: full sparse reference merged; full-text literature, locality objectives, chronological trace/cache simulation and measured PCIe/offload experiments remain open.
- #2: exact all-GQA reference merged; full literature, broader sampling/precision tests, shallow-cache reuse and empirical work remain open.
- #8: actual corpus source approvals and tokenizer remain required before real training.
- #11 publication/license approval pending. Dense standalone GPU smoke and matched controls under #10; GPU inference/cache, long-context, activation checkpointing under #7/#10.

## Current unresolved decisions
Experimental injection/stability choice, MTP strategy beyond initial offset-2, hybrid verifier rollback and shallow cache extraction, optimized GDN/MoE kernels, router objective and cache policy, data/tokenizer rights and actual pipeline, numerical formats, training budget, CC BY 4.0 scope. No assumed NASK/PLLuM relationship.

## Restart
Read AGENTS.md and query GitHub. Do not rely on this snapshot for current PR/issue state. Continue the highest-priority unblocked task unless maintainer chooses otherwise.

## Latest execution handover (2026-11-27)
T0002 #28 attempt-001 #34 reviewed: all three gates and seven evidence hashes pass; Nano 43,055,076 parameters, 840.7 MiB allocated / 986 MiB reserved, one FP32 step, 183 optimizer-state entries. Evidence PR pending merge; no closure yet. T0003 #35 specification in review: one full synthetic epoch (35 steps), state growth and repeated checkpoint integrity, no new architecture. Next parallel research: #7 BF16 autocast with FP32 master weights/Adam/GDN matrix/router and loss; require implementation PR, then matched FP32/BF16 tests; #8 data/tokenizer provenance; #10 realistic long-document fixture, inference and matched controls. No production training before corpus/tokenizer gate.

## Parallel execution handover (2026-11-27)
T0002 #28 and T0003 #35 closed after #34/#37 merged. T0003 35/35 batches, 2534 NTP labels, 198 Adam states, 1082 MiB peak reserved, exact T0002 step-1 reproduction; checkpoint local-only. T0004 #38 spec #39 merged, Mac CPU synthetic corpus reference only; production materialization Linux, see research/data/portability.md. T0005 #40 BF16 implementation spec in review; Linux independent of Mac. #7/#8/#10 remain open. Next precision validation, checkpoint cadence and actual contiguous-context tests; corpus rights/pins before real training.

## Latest parallel handover
T0004 #38/#45 and T0006 #48/#50 accepted/merged/closed; T0005 #40 and CI #46 previously closed. CI #47 active. Muon 210943940 vs AdamW 320848380 bytes state after one Nano step; exact pre-update/restore evidence, no superiority. Mac corrected synthetic oracle, swapped-key edge case carried into T0008. FP8-first priority: T0007 #51 Linux native GEMM feasibility; T0008 #52 Mac byte BPE reference; no actual corpus acquisition. #7/#8/#10/#42/#43 remain open. Source rights/pins and production tokenizer gate before substantive pretraining.

## Native FP8 handover
T0007 #51/#54 accepted and merged; 33/33 GEMMs native on Ada, max relative error 0.062166 vs FP32 /0.001681 vs dequant reference. Current probe includes host scalar scale copies, diagnostics and synchronization; not optimized performance. T0009 #56 integrates native autograd in selected projections, BF16/FP32 controls. T0008 #52/#55 correction requested: cap artifact expansion and codec inputs, canonical reload; original evidence retained. Mac next after acceptance: real-source provenance gate and portable tokenizer-to-shard packing pipeline, no real data until explicit rights and acquisition approval. #7/#8/#10/#42/#43 remain open.

## T0008/T0009 review handover
T0008 #52 accepted after #58 correction merged; original #55 already merged, attempt-001 retained. T0009 #56/#59 native 156-projection FP8 training demonstrated, paired one-step and exact smoke resume reported; correction required for finite flags, per-depth clipping/underflow/padding, report exit status and full-vector gradient/routing audit. No merge until reviewed correction. T0010 #60 Mac portable tokenized shards/packing bridge specification in review, independent of Linux. Actual corpus acquisition/production tokenizer still gated under #8. FP8+Muon next only after T0009 accepted.

## Current handover
T0009 #56 accepted/closed after #59 merge; corrected per-depth audits and exact same-mode resume, no saturation/nonfinite, Nano R4 full gradient relative L2 .342295. #61 T0010 #60 Mac spec merged, no implementation submission yet. T0011 #62 Linux native FP8+Muon correctness and three one-epoch synthetic comparisons in review. Broader #7/#8/#10/#42/#43 remain open. Data rights/pins and production tokenizer are the next research gate; no acquisition approved. Historical status sections above are records, live issues and latest handover authoritative.

## T0010/T0011 review handover
T0010 #60 accepted/closed, #64 merged. T0011 #62/#65 scientifically accepted; shared evidence INDEX conflict resolved without tested source change; merge only after renewed CI. FP8+Muon one synthetic epoch validation NLL 1.490035 (FP8 AdamW 3.499237, BF16 Muon 1.696633), single seed, no superiority. T0012 #66 actual contiguous-context paired resource qualification and cross-platform T0010 demo; depends on #65 merge/closure and reviewed spec. T0013 #67 Mac metadata-only source qualification and trusted shard boundary, no real corpus acquisition. Parent #7/#8/#10/#42/#43 remain open. Prioritize source approval and real pilot over further toy benchmarks.
