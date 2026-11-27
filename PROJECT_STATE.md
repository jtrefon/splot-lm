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

## Next work
- #10: synthetic FP32 harness, executable candidate configs, checkpoint/resume and validation on review branch; five toy harness tests pass, actual SPLOT CLI/integration pending (local checkout blocked by DNS). Next execute integration and add inference benchmarks/budget matching. #7 precision and activation memory policy in parallel; #8 approved corpus/tokenizer gate before real training.
- #6: full sparse reference in review; full-text literature, locality objectives, chronological trace/cache simulation and measured PCIe/offload experiments remain open.
- #5: learned hybrid implementation merged; GPU, precision, long context, kernel and empirical tests remain.
- #2: exact all-GQA reference merged; full literature, broader sampling/precision tests, shallow-cache reuse and empirical work remain open.
- #4: recurrent all-GQA and GDN integration merged; activation checkpointing and GPU acceptance outstanding.
- #9: synthetic fixture integration now exercised in recurrent reference; GPU smoke still outstanding. #8: actual corpus source approvals and tokenizer remain required before real training.
- #3 exact candidate parameter inventories now verified; matched controls and empirical work remain; #7/#10 precision and hardware measurement; #11 publication/license approval.

## Current unresolved decisions
Experimental injection/stability choice, MTP strategy beyond initial offset-2, hybrid verifier rollback and shallow cache extraction, optimized GDN/MoE kernels, router objective and cache policy, data/tokenizer rights and actual pipeline, numerical formats, training budget, CC BY 4.0 scope. No assumed NASK/PLLuM relationship.

## Restart
Read AGENTS.md and query GitHub. Do not rely on this snapshot for current PR/issue state. Continue the highest-priority unblocked task unless maintainer chooses otherwise.
