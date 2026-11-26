# Project state / Stan projektu
Snapshot: 2026-11-26. **Live GitHub issues and PRs are authoritative.** This file is an orientation map, not a duplicate issue tracker.

## Read first
[AGENTS.md](AGENTS.md) · [Roadmap](ROADMAP.md) · [Issue index](research/BACKLOG.md) · [Decision log](research/decisions/README.md) · [Evidence rules](research/literature/EVIDENCE.md)

## Established
- PRs #1/#12/#14/#15/#16/#17/#18 merged: foundational RFCs, native MTP and locality reviews, continuity, recurrence and hybrid equation tests, architecture contract and working whitepaper. Research issues remain open where acceptance is incomplete.
- PR #19 merged: bilingual data contract, candidate source registry and synthetic data fixture. Actual source rights/pins, near-dedup, corpus and BPE remain gated (#8).
- PR #20 merged: trainable dense GQA reference, six CPU tests and FP32 overfit smoke, GPU protocol outstanding.
- This review branch implements fixed-depth all-GQA shared recurrent reference, seven new CPU tests, six dense regression tests, 40-step synthetic A1 smoke. See [implementation](research/implementation/recurrent-reference.md) and [evidence](experiments/results/recurrent-reference-v0.1.md). No full hybrid/MoE model, real corpus training, trained release or measured hardware improvement.

## Next work
- #2 next: implement shared offset-2 head and A0/A1/A2 masked objectives, verify exact fixture labels and gradient flow, then separate exact speculative sampler and state transactions.
- #4: recurrent all-GQA reference in review; checkpointing, full GDN integration and other acceptance tests outstanding.
- #5: GDN/GQA hybrid mixer integration and rollback after MTP reference. #6: MoE and locality research after correctness; full-text priority sources outstanding.
- #9: synthetic fixture integration now exercised in recurrent reference; GPU smoke still outstanding. #8: actual corpus source approvals and tokenizer remain required before real training.
- #3 reconcile actual full-candidate parameter counts when GDN/MoE/MTP exist; #7/#10 precision and hardware measurement; #11 publication/license approval.

## Current unresolved decisions
Experimental injection/stability choice, MTP strategy beyond initial offset-2, exact verifier and shallow cache extraction, GDN/MoE integration, router objective and cache policy, data/tokenizer rights and actual pipeline, numerical formats, training budget, CC BY 4.0 scope. No assumed NASK/PLLuM relationship.

## Restart
Read AGENTS.md and query GitHub. Do not rely on this snapshot for current PR/issue state. Continue the highest-priority unblocked task unless maintainer chooses otherwise.
