# Project state / Stan projektu
Snapshot: 2026-11-26. **Live GitHub issues and PRs are authoritative.** This file is an orientation map, not a duplicate issue tracker.

## Read first
[AGENTS.md](AGENTS.md) · [Roadmap](ROADMAP.md) · [Issue index](research/BACKLOG.md) · [Decision log](research/decisions/README.md) · [Evidence rules](research/literature/EVIDENCE.md)

## Established
- PR #1 merged: foundation, bilingual documentation, RFC 0001 and 0005 drafts, research hypotheses and protocols.
- PR #12 merged: native MTP focused review and proposed exact self-speculation protocol. Issue #2 remains open: full literature coverage, architecture decisions and acceptance criteria incomplete.
- Issues #2–#11 are live workstreams; #13 establishes repository-first continuity.
- No implementation, training, verified model capability or measured SPLOT performance yet.

## Next work
- #6: focused prior-art review of cache-aware MoE routing, depth reuse and speculative verification. Establish a bounded, falsifiable question, not a novelty claim. Check live PRs for progress.
- #4 and #5: precise recurrence and GDN/GQA equations/cache semantics, then #3 reconcile parameter accounting and architecture. #8 data provenance can advance independently.
- #9: dense baseline once its minimal data/correctness contract is defined. Native MTP remains first-generation scope; validate baseline, fixed depth and exact speculation before optimization.
- #7/#10 precision and hardware measurement; #11 publication/license approval.

## Current unresolved decisions
Exact GDN dimensions and state, coda readout sharing, MTP loss/strategy, router objective and cache policy, data/tokenizer, numerical formats, training budget, CC BY 4.0 scope. All draft decisions subject to maintainer review. No assumed NASK/PLLuM relationship.

## Restart
Read AGENTS.md and query GitHub. Do not rely on this snapshot for current PR/issue state. Continue the highest-priority unblocked task unless maintainer chooses otherwise.
