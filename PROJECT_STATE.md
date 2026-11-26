# Project state / Stan projektu
Snapshot: 2026-11-26. **Live GitHub issues and PRs are authoritative.** This file is an orientation map, not a duplicate issue tracker.

## Read first
[AGENTS.md](AGENTS.md) · [Roadmap](ROADMAP.md) · [Issue index](research/BACKLOG.md) · [Decision log](research/decisions/README.md) · [Evidence rules](research/literature/EVIDENCE.md)

## Established
- PR #1 merged: foundation, bilingual documentation, RFC 0001 and 0005 drafts, research hypotheses and protocols.
- PR #12 merged: native MTP focused review and proposed exact self-speculation protocol. Issue #2 remains open: full literature coverage, architecture decisions and acceptance criteria incomplete.
- PR #14 merged: repository-first handover; PR #15 merged: expert-locality review and draft RFC 0004 (#6 remains open).
- Issues #2–#11 are live workstreams; #13 continuity is now in main.
- No actual model implementation, training, verified capability or measured SPLOT performance yet. Toy scheduling tests are not model tests.

## Next work
- #4: PR #16 merged, RFC 0002 draft and five passing toy CPU semantics tests; actual mixer tests remain open.
- #5: RFC 0003 proposed shapes, equations, rollback and six passing CPU equation tests on review branch. Actual mixer integration remains open.
- #3 next: reconcile full architecture and exact parameter/FLOP accounting after RFC 0003 review. #8 data provenance can advance independently.
- #6: full-text priority sources and trace-based cache experiments remain open; see RFC 0004.
- #9: dense baseline once its minimal data/correctness contract is defined. Native MTP remains first-generation scope; validate baseline, fixed depth and exact speculation before optimization.
- #7/#10 precision and hardware measurement; #11 publication/license approval.

## Current unresolved decisions
GDN dimensions and state proposed (RFC 0003), coda readout sharing, MTP loss/strategy, router objective and cache policy, data/tokenizer, numerical formats, training budget, CC BY 4.0 scope. All draft decisions subject to maintainer review. No assumed NASK/PLLuM relationship.

## Restart
Read AGENTS.md and query GitHub. Do not rely on this snapshot for current PR/issue state. Continue the highest-priority unblocked task unless maintainer chooses otherwise.
