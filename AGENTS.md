# SPLOT — repository-first instructions for contributors and AI agents
Scope: entire repository. Maintainer: jtrefon. English canonical; Polish translations welcome. Read this file first in any new session. This is project guidance, not a substitute for user authorization or tool permissions.

## Resume from GitHub, never chat memory
1. Inspect current main HEAD, open/merged PRs and open issues via GitHub API or git. Do not assume previous sessions' state.
2. Read PROJECT_STATE.md, research/BACKLOG.md, ROADMAP.md, relevant RFCs and decision records. Check issue comments and linked PRs; compare branch heads to main. If the state file is stale, reconcile against live GitHub.
3. Pick the highest-priority unblocked issue (or the maintainer's explicit priority). Identify acceptance criteria, dependencies, evidence gaps and exact next action. Avoid duplicate issues/PRs/branches.
4. Work on a named branch (research/<topic> or feat/<topic>) from current main unless explicitly stacked on a reviewed dependency. Preserve unrelated changes. Use issue number in commits and PRs. No merge, force push or main write without explicit maintainer approval.
5. Separate hypotheses, cited literature, design decisions, implementation, and measured results. Never report a search snippet as full-text review; verify title, authors, version/date, link, relevant section, limitations. Treat prior summaries as leads, not evidence. Store source register and BibTeX; no copyrighted paper dumps.
6. Research documents in English plus Polish for major RFCs/reviews. English canonical. Note translation status and revision. Code, configs, manifests and scientific data are canonical in English.
7. Before ending each work unit, commit artifacts, open/update PR, link it from the issue, state completed/remaining acceptance criteria, blockers, validation status and next action. Update PROJECT_STATE.md when direction changes; live issue/PR state always wins. Never claim work uploaded, verified or run without tool evidence.
8. For experiments, record exact commit, config/data hashes, seeds, environment, hardware, metrics and result locations. Do not publish secrets, personal chats, private data, unsupported affiliation or novelty claims. Negative results remain in history.

## Definition of done
Issue closes only after acceptance criteria satisfied, evidence/test results linked and relevant PR merged with maintainer approval. Draft, simulated, external-reported and measured findings must be distinguishable. Unresolved work stays open. A PR may advance an issue without closing it. No model performance claims until measured. If tool access is blocked, report it and preserve deliverables without pretending remote state changed.

## Restart prompt
"Continue SPLOT from https://github.com/jtrefon/splot-lm. Read AGENTS.md, PROJECT_STATE.md, live open issues and PRs; reconcile status against main, select the next unblocked task, work in a branch, open/update a PR, and record issue progress. Do not merge without my approval. Use the repo as the sole project source of truth; no prior chat required."

## Priority
Correctness and reproducibility before performance. Text-only, native MTP first generation, fixed depth before adaptive. RTX 4090 is initial target, not a guaranteed training budget. Preserve Apache-2.0; proposed research license subject to approval. Independent project; no affiliation with PLLuM/NASK implied.
