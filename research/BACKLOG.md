# Research backlog / Plan prac
The founding backlog is now tracked in live GitHub issues. Priorities are proposed, not deadlines. P0 architecture and correctness; P1 extensions; P2 publication.

| Issue | Workstream | Priority |
|---|---|---|
| [#2](https://github.com/jtrefon/splot-lm/issues/2) | Native MTP and recurrent self-speculation | P0 |
| [#3](https://github.com/jtrefon/splot-lm/issues/3) | RFC 0001: architecture and parameter accounting | P0 |
| [#4](https://github.com/jtrefon/splot-lm/issues/4) | RFC 0002: recurrent execution | P0 |
| [#5](https://github.com/jtrefon/splot-lm/issues/5) | RFC 0003: GDN/GQA | P0 |
| [#6](https://github.com/jtrefon/splot-lm/issues/6) | RFC 0004: MoE and locality prior art | P1 |
| [#7](https://github.com/jtrefon/splot-lm/issues/7) | RFC 0006: precision | P1 |
| [#8](https://github.com/jtrefon/splot-lm/issues/8) | Data/tokenizer and provenance | P0 |
| [#9](https://github.com/jtrefon/splot-lm/issues/9) | Dense baseline implementation | P0 |
| [#10](https://github.com/jtrefon/splot-lm/issues/10) | Experiment and hardware harness | P1 |
| [#11](https://github.com/jtrefon/splot-lm/issues/11) | Open science and licensing | P2 |

Dependency: #2–#5 and #8 inform #9; dense implementation precedes fixed recurrence, then MTP, GDN, MoE and precision. Follow issues for state; do not duplicate them in prose. Draft review: [native MTP](literature/reviews/native-mtp/en.md).

Continuity: [#13](https://github.com/jtrefon/splot-lm/issues/13) / merged PR #14. RFC 0002 (#4) draft and toy oracle now available in its review branch; actual mixer correctness remains pending. Next #5 then #3; consult live issues/PRs for status.

RFC 0002 merged in #16; RFC 0003 draft and equation oracle on review branch (#5). Next #3 reconcile architecture and exact parameter accounting; #8 independent. Actual mixer integration remains open; consult live issues/PRs.

PR #17 merged. Architecture contract v0.2, Nano/Mini candidate inventories, accounting tests and whitepaper draft in #3 review branch. Next #8 data/tokenizer, then #9 dense baseline. Live issues override snapshots.

PR #18 merged. #8 data contract, candidate registry and tested synthetic fixture in review branch. Next #9 synthetic dense baseline; production corpus/tokenizer remains #8, not assumed approved. No real-data training until rights and split gates.

PRs #19 and #20 merged. #4 recurrent all-GQA implementation, actual fixture integration, seven new CPU tests and A1 smoke in review branch. Next #2 offset-2 head and A0/A1/A2, then exact verifier; #5/#6 after correctness. #8 production data remains gated; #9 GPU smoke outstanding. Live issue/PR state wins.

PR #21 merged. Native MTP offset-2 and A0/A1/A2 implementation in review branch (#2); seven CPU tests, three synthetic 40-step smoke runs. Next exact D1/D2 verifier, EOS/budget, sampling correctness, cache replay; then #5 hybrid and #6 MoE. #8 production data and #9 GPU smoke pending.

PR #22 merged. Exact D1/D2 all-GQA speculative reference in review (#2), six CPU tests and seven MTP regressions pass. Next #5 learned GDN v1 standalone and hybrid integration, matrix/convolution rollback; then #6 MoE. #2 remains open for broader validation and performance. #8 real corpus gate unchanged.
