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
