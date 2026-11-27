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

PR #23 merged. #5 learned hybrid dense GDN/GQA + MTP implemented on review branch; six CPU tests, 13 regressions, synthetic A2 smoke and short Nano forward/backward. Next #6 learned sparse MoE and full model integration; GPU, real corpus and optimization pending.

PR #24 merged. Full SPLOT sparse hybrid reference in review (#6/#3); exact candidate counts, six tests, 19 regressions, synthetic A2+balance smoke and actual short Nano forward/backward. Next #10 reproducible harness, checkpoint/resume and GPU smoke; #7 precision, #8 approved data/tokenizer in parallel. Locality experiments remain #6.

PR #25 merged. #10 synthetic FP32 experiment harness, exact resume mechanics, executable full candidate configs, token-weighted evaluation and analytical MAC report in review. Five toy harness tests pass; full SPLOT integration/CLI and CUDA not executed (sandbox clone DNS failure). Next run integration, benchmark D0/D1/D2, add matched controls. #7 precision and #8 real data remain gated.

PR #26 merged. REP v0.1 in review: one-repository task/evidence handoff, no runner infrastructure. T0001 #27 pinned GPU correctness task requires merged spec and explicit local launch; T0002 #28 blocked on reviewed evidence. Read EXECUTION.md and evidence index. #10 remains open.

2026-11-27: #3/#4/#5/#9 closed for original specification/reference acceptance, #27 T0001 closed after merged #31/#32 and reviewed evidence. #2 literature verification, #6 locality/PCIe, #7 precision, #8 real data, #10 empirical benchmarks/matched controls and #11 licensing remain open. T0002 #28 finalized in review; owner launch only after merge.

T0002 #28/#34 reviewed accepted within one-step scope, awaiting merge. T0003 #35 one complete synthetic FP32 Nano epoch proposed, 35 steps, memory/time and optimizer coverage. Follow #7 with explicit BF16 implementation (current CLI rejects BF16), then numerical and resource validation; #8 corpus/tokenizer licensing and #10 inference/matched controls remain open. Do not interpret padded T96 as contiguous 96-token context.

T0002/T0003 accepted/merged and closed (#28/#35); #38 Mac T0004 spec merged, no large corpus transfer; #40 Linux T0005 BF16 implementation specification in review. Portability policy in research/data/portability.md. Keep #7/#8/#10 open. Checkpoint cadence and matched full-epoch precision validation after T0005.
