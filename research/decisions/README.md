# Decision log / Rejestr decyzji
Append numbered decision records; never rewrite accepted rationale silently. Each record: ID/date, status (proposed/accepted/rejected/superseded), issue/PR, alternatives, rationale/evidence, risks, consequences, follow-up and supersedes. Accepted requires maintainer review; merged draft research is not proof of empirical claims.

| ID | Decision | Status | Source |
|---|---|---|---|
| D0001 | Repository-first research continuity, PR review and issue-linked evidence | Proposed in #13 | AGENTS.md |
| D0002 | Bilingual EN canonical/PL translations, independent project, text-first | Foundational direction | #1 |
| D0003 | Fixed recurrent depth + native MTP in first generation; exact reference before optimization | Proposed design | #12, #2 |
| D0004 | Hardware locality hypotheses require baseline and prior-art review; no novel claim | Proposed research direction | #6 |
| D0005 | Apache 2.0 code; CC BY 4.0 research scope pending approval | Open licensing decision | #11 |
| D0006 | Fixed R per sequence; prefix-compatible pass functions use immutable R_cap, never runtime R; distinct logical caches and full unroll baseline | Proposed; toy oracle only | #4, RFC 0002 | 
| D0007 | GDN v1 reference, 12 equal 64-d heads, 4-tap convolution; 3 GDN + gated GQA core, ungated GQA prelude/coda; FP32 matrix state, separate logical state and rollback | Proposed; six CPU equation tests only | #5, RFC 0003 |
| D0008 | Nano 43,055,076 and Mini 146,987,784 proposed; shared offset-2 projection + RMSNorm, tied E; I0/S0 controls; separate unique active and contraction MAC accounting | Proposed, six accounting tests only | #3, RFC 0001 v0.2 |
| D0009 | Proposed bilingual 40/40/20 PL/EN/code overall content mix, 32K shared byte BPE on train only; source registry gated, exact/group split before packing | Proposed; synthetic tests only | #8, research/data/README.md |
| D0010 | All-GQA recurrent implementation uses fixed loops/readouts per cache, distinct P/C/Q logical keys, I0/S0 default, optional depth vector and independent injection/scaling ablations; hidden states exposed for MTP | Implemented in review; CPU synthetic evidence only | #4, research/implementation/recurrent-reference.md |

Evidence status and design status are orthogonal. Link the exact commit or PR when a decision is accepted. Changes in hypothesis or protocol require dated amendments before examining results.
