# REP task specifications
Read [AGENTS.md](../../AGENTS.md) and [EXECUTION.md](../../EXECUTION.md). Snapshot 2026-11-28; live issues and PRs are authoritative. Reviewed merged spec plus explicit local owner launch required. No automatic next tasks. [Evidence](../results/INDEX.md) · [Template](TEMPLATE.md).

| Task | Specification | Issue | Status |
|---|---|---|---|
| T0001 | [GPU validation](T0001-gpu-validation.md) | [#27](https://github.com/jtrefon/splot-lm/issues/27) | Closed |
| T0002 | [Nano CUDA](T0002-nano-cuda.md) | [#28](https://github.com/jtrefon/splot-lm/issues/28) | Closed |
| T0003 | [Nano epoch](T0003-nano-epoch.md) | [#35](https://github.com/jtrefon/splot-lm/issues/35) | Closed |
| T0004 | [Mac corpus](T0004-mac-corpus-reference.md) | [#38](https://github.com/jtrefon/splot-lm/issues/38) | Closed |
| T0005 | [BF16 harness](T0005-bf16-harness.md) | [#40](https://github.com/jtrefon/splot-lm/issues/40) | Closed |
| T0006 | [Muon hybrid](T0006-muon-hybrid.md) | [#48](https://github.com/jtrefon/splot-lm/issues/48) | Closed |
| T0007 | [Native FP8 probe](T0007-native-fp8-probe.md) | [#51](https://github.com/jtrefon/splot-lm/issues/51) | Closed |
| T0008 | [Byte BPE](T0008-byte-bpe-oracle.md) | [#52](https://github.com/jtrefon/splot-lm/issues/52) | Closed |
| T0009 | [Native FP8 autograd](T0009-fp8-autograd.md) | [#56](https://github.com/jtrefon/splot-lm/issues/56) | Closed |
| T0010 | [Tokenized shards](T0010-tokenized-shards.md) | [#60](https://github.com/jtrefon/splot-lm/issues/60) | Closed |
| T0011 | [FP8 + Muon](T0011-fp8-muon.md) | [#62](https://github.com/jtrefon/splot-lm/issues/62) | Closed |
| T0012 | [Contiguous context](T0012-contiguous-context.md) | [#66](https://github.com/jtrefon/splot-lm/issues/66) | Closed |
| T0013 | [Source qualification](T0013-corpus-qualification.md) | [#67](https://github.com/jtrefon/splot-lm/issues/67) | Closed |
| T0014 | [Verified ingestion](T0014-dataset-ingestion.md) | [#71](https://github.com/jtrefon/splot-lm/issues/71) | Closed |
| T0015 | [Production tokenizer](T0015-production-tokenizer.md) | [#72](https://github.com/jtrefon/splot-lm/issues/72) | Closed |
| T0016 | [Real bilingual corpus](T0016-real-corpus.md) | [#76](https://github.com/jtrefon/splot-lm/issues/76) | Closed |
| T0017 | [FP8 reuse and pilot](T0017-fp8-weight-reuse-pilot.md) | [#77](https://github.com/jtrefon/splot-lm/issues/77) | Closed; #92 merged, bounded feasibility accepted with documented procedural deviations |
| T0018 | [MTP observability](T0018-mtp-observability.md) | [#81](https://github.com/jtrefon/splot-lm/issues/81) | Closed; attempt-002 accepted (#91); attempt-001 review-rejected |
| T0019 | [Matched baseline design](T0019-baseline-design.md) | [#89](https://github.com/jtrefon/splot-lm/issues/89) | Closed; attempt-002 design accepted, attempt-001 partial |
| T0020 | [CUDA cost baseline](T0020-cuda-cost-baseline.md) | [#90](https://github.com/jtrefon/splot-lm/issues/90) | Closed; 12/12 paired cases accepted (descriptive) |
| T0021 | [Dense NTP harness](T0021-dense-ntp-harness.md) | [#97](https://github.com/jtrefon/splot-lm/issues/97) | Closed; scoped implementation accepted, actual-vocab compatibility transferred to T0023 |
| T0022 | [Backward attribution](T0022-backward-attribution.md) | [#98](https://github.com/jtrefon/splot-lm/issues/98) | Closed; four exactness-qualified cases, module attribution unassigned |
| T0023 | [Actual-vocabulary pilot](T0023-actual-vocab-pilot.md) / [continuation](T0023-continuation.md) | [#101](https://github.com/jtrefon/splot-lm/issues/101) | Open; attempt-001 failed CI, attempt-002 test correction after owner launch |
| T0024 | [Chunked GDN feasibility](T0024-gdn-chunk-feasibility.md) | [#102](https://github.com/jtrefon/splot-lm/issues/102) | Open; component evidence accepted, reset/tail gate pending T0025 prerequisite |
| T0025 | [Opt-in chunk64 GDN](T0025-gdn-opt-in.md) | [#105](https://github.com/jtrefon/splot-lm/issues/105) | Linux; prerequisite reset/tail, then full-model qualification after owner launch |
