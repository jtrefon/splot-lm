# REP task specifications
Read [AGENTS.md](../../AGENTS.md) and [EXECUTION.md](../../EXECUTION.md). Issues and live PRs are authoritative. A reviewed spec on main plus explicit owner launch is required. No automatic next tasks. [Evidence index](../results/INDEX.md); [template](TEMPLATE.md).

| Task | Specification | Issue | Status at this handover |
|---|---|---|---|
| T0001 | [GPU validation](T0001-gpu-validation.md) | #27 | Closed |
| T0002 | [Nano CUDA](T0002-nano-cuda.md) | #28 | Closed |
| T0003 | [Nano epoch](T0003-nano-epoch.md) | #35 | Closed |
| T0004 | [Mac corpus](T0004-mac-corpus-reference.md) | #38 | Closed |
| T0005 | [BF16 harness](T0005-bf16-harness.md) | #40 | Closed |
| T0006 | [Muon hybrid](T0006-muon-hybrid.md) | #48 | Closed |
| T0007 | [Native FP8 probe](T0007-native-fp8-probe.md) | #51 | Closed |
| T0008 | [Byte BPE](T0008-byte-bpe-oracle.md) | #52 | Closed |
| T0009 | [Native FP8 autograd](T0009-fp8-autograd.md) | #56 | Closed |
| T0010 | [Tokenized shards](T0010-tokenized-shards.md) | #60 | Closed |
| T0011 | [FP8 + Muon](T0011-fp8-muon.md) | #62 | Accepted, merge pending renewed CI; see #65 |

| T0012 | [Contiguous context](T0012-contiguous-context.md) | #66 | Linux, after #65 merge and spec merge |
| T0013 | [Source qualification](T0013-corpus-qualification.md) | #67 | Mac, spec review then owner launch |

| T0014 | [Verified ingestion](T0014-dataset-ingestion.md) | #71 | Linux, requires #70 and reviewed spec merge |
| T0015 | [Production tokenizer](T0015-production-tokenizer.md) | #72 | Mac, reviewed spec and owner launch; optional package authorization separate |
