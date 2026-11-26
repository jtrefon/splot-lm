# SPLOT — Sparse Predictive Looped Optimized Transformer

[Polski](README.pl.md) | [Architecture RFC](research/rfcs/0001-architecture/en.md) | [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md)

Independent, Polish-origin, internationally accessible research into efficient language model architectures. SPLOT explores recurrent depth, hybrid attention, sparse mixture-of-experts (MoE), native multi-token prediction (MTP), low-precision training and hardware-aware inference on commodity hardware.

**Status: research proposal / pre-implementation.** No trained SPLOT model, benchmark result, or claim of architectural novelty exists yet. The project is not affiliated with PLLuM, NASK or another institution.

## Research question
Can a language model learn to allocate parameter activation, computational depth and context representation while controlling memory traffic and numerical error, improving the quality–latency–memory Pareto frontier?

## Start here
- [Research foundation and decisions](research/foundation/en.md) ([PL](research/foundation/pl.md))
- [RFC 0001: architecture](research/rfcs/0001-architecture/en.md) ([PL](research/rfcs/0001-architecture/pl.md))
- [RFC 0005: native MTP](research/rfcs/0005-native-mtp/en.md) ([PL](research/rfcs/0005-native-mtp/pl.md))
- [Literature register](research/literature/README.md) and [BibTeX](research/literature/references.bib)
- [Hypotheses](research/hypotheses/README.md), [experimental protocol](experiments/protocols/initial-ablation.md), [research backlog](research/BACKLOG.md)
- [Reproducibility](experiments/README.md), [illustrative model configuration](configs/models/splot-mini-proposal.yaml)

English is canonical for scientific specifications, source and experiments. Polish translations and discussion are welcome. Code is Apache-2.0; original research text is proposed for CC BY 4.0 (see [licensing](LICENSING.md)). No third-party data or weights are distributed here.
