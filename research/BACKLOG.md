# Initial issue backlog
Create GitHub issues after foundation review; these are not yet live issues.

- RFC 0001: review architecture and parameter count; define exact shapes and initial fixed-depth state semantics.
- RFC 0002: recurrent depth topology, injection, normalization, cache indexing and adaptive halting later.
- RFC 0003: GDN/GQA mixer specification, exact recurrence equations, correctness oracle.
- RFC 0004: MoE routing and locality prior-art search, load balance and dispatch.
- RFC 0005: MTP literature full-text review, compare Medusa, DeepSeek, EAGLE-3, LoopMTP, LoopSpec, Floodgates; decide training and verification method; verify LoopFill.
- RFC 0006: precision and memory accounting, BF16 reference, FP8 benchmark, simulated FP4, quantized recurrence drift.
- Dataset/tokenizer: provenance, language mix, document splits, contamination, licensing.
- Implement dense PyTorch baseline and numerical/causality tests; then fixed recurrence and MTP.
- Experiment harness and benchmark manifests; benchmark RTX 4090; do not extrapolate unmeasured rates.
- Publication workflow: model cards, weights license, reproducible releases, Zenodo DOI, bilingual translations.
