# RFC 0002 toy correctness smoke test
2026-11-26 | #4 | Not a model training or inference benchmark.

Command: python tests/numerical/test_recurrent_semantics.py
Environment: local tool sandbox, CPU, PyTorch 2.10.0+cpu, float64, seed 42; toy hidden width 3, prelude=1/core=2/coda=1, five random inputs, R=1/2/4, tolerance atol=rtol=1e-12. Synthetic mixer, no corpus or external checkpoint. Code executed locally before commit; rerun the version in this PR. Exact commit is the commit containing this report (use git log). No GPU used.

Observed: five unittest tests passed (0.049 s locally, not a performance measurement).
- test_document_reset
- test_prefix_and_depth_invariance
- test_rollback
- test_shared_gradient_equals_sum_of_untied_copies
- test_stage_major_vs_token_major

Limitations: toy mixer only, no GQA/GDN, MoE, MTP distribution, batching, padding, actual logits, real numerical precision or production cache. Does not establish any capability or hardware improvement. Real implementation acceptance remains open.
