# RFC 0003 independent equation oracle
2026-11-26 | #5 | CPU only; no performance benchmark.
Command: python tests/numerical/test_hybrid_mixer.py
Environment: local CPU tool sandbox, NumPy 2.3.5, float64, seed 42, atol=rtol=1e-12. Test source Git blob a1484b82e6e10cfbafbebd294b947752a8451597. Small synthetic input arrays; no datasets, checkpoints, GPU or external code. Run against exact version in PR.

Observed 6/6 passing:
- test_chunk_vs_recurrent_nonzero_state
- test_irregular_chunks_and_prefix
- test_gate_limits
- test_matrix_rollback
- test_conv_reset_and_rollback
- test_grouped_rope_attention_vs_explicit_causal

Independent chunk solve uses a unit lower triangular matrix with log decay ratios, versus direct rank-one recurrence. Convolution and GQA tested separately. This is not a full mixer implementation: no learned projections/gating network, normalization of complete block, training gradients, batched/padded/packed integration, MoE, MTP target distribution, GPU, or speed claim. Real integration acceptance remains open.
