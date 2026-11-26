# Dense reference smoke evidence
2026-11-26 | Issue #9 | CPU only.
Environment: Python 3.13.5, PyTorch 2.10.0+cpu, torch.cuda.is_available() false. Seed 42, CPU threads 1. No external corpus, tokenizer, checkpoint or GPU. Source blobs tested locally and uploaded:
- src/splot/dense.py 494e12476ae8cefb6b84b2a36307497356c3cc86
- src/splot/__init__.py 0d357de3aef67bfc181fbee317238ae948e15c46
- tests/test_dense.py 2aa8f4ae7c2816416350b25805f2369c48391d55
- tools/smoke_dense.py 1132358688a81bd02448263bd1a58d5f841d11ca

Commands: PYTHONPATH=src python tests/test_dense.py; PYTHONPATH=src python tools/smoke_dense.py (default config equal to dense-smoke.json).
Six tests passed (0.072 s final local test runner, not a benchmark):
- test_cache_branch_and_padding_noop
- test_cached_chunks_and_document_reset
- test_count_and_tied_output
- test_future_and_document_isolation
- test_invalid_metadata_and_cache_contract
- test_loss_gradients_and_empty

FP64 test tolerances atol=rtol=1e-10. Actual forward/backward test, no real corpus. Training stdout:
```json
{
  "torch": "2.10.0+cpu",
  "device": "cpu",
  "bf16": false,
  "steps": 40,
  "seed": 42,
  "parameters": 27040,
  "initial_loss": 5.5875630378723145,
  "final_loss": 2.0333304405212402,
  "cache_max_abs_error": 1.1920928955078125e-06
}
```
AdamW lr=.003, weight_decay=0, synthetic byte IDs 8+UTF8, BOS=1, EOS=2, PAD=0. Two fixed short documents, same batch repeated. Loss values measured before each optimizer step, so final value is pre-step 40. Final cache comparison after step 40. This confirms ability to overfit, not generalization or language capability. No weights saved. FP32 cache assert_close atol=1e-5, rtol=1e-4.

Separate manual checks: instantiated Dense-Nano and Dense-Mini on meta device and compared counts 25,172,352 and 75,510,528 respectively to formula (exact matches; no actual forward). CPU BF16 autocast eval with one 4-token sequence, seed 42, compared cached/uncached outputs, max absolute difference 0.0 on that fixture, tolerance .05/.05; exploratory only. No GPU, long-context, fused kernel, full model scale, MTP/GDN/MoE, or actual data integration tests. Pending GPU smoke protocol in research/implementation/dense-reference.md.
