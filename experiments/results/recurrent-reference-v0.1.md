# Fixed-depth recurrent all-GQA CPU evidence
2026-11-26 | #4 | Synthetic tests, not a benchmark.
Environment Python 3.13.5, PyTorch 2.10.0+cpu, no CUDA; seed 42, one CPU thread. Fixture blob from merged #19: 4cd314cd765a90861a84a364220f6dddf9d2ab2c. Code blobs tested locally, verified on GitHub tree:
- src/splot/dense.py 2ba9ac5e5b9c5990bf3e1c69e6c81e27bf54fd75
- src/splot/__init__.py 82239aef1988ce216969556b78422fc1c4c4ef5c
- src/splot/recurrent.py c5920eb33155e1e517d3a556b2aedfc72cc3c19e
- tests/test_recurrent.py f9fb6c5a873e8365f57a3e74348242845bcea147
- tools/smoke_recurrent.py 832f9eef57f012d522fe7d612eebfb929c858663

Commands: PYTHONPATH=src:tools python tests/test_recurrent.py; PYTHONPATH=src python tests/test_dense.py; PYTHONPATH=src:tools python tools/smoke_recurrent.py.
7/7 new recurrent tests (2.386s local test runner, not benchmark): test_cache_branch_replay_and_contract, test_cache_equivalence_and_distinct_logical_states, test_causal_document_isolation_and_padding, test_depth_prefix_invariance_variants, test_fixture_training_and_finite_gradients, test_physical_count_and_r1_dense_equivalence, test_shared_gradients_equal_untied_sum. 6/6 dense regressions rerun (0.066s). FP64 atol=rtol=1e-10, recurrent gradient test with 2 passes and all-depth readouts, separate untied core/coda copies. 12-step test uses FP32 AdamW and actual pack() fixture; verifies finite gradients and declining objective, not quality.

40-step FP32 smoke stdout:
```json
{
  "torch": "2.10.0+cpu",
  "device": "cpu",
  "seed": 42,
  "steps": 40,
  "loops": 2,
  "parameters": 82720,
  "initial_loss": 6.999616622924805,
  "final_loss": 3.109294891357422,
  "cache_max_abs_error": {
    "1": 1.1920928955078125e-06,
    "2": 1.1920928955078125e-06
  }
}
```
Config is executable defaults: vocab=264, d=32, layers=8, Q/KV=4/2, head=8, FFN=64, rope=1e6, eps=1e-5, init std=.02, P/M/Q=2/4/2, R_cap=4, R=2, I0/S0, no depth embedding. pack() original two fixed EN/PL examples length 64 with document boundary/reset and masked offset-1 targets. AdamW lr=.003, weight_decay=0, A1 objective final CE + .25 mean previous CE. Losses are pre-update step 1 and step 40, cache compared after step 40. Gate final <0.5 initial; FP32 cache assert_close atol=1e-5 rtol=1e-4. No checkpoints saved. Not comparable to #20 loss: different objective, physical model and packing. No external corpus, real tokenizer, GPU, BF16, larger config, checkpointing, offset-2 MTP, stochastic verifier, GDN or MoE. No quality/speed/generalization claim.
