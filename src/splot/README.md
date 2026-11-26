# SPLOT reference implementations
Current: dense GQA decoder-only control (`dense.py`). This is the slow correctness reference, not the recurrent hybrid MoE candidate. No checkpoint or production tokenizer.

```bash
PYTHONPATH=src python tests/test_dense.py
PYTHONPATH=src python tools/smoke_dense.py --config configs/baselines/dense-smoke.json
```

PyTorch 2.10.0 CPU tested; Python 3.13.5. See [implementation contract](../../research/implementation/dense-reference.md) and [evidence](../../experiments/results/dense-reference-v0.1.md). No external downloads required after PyTorch installation. Use explicit device-specific PyTorch installation instructions; CUDA smoke remains unrun. Do not benchmark this Python-loop oracle as an optimized baseline.
