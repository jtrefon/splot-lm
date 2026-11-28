# SPLOT reference implementation

Trainable PyTorch research architecture. The dense baseline is retained for correctness and future matched comparisons; the full SPLOT implementation is not an optimized production backend.

| Module | Responsibility | Contract |
|---|---|---|
| `dense.py` | Dense GQA control, norms, tied embeddings, causal masks/cache | [Dense](../../research/implementation/dense-reference.md) |
| `recurrent.py` | Shared 2/4/2 stack, logical per-pass states, fixed depths | [Recurrence](../../research/implementation/recurrent-reference.md) |
| `mtp.py` | Native offset-2 head, A0/A1/A2 objectives | [MTP](../../research/implementation/native-mtp.md) |
| `gdn.py`, `hybrid.py` | Learned GDN recurrence, hybrid GDN/GQA, dense FFN control | [Hybrid](../../research/implementation/hybrid-gdn.md) |
| `moe.py`, `sparse.py` | Eight routed experts, top-2 + shared, FP32 routing and balance | [Sparse SPLOT](../../research/implementation/sparse-moe.md) |
| `speculative.py` | Exact D1/D2 sampling and retained-prefix replay, single row | [Speculation](../../research/implementation/exact-speculation.md) |
| `fp8.py` | Native selected linear autograd, device scales, audits, opt-in weight reuse | [FP8](../../research/implementation/fp8-autograd.md), [reuse](../../research/implementation/fp8-weight-reuse-pilot.md) |

Nano (43,055,076 physical parameters at V32768) has passed RTX 4090 synthetic CUDA training including native FP8 + Muon and contiguous T4096 qualification. Mini (146,987,784) inventory is implemented; sustained GPU training is not qualified. FP32/BF16 controls retained. The real-data loader and 16K corpus are available; first actual run still pending. See [current state](../../PROJECT_STATE.md) and [experimental evidence](../../experiments/results/INDEX.md).

From a clean checkout with a compatible existing PyTorch runtime:
```bash
PYTHONPATH=src:tools python tools/ci_cpu.py
PYTHONPATH=src:tools python tests/test_moe.py
PYTHONPATH=src:tools python tests/test_speculative.py
```

The GDN scan, attention and routing are slow correctness references, not optimized kernels; D1/D2 sampling is CPU float64 and retains separate shallow/target state. Do not interpret synthetic losses or single-step timings as production performance. GPU runs require reviewed task and explicit local authorization.
