# SPLOT reference implementations
Current: dense GQA decoder-only control (`dense.py`). This is the slow correctness reference, not the recurrent hybrid MoE candidate. No checkpoint or production tokenizer.

```bash
PYTHONPATH=src python tests/test_dense.py
PYTHONPATH=src python tools/smoke_dense.py --config configs/baselines/dense-smoke.json
```

PyTorch 2.10.0 CPU tested; Python 3.13.5. See [implementation contract](../../research/implementation/dense-reference.md) and [evidence](../../experiments/results/dense-reference-v0.1.md). No external downloads required after PyTorch installation. Use explicit device-specific PyTorch installation instructions; CUDA smoke remains unrun. Do not benchmark this Python-loop oracle as an optimized baseline.

## Fixed-depth recurrent all-GQA control

`recurrent.py` implements the shared 2/4/2 sandwich with per-pass caches, all-depth NTP readouts and configurable injection/scaling. [Contract](../../research/implementation/recurrent-reference.md) and [evidence](../../experiments/results/recurrent-reference-v0.1.md).

```bash
PYTHONPATH=src:tools python tests/test_recurrent.py
PYTHONPATH=src:tools python tools/smoke_recurrent.py --loops 2 --steps 40
```

Requires merged synthetic fixture (tools/data_fixture.py). No GDN or MoE yet. GPU smoke unrun; real training blocked on #8.

## Native MTP

`mtp.py`: shared offset-2 head, A0/A1/A2 objectives. [Contract](../../research/implementation/native-mtp.md) and [evidence](../../experiments/results/native-mtp-v0.1.md).

```bash
PYTHONPATH=src:tools python tests/test_mtp.py
PYTHONPATH=src:tools python tools/smoke_mtp.py --mode A2 --loops 2 --steps 40
```

GPU and real corpus training unrun.

## Exact speculative reference

`speculative.py` implements D1 serial shallow drafting and D2 independent MTP drafting, full-depth verification and transactional replay. [Contract](../../research/implementation/exact-speculation.md) · [Evidence](../../experiments/results/exact-speculation-v0.1.md).

```bash
PYTHONPATH=src:tools python tests/test_speculative.py
```

Single sequence, CPU probability arithmetic, no optimization or speed claim. Hybrid GDN rollback remains pending.
