# Tests

CPU suites cover dense/GQA, recurrence, MTP, hybrid GDN, sparse MoE, causal/cache semantics and exact speculative verification. Additional harness, optimizer, precision, data and REP tests live in `tools/`.

Run `PYTHONPATH=src:tools python tools/ci_cpu.py` from a clean checkout with compatible PyTorch. Hosted CPU workflow discovers suites and checks structured evidence hashes. Optional CUDA/tokenizer qualification is separate and must not be reported as passed when not executed. See [task specs](../experiments/tasks/README.md) and [evidence index](../experiments/results/INDEX.md).
