# Harness mechanics evidence
2026-11-27 | #10
Local Python 3.13.5, PyTorch 2.10.0+cpu, CUDA unavailable, one thread. Base repository HEAD 485567dcff11b18bedd40dc4b48ffe87559b39ab inspected via GitHub connector. Local Git clone failed DNS resolution; repository model modules were not mounted. Tests deliberately use a small independent PyTorch embedding/dropout/linear network, not SPLOTLM. Source blobs locally executed:
- tools/experiment.py 6929c08fc9f8b74253fe2e3cc7cbcd98b3c99d67
- tools/test_experiment.py 4aaf4832cf0fd2a2aee5dfb17ef4986e27e161b5

Command: python tools/test_experiment.py (local equivalent file layout). 5/5 passed, 0.618 s unittest runner (not benchmark):
- test_hash
- test_logical_macs
- test_resume_exact_and_reject_mismatch
- test_schedule
- test_token_weighted_eval

Six uninterrupted updates compared with two updates + checkpoint restore + four updates, seed 42, dropout .2, AdamW lr .01, five blocks, batch 2. Resumed model initially seeded differently; restoring RNG and optimizer yields bitwise equality of parameters, AdamW states and non-timing history. Reject mismatched identity, accidental overwrite and decreasing total steps. Weighted NLL invariant to validation batch size within tested tolerance. No full SPLOT CLI or integration test executed; no SPLOT run.json or checkpoint generated in this work unit. Previous #25 evidence not re-run. GPU unavailable, no energy, latency or throughput measurement. Integration test remains pending; commands and proposed configs in contract.
