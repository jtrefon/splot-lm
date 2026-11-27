# Learned hybrid GDN/GQA CPU evidence
2026-11-27 | #5 | Synthetic correctness, not benchmark.
Python 3.13.5, PyTorch 2.10.0+cpu, CPU one thread, seed 42. New/modified source blobs verified against exact locally tested files:
- src/splot/gdn.py 33611f5cfcdfb84c48b701f4af9807d66f81ac64
- src/splot/hybrid.py 69c4a49b195b53c9073d8cc0ef56ae03112a2f8f
- src/splot/dense.py a3fdd655b67bc950001bf8eb95c7aabe6f1cf4ea
- src/splot/__init__.py ddac4fc2ce414c190e8c62951c83c0d1ccfdb79f
- tests/test_hybrid.py 91a834093152977e112c5f52c8acd3bb5e92569e
- tools/smoke_mtp.py 59dd67aee067087b222c08c2b31c1194e02f2d1a
Inherited verified: recurrent.py c5920eb33155e1e517d3a556b2aedfc72cc3c19e; mtp.py 8c4a1a21541a3cb1142a7293cf14a31f114991d5; speculative.py e9a193d3864f4edffddaad4b0cd9ff0b8ef6eacb; tools/data_fixture.py 4cd314cd765a90861a84a364220f6dddf9d2ab2c; tests/test_speculative.py 2b10d9fcd69c410850485589da604fcaae501449; tests/test_mtp.py fc9fb4e32c26cc757f1f6b3ea9e1aa4bf23a4a05.

Commands as implementation contract. Six hybrid tests pass, 7 MTP and 6 speculative regressions rerun pass. FP64 atol=rtol=1e-10. Test scan vs independent triangular solve with nonzero initial matrix/history and learned projections, compare gradients of output and final state loss. At R=1,2,4, compare all readout logits, offset2 and final per-row states for full vs token and irregular chunks. Greedy D1/D2 equals ordinary target; stochastic runs compare committed caches/logits to fresh prefix, not identical sampled strings. Hybrid six-token fixture, D1 greedy accepted 4 rejected 0, D2 greedy accepted 3 rejected 3; stochastic D1/D2 each accepted 4 rejected 0. Untrained weights, no trained acceptance conclusion. Inherited toy distribution test rerun: 3000 trials each, max errors .010/.004. Other historical suites not rerun.

40-step A2 smoke: default recurrent config (d32, vocab264, Q/KV4/2, head8, P/M/Q2/4/2, R=2, no injection, unscaled residual), HybridLM 92912 params. Two fixed EN/PL original docs packed length64, 59 NTP / 57 offset2 valid labels. AdamW lr .003 weight_decay 0; loss weights .25/.25; seed42. Loss values before step1 and step40:
| Component | Initial | Final |
|---|---:|---:|
| Total | 8.387558937 | 2.282608747 |
| Final NTP | 5.595514297 | 1.463391185 |
| Early NTP | 5.585738659 | 1.490600824 |
| Offset2 | 5.582439899 | 1.786268950 |
All active gradients finite. No checkpoint. Not language quality evidence.

Additional manual CPU checks (not standalone scripted tests): after identical 40-step smoke, BF16 autocast full vs chunks (0,13,37,64) with R=2 and all-depth NTP+offset2; max abs difference 0.0 on both heads at both depths, atol=rtol=.05. State dtype test in test_hybrid.py confirms FP32 matrix and BF16 raw history. Nano/Mini meta counts 26527716 / 80902920, actual Nano dense hybrid forward/backward on IDs [1,80,81,2], labels offset1 [80,81,2,-100], offset2 [81,2,-100,-100], R=2, total 15.581604, all parameter grads finite. Nano initialization used RNG state after prior smoke; standalone numerical value not guaranteed. No optimizer step at Nano size. Mini no forward. GPU/BF16 training, long context, speed, actual corpus, MoE, fused kernels and checkpointing untested.
