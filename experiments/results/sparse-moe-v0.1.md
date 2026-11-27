# Full sparse SPLOT CPU evidence
2026-11-27 | #6/#3 | Synthetic correctness, not a benchmark.
Environment Python 3.13.5, PyTorch 2.10.0+cpu, one thread, seed 42, CUDA unavailable. Exact locally tested Git source blobs verified:
- src/splot/__init__.py c66d7c81727b9916d6bff34251176a722faedbc5
- src/splot/moe.py d01fbd043e19b8d774dd8bbcf3316bdc221adcd8
- src/splot/sparse.py 49fb44725eceae9f34d2d0a24e29e5f8ab394cdc
- src/splot/recurrent.py 6396de37894ded1e0b49b6d580ab60beb5067efa
- src/splot/mtp.py 9186e8696ae8a53cc86e07eefeca2fbbd58e552d
- tests/test_moe.py dcd8b4bcf7056addd88d3383a76bdc8a523a035d
- tools/smoke_mtp.py 79bc02b12598faaf2359d9d10990ad42a2254159
Inherited: dense.py a3fdd655b67bc950001bf8eb95c7aabe6f1cf4ea; gdn.py 33611f5cfcdfb84c48b701f4af9807d66f81ac64; hybrid.py 69c4a49b195b53c9073d8cc0ef56ae03112a2f8f; speculative.py e9a193d3864f4edffddaad4b0cd9ff0b8ef6eacb; tools/data_fixture.py 4cd314cd765a90861a84a364220f6dddf9d2ab2c; tests/test_hybrid.py 91a834093152977e112c5f52c8acd3bb5e92569e; tests/test_mtp.py fc9fb4e32c26cc757f1f6b3ea9e1aa4bf23a4a05; tests/test_speculative.py 2b10d9fcd69c410850485589da604fcaae501449.

Commands: PYTHONPATH=src:tools python tests/test_moe.py (6/6, 3.098s runner); same for test_hybrid.py (6/6), test_mtp.py (7/7), test_speculative.py (6/6). Runtime not benchmark. Float64 tolerances 1e-10. Actual SPLOT six-token greedy D1/D2 sequences equal ordinary full-depth decoding, D1 accepted 4 rejected 1, D2 accepted 3 rejected 3; stochastic D1/D2 each accepted 4 rejected 0. All resulting target/shallow caches and next logits match fresh prefill. Not acceptance measurements on trained models. Regression finite toy joint sampler still max absolute frequency errors .010/.004 across 3000 draws each. Other historical suites not rerun.

Forty-step smoke: PYTHONPATH=src:tools python tools/smoke_mtp.py --sparse --mode A2 --loops 2 --steps 40. Default d32, vocab264, Q/KV4/2, D8, dense FFN64, expert FFN32, 8 routed +1 shared, I0/S0, R=2/R_cap4. AdamW lr=.003, weight_decay=0, same two synthetic EN/PL docs packed length64, NTP valid59 and offset2 valid57. Weights .25/.25/.01. Actual model count179952. Pre-update step1 and step40:
| Component | Initial | Final |
|---|---:|---:|
| Total | 8.395731926 | 2.268871546 |
| Final NTP | 5.593532562 | 1.447545290 |
| Early NTP | 5.584756851 | 1.466769814 |
| Offset2 | 5.583521843 | 1.778364420 |
| Balance | 1.013000250 | 1.004276037 |
All active gradients finite; absent expert gradients allowed. No checkpoint saved. Single seed, fixed batch; no held-out quality or matched compute claim. CPU BF16 autocast exploratory forward confirms routing probabilities FP32 and finite logits.

Nano manual smoke (not yet standalone script): seed42, d384,vocab32768,layers8,Q/KV6/2,head64,dense intermediate1024, expert512, default R2. IDs [[1,80,81,2]], docs all 0, pos 0..3, all valid; labels +1 [[80,81,2,-100]], +2 [[81,2,-100,-100]]. A2+balance total 15.06480598449707, actual count43,055,076, 153/198 parameter tensors receive gradients, all present gradients finite. No optimizer step. Mini meta inventory146,987,784; no full forward. Both exact RFC counts.

Illustrative untrained Nano routes from same 4-token sample at R2, pass2 depth overlap per physical block 0.75,0.875,0.875,0.75 (mean .8125); 3–6 unselected experts per individual pass/block in this tiny sample. Not evidence of trained locality or collapse; empirical marginal control, larger samples, chronological access traces and transfer/latency remain required. No real corpus, GPU, optimized dispatch, offload, quantized model or performance results.
