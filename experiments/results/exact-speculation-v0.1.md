# Exact speculation correctness evidence
2026-11-27 | #2 | CPU correctness, no timing benchmark.
Python 3.13.5, PyTorch 2.10.0+cpu; one CPU thread. Local /tmp/splot-work inherited sources verified against main Git blobs, then new sources tested and uploaded with exact hashes.
- src/splot/speculative.py e9a193d3864f4edffddaad4b0cd9ff0b8ef6eacb
- src/splot/__init__.py 7468a82d72e4cace0a6bf34445efeb2692ee8959
- tests/test_speculative.py 2b10d9fcd69c410850485589da604fcaae501449
- inherited dense.py 2ba9ac5e5b9c5990bf3e1c69e6c81e27bf54fd75, recurrent.py c5920eb33155e1e517d3a556b2aedfc72cc3c19e, mtp.py 8c4a1a21541a3cb1142a7293cf14a31f114991d5, tools/data_fixture.py 4cd314cd765a90861a84a364220f6dddf9d2ab2c, tests/test_mtp.py fc9fb4e32c26cc757f1f6b3ea9e1aa4bf23a4a05.

Commands: PYTHONPATH=src:tools python tests/test_speculative.py; PYTHONPATH=src:tools python tests/test_mtp.py.
6/6 new tests (2.256 s test runner), 7/7 MTP regressions (0.802 s runner). Times not throughput measurements. Other historical suites not rerun.
- Single-token residual identity, identical and disjoint supports: float64 atol/rtol 1e-14 for algebra.
- Toy conditional joint distribution: prefix [0], target P(0|0)=.8, P(0|1)=.35. Exact probabilities of (00,01,10,11) = (.64,.16,.07,.13). Proposal different from target; D2 offset probabilities (.6,.4). 3000 two-token generations per strategy, seed 123, prespecified test tolerance .04 absolute max empirical frequency error.
| Sequence | Target probability | D1 count | D2 count |
|---|---:|---:|---:|
| 00 | .64 | 1907 | 1924 |
| 01 | .16 | 464 | 484 |
| 10 | .07 | 240 | 198 |
| 11 | .13 | 389 | 394 |
Max absolute discrepancy D1 .010, D2 .004. Both pass. Every sampled target/shallow toy state checked against committed tokens.
- Forced first and second position rejection: replay from original prefix verified via recorded call arguments. EOS tests accepted EOS, corrected EOS, rejected draft EOS; zero budget and all-accepted bonus.
- Actual untrained MTPLM: default RecurrentConfig (264 vocabulary, d32, 2/4/2, R_cap4, I0/S0), 83776 physical parameters, FP64, weights seed 42, target R4/shallow R1, prefix [1,80,81,82], budget six tokens, temperature 1, sampling seed 9. D1 and D2 greedy both emitted (20,20,20,20,20,20), same as ordinary cached target autoregression. Stochastic D1 and D2 both emitted (168,170,49,22,117,0) in this seed, not required or a distribution proof. For each run, both final caches (all key/value tensors), pending logits and shallow offset-2 logits compared to fresh full-prefix prefill at atol=rtol=1e-10. D1 greedy proposed/accepted/rejected 6/4/1; D2 7/3/3; stochastic both 4/4/0. These untrained tiny-sequence counts are recorded for reproducibility only, not acceptance estimates.
- Invalid inputs and extreme-temperature stable softmax.
No real corpus, trained weights, CUDA, BF16, GPU or throughput, top-k/top-p, batch, GDN/MoE or optimized shallow cache sharing. Mathematical distribution preservation for valid p/q under ideal arithmetic; empirical toy check finite and model integration tests not exhaustive.
