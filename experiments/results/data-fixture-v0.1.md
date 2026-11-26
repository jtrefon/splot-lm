# Synthetic data contract smoke test
2026-11-26 | #8 | Python 3.13.5, local CPU, standard library, no network or tokenizer dependency.
Commands: python tools/test_data_fixture.py; python tools/data_fixture.py
Exact locally tested Git blobs: fixture 4cd314cd765a90861a84a364220f6dddf9d2ab2c; tests 90fbd26b6a66493a0f02fde2aa6f401e0ca9265f. Generated manifest Git blob f9eba647011acf47e96f8fd3e548157d73518029 (regenerate with CLI; not checked in as raw file).
Six tests passed: manifest repeatability, no silent truncation/duplicate IDs, packing and offsets, deterministic split/disjoint fingerprints and groups, transitive group dedup, exact UTF-8 round trips. 0.003s test runner, not benchmark.
| Split | Documents | Content byte tokens | Manifest SHA-256 |
|---|---:|---:|---|
| Train | 57 | 2477 | ca2d80264a350429f079befdfc852cd476de3b2edfee06e89558fd4ee29d3c42 |
| Validation | 9 | 367 | 94465781a1fb808c51a3d6ae76e2a464de442fba947f140cb9fcca24f4851131 |
| Test | 14 | 666 | 9dcef6b6085352370e929b386a4cdfbfeaa6cc34b1c009029f4d8d0e087b543e |

Seed splot-fixture-v1, deterministic hash assignment 80/10/10 targets (realized counts differ). Synthetic templated texts repeat topics across splits; exact and declared group separation only, not near-duplicate or semantic isolation. Not evidence for generalization, BPE quality, real-data contamination, actual corpus licenses or performance. No external corpus acquired; actual tokenizer and scalable preprocessing remain open. Fixture supports #9 causal model smoke.
