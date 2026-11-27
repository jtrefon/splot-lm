# Native MTP CPU evidence
2026-11-27 | #2 | Synthetic correctness/overfit, not quality benchmark.
Python 3.13.5, PyTorch 2.10.0+cpu, CPU one thread, seed 42; no CUDA. Commands: PYTHONPATH=src:tools python tests/test_mtp.py; PYTHONPATH=src:tools python tools/smoke_mtp.py --mode A0 (repeat A1 and A2).
Locally tested source Git blobs, verified against uploaded tree:
- src/splot/mtp.py 8c4a1a21541a3cb1142a7293cf14a31f114991d5
- src/splot/__init__.py e01e9c2b59f3a074ef5986b48b870a006bc9b2df
- tests/test_mtp.py fc9fb4e32c26cc757f1f6b3ea9e1aa4bf23a4a05
- tools/smoke_mtp.py fb4d2f3c663c2a47a14f81f541195745b221fb91
- inherited dense.py 2ba9ac5e5b9c5990bf3e1c69e6c81e27bf54fd75; recurrent.py c5920eb33155e1e517d3a556b2aedfc72cc3c19e; tools/data_fixture.py 4cd314cd765a90861a84a364220f6dddf9d2ab2c.
Container had no GitHub DNS access; inherited sources reconstructed locally and verified by matching Git blob hashes. Seven tests passed (0.896s local runner, not timing benchmark); names in test file. FP64 atol=rtol=1e-10. Older suites not rerun.
Config default RecurrentConfig (264 vocab,d32,P/M/Q=2/4/2,R_cap=4,R=2, I0/S0,no depth embedding) plus offset projection and norm, 83776 parameters for all modes. AdamW lr=.003, weight_decay=0, 40 steps, pack two fixed documents "A prism bends light." and "Żółty pryzmat załamuje światło." into length 64. Valid offset-1 count 59; offset-2 count 57 for A2. Loss entries before optimizer steps 1 and 40. Both weights .25. No checkpoint saved.
| Mode | Initial total | Final total | Initial final NTP | Final final NTP | Initial early NTP | Final early NTP | Initial offset2 | Final offset2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 5.602704525 | 2.419738770 | 5.602704525 | 2.419738770 | inactive | inactive | inactive | inactive |
| A1 | 6.999616623 | 3.109294891 | 5.602704525 | 2.486731052 | 5.587647915 | 2.490254879 | inactive | inactive |
| A2 | 8.397461891 | 3.676704407 | 5.602704525 | 2.390770912 | 5.587647915 | 2.401129961 | 5.591382504 | 2.742604733 |
All three passed total <60% initial and finite active gradients; inactive auxiliary grad None. Different objective costs and scalar scales, no compute matching, one seed, no holdout, no comparison claim. No GPU, BF16, actual tokenizer, corpus, GDN/MoE, exact verifier or speed measurement. See implementation contract for next gates.
