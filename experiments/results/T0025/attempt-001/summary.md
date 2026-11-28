# T0025 attempt-001 — opt-in chunk64 GDN full-model qualification (negative candidate)

Task T0025 (IMP+VAL), issue #105, parents #85/#7/#10/#87/#102. Instruction/base `a87adbd1e7186926fa1766adc4aa8470b71d8579`; tested implementation `7dac49e083eae0068ff361e7a81b3a4f41055b3f` in `feat/T0025-gdn-chunk64-attempt-001`. UTC 2026-11-28T17:03:53Z to 2026-11-28T17:16:34Z. Overall: **failed** — prerequisite, implementation and unit tests passed; the T512 full-model chunk64 candidate failed the preregistered hidden/logit and routing gates. T2048 and resources were not run.

## Checks

| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `source-verification.json`, `execution.log` |
| prerequisite_reset | passed | `prerequisite-reset.json`, `gdn_prereq.py`, `execution.log` |
| implementation | passed | `implementation.diff`, `implementation.files.json`, `source-verification.json`, `summary.md` |
| unit_tests | passed | `unit-tests.txt`, `resume-gate.json`, `resume-scan-compare.json`, `resume-chunk-compare.json`, `execution.log` |
| full_model | failed | `full-model-512.json`, `execution.log` |
| resources | not_run | T512 candidate gate failed; no T2048/resource run |
| handover | passed | flat package; validator result in PR body |

## Environment and source

- GPU 0, NVIDIA GeForce RTX 4090, 24564 MiB total / 24077 MiB free at preflight; driver 580.178.04.
- Python 3.12.7, torch 2.10.0+cu128, CUDA 12.8; deterministic algorithms, TF32 off, CUBLAS workspace `:4096:8`.
- Host RAM available 26795646976 B; scratch free 181477859328 B; limits 12 GiB / 8 GiB exceeded.
- Base `a87adbd1e7186926fa1766adc4aa8470b71d8579` clean; tested `7dac49e083eae0068ff361e7a81b3a4f41055b3f` adds 7 authorized files. No packages, real data, checkpoints, Mini, FP8 qualification, compilation, background work or automatic retries.

## Prerequisite reset/tail

FP64, T129, seed42 carry document matching the T0024 input, C16/32/64:

| Chunk | Doc A ref rel/abs | Doc B ref rel/abs | A rerun after B rel/abs | 37+92 carried split rel/abs | Status |
|---:|---|---|---|---|---|
| 16 | 6.429e-16 / 5.551e-17 | 6.672e-16 / 5.551e-17 | 6.429e-16 / 5.551e-17 | 6.432e-16 / 4.857e-17 | passed |
| 32 | 7.018e-16 / 8.327e-17 | 7.264e-16 / 6.939e-17 | 7.018e-16 / 8.327e-17 | 6.916e-16 / 5.551e-17 | passed |
| 64 | 9.004e-16 / 6.592e-17 | 8.365e-16 / 8.327e-17 | 9.004e-16 / 6.592e-17 | 7.033e-16 / 5.551e-17 | passed |

All independent-document, no-contamination and carried split/tail checks passed; gates were FP64 relative L2 and max abs ≤1e-9. **T0024 prerequisite cleared.**

## Implementation

- `delta_chunk64` composes the existing `delta_chunk` oracle over consecutive ≤64-token slices with carried FP32 state and no detach.
- `RecurrentConfig.gdn_algorithm` is a validated optional field: default `scan`; opt-in `chunk64`. Existing oracle `GatedDeltaNet.run(..., algorithm='chunk')` is unchanged.
- `HybridLM` wires the config into the three GDN blocks. No new parameters/buffers/state_dict entries; legacy configs omit the field and preserve their old identity. FP8 + chunk64 is rejected by `tools/experiment.py`.
- Scan default preserves the accepted T0022 T512 input/init/partition/losses exactly in the full-model probe.
- Full driver and tests: `gdn_chunk_probe.py`, `tests/test_gdn_chunk64.py`; source diff in `implementation.diff`.

## Unit tests and mode identity

- Local full CPU: **31 scripts, 288 tests, 31 REP reports**, 0 failures/errors. New chunk64 tests 4/4: FP64/FP32 numerics and gradients, default/config inventory, packed rows/document reset/padding, prefix cache and logical passes, cache ownership.
- Tiny CPU 4 vs 2+2: scan exact and chunk64 exact for model, optimizer, RNG, non-timing history and validation (`resume-gate.json`).
- Cross-mode chunk64 checkpoint resumed with scan config exited 1 with `ValueError: run identity mismatch`; checkpoint SHA unchanged.
- No legacy config digest/state_dict inventory change; scan defaults unchanged.

## T512 full-model candidate result

The scan control matched the accepted T0022 T512 baseline exactly (input/init/partition and all losses); candidate and control used identical initialization/input/partition and the same config except algorithm.

Loss components all passed the ≤0.02 nats gate:

| Component | Absolute difference | Status |
|---|---:|---|
| balance | 0.000063896 | passed |
| early_ntp | 0.000059128 | passed |
| final_ntp | 0.000124931 | passed |
| offset2 | 0.000134468 | passed |
| total | 0.000171661 | passed |

Gradients passed: same present/missing inventory, global relative L2 `0.005046450` (≤0.05), all per-tensor gates passed, finite state. **Tensor and routing gates failed:**

| Tensor | Relative L2 | Max abs | Reference max abs | Max-abs limit | Status |
|---|---:|---:|---:|---:|---|
| hidden_1 | 0.015144 | 0.090125 | 0.781867 | 0.050000 | failed |
| hidden_2 | 0.032722 | 0.175865 | 1.093920 | 0.054696 | failed |
| logits_1 | 0.013531 | 0.239258 | 2.203125 | 0.110156 | failed |
| logits_2 | 0.032275 | 0.392578 | 2.140625 | 0.107031 | failed |
| offset2_1 | 0.015817 | 0.312256 | 2.125000 | 0.106250 | failed |
| offset2_2 | 0.033236 | 0.396484 | 2.093750 | 0.104688 | failed |

Routing top2 sets must be exact; disagreement was **1.953125%** overall, by pass `{'C': 0.00830078125}` and pool `{'0': 0.00390625, '1': 0.0087890625, '2': 0.0126953125, '3': 0.0078125}`. Because routing differs, full-model equivalence is rejected rather than relaxed. All model/optimizer/gradient tensors were finite.

Per spec, the first candidate gate failure stopped T2048 and all resource measurements. **T2048 and resources are `not_run`.** No threshold, tolerance or routing gate was changed.

## Interpretation and next action

The opt-in chunk64 implementation is present, default-off and unit-qualified, but it is **not full-model equivalent** at T512 under the preregistered gates. The scan default is preserved; no production, quality, throughput or architecture-superiority claim is made. The candidate should not proceed to T2048/resource or adoption without a separately reviewed analysis of the hidden/logit amplification and routing sensitivity. No further task was launched here.

## Limitations

- Full-model evidence is one BF16 synthetic update at T512; no T2048, resource, learning or quality evidence.
- The candidate is numerically different contraction order; tensor/routing divergence is a real negative result, not a tolerance to relax.
- `delta_chunk` remains a small finite-log-decay dense triangular oracle, not a production kernel.
- T0024 component feasibility does not imply full-model equivalence.

## Handover

Evidence contains prerequisite metrics, implementation diff/files, unit-test log, resume/mode-identity gates, the T512 full-model failure metrics and summary. No raw tensors, token arrays, private paths or checkpoints are published.

Validator command:

```bash
python3 tools/rep/validate_result.py experiments/results/T0025/attempt-001 \
  --task T0025 \
  --source 7dac49e083eae0068ff361e7a81b3a4f41055b3f \
  --checks environment,prerequisite_reset,implementation,unit_tests,full_model,resources,handover
```

T0025 remains open; no merge or closure. No further task is started.
