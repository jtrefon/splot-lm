# T0001 attempt-002 — passed (FP32 CUDA correctness and checkpoint equivalence)

Task T0001 (VAL), parent #10, issue #27. Instruction commit `1cb139df8e870c42a7a01eaf57624c85e3aa8a80`; source pin and tested commit `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa`. UTC 2026-11-27T13:49:18Z to 2026-11-27T13:51:34Z.

## Authorization and scope
Owner explicitly authorized the local T0001 launch in this session. Attempt-002 ran the task as written with the step-2 `PYTHONPATH=src:tools` environment set for every command, including CUDA. No source edits, no tolerance changes, no new dependencies, no real corpus, no Nano/Mini training. Local inference was stopped for the GPU window under owner authorization and left stopped for manual owner restart.

## Outcome
**Passed.** All seven required checks passed. No task deviation; attempt-001 is a separate preserved setup-blocked attempt and was not retried in place. This is correctness evidence for the 179,952-parameter synthetic smoke topology only.

## Checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| cpu_regression | passed | `cpu-regression.json`, `execution.log` |
| cpu_integration | passed | `cpu-integration.json`, `execution.log` |
| cuda_continuous | passed | `run-continuous.json`, `execution.log` |
| cuda_split | passed | `run-split.json`, `execution.log` |
| cuda_resume | passed | `run-resumed.json`, `execution.log` |
| cuda_equivalence | passed | `comparison.json`, `execution.log` |

## Observed results
- Source HEAD was exactly `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa` and the worktree was clean at start and end.
- Environment: Python 3.12.7; torch 2.10.0+cu128; CUDA runtime 12.8; numpy 1.26.4; RTX 4090, 24,564 MiB, driver 580.178.04; GPU 0; 24,077 MiB free after the authorized inference stop.
- CPU: 11/11 suites exited 0; 66 unittest tests. Integration exited 0 with one test; actual and declared parameters both 179,952.
- Continuous CUDA: 4 steps; 266 valid tokens; 260 supervised NTP labels; validation NLL sum 1949.4920654296875 over 376 labels, mean NLL 5.184819322951297, perplexity 178.54118858045874. Objective moved 8.40405559539795 -> 7.995279788970947.
- Split CUDA: 2 steps; 141 valid tokens; 138 supervised NTP labels; validation NLL sum 2012.6712951660156 over 376 labels, mean NLL 5.352849189271319, perplexity 211.2092164282508. Objective moved 8.40405559539795 -> 8.245919227600098.
- Resumed CUDA: 4 total steps; 266 valid tokens; 260 supervised NTP labels; validation NLL sum 1949.4920654296875 over 376 labels, mean NLL 5.184819322951297, perplexity 178.54118858045874, identical to continuous.
- Peak CUDA allocated/reserved bytes were finite and positive for every step; exact per-step values are in `metrics.jsonl`.
- Checkpoints: continuous `18a16598001bd04a19eb708cff900d121c5fd33d9d573f4f14cbb00d5f04d9fe` (2,414,603 bytes); split-before-resume `6aa3b67cb96afc5b4fe4c8935f14422353148d9ec5cad8760897c90eb6b587c5` (2,414,283 bytes); resumed-final `6f3e59e3a9d92c9a2e43a75c2e6203aea95e186dbc370deacae57c008aceb984` (2,414,795 bytes). All are local-only and not published.
- Equivalence: `compare_resume.py` exited 0, reporting model/optimizer/RNG equality, identical non-timing history and identical validation. Timing and allocator peaks are explicitly excluded.

## Metrics
`metrics.jsonl` contains the raw per-step observations with run/check labels, units, values and CUDA peak bytes, plus validation and equivalence observations. The resumed history repeats the first two steps from the split run; those rows are marked `repeated_prefix` and must not be summed with the split run. Wall-clock `step_seconds` are reference-only and excluded from equality; no throughput or performance claim is made.

## First failure
No failure in attempt-002. Attempt-001 is a separate blocked preflight: its first CUDA command omitted `PYTHONPATH=src:tools` and failed before model import with `ModuleNotFoundError: No module named 'splot'`; no model code or checkpoint. It remains preserved at PR #31 and was not retried in place.

## Limitations
- Smoke topology (179,952 parameters) and synthetic-v1 fixture only; not Nano/Mini feasibility.
- FP32 only; no BF16/FP8, activation-checkpointing or optimizer-memory study.
- Single machine, single GPU and one software stack; no cross-device or cross-version determinism claim.
- Validation is synthetic-template validation, not language quality or generalization.
- No measured FLOPs, energy, throughput or PCIe/offload claim.
- No real corpus or production tokenizer.

## Proposed next action
Maintainer review of this evidence PR and the prior attempt. Do not run T0002 automatically; it remains blocked until this evidence is reviewed and the owner provides a new explicit authorization, finalized source pin and limits. No source change is proposed by this VAL attempt.
