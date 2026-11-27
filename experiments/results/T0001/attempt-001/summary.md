# T0001 attempt-001 — blocked on CUDA environment setup

Task T0001 (VAL), parent #10, issue #27. Instruction commit `1cb139df8e870c42a7a01eaf57624c85e3aa8a80`; source pin and tested commit `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa`.

## Scope and authorization
Owner explicitly authorized the local T0001 launch in the current session on 2026-11-27. Scope was the task as written: CPU regressions, full synthetic harness integration, bounded FP32 CUDA smoke and same-device checkpoint/resume equivalence, with a 60-minute foreground limit. No source edits, no tolerance changes, no real corpus, no Nano/Mini training.

## Outcome
**Blocked.** All CPU checks passed. The first CUDA invocation exited before importing/modeling SPLOT because `PYTHONPATH=src:tools` from task step 2 was not present in that shell; the exact error was `ModuleNotFoundError: No module named 'splot'`. No model code executed, no checkpoint was created, and the dependent CUDA checks were not run. This is an operator environment setup error, not a model or hardware result.

## Observed checks
| Check | Status | Evidence |
|---|---|---|
| environment | passed | `environment.json`, `execution.log` |
| cpu_regression | passed | `cpu-regression.json`, `execution.log` |
| cpu_integration | passed | `cpu-integration.json`, `execution.log` |
| cuda_continuous | blocked | `execution.log` |
| cuda_split | not_run | dependency on `cuda_continuous` |
| cuda_resume | not_run | dependency on `cuda_split` |
| cuda_equivalence | not_run | dependency on both 4-step runs |

## CPU results (observed)
- 11/11 suites exited 0; **66 tests** reported (sum of per-suite unittest counts).
- Integration exited 0 with one test; actual and declared parameter counts both **179,952**.
- Source HEAD was exactly `d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa` and `git status --porcelain` was empty at start and end.

## CUDA first failure and reproduction
Command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 300s python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out <RUNS>/continuous --steps 4 --device cuda
```
Exit: 1. Stderr ended with:
```
ModuleNotFoundError: No module named 'splot'
```
Full command record and traceback are in `execution.log`. No model code or CUDA kernel executed.

## Metrics
`metrics.jsonl` is empty: no training step or validation metrics were produced because the CUDA command failed before model import. CPU suite counts are in `cpu-regression.json` and `cpu-integration.json`; they are not training metrics. No performance, memory or quality inference is made.

## Deviations and limitations
- Deviation: the CUDA shell lacked the step-2 `PYTHONPATH=src:tools` environment; the command failed before model import. The attempt was stopped rather than retried in place.
- No CUDA model execution, checkpoint hash, resume comparison or equivalence evidence.
- Inferences above the smoke topology and FP32 synthetic-v1 data are not supported.
- The local inference process was stopped to free GPU 0 under owner authorization and left stopped for manual owner restart.

## Proposed next action
Execute a new attempt (002) with `PYTHONPATH=src:tools` explicitly set for every command in its shell, same source pin and exact task checks. Treat attempt-001 as a setup-blocked handover, not as CUDA evidence. No source change is proposed.
