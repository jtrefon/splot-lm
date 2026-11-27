# REP execution evidence index
Append one row per submitted attempt; live issues/PRs determine review state. Never replace prior failed attempts. See [EXECUTION.md](../../EXECUTION.md). This index is navigation, not scientific acceptance. Historical pre-REP evidence remains in this directory and is indexed by PROJECT_STATE.md and the relevant RFC/implementation contracts.

| Task | Attempt | Tested source | Outcome | Headline observation | Evidence | Issue / PR |
|---|---|---|---|---|---|---|
| T0001 | 001 | d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa | blocked | CPU 11/11 suites (66 tests) and integration passed; CUDA command failed before model import because PYTHONPATH=src:tools was absent in that shell; no checkpoint or CUDA evidence. | [attempt-001](T0001/attempt-001/summary.md) / [result.json](T0001/attempt-001/result.json) | #27 / PR #31 |
| T0001 | 002 | d424a2ad75619dfb8821f260edaa6ab5fc1bb3aa | passed | All seven checks passed: CPU 11/11 suites (66 tests), integration, 4-step CUDA continuous, 2+2 split/resume and exact same-device checkpoint equivalence; 179,952-parameter synthetic smoke. | [attempt-002](T0001/attempt-002/summary.md) / [result.json](T0001/attempt-002/result.json) | #27 / PR #32 |
