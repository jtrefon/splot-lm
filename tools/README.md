# tools

Future reproducibility and analysis tools.

## Reproducible synthetic experiments
See [harness contract](../research/implementation/experiment-harness.md). `python tools/test_experiment.py` tests mechanics; `PYTHONPATH=src:tools python tools/test_experiment_integration.py` tests full SPLOT (pending execution). `PYTHONPATH=src:tools python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out ../splot-runs/smoke-42 --steps 4` from clean checkout. Real corpus and BF16 not enabled. Keep outputs outside repo.
