# Reproducible synthetic experiment harness v0.1
2026-11-27 | #10 | [Polski](experiment-harness.pl.md)

## Scope
PR #25 merged: complete all-resident SPLOTLM reference, exact Nano/Mini inventories and earlier CPU correctness/smoke evidence. This work unit adds experiment orchestration, not optimized training, real data or new model-quality results. The current CLI supports SPLOTLM only, synthetic-v1 byte fixture only, FP32, fixed depth and A0/A1/A2. The three JSON configs are executable constructor inputs rather than the illustrative YAML architectural proposals. Smoke is 179,952 parameters (264 vocabulary); Nano 43,055,076 and Mini 146,987,784 (32,768 capacity). Counts from previously merged evidence; new harness asserts actual count on construction. Larger configurations are not promises of feasible memory/runtime; full-vocabulary logits and explicit scans make them expensive. LR .003, zero weight decay, seed 42 and batch 1 are smoke choices, not preregistered production hyperparameters.

## Reproducibility
Require clean Git checkout; record actual HEAD, canonical JSON config SHA-256, synthetic manifest SHA-256, packed train/validation SHA-256, environment (Python, torch, device, CPU architecture/platform, CUDA runtime, GPU name, thread count), settings and source data manifest. No invented hashes. Set Python/torch seeds and CUDA RNG where applicable, one CPU thread, deterministic PyTorch algorithms, disable TF32 on CUDA. For CUDA set CUBLAS_WORKSPACE_CONFIG before process launch; record driver, GPU UUID/VRAM, clocks, OS and complete dependency lock separately (not automatically captured). Determinism is scoped to same environment and supported kernels, not cross-device or version equivalence. A deterministic GPU kernel failure must be reported rather than silently disabling enforcement.

Packer uses fixed train and validation documents; no test evaluation during tuning. Seed+epoch local Python shuffle over packed block indices, no replacement within epoch, retain shorter final batch. Step means one AdamW optimizer update, no gradient accumulation or scheduler. Record every batch index, valid tokens (including BOS/EOS), supervised NTP labels, available offset-2 labels, pre-update total objective and components, synchronized training-step elapsed wall time and CUDA peak allocated/reserved bytes. Skip no batches silently; fail on empty NTP batch or nonfinite objective/observed gradients. Unused experts may have no gradient. Validation uses final target NTP only; sum masked CE over labels then divide by total label count; do not average batch perplexities or compare total A2 objective to A0 NTP. Synthetic templates cross splits; not evidence of generalization. Test partition reserved.

## Checkpoint and resume
Atomically replace checkpoint.pt after each optimizer step (temporary sibling + rename; not a guarantee of power-loss durability). Stores model, optimizer, RNG state, immutable run identity and full step history; checkpoint is authoritative, run.json regenerated on successful completion. Resume requires exact identity (including code SHA/config/data/device/settings); --steps is total target step count, not additional updates. New run directory must be empty; no silent overwrite of unrelated runs. One writer per directory; no concurrent training or locking. Only load self-produced trusted checkpoint files; weights_only=True. Save every step for smoke correctness, not scalable checkpoint policy. Timing excludes checkpoint I/O, collation, logging and validation; no claim of end-to-end training throughput. CUDA peak values include persistent weights/optimizer/cache allocator reservation; CPU memory not measured. Missing metrics use null (not zero).

## Run after merge/review
Use clean checkout at exact commit and supported PyTorch; put runs outside repo.
```bash
python tools/test_experiment.py
PYTHONPATH=src:tools python tools/test_experiment_integration.py
PYTHONPATH=src:tools python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out ../splot-runs/smoke-42 --steps 4
PYTHONPATH=src:tools python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out ../splot-runs/smoke-42 --steps 8 --resume
```
GPU smoke (not executed):
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:tools python tools/experiment.py --config configs/experiments/splot-smoke-synthetic.json --out ../splot-runs/cuda-smoke-42 --steps 4 --device cuda
```
Run nvidia-smi and record driver/GPU/VRAM; compare CPU and CUDA loss/caches using separately declared tolerances, not exact training equality. Nano then Mini only after reviewing available memory and a short forward/backward; avoid long runs until #7 activation-memory policy. No GPU access in local sandbox. FP32 only until precision policy is reviewed.

## Accounting and comparability
Per-run analytical forward MACs sum RFC 0001 accounting over each valid packed document length and actual scheduled batch indices with selected readouts/offsets. It is an idealized unpadded contraction count, excluding scalar ops, actual dense masked/padded work, backward, optimizer, recomputation, transfer and kernel overhead; not measured FLOPs. Record realized label/valid-token budgets independently of steps. Matched-token study equalizes the actual content/label exposure (BOS/EOS counted separately in future real-corpus loader); parameter-matched studies use separately derived model controls; matched-training-compute study needs validated backward/profiler accounting rather than this forward estimate alone. This first CLI does not automate matching or build the dense/hybrid control experiments. Compare independent conditions/seeds only after full integration and data gate. No energy or profiler metrics yet.

## Evidence and limitations
Five local harness mechanics tests passed on a toy PyTorch embedding/dropout/linear model with AdamW: exact resume state/optimizer/history equivalence, identity and overwrite guards, schedule, weighted validation, logical MAC aggregation and digest stability (some assertions grouped). Actual SPLOT integration test and CLI run added, NOT executed locally: sandbox cannot clone GitHub (DNS resolution failure). Connector supplied exact current source/API for integration; local executed harness/test blobs match uploaded bytes. No SPLOT training metrics from this work unit. Existing SPLOT smoke evidence remains separate. See [results](../../experiments/results/experiment-harness-v0.1.md).

## Next
Execute integration test and CLI on clean checkout; record resulting run.json and validate parameter counts, resume and CUDA behavior. Add inference benchmark using committed target/shallow caches with separate prefill/TTFT/decode, D0/D1/D2 greedy parity, proposed/accepted/rejected counts, per-offset acceptance, replay duration, CUDA synchronization, memory and energy instrumentation; distinguish reference from optimized path. Implement budget matching and dense/hybrid controls, precision/checkpointing #7, actual approved data/tokenizer #8. Keep #10 open until these empirical criteria are met. No claimed speedup.
