# Experiments and reproducibility

The repository contains measured CPU/CUDA experiments, not just proposed protocols. See [evidence index](results/INDEX.md) for immutable attempt records, [task specifications](tasks/README.md) for preregistered scope and [REP](../EXECUTION.md) for execution and acceptance rules.

- Reference correctness, synthetic training, same-mode resume, precision, optimizer, context resource and weight-reuse evidence are available.
- Real bilingual data is materialized ([T0016](results/T0016/attempt-001/summary.md)); actual pilot execution is pending ([T0017 continuation](tasks/T0017-continuation.md)).
- Hosted CPU CI checks regressions and structural evidence hashes; it does not replicate local GPU results.

Store exact tested SHA, config/data/tokenizer hashes, seeds, environment, labels, metrics and limitations. Keep datasets, checkpoints, large logs and tensors local or in explicitly approved external storage; record hashes, bytes and availability. Never overwrite earlier failures or blocked attempts. Distinguish submitted, structurally valid and independently accepted evidence. No new local compute without the named task and owner authorization.
