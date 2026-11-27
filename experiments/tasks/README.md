# REP task specifications
Read [AGENTS.md](../../AGENTS.md) and [EXECUTION.md](../../EXECUTION.md) before any task. Issues are authoritative for lifecycle; tasks are versioned procedures, not a separate queue. Specification on reviewed main + explicit local owner launch required. Do not execute draft/blocked or dependent tasks automatically.
- [T0001](T0001-gpu-validation.md) / [#27](https://github.com/jtrefon/splot-lm/issues/27): completed/closed; both attempts on main.
- [T0002](T0002-nano-cuda.md) / [#28](https://github.com/jtrefon/splot-lm/issues/28): Nano one-step validation; ready after reviewed amendment merges and explicit owner launch.
Use [TEMPLATE.md](TEMPLATE.md) for future tasks. Evidence path per attempt; [results index](../results/INDEX.md). Read live issues and evidence index for status.

- [T0003](T0003-nano-epoch.md) / [#35](https://github.com/jtrefon/splot-lm/issues/35): full synthetic Nano FP32 epoch; depends on reviewed/merged #34 and reviewed task amendment, explicit owner launch. No automatic execution.

- [T0004](T0004-mac-corpus-reference.md) / [#38](https://github.com/jtrefon/splot-lm/issues/38): Mac CPU synthetic data tooling, spec #39 merged; independent owner launch.
- [T0005](T0005-bf16-harness.md) / [#40](https://github.com/jtrefon/splot-lm/issues/40): Linux BF16 harness implementation, ready only after reviewed spec merge and owner launch.

- [T0006](T0006-muon-hybrid.md) #48: accepted/closed (#50 merged).
- [T0007](T0007-native-fp8-probe.md) #51: Linux native FP8 qualification, owner launch after merge.
- [T0008](T0008-byte-bpe-oracle.md) #52: independent Mac tokenizer oracle and mapping fix, owner launch after merge.
