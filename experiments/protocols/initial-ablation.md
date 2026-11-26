# Initial ablation protocol (draft)
Use the same tokenizer, held-out document split, dataset mixture, evaluation harness and optimizer tuning budget. Compare distinct matched-token, matched-resident-parameter and matched-training-FLOP conditions; do not pretend all can match simultaneously. Report logical FLOPs, physical and activated parameters, wall clock and kernel overhead. Use >=3 seeds for principal comparisons if budget permits, paired evaluation and uncertainty intervals. Record exceptions. Begin with smoke tests, then ~100M–1B tokens per pilot as budget allows.

| ID | Intervention |
|---|---|
| E0 | Dense GQA baseline, NTP |
| E1 | Hybrid GDN/GQA only |
| E2 | Sparse MoE only |
| E3 | Hybrid + MoE |
| E4 | Fixed-depth recurrence (also dense recurrent control) |
| E5 | Adaptive depth after cache semantics |
| E6 | Expert locality objective and offloading |
| E7 | FP8 training vs BF16 |
| E8 | Loop-aware simulated FP4 QAT vs PTQ |
| E9 | Native MTP: factorial across recurrence and mixer, M0–M5 in RFC 0005 |

MTP correctness is an early implementation gate, even though E9 is the historical experiment ID. Metrics: loss/perplexity, reasoning and code accuracy, long-context retrieval, throughput (prefill/decode), TTFT, peak VRAM/RAM, energy/token, FLOPs, PCIe bytes, cache hits, expert entropy/utilization/switches, acceptance by offset and rollback latency. Report full distribution equivalence for sampling, greedy exact match, quantization depth drift. Pilot thresholds TBD before training. Avoid benchmark contamination, document truncation and train/validation overlap. Publish negative results, manifests and commit hashes.
