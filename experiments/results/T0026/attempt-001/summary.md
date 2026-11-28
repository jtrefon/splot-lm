# T0026 attempt-001 — prospective matched learning evaluation and corpus protocol

Task T0026 (ANA), issue #108, parents #85/#8/#10/#86/#87. Instructions/source `003cf8db6126f0aaefdd599ac159f2503457f709`. Branch `research/T0026-matched-eval-attempt-001`. UTC 2026-11-28T17:40:36Z to 2026-11-28T17:45:24Z for the authorized design window; evidence assembly followed.

**Outcome:** **passed as an analytical design handover**. Proposed a prospective matched learning protocol, byte-auditable BPB schema, 1–5M-token bilingual corpus plan, frozen capability probes, paired-seed precision plan, optimized dense comparator requirements and a separate GDN localization diagnostic. No training, acquisition, tokenizer fit, install, test/checkpoint access, source edit or shared state-document edit. This is not scientific acceptance or preregistration of final margins.

## 1. Preflight

- `origin/main` and local `HEAD`: `003cf8db6126f0aaefdd599ac159f2503457f709`.
- Open PR queue: **0** verified at start.
- T0023 accepted/integrated; T0025 full-model chunk64 rejected and excluded; T0027 independent and unreviewed, its outcomes not used.
- Local macOS 27.2.1 arm64 Apple M4, 10 cores / 17,179,869,184 B RAM, CPU only, CPython 3.15.2; torch absent and not installed.
- Working tree clean apart from the new T0026 evidence directory.

## 2. Model accounting

| Arm | Physical params | NTP base / gradient-present | Active params | Prefill 512 MACs | vs S |
|---|---:|---:|---:|---:|---:|
| S: SPLOT A0 scan R2 | 36,763,620 | 36,615,780 | 22,460,004–27,178,596 | 17,381,326,848 | — |
| P: dense Q8/KV4 FFN2432 | 35,527,168 | 35,527,168 | 35,527,168 | 18,993,381,376 | +9.275% |
| C: dense Q8/KV2 FFN2176 | 32,381,440 | 32,381,440 | 32,381,440 | 17,382,768,640 | +0.0083% |

SPLOT physical includes 147,840 dormant MTP parameters under A0; the NTP base excludes them. Dense has no dormant head. Primary contrast is S−C (forward-MAC-matched); S−P is secondary parameter-matched. Decode MACs are reported separately (S 35,125,248; P/C 41,416,704 at T512, +17.9% dense).

## 3. Evaluation design

- **Primary endpoint:** token-weighted final validation NTP NLL, exact counted-label denominator, padding/cross-document labels excluded.
- **Aggregation:** token-weighted primary; per-language and per-family; family-macro secondary. Family bootstrap is sensitivity only with four validation families.
- **BPB:** content-token NLL divided by `ln(2) ×` exact UTF-8 bytes covered by those targets; EOS/control NLL reported separately; no zero-byte division.
- **Missing provenance identified:** v1 lacks per-token byte lengths/offsets, document byte ranges, per-token family/language IDs, control mask and tokenizer byte table. Proposed `splot-pilot-blocks-v2` adds these plus shard byte/family/language reconciliation and a reversibility audit.
- **Regimes:** matched token (primary quality), matched forward MAC (primary cost; C within 0.01% at T512), matched parameter (secondary), optional matched training compute with achieved gaps disclosed.
- **Dense comparator:** future optimized SDPA/causal-segmented dense path, same tokenizer/RMSNorm/RoPE/SwiGLU/tied embeddings, BF16 autocast FP32 master, same optimizer/data order; equal tuning trials and validation-only selection. Current dense path is a correctness reference, not an optimized baseline.
- **Seeds/cadence:** paired seeds, same data order; checkpoint every 16–32 updates; validation at fixed milestones and final; no early stopping for primary; test frozen.
- **Probes:** key–value retrieval (chance 1/16, 256 instances), compositional rule completion (chance 1/8, 256), document-order reconstruction (chance 1/2, 128); frozen, disjoint, n-gram audited, floor/ceiling handling, hidden test set.
- **GDN follow-up:** FP32 recurrence vs BF16 cast points, router logits/margins/rank flips by depth/pool/pass, fixed-route teacher forcing diagnostic only; no production change or equivalence claim.

## 4. Precision and margins

Paired-seed planning (assumed SD; no observed variance):

| Seeds | 95% CI half-width, SD 0.05 | MDE80, SD 0.05 |
|---:|---:|---:|
| 3 | 0.1242 | 0.1787 |
| 5 | 0.0621 | 0.0964 |
| 8 | 0.0418 | 0.0668 |
| 12 | 0.0318 | 0.0514 |
| 20 | 0.0234 | 0.0382 |

n=3 is descriptive only. A 0.05-nat margin at SD 0.05 needs about 12 seeds; a 0.03-nat advantage needs roughly 22–30. Proposed margins (not preregistered): NLL noninferiority +0.05, meaningful advantage −0.03; BPB +0.01/−0.005; capability −2 pp; training cost +10%. Noninferiority requires the two-sided 95% CI upper bound below +0.05; superiority below −0.03. No p-value or equivalence claim is made from T0016's two validation families.

## 5. Corpus and storage plan

| Train tokens | Validation | Test | Extracted UTF-8 (est.) | Raw (est.) | Compact 16 B/token | Current JSONL ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 50k | 50k | 4.30 MB | 5.36 MB | 17.6 MB | 53.1 MB |
| 3M | 150k | 150k | 12.89 MB | 16.07 MB | 52.8 MB | 159.3 MB |
| 5M | 250k | 250k | 21.48 MB | 26.78 MB | 88.0 MB | 265.4 MB |

Plan: 8/4/4 minimum families (12/6/6 preferred), family-disjoint splits, exact/normalized/near-duplicate dedup, 13-gram cross-split audit, one frozen train-only 16k byte-level BPE, source registry with rights evidence and immutable revisions, no acquisition. The 1M stage supports S0/S1 descriptive budgets; 5M is confirmation only if observed variance and family spread justify it.

## 6. Budget and compute estimates

| Stage | Positions/run | Updates/run | Runs | Total valid tokens (est.) | Wall at T0017 anchor |
|---|---:|---:|---:|---:|---:|
| S0 feasibility | 65,536 | 128 | 9 | 415,836 | 24.3 min |
| S1 learning curve | 196,608 | 384 | 9 | 1,247,508 | 72.9 min |
| S2 confirmation | 1,048,576 | 2,048 | 9 | 6,653,376 | 389 min |
| S3 scaling (optional) | 5,242,880 | 10,240 | 9 | 33,266,880 | 1,945 min |

At 5M tokens, analytical forward MACs are S 169.74T, P 185.48T, C 169.75T; estimated training FLOPs (6× forward MACs) are 1.018P, 1.113P and 1.019P; at an assumed 10 TFLOP/s that is 101.8/111.3/101.9 s per arm. These are not measured training costs; small-model batch-1 overhead dominates (T0017 anchor 1.266 s/update). S2/S3 need separate multi-window authorization.

## 7. First failures

The evidence-capture runner initially launched three local commands from the repository root instead of the evidence directory: `wc -c` (exit 1, files not found), `py_compile protocol_analysis.py` (exit 1, file not found) and `protocol_analysis.py` (exit 2, file not found). These were local command-context errors, not analysis or acceptance-gate failures; the analytical output did not change. They are preserved in `execution-first-failures.log`, and the corrected `execution.log` records exit 0.

## 8. Limitations and non-claims

- Design/analysis only: no training, measured quality, acquisition, tokenizer fit, test access or dependency install.
- Precision tables are assumption-driven planning aids, not power guarantees or observed variance.
- T0016 is a small eight-family pilot; its two validation families cannot support population inference.
- The current dense path is a correctness reference; no optimized comparator or measured cost exists yet.
- The v2 byte schema, corpus registry, optimized dense implementation and final margins are unresolved and require reviewed follow-up tasks.
- No model quality, superiority, equivalence, scaling or architecture-adoption claim. No merge, closure or next-task launch.

## 9. Handover

Evidence is flat under `experiments/results/T0026/attempt-001/`: `design.md`, `corpus-design.md`, `proposed-protocol.json`, `analysis.json`, `protocol_analysis.py`, `source-refs.json`, environment, result, metrics, execution log and preserved first-failure log. `proposed-protocol.json` carries the machine-readable arms, regimes, endpoints, margins, corpus, probes, GDN follow-up, unresolved decisions and dependency graph.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0026/attempt-001 \
  --task T0026 --source 003cf8db6126f0aaefdd599ac159f2503457f709 \
  --checks environment,evaluation_design,corpus_design,handover
```

Output: STRUCTURALLY VALID: passed (not scientific acceptance). One INDEX row is appended; PR and issue #108 carry the handover. No merge, closure or next-task launch.
