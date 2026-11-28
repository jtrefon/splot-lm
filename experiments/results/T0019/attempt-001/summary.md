# T0019 attempt-001 — matched baseline and capability evaluation design
2026-11-28 | ANA | Issue #89 | Parents #85/#10/#8/#87 | PR (see issue #89)

**Outcome:** passed as an analysis/design handover. No training, model implementation, test-data access, checkpoint access, package installation or hardware claim.
**Instructions:** `f49af5bfb0085cbbc7161d5b3354b8e8fe309696` (reviewed main containing the merged T0019 spec).
**Source pin:** `6e8911662ac56df665147cb79e412f0383cf71d8` (read-only detached checkout). The pin predates the merged T0019 specification by design; the spec was read from instruction main.
**Dependency:** T0018 issue #81 accepted/closed; PR #91 merged at `f49af5b`.

## 1. Inventory

From the pinned checkout:

- Nano physical parameters: **43,055,076** at V32768, base 42,907,236, MTP auxiliary 147,840. NTP-only/A2 logical blocks are 12/14.
- NTP-only forward MACs: 41,416,704 decode and 20,602,552,320 prefill at context 512; 48,494,592 and 89,657,966,592 at 2048. A2: 83,392,512 / 41,893,232,640 at 512; 92,829,696 / 177,236,606,976 at 2048.
- Existing dense 8-block control: **25,172,352** without MTP (25,320,192 with a matched head), confirmed by component formula `12,582,912 + 8×1,573,632 + 384`. It is a correctness control, not matched to 43M.
- The current dense reference supports forward/cache and a two-example smoke trainer, but not the full pilot training/evaluation path, MTP/offset-2, dense accounting in product tooling or dense MTP optimizer classification. The detailed gap table is in `design.md`.

All MAC numbers are ideal contraction counts, not training FLOPs, latency or memory. Local Mac has no torch; counts are formula arithmetic and commits, not local instantiation.

## 2. Candidate design

Primary vocabulary is **V16384** (actual T0016 tokenizer); V32768 is compatibility only. Formula and full enumeration are in `candidate-counts.py` / `candidate-counts.json`.

| Candidate | Width×Layers | Q/KV | FFN | Params | vs target | Prefill 512 | vs NTP | Prefill 2048 | vs NTP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Iso-parameter V16384 | 512×9 | 8/2 | 1,920 | 41,101,312 | −4.54% | 22,114,729,984 | +7.34% | 102,954,434,560 | +14.83% |
| Iso-forward-MAC V16384 | 512×6 | 8/2 | 2,816 | 38,542,336 | −10.48% | 20,402,667,520 | −0.97% | 91,274,346,496 | +1.80% |
| Iso-parameter V32768 | 512×6 | 8/2 | 2,176 | 41,032,704 | −4.70% | 21,677,735,936 | +5.22% | 96,374,620,160 | +7.49% |
| Iso-forward-MAC V32768 | 512×6 | 8/2 | 1,920 | 38,673,408 | −10.18% | 20,469,776,384 | −0.64% | 91,542,781,952 | +2.10% |

The selected shapes meet the 5% parameter or forward-MAC target where feasible; the V16384 iso-parameter forward gap and both parameter gaps are explicit. Parameter parity and compute parity cannot both be exact because SPLOT reuses core weights across loops. The design separates primary NTP-only from secondary A2 auxiliary comparison and requires a matched dense offset head for A2.

## 3. Evaluation design

- Data: T0016 `splot-pilot-blocks-v1` only after exact local manifest/tokenizer/source verification; family-level train/validation/test separation; test untouched until freeze.
- Metrics: token-weighted held-out NTP NLL (nats/valid label), per language and aggregate; bits/byte with explicit UTF-8 coverage and BOS/EOS/pad reconciliation; PPL only with identical tokenizer.
- Guardrails: key–value retrieval with distractors and a compositional pattern/arithmetic probe, each with unseen generators/instances and chance baseline.
- Design: 3 arms × 3 planning seeds × 128 updates, B1/T512, 46,076 training NTP labels per pass, equal data order/token/tuning budgets, BF16 + FP32 master state, same semantic optimizer family, validation-only selection, paired bootstrap over held-out families/blocks.
- Proposed margins (not preregistered): NTP NLL noninferiority +0.05 nats/token, minimum meaningful advantage −0.03; bits/byte +0.01; guardrail accuracy −2 percentage points; cost guardrail +10% descriptive. Three seeds are planning evidence, not a powered confirmatory test.
- Budget: 1,152 updates / 589,824 valid-token presentations / 414,684 NTP label evaluations. Accepted T0012/T0011 anchors suggest roughly 45–75 minutes, up to about 2 hours with equal tuning; this is a rough planning range, not a matched-pilot measurement.

## 4. Reproduction and first failure

Commands and exact outputs are in `execution.log`, `accounting.json` and `candidate-counts.json`. The four pinned accounting commands exited 0. `consistency-check.py` initially failed with `FileNotFoundError` because it used an evidence path relative to an already-relative working directory; it was corrected to be `__file__`-relative and rerun successfully. No analytical value changed. The failed attempt and correction remain in `execution.log`.

## 5. Limits

- Analysis/design only: no model training, architecture implementation, data acquisition, package install, GPU/MPS use, test access or private checkpoint access.
- Local torch is absent; parameter/MAC formulas were not instantiated as modules locally.
- Candidate counts are analytical and exclude backward, optimizer, padding, memory traffic and kernel efficiency.
- The pilot corpus is small and not representative; the budget is a planning estimate from different accepted synthetic measurements.
- T0017 is not accepted and its results were not used.
- Margins, seeds, probes and budgets require review before becoming preregistered gates. No quality, speed, memory or scaling claim.

## 6. Handover

Evidence is flat under `experiments/results/T0019/attempt-001/`: `design.md`, `summary.md`, `result.json`, `environment.json`, `metrics.jsonl`, `execution.log`, `accounting.json`, `candidate-counts.json`, `candidate-counts.py`, `consistency-check.py` and this summary. No raw text, token arrays, weights or private data are present.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0019/attempt-001 \
  --task T0019 --source 6e8911662ac56df665147cb79e412f0383cf71d8 \
  --checks environment,inventory,matched_design,evaluation_design,handover
```

Its output is recorded in `result.json` and the PR. One INDEX row is appended; issue #89 receives the handover. No merge, closure, training or next-task launch.
