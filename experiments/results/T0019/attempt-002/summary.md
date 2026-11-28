# T0019 attempt-002 — corrected matched baseline and capability design
2026-11-28 | ANA | Issue #89 | Source pin `6e8911662ac56df665147cb79e412f0383cf71d8` | Instructions `a771c370efe662f17ddf8c06769cbd06a783388a`

**Outcome:** passed as a corrective analysis/design handover. Attempt-001 is preserved and remains partially reviewed. No candidate is selected; no implementation, training, package install, test data or private checkpoint was used.
**Review record:** [research/evaluation/T0019-review.md](../../../research/evaluation/T0019-review.md).
**Dependency:** T0018 #81 closed; PR #91 merged at `f49af5b`.

## 1. Same-vocabulary corrections

The prior V16384 candidates were compared with V32768 SPLOT totals. The pinned `account(Shape(vocab=16384), loops=2, ...)` gives:

| Vocab | NTP physical | NTP base | MTP aux | NTP prefill 512 | NTP prefill 2048 | A2 prefill 512 | A2 prefill 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 36,763,620 | 36,615,780 | 147,840 | 17,381,326,848 | 76,773,064,704 | 29,008,330,752 | 125,696,999,424 |
| 32768 | 43,055,076 | 42,907,236 | 147,840 | 20,602,552,320 | 89,657,966,592 | 41,893,232,640 | 177,236,606,976 |

NTP decode MACs: V16384 35,125,248 / 42,203,136 at 512/2048; V32768 41,416,704 / 48,494,592. Cache bytes are vocabulary-independent. The prior 41,101,312-parameter option is **+11.80%** versus the V16384 total with auxiliary head and **+12.26%** versus the NTP base, not −4.54%.

## 2. Candidate options (not selected)

Dense NTP counts exclude a dormant MTP head; optional auxiliary counts are separate and are not used to claim NTP parity.

| Vocab | Option | Width×Layers | Q/KV | FFN | NTP params | vs NTP base | Prefill 512 | vs NTP 512 | Prefill 2048 | vs NTP 2048 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16384 | Nearest parameter | 512×6 | 8/4 | 2,432 | 35,527,168 | −2.97% | 18,993,381,376 | +9.27% | 85,637,201,920 | +11.55% |
| 16384 | Nearest forward MAC | 512×6 | 8/2 | 2,176 | 32,381,440 | −11.56% | 17,382,768,640 | +0.0003% | 79,194,750,976 | +3.15% |
| 32768 | Nearest parameter | 512×6 | 8/2 | 2,176 | 40,770,048 | −4.98% | 21,677,735,936 | +5.22% | 96,374,620,160 | +7.49% |
| 32768 | Nearest forward MAC | 512×6 | 8/2 | 1,920 | 38,410,752 | −10.48% | 20,469,776,384 | −0.64% | 91,542,781,952 | +2.10% |

These are options for review, not matched selections. Parameter and forward-MAC parity cannot both be exact because SPLOT reuses core weights across loops. Decode gaps and full enumeration are in `candidate-counts.json`.

## 3. Regimes, budget and evaluation corrections

- **NTP-only primary:** compare SPLOT NTP base/active parameters and MACs against dense NTP candidates without a dormant head. Report total physical with dormant MTP and active/optimized parameters separately; do not add dormant parameters to claim parity.
- **A2 deferred:** SPLOT A2 has two NTP readouts plus two offset readouts with `depth_weight=0.25` and `offset_weight=0.25`. A single dense offset head plus one readout is not a credible comparator. No A2 candidate is selected; A2 accounting is retained only as cost context.
- **Budget units corrected:** one T0016 training pass is 128×512 = 65,536 **positions**, 46,204 valid tokens, 46,076 NTP labels and 45,948 offset labels. Nine runs = 589,824 positions, 415,836 valid tokens, 414,684 NTP labels; NTP-only uses 0 offset labels. The previous 589,824 “valid tokens” was positions.
- **Probes:** key–value retrieval uses 16 single-token values, four distractors and chance 1/16; compositional completion uses an 8-symbol held-out rule and chance 1/8. Probe instances/templates are disjoint from training, never used for tuning, and audited against train n-grams. If all arms are within 2 pp of chance, the probe is uninformative.
- **Uncertainty:** the validation split has only two families, so no population/family-level confidence or equivalence claim is made. Report per-family/per-block descriptive NLL and seed ranges; bootstrap only independent probe instances. Margins (+0.05 NLL noninferiority, −0.03 meaningful advantage, +0.01 bits/byte, −2 pp guardrails, +10% cost) remain planning proposals, not preregistered gates.
- **Budget estimate:** 45–75 minutes, up to about 2 hours with equal tuning, from accepted T0012/T0011 anchors; rough planning only.

## 4. First failures and preservation

The first candidate-counts run failed with `KeyError: base` and the second with `KeyError: auxiliary` because same-vocab counts are nested under `ntp_512`/`ntp_2048`. Both were schema-access bugs in the new evidence script, corrected before the final transcript; no analytical value changed. The failure-only log is preserved as `execution-first-failures.log`; the complete corrected transcript is `execution.log`. Attempt-001 is untouched.

## 5. Limits

- Analysis/design only; no candidate selection, model/training implementation, install, GPU/MPS, test or checkpoint access.
- Local Mac has no torch; counts are pinned accounting/formula values, not locally instantiated modules.
- MACs exclude backward, optimizer, padding, memory traffic and kernel efficiency.
- The pilot is small; two validation families cannot establish population generalization.
- The budget is extrapolated from different accepted synthetic measurements.
- T0017 results were not used. No quality, speed, memory or scaling claim.

## 6. Handover

Evidence is flat under `experiments/results/T0019/attempt-002/`: `design.md`, `summary.md`, `result.json`, `environment.json`, `metrics.jsonl`, `execution.log`, `execution-first-failures.log`, `accounting.json`, `account-vocab.py`, `candidate-counts.py`, `candidate-counts.json`, `consistency-check.py`.

Validator:

```bash
python3 tools/rep/validate_result.py experiments/results/T0019/attempt-002 \
  --task T0019 --source 6e8911662ac56df665147cb79e412f0383cf71d8 \
  --checks environment,inventory,matched_design,evaluation_design,handover
```

Its output is recorded in `result.json` and the PR. One INDEX row is appended; issue #89 receives the handover. No merge, closure, training or next-task launch.
