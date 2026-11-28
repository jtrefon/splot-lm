# T0019 attempt-002 — corrected matched baseline and capability design
2026-11-28 | ANA | Issue #89 | Source pin `6e8911662ac56df665147cb79e412f0383cf71d8` | Instructions `a771c370efe662f17ddf8c06769cbd06a783388a`

**Status:** corrected analytical design only. No candidate is selected; no model or training was implemented; no test data, private checkpoint or package install was used. All MACs are ideal contraction estimates, not training FLOPs, latency or memory.

This attempt responds to [the coordinator review](../../../research/evaluation/T0019-review.md). Attempt-001 is preserved untouched. The material correction is that V16384 comparisons must use the pinned SPLOT accounting with `Shape(vocab=16384)`, not V32768 totals. The prior 41,101,312-parameter option is **+11.80%** versus the V16384 total with auxiliary head (36,763,620), or **+12.26%** versus the NTP base (36,615,780), not the previously reported −4.54%.

## 1. Same-vocabulary inventory

`account-vocab.py` loads the pinned `tools/accounting.py` and calls `account(Shape(vocab=...), loops=2, ...)`. `accounting.json` records the four exact V32768 CLI commands plus both vocabularies.

| Vocab | Regime | Context | Physical | NTP base | MTP aux | Logical blocks | Decode MACs | Prefill all-logit MACs | Cache bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | NTP-only | 512 | 36,763,620 | 36,615,780 | 147,840 | 12 | 35,125,248 | 17,381,326,848 | 2,204,160 |
| 16384 | NTP-only | 2048 | 36,763,620 | 36,615,780 | 147,840 | 12 | 42,203,136 | 76,773,064,704 | 6,922,752 |
| 16384 | A2 | 512 | 36,763,620 | 36,615,780 | 147,840 | 14 | 58,226,688 | 29,008,330,752 | 2,728,448 |
| 16384 | A2 | 2048 | 36,763,620 | 36,615,780 | 147,840 | 14 | 67,663,872 | 125,696,999,424 | 9,019,904 |
| 32768 | NTP-only | 512 | 43,055,076 | 42,907,236 | 147,840 | 12 | 41,416,704 | 20,602,552,320 | 2,204,160 |
| 32768 | NTP-only | 2048 | 43,055,076 | 42,907,236 | 147,840 | 12 | 48,494,592 | 89,657,966,592 | 6,922,752 |
| 32768 | A2 | 512 | 43,055,076 | 42,907,236 | 147,840 | 14 | 83,392,512 | 41,893,232,640 | 2,728,448 |
| 32768 | A2 | 2048 | 43,055,076 | 42,907,236 | 147,840 | 14 | 92,829,696 | 177,236,606,976 | 9,019,904 |

The cache byte counts are vocabulary-independent. The MTP head is **147,840** parameters at d384 (`384² + 384`). For NTP-only, the comparison target is **36,615,780** at V16384 and **42,907,236** at V32768; total physical counts including the dormant head are reported separately, never added to a dense NTP candidate to manufacture parity.

The existing dense 8-block control is 25,172,352 without MTP (25,320,192 with a matched head), formula-verified as `12,582,912 + 8×1,573,632 + 384`; it is not matched to either SPLOT target.

## 2. Corrected candidate options (not selected)

Formulas in `candidate-counts.py`; full output is `candidate-counts.json`. Candidates here are **options for review**, not a selection or launch recommendation. Dense NTP counts exclude a dormant auxiliary head; optional auxiliary counts are shown separately only for later A2 discussion.

| Vocab | Option | Width×Layers | Q/KV | FFN | NTP params | vs NTP base | Prefill 512 | vs NTP 512 | Prefill 2048 | vs NTP 2048 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16384 | Nearest parameter | 512×6 | 8/4 | 2,432 | 35,527,168 | −2.97% | 18,993,381,376 | +9.27% | 85,637,201,920 | +11.55% |
| 16384 | Nearest forward MAC | 512×6 | 8/2 | 2,176 | 32,381,440 | −11.56% | 17,382,768,640 | +0.0003% | 79,194,750,976 | +3.15% |
| 32768 | Nearest parameter | 512×6 | 8/2 | 2,176 | 40,770,048 | −4.98% | 21,677,735,936 | +5.22% | 96,374,620,160 | +7.49% |
| 32768 | Nearest forward MAC | 512×6 | 8/2 | 1,920 | 38,410,752 | −10.48% | 20,469,776,384 | −0.64% | 91,542,781,952 | +2.10% |

Decode gaps are also explicit in `candidate-counts.json`: V16384 parameter option +10.08%/+13.98% at 512/2048; V16384 MAC option +1.13%/+6.53%; V32768 parameter option +6.02%/+10.00%; V32768 MAC option +0.32%/+5.14%.

The nearest-parameter V16384 option is still 9.27% above NTP prefill at 512, and the nearest-MAC option is 11.56% below NTP base parameters. This is the structural trade-off: SPLOT reuses core weights across loops, so a conventional dense shape cannot simultaneously match parameter and forward-MAC targets. No option is called matched or selected.

Optional auxiliary head, if a later A2 comparator is approved: `d²+d = 262,656` parameters at d512 and `T*(d²+dV)` MACs per offset. Adding it to the V16384 parameter option gives 35,789,824 total (−2.65% vs SPLOT total 36,763,620), but it is not used for NTP parity.

## 3. NTP and A2 regimes

### NTP-only primary

The credible primary regime is NTP-only, final readout, no offset loss. Compare SPLOT NTP base parameters and MACs to dense NTP candidates without a dormant head. Report both:

- `physical_ntp_base`: parameters active in the NTP model/objective.
- `physical_total_with_dormant_mtp`: total allocated parameters if the MTP head remains in the module but is unused.
- `active/optimized`: parameters receiving a nonzero gradient and optimizer state under the objective. With zero weight decay and no offset path, the dormant head is not active; report it, but do not count it as an active NTP parameter.

Dense candidates are not given a dormant MTP head in the primary comparison.

### A2 auxiliary — deferred

SPLOT A2 uses **two NTP readouts** (depths 1 and 2) and **two offset-2 readouts** with `depth_weight=0.25`, `offset_weight=0.25`, `balance_weight=0.01`. A single dense offset head plus one readout cannot match that computation. A credible A2 comparator would need two readout paths, the same offset projection/head and the exact loss weights, plus a matched optimizer/precision path. This is a separate implementation and comparator question. **A2 is deferred; no A2 candidate is selected or claimed as matched.** The SPLOT A2 accounting is retained for cost context:
- V16384: 29,008,330,752 prefill@512, 125,696,999,424 prefill@2048, 58,226,688 decode@512, 67,663,872 decode@2048.
- V32768: 41,893,232,640 / 177,236,606,976 / 83,392,512 / 92,829,696.
The A2 increase is not attributable to a single offset head; it includes extra readout/coda work.

## 4. Corrected budget units

T0016 training split: 128 blocks × 512 positions = **65,536 positions**, **46,204 valid tokens**, **46,076 NTP labels**, **45,948 offset-2 labels** per full epoch. Positions include padding; valid tokens include BOS/EOS; NTP labels are the token-weighted denominator; offset labels are separate and are 0 in the NTP-only regime.

The planning matrix reserves nine runs (SPLOT plus two dense options chosen after review, 3 seeds each) at one pass each:
- **589,824 positions**
- **415,836 valid tokens**
- **414,684 NTP labels**
- **0 offset labels in NTP-only** (45,948 per run only if a future accepted A2 regime is used)

The prior report's 589,824 "valid tokens" was wrong; it is positions. The 45–75 minute estimate remains a rough planning range from accepted T0012/T0011 single-update/synthetic-epoch anchors, not a matched measurement.

## 5. Evaluation protocol corrections

### Metrics

- Token-weighted held-out NTP NLL in nats per valid NTP label, per language and aggregate.
- Bits/byte `= NLL_sum / (ln(2) × covered UTF-8 bytes)`. Byte coverage is computed for target text represented by valid non-ignore labels, excluding BOS/EOS/pad and masked positions; report coverage ratio and missing bytes. Compare bits/byte only with reconciled coverage.
- PPL only with identical tokenizer and masks.
- Report positions, valid tokens, NTP labels, offset labels, padding and byte coverage separately.

### Probes: exact construction, exposure and chance

1. **Key–value retrieval with distractors.** Fixed generator: 16 single-token keys and 16 single-token values, four distractor pairs, one query; score the value token sequence, exact match. Held-out generator seed/instances; chance denominator **1/16** for a single-token answer. Training exposure audit: tokenize each probe and verify no probe prompt/template or exact 8-gram appears in the local train blocks; probe instances are never used for training or tuning.
2. **Compositional pattern completion.** Fixed generator: 8-symbol alphabet, a held-out successor/induction rule, 256 instances; score the next symbol, exact match; chance denominator **1/8**. Templates, symbol mappings and generation seeds are disjoint from any training data and from probe 1; exposure audit as above.

Probes run after the checkpoint and configuration are frozen, once, with no tuning on probe results. If every arm is within 2 percentage points of chance, the probe is **uninformative** and no capability claim is made. Report floor/ceiling and per-instance bootstrap intervals; the generator distribution, not the prose population, is the scope.

### Uncertainty and margins

The validation split has only **two families** (one EN, one PL). It cannot establish population or family-level generalization. Report per-family/per-block descriptive NLL and seed ranges; do not present a family-level confidence interval or a population equivalence claim. Probe uncertainty may bootstrap held-out probe instances because they are independent generator draws.

The following remain **planning proposals**, not preregistered confirmatory gates:
- NTP NLL noninferiority +0.05 nats/valid label; minimum meaningful advantage −0.03.
- Bits/byte +0.01; guardrail accuracy −2 percentage points.
- Cost +10% descriptive wall/memory guardrail.
- Three planning seeds; a confirmatory study needs more families, blocks and seeds, with the margins fixed before outcomes.

Failure rules: nonfinite loss/state, manifest/tokenizer/source mismatch, test access, unequal token/tuning budget, missing identity, or evaluation/cache mismatch fails the affected run; no retry or gate relaxation.

### Budget

Reserved planning matrix: 3 arms × 3 seeds × 128 updates = 1,152 updates; one pass each. The two dense options are not selected in this attempt. The estimate remains 45–75 minutes, up to about 2 hours with a small equal tuning stage, based on accepted T0012/T0011 anchors. It is not a matched-pilot measurement and does not authorize training.

## 6. Remaining implementation gates

1. Reviewed dense model-family selector and architecture identity in `tools/experiment.py`.
2. Dense NTP training/evaluation adapter with verified pilot loader, checkpoint/resume identity and exact label/mask contract.
3. Dense MTP/readout design only if A2 is later approved; otherwise keep A2 deferred.
4. Extend semantic optimizer partitioning/classification for any dense auxiliary owner; verify dense NTP partition.
5. Product dense accounting, replacing the evidence scripts; instantiate candidate configs and verify parameter counts.
6. Per-block UTF-8 byte coverage or local detokenization for bits/byte.
7. Probe generators/exposure audit and analysis tooling.
8. Paired run matrix, seed/data-order manifest, equal tuning budget and analysis script.
9. CPU/CI tests for accounting, identity, data path and evaluation masks.

## 7. Limitations

- Analysis/design only; no candidate selected, implementation, training, install, GPU/MPS, test or checkpoint access.
- Local Mac has no torch; formulas are not locally instantiated as modules.
- Candidate MACs exclude backward, optimizer, padding, memory traffic and kernel efficiency.
- The pilot corpus is small; two validation families cannot support population claims.
- The budget is a planning extrapolation from different accepted synthetic measurements.
- T0017 results were not used; no T0017 checkpoint or test data was accessed.
- No quality, speed, memory or scaling claim is made.

## 8. Proposed next action

Coordinator review of the corrected same-vocab accounting and unselected candidate options. Do not implement or train from T0019. If a matched study is later approved, first complete the §6 gates in a separate reviewed implementation task, then preregister the NTP-only margins/budget and decide whether A2 is credible enough to include. T0020 remains separate.

## Polski — podsumowanie

Poprawka T0019 wyrównuje cele do tego samego słownika: przy V16384 SPLOT ma 36 615 780 parametrów bazowych NTP (36 763 620 z głowicą MTP) i 17 381 326 848 MAC prefill@512, a nie wartości V32768. Kandydaci gęści są jedynie opcjami: najbliższy parametrom ma +9,27% MAC@512, a najbliższy MAC ma −11,56% parametrów. A2 odroczono, bo dwa odczyty NTP i dwa offsetowe nie mają wiarygodnego odpowiednika w jednej głowicy gęstej. Budżet rozdziela pozycje, ważne tokeny i etykiety NTP; dwa rodziny walidacyjne nie dają uogólnienia populacyjnego. To projekt, nie trening ani wynik jakościowy.
