# Native MTP reference v0.1
2026-11-27 | #2 | [Polski — summary](native-mtp.pl.md)

## Contract
Implements the direct offset-2 option of RFC 0001 v0.2 and RFC 0005 on the fixed-depth all-GQA recurrent control merged in #21. MTPLM subclasses RecurrentLM; existing target logits unchanged at identical weights. A shared bias-free d-by-d projection and independent RMSNorm operate on requested pre-coda h_r, then F.linear with the SAME input embedding matrix E. No independent vocabulary head, no future-token embedding or label in forward. Auxiliary logits predict x_(t+2) conditional on x_<=t; this is not a conditional autoregressive second-token proposal conditioned on the sampled first token. The true joint proposal and exact verifier must be specified next.

Physical extra parameters d*d+d: smoke d=32 adds 1056 to 82720 (83776 total). Nano d=384 adds 147840, Mini d=768 adds 590592 to their respective future base architectures. The latter are analytical; no complete hybrid model yet. Head is always instantiated in MTPLM; inactive in A0/A1 (grad=None, AdamW does not update inactive params). RecurrentLM remains the no-head control. Same seed and initial base weights verified. MTP projections are tokenwise and add no persistent cache; requesting an auxiliary depth currently also requires NTP readout at that depth, so coda work/cache may increase. No optimized hidden-only path.

Forward signature extends recurrent with offset2_depths (default empty, subset of selected NTP readouts; unique sorted depths). MTPOutput contains logits, hidden, cache and offset2 dict. Padding output zero. Existing cache ownership, fixed loops/readouts, caller invalidation and nonmutating branch rules remain; offset2 selection may vary within same readouts because it is stateless. No speculative decoding implementation here.

## Training objectives
training_objective(model, inputs, labels, mode, loops, depth_weight=.25, offset_weight=.25) returns total, final_ntp, early_ntp, offset2 and valid counts for each offset. Uses already shifted packer labels (ignore -100), per-head mean over valid tokens; no extra shift. Caller responsible for label integrity, no cross-document labels; fixture tested. Counts are per offset, not multiplied by depth; empty earlier mean at R=1 is zero; {1,R} deduplicated. All-ignored targets produce zero gradients; skip empty batches in real training to avoid optimizer state/decay updates. A0 final CE only; A1 adds depth_weight times mean earlier NTP CE; A2 adds offset_weight times mean offset-2 CE at unique depths {1,R}. The weights are provisional. Separate component losses logged; total across modes not directly comparable. A0/A1 may omit offset-2 labels. A2 requires both. Invalid mode, depth and nonfinite/negative weights rejected.

## Validation
Seven CPU tests executed (Float64 atol=rtol=1e-10): exact inventory and unchanged base logits, objective algebra and R1 dedup, prefix/document isolation and true fixture labels, active/inactive gradients and auxiliary-only gradient path (core and tied embedding, not coda), empty masks, cached offset equivalence/no extra logical state, invalid input rejection. Three FP32 40-step smoke runs with same seed, model inventory, data and optimizer settings, A0/A1/A2; all passed finite gradients and synthetic loss-decrease gate. See [results](../../experiments/results/native-mtp-v0.1.md). A2 final-depth NTP 5.602705 -> 2.390771; auxiliary 5.591383 -> 2.742605. These are two fixed synthetic documents, single seed, no held-out evaluation or compute matching. No evidence of quality gain or speedup. Earlier suites not rerun in this work unit; new tests exercise actual inherited base code and verified fixture.

Run from repo:
```bash
PYTHONPATH=src:tools python tests/test_mtp.py
for mode in A0 A1 A2; do
  PYTHONPATH=src:tools python tools/smoke_mtp.py --mode "$mode" --loops 2 --steps 40
done
```
GPU optional --device cuda, unrun; BF16/large config, long context, actual corpus, checkpointing, profiling remain open.

## Next: exact reference sampler
Separate issue #2 work unit: implement exact serial shallow autoregressive proposal D1 and independent-offset D2. For D2 q1 from shallow NTP, q2 from offset-2 head evaluated on same committed prefix (q2 deliberately ignores sampled y1; this defines a valid independent conditional proposal, not a learned conditional joint). Sequential acceptance uses actual transformed proposal distributions, full-depth target p_i with same prefix and previously accepted drafts; residual correction normalized positive part p-q, bonus from target when all accepted and budget permits. Implement greedy oracle and stochastic finite-vocabulary distribution checks, EOS/max budget handling, immutable prefix cache/replay, rejection at every position. Start separate shallow/target prefill, no unsafe cross-depth cache reuse. Compare exact target probabilities, not RNG-identical strings. GDN matrix/history rollback after integration. Do not report speedup until optimized baseline and hardware measurement. #2 stays open: full literature verification, verifier, pilot design and empirical comparisons outstanding.
