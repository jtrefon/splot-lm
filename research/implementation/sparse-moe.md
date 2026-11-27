# Full SPLOT sparse hybrid reference v0.1
2026-11-27 | #6, #3 | [Polski — summary](sparse-moe.pl.md)

## Scope
SPLOTLM completes the initial RFC 0001 v0.2 topology: 2 dense GQA prelude blocks, 4 shared recurrent core blocks (3 GDN + 1 gated GQA, each with sparse FFN), 2 dense GQA coda blocks, tied embeddings, native offset-2 MTP. Fixed R=1..4. Existing HybridLM is the dense-FFN control. Each physical core block has one router, 8 independent routed SwiGLU experts and one always-on shared expert, all bias-free; weights reused at every recurrent pass. Default expert intermediate width is half the dense intermediate width, exactly the Nano/Mini proposal. No locality regularizer, capacity dropping, per-pass router, expert offloading, quantization or fused kernel. All-resident reference is intentionally slow.

## Router, dispatch and objective
For valid token h, p=softmax(W h), stable descending sort of p with lower expert index first on exact ties. Select top 2, w=p_selected/sum(p_selected). Output shared(h)+sum(w_e expert_e(h)). Router projection, softmax and selected weight normalization are FP32 even under autocast (FP64 in oracle); FP32 arithmetic does not imply identical near-tie routing across devices/kernels, nor does stable tie-breaking guarantee fully deterministic GPU reductions. Use FP32 master weights when testing mixed precision. Expert execution follows ambient dtype; accumulate expert contributions in FP32 or FP64 then cast to stream dtype. Gather only valid tokens; exactly 2 routed assignments plus shared expert per valid token, zero FFN output on invalid padding, no dropped tokens. No gradient required for an unused expert. Tests compare against independently evaluated per-token expert selection and outputs and parameter/input gradients.

Route records are returned per forward as ((stage,pass,block), Route(indices,probabilities,selected,weights)). No mutable module-global trace, no routing state in the persistent inference cache. No raw text stored. Records are call-local and retain differentiable probabilities; do not accumulate training graphs in logging. routing_summary uses detached statistics: hard counts, unused experts within observed window (not a global dead-expert claim), mean categorical entropy in nats, same-document adjacent valid temporal overlap and same-position cross-depth overlap. Time overlap does not span separate calls or invalid gaps; depth overlap is within the same physical pool. Not a chronological hardware trace. Uniform chance k/E=0.25 is only a theoretical independent-uniform reference. Empirical marginal controls and per-domain aggregation remain outstanding.

Global balance per physical pool aggregates all valid tokens and logical passes in the current forward, f_e=hard assignments/(k*N), pbar_e=mean soft probability, L_b=E sum f_e*pbar_e, then mean across four physical pools. Hard assignments nondifferentiable; gradients through pbar. Empty pool contributes zero. No per-pass uniformity constraint; specialization across passes may emerge. Not global across distributed workers or microbatches; future trainer must explicitly define aggregation. Objective extends A0/A1/A2 with provisional coefficient .01 times balance. Existing non-MoE models return zero balance and unchanged token losses. All-ignored labels with valid tokens may still have router loss; skip empty-target batches if the intended objective is token-supervised only. A scalar balance near 1 alone does not demonstrate healthy routes: uniform p can coexist with deterministic collapsed hard assignments. Report hard loads and entropy jointly.

## Accounting
| Model | Parameters | Verification |
|---|---:|---|
| Smoke (width32, vocab264) | 179,952 | actual count, forward/backward/training |
| SPLOT-Nano (width384, vocab32768) | 43,055,076 | actual count and short FP32 forward/backward |
| SPLOT-Mini (width768, vocab32768) | 146,987,784 | meta-tensor inventory |

Both candidate totals exactly match RFC 0001. At each core invocation three of nine expert FFNs execute for a token (two routed and one shared), not necessarily the same experts at subsequent passes. Across 4 core pools, 4R logical invocations and 12R expert evaluations per token; not 12R unique physical experts. Weight-cache keys must omit logical pass for shared immutable weights; no residency changes implemented. No claim of measured active footprint, transfer reduction or speed. Raw BF16 parameter payload ~86.1 MB Nano, ~294.0 MB Mini, excluding all overhead.

## Validation
Six new tests pass: dispatch/output/gradient oracle; deterministic ties/pad/empty pool/balance; exact candidate counts; R=1/2/4 cache and route equality with irregular chunks, document isolation and depth invariance; D1/D2 greedy parity and stochastic committed-state replay; FP32 training, router gradients and CPU BF16 autocast routing dtype. Six hybrid, seven MTP and six speculative tests rerun and pass. No GPU. Forty-step synthetic A2+balance smoke 8.395732 -> 2.268872. Full Nano four-token FP32 forward/backward with finite active gradients (153 of 198 parameter tensors received gradients, unused experts legitimately have no gradient). Mini meta only. No checkpoint or held-out quality. See [evidence](../../experiments/results/sparse-moe-v0.1.md).

Run:
```bash
PYTHONPATH=src:tools python tests/test_moe.py
PYTHONPATH=src:tools python tools/smoke_mtp.py --sparse --mode A2 --loops 2 --steps 40
```
Optional --device cuda remains unrun. Default smoke configuration is not Nano. Slow Python scans and expert loops are correctness references, not performance baselines.

## Next
First establish reproducible experiment harness (#10), executable full candidate configs and checkpoint/resume, deterministic data order, evaluation and GPU smoke; address precision/activation-memory policy (#7), optimize only against correctness oracle. Production source acquisition/tokenizer remains gated #8. #6 remains open for full literature verification, chronological traces, cache-only simulation and measured PCIe/offload experiments; no locality optimization yet. #3 actual physical count milestone met, matched controls and empirical verification remain. Do not infer architectural superiority or release readiness from smoke.
