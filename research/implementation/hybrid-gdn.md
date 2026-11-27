# Learned hybrid GDN/GQA reference v0.1
2026-11-27 | #5 | [Polski — summary](hybrid-gdn.pl.md)

## Scope and implementation
Implements RFC 0003 GDN v1, replacing the first three shared core mixers of the validated all-GQA MTP model with learned GDN and the fourth with gated full GQA. Prelude/coda remain ungated GQA and all FFNs remain dense. HybridLM subclasses MTPLM; existing A0/A1/A2 and D1/D2 APIs work unchanged. No sparse experts, optimized kernel, adaptive depth, new DeltaNet variant or production corpus. GDN head count is set equal to GQA query-head count for this control; key/value dimensions equal head_dim. Four-tap depthwise causal convolution over raw projected QKV, per-head L2 normalization and q scale, scalar log-decay/write gates, post-update readout, shared head RMSNorm and SiLU output gate. No GDN RoPE. Gated GQA multiplies the concatenated attention outputs by sigmoid of a learned bias-free input projection before output projection. Existing ungated default preserved.

For each head, D=exp(g)S, u=beta*(v-k^T D), S'=D+k u^T, o=q^T S'. Decay gate uses -exp(log_A)*softplus(a(x)+dt_bias). Initialization: log_A=log(U(1,16)), dt log-uniform .001–.1 and inverse-softplus bias, convolution U(-.5,.5), linear std from existing DenseConfig. Convolution initialization is a provisional new implementation choice, not a tuned literature result. GDN run() provides sequential scan (actual model path) and independently derived dense triangular solve (small test oracle). The solve is not an optimized chunk kernel; extreme decay/long sequence stability remains untested. Learned forward and gradients (input and every parameter, including final matrix state contribution) match both algorithms on the tested nonzero initial state in FP64.

## Logical cache and precision
Each GDN logical invocation stores (matrix [H,D,D], raw QKV history [3,3*H*D]) per batch row. At each new document zero both; padding never consumes a token or changes state. GQA stores rotated K and unrotated V; existing logical (stage,pass,block) keys remain. Matrix updates create new tensors, history slice cloned, no in-place mutation of committed prefix. This allows existing speculative snapshot/replay driver to restore both states without inversion or truncation of GDN state. Cached per-pass and per-readout depth remains fixed; shallow and target caches independent. Cache tensors are read-only by convention; discard after weights/config/device/dtype changes. GDN matrix contractions run outside autocast in FP32 (FP64 for double oracle); raw projection history follows projection dtype, BF16 under CPU autocast. Gates' nonlinear computations FP32 in mixed precision, but projections may be autocast. Not a production mixed-precision policy. CPU BF16 forward and short chunk parity exploratory, GPU/long-context/training BF16 untested.

## Parameter accounting
Base is the all-GQA MTP control plus 3*(GDN-GQA) plus one gated GQA d*q projection. For q=Hq*D, k=Hkv*D, GQA=d*(2q+2k); GDN=d*(5q+2Hq)+12q+2Hq+D. All other FFNs dense.
| Control | Physical parameters | Verification |
|---|---:|---|
| Smoke | 92,912 | actual forward/backward and count |
| Hybrid-Nano dense | 26,527,716 | meta count plus four-token FP32 forward/backward |
| Hybrid-Mini dense | 80,902,920 | meta count only |
These are NOT the full SPLOT candidates; #6 MoE replaces the four core dense FFNs later. The separate candidate architecture count remains 43,055,076 / 146,987,784 pending full integration.

## Validation and limitations
Six new tests: learned scan/chunk outputs/state/gradients, exact parameter inventories, R=1/2/4 full vs tokenwise/irregular chunk logits and all final states incl offset2, depth/prefix and document/batch isolation, retained-prefix replay plus D1/D2 actual model greedy equality and stochastic cache parity, FP32 training/CPU BF16 state dtype. Seven MTP and six speculative regressions rerun. All passed. Six-token greedy hybrid D2 encountered three rejections; both caches and pending logits match fresh prefill after the committed sequence. No trained acceptance or speed claim. 40-step FP32 A2 synthetic smoke total 8.387559 -> 2.282609. Not comparable to earlier models' losses as proof of improvement. See [evidence](../../experiments/results/hybrid-gdn-v0.1.md).

Run:
```bash
PYTHONPATH=src:tools python tests/test_hybrid.py
PYTHONPATH=src:tools python tests/test_mtp.py
PYTHONPATH=src:tools python tests/test_speculative.py
PYTHONPATH=src:tools python tools/smoke_mtp.py --hybrid --mode A2 --loops 2 --steps 40
```
GPU --device cuda optional; no CUDA in current environment. Sequential Python scan, no fused kernel; long context and memory benchmarks pending. Actual GDN end-to-end training gradients work but checkpointing, matched controls, extreme gates, wider stochastic rejection coverage and optimized kernels remain open.

## Next
#6 implement routed and shared SwiGLU experts, FP32 router/top-k, deterministic ties, balance objective and no token drops. First verify dense-to-sparse accounting and explicit oracle, then full hybrid MoE + MTP model, cache/speculation regression and actual Nano parameter count. #5 remains open for broader precision/kernel and empirical work, #2 for exactness coverage and empirical speed; #8 real-data gate persists.
