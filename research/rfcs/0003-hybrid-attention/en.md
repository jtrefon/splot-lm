# RFC 0003 — Hybrid Gated DeltaNet and GQA
Draft v0.1 | 2026-11-26 | #5 | [Polski — summary](pl.md)

## 1. Decision and scope
Reference candidate: two independent ungated full GQA prelude blocks, a shared core of three Gated DeltaNet (GDN) blocks followed by one gated full GQA block, and two independent ungated full GQA coda blocks. This makes the 3:1 ratio a property of the core, not the entire logical stack. Each mixer receives the pre-normalized residual stream [B,T,d] and returns [B,T,d]; residual updates and fixed depth are defined by RFC 0002. Keep a dense all-GQA baseline and an all-GQA recurrent control. No sparse attention, windowing, latent KV compression, grouped value expansion, adaptive depth or kernel fusion in the first oracle. Native MTP remains in scope, but does not change target mixer semantics.

Use GDN v1 (Yang et al., 2025, Eq. 10) as the reference. Gated DeltaNet-2 and Integral DeltaNet are promising alternatives, not defaults; they change recurrent memory and potentially rollback/precision contracts. See [source notes](sources.md). Proposed SPLOT block is inspired by GDN and Qwen3-Next; it is not a reproduction of Qwen checkpoints.

## 2. Mini dimensions and projections
All linear maps bias-free except explicit dt bias; no biases on convolution. Set d=768, GDN H=12, d_k=d_v=64, q/k/v/z widths 768 each, convolution K_c=4. Equal key/value head counts simplify reference. GQA H_q=12,H_kv=4,D=64, group=3. RMSNorm epsilon 1e-5, L2 epsilon 1e-6, interleaved RoPE theta=1e6. All provisional until architecture reconciliation (#3).

For each normalized token x:
- GDN raw q0,k0,v0,z projections: W_q,W_k,W_v,W_z each [d,768] in mathematical input-by-output orientation. W_a,W_b each [d,H]; log_A, dt_bias each [H].
- Concatenate raw q0,k0,v0 into c in R^2304. Depthwise causal convolution kernel weights [K_c,2304] (mathematical layout oldest to newest). For token t, y_t=SiLU(sum_{j=0..K_c-1} w_j * c_{t-K_c+1+j}), with zero left padding at document start. Keep raw pre-convolution history [B,K_c-1,2304], not activated outputs. Split y into q,k,v and reshape [B,T,H,64]. z and the two scalar gates bypass convolution.
- Normalize q,k per head with q/sqrt(sum q²+eps), k/sqrt(sum k²+eps); scale q additionally by 1/sqrt(d_k) as an explicit reference convention. No RoPE on GDN initially. v is not L2-normalized.
- beta=sigmoid(x W_b), g=-exp(log_A)*softplus(x W_a+dt_bias), alpha=exp(g); per token/head scalar. Compute in FP32 in mixed precision. Initialize log_A from log(U(1,16)), dt_bias=softplus_inverse(dt) with dt log-uniform on [0.001,0.1] using seeded draws, log/exp/softplus in FP32; record initialization seed and actual distributions. This is a proposed initialization, not a copied checkpoint. No arbitrary gate clipping that changes the model; nonfinite gates fail tests.

## 3. Gated delta update
Use key-by-value orientation S in R^(d_k x d_v) per head (transpose of paper convention). For each valid token, reset before update if document changes:
```text
D_t = alpha_t * S_(t-1)
u_t = beta_t * (v_t - k_t^T D_t)      # value vector
S_t = D_t + k_t u_t^T
o_t = q_t^T S_t
```
State decay happens BEFORE computing the prediction error. Output reads post-update state, so current token participates. Equivalent to S_t=alpha_t(I-beta_t k_t k_t^T)S_(t-1)+beta_t k_t v_t^T. beta=0 is pure decay; alpha=0 resets the old state before writing. With normalized k, 0<=beta<=1 gives a non-expansive erase transition before the additive write (not a bound on overall model norm).

Per head, apply RMSNorm over d_v with one shared learned vector [64], then multiply by SiLU(z_t reshaped per head). Flatten H*d_v and project with W_o [768,d]. No extra norm on gate. FP32 state and reductions by default, BF16 activation/output projection and convolution history for BF16 inference; FP64/FP32 reference arithmetic first. State [B,H,d_k,d_v], convolution [B,K_c-1,2304] independent for every logical core invocation. Never share mutable state across depth or requests.

## 4. Chunkwise oracle derivation
Within a contiguous chunk of valid same-document tokens, initial state S_0 need not be zero. Define G_i=sum_{j<=i} g_j, gamma_i=exp(G_i), delta_ij=exp(G_i-G_j) for i>=j. Then:
S_i=gamma_i S_0+sum_{j<=i} delta_ij k_j u_j^T.
The rank-one write vectors solve the unit-lower-triangular system L U=Y:
L_ii=1; L_ij=beta_i delta_ij (k_i^T k_j) for j<i; L_ij=0 for j>i.
Y_i=beta_i (v_i^T-gamma_i k_i^T S_0).
O_i=gamma_i q_i^T S_0+sum_{j<=i} delta_ij(q_i^T k_j)u_j^T.
Final state is the expression for S at the last token. This is a dense algebraic oracle (not a fast WY implementation); solve per head. Use differences of log decay rather than divide cumulative products. Never exponentiate positive upper-triangle differences unnecessarily; clip only unused upper triangle before exponentiation. Exact-zero alpha is handled by sequential oracle or splitting at resets; do not feed log(0) into dense difference computation. Finite precision extreme decays, long chunks, gradients and fused kernel equivalence remain tests for implementation. Start chunk lengths 1/2/7/16/64 and nonzero initial state; handle final partial chunk without changing state. Packed docs split at boundaries or implement mathematically equivalent segmented scan; invalid padding has no transition. Future optimized WY/FLA/Triton path requires correctness comparison and pinned dependency/license review.

## 5. Full GQA and RoPE
Project Q:[B,T,12,64], K,V:[B,T,4,64]. W_q:[768,768], W_k,W_v:[768,256], W_o:[768,768]. Rotate interleaved pairs (2i,2i+1) in Q,K at document-relative position t with frequency theta^(-2i/64); V unrotated. FP32 angle calculation and reduction; store rotated K and unrotated V in cache. Query head h maps to KV head floor(h/3). For token t, attention logits s_tj=(Q_t dot K_j)/sqrt(64) for j<=t, valid same-document tokens only; masked positions excluded before stable softmax. For an invalid query, output zero and no state append, not softmax over all masked positions. No Q/K norm in reference, no attention bias/dropout. Ungated prelude/coda return concat(head outputs) W_o. Gated core additionally W_gate:[768,768]; multiply flattened head outputs by sigmoid(x W_gate) before W_o. Gate is tokenwise, not another cache. This is a chosen simplified gate, not a claim of Qwen identity. Test gated vs ungated independently. Use explicit math then SDPA/FlashAttention when verified, ensuring absolute offsets for chunked prefill and cached decoding. Never materialize repeated KV heads in stored cache. Full GQA remains quadratic in total prefill attention work and linear in decode context per token; hybrid does not remove this asymptotic behavior.

## 6. State ABI, rollback and batching
Follow RFC 0002 logical activation-state keys. GDN has two mutable tensors per invocation: recurrent S and raw convolution history, plus valid lengths/position/document metadata. GQA has rotated K/V and lengths; separate coda cache per active readout. Keep independent per-sequence state; padding must not shift convolution, decay S or append KV. Reset all mixer states and positions on new document. For chunk prefill continuation, carry BOTH GDN tensors. Initial empty history is zeros, empty K/V length zero, S zero. No in-place training cache mutation.

Exact speculative reference: checkpoint committed prefix (lengths, positions, pending logits, clone of S/history at each GDN invocation; GQA cache prefix immutable or clone). Verify proposed suffix in scratch state. On rejection restore checkpoint and replay accepted suffix plus correction once. On all-accepted commit verified scratch; process optional bonus once. GQA may truncate if prefix unmodified; GDN cannot truncate a fixed-size matrix. Algebraic inversion is not a safe default: alpha can be tiny, erase transition nearly singular, low precision loses information. Measure snapshot/replay vs per-token checkpoints, with memory and tail latency. Do not alias draft/target mutable caches; deep-prefix adaptation in RFC 0005 is a distinct experimental mechanism. Weight residency is independent (RFC 0004).

## 7. Parameter and cache accounting (analytical)
For one GDN mixer: d*(2H d_k+2H d_v+2H)+K_c*(2H d_k+H d_v)+2H+d_v+(H d_v)*d=2,976,856 parameters. GQA ungated: d*(H_q D+2 H_kv D)+H_q D*d=1,572,864; gated adds d*H_q D=589,824, total 2,162,688. Thus 3 physical GDN + 4 ungated GQA + 1 gated GQA =17,384,712 mixer parameters. With existing proposed expert/dense FFNs and embeddings (128,974,848), router weights (24,576), two block RMSNorm vectors each plus final norm (13,056), approximate subtotal 146,397,192 BEFORE MTP auxiliary modules, depth vectors, extra norms or other variants. This is an analytical count under stated bias/weight assumptions; #3 must reconcile exact implementation counts. GDN gate/norm parameters included once.

Per sequence and logical invocation, FP32 S bytes=H*d_k*d_v*4=196,608; BF16 history=(K_c-1)*2304*2=13,824; total 210,432 bytes (205.5 KiB). GQA BF16 KV=2*H_kv*D*2*T=1024*T bytes. One GDN state becomes smaller than one GQA cache at integer T>=206. With only final coda active, total cache=3R*210432+(4+R)*1024*T bytes. All-GQA same topology= (4+4R)*1024*T. Additional shallow coda readouts add 2*1024*T each. For R=4:
| Context | Hybrid MiB | All GQA MiB |
|---|---:|---:|
| 128 | 3.408 | 2.500 |
| 512 | 6.408 | 10.000 |
| 2048 | 18.408 | 40.000 |
| 8192 | 66.408 | 160.000 |
| 32768 | 258.408 | 640.000 |
At T=2048, R=4, three additional coda readouts add 12 MiB. A full GDN snapshot costs another 2.408 MiB per sequence; per-token snapshots multiply this cost. Excludes allocation rounding, metadata, batching, temporary activations, logits, draft caches, weights, optimizer and workspaces. At long context hybrid ratio tends to (4+R)/(4+4R)=40% for R=4, not 25%. These are analytical state estimates, not measured VRAM or speed. GDN matrix update is O(H*d_k*d_v) per token, independent of T; actual performance depends on memory bandwidth, kernel fusion and parallelism.

## 8. Validation and research gates
The independent NumPy float64 oracle validates gated delta scan versus triangular solve, irregular chunks/nonzero initial state, limiting gates, matrix rollback, convolution history/reset/rollback and grouped RoPE attention against explicit causal reference. It tests projected inputs, not learned projections, mixer integration, gradients, batch packing, quantization or fused GPU kernels. See results. Actual mixer implementation gates: CPU FP64/FP32 forward and gradient parity on small tensors, batch/padding/document isolation, nonzero initial states, extreme gate/length tests, cached vs full logits and state, rollback at every rejection index, recurrent-depth integration and MTP exactness. Fused BF16/FP8 comparisons need predeclared tolerances and hardware benchmarks. Include all-GQA and alternative core patterns as matched-parameter/compute controls; measure retrieval, code/math, loss and long-context memory/latency. No performance or novelty claim.

Next #3 parameter reconciliation then #8 data contract/#9 dense reference; keep #5 open until real mixer integration and tests satisfy acceptance. Newer GDN-2/Kimi Delta/Integral Delta variants tracked after reference.
