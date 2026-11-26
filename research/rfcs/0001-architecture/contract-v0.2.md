# SPLOT architecture contract v0.2
Draft, 2026-11-26 | Issue #3 | [Polski](contract-v0.2.pl.md)

This consolidates the merged RFC 0002–0005 proposals into a buildable reference contract. It supersedes conflicting v0.1 defaults only upon maintainer acceptance. No trained model, measured FLOP/s, GPU allocation or novelty claim. Preserve original RFCs as historical rationale. RFC 0006 precision, dataset issue #8 and actual mixer/inference tests remain outstanding.

## 1. Fixed candidate shapes
| Setting | Nano | Mini |
|---|---:|---:|
| Vocabulary capacity | 32768 | 32768 |
| Residual width d | 384 | 768 |
| GQA Q/KV heads | 6/2 | 12/4 |
| GDN key/value heads | 6/6 | 12/12 |
| Head width | 64 | 64 |
| Dense SwiGLU intermediate | 1024 | 2048 |
| Expert SwiGLU intermediate | 512 | 1024 |
| Routed experts / top-k / shared | 8/2/1 | 8/2/1 |
| Prelude/core/coda physical blocks | 2/4/2 | 2/4/2 |
| R_cap / default R | 4/2 | 4/2 |
| GDN convolution taps | 4 | 4 |
| Initial training context | 2048 | 2048 |

Vocabulary is capacity, not a finalized tokenizer. Nano halves width and FFN widths, preserves depth/topology and routing. All learned projections bias-free except GDN dt bias. Norm epsilon 1e-5; Q/K L2 epsilon 1e-6; interleaved RoPE theta=1e6; tied input/output E. No learned depth vector, dropout, runtime adaptive halting, locality loss or quantization in initial reference. Training BF16 after CPU FP32/FP64 oracle; GDN matrix and router reductions FP32. Exact initialization and optimizer budget to be preregistered; general linear/embedding initialization still open in #9.

## 2. Forward contract
Input token IDs [B,T], validity and document IDs [B,T]. Positions reset at boundaries. E[V,d], z=P(E[x]), h_0=z. Prelude P and coda Q: two distinct ungated GQA + dense SwiGLU blocks each. Core C: three GDN + MoE blocks then gated full GQA + MoE. Each residual branch pre-RMSNorm. I0 no repeat injection and S0 unscaled residual are controls; compare I1/I2 and S1/S2 in RFC 0002 separately before long training. Run fixed R passes of same physical core; do not condition pass functions on requested R. Each logical invocation has its own mutable state. h_r=C(h_(r-1)); Q weights shared across readout depths, caches distinct. Normalized coda and E^T yield logits [B,T,V].

GDN raw QKV convolution and gated delta recurrence exactly per RFC 0003; state [B,H,64,64] FP32 plus raw convolution history [B,3,3H*64] BF16. GQA Q:[B,T,Hq,64], K/V:[B,T,Hkv,64], group factor 3; core sigmoid output gate. Full attention includes current token, no cross-document masking leakage. MoE route FP32 softmax over eight routed experts, deterministic top-2 normalized weights, one always-on shared expert, no capacity dropping in reference. Shared router per physical block; optional depth-conditioned or per-pass routers separate experiments. Balance coefficient proposed 0.01, verify scale and tune before confirmatory runs. No hardware-directed target rerouting. Training uses complete unrolled graph, unique parameter tensors and one optimizer state per parameter.

## 3. Native MTP, executable starting choice
Keep RFC 0005 A0/A1/A2 as controls. Define H_(r,1)=E^T RMSNorm(Q(h_r)); H_(r,2)=E^T RMSNorm_aux(W2 h_r) with column-vector notation, W2 [d,d], norm vector [d], one module shared across depths. No extra vocabulary matrix, no auxiliary coda, no future ground-truth inputs. Readout offset k=2 is the token two positions ahead, not two extra tokens. A2 computes NTP all depths and offset-2 at distinct depths {1,R}; for R=1 deduplicate. A1 mean over earlier depths empty => zero. CE mask excludes padding, document boundaries and invalid labels, normalize per head by valid labels; empty => zero without NaN. Proposed lambda_depth=0.25, lambda_mtp=0.25. Keep physical MTP parameters even if disabled for A0/A1 in same checkpoint inventory; for independent no-MTP controls omit them and report count. Offset-2 module adds d²+d. Latent alignment, sequential MTP, cheap shallow NTP readout and masked-slot adaptation are separate interventions. D1 serial shallow proposal and D2 independently sampled parallel offsets require actual conditional q_i and exact target verification as in RFC 0005. Only target final NTP head used in ordinary decoding; training auxiliary reads need not allocate persistent inference state. No production acceleration assumed.

## 4. Accounting definitions
P=2 prelude, M=4 core, Q=2 coda. For physical parameters, let q=Hq*D, k=Hkv*D, g=Hg*D, E_r=8, K=2, m_e/m_d expert/dense intermediate. GQA G=d(2q+2k), gated Gg=G+dq. GDN N=d(5g+2Hg)+3g*Kconv+2Hg+D; one shared per-head output norm vector D. SwiGLU expert F=3d*m_e, dense Fd=3d*m_d, router J=d*E_r. Base=Vd+4(G+Fd+2d)+3N+Gg+4((E_r+1)F+J+2d)+d. Aux=d²+d. No double-count tied E or repeated core/coda weights.
| Component | Nano | Mini |
|---|---:|---:|
| Base without MTP | 42,907,236 | 146,397,192 |
| MTP auxiliary | 147,840 | 590,592 |
| Total | 43,055,076 | 146,987,784 |

Target unique-active *parameter footprint*: all nonexpert base parameters (including E for full output projection) plus shared experts and union of routed experts selected across R passes for ONE token. Not measured read bytes or simultaneous residency. Min=Base-4(E_r-K)F, max=Base-4(E_r-min(E_r,RK))F. At Mini R=1 min=max=89,774,088; R=2 89,774,088–108,648,456; R=4 89,774,088–146,397,192. Batch/sequence union can be larger; MTP projection adds d²+d if executed. Active count excludes extra inactive offset head for ordinary target. Report actual route unions separately. Repeated compute increases even when same weights selected.

## 5. Analytical contraction work (MACs)
A multiply-accumulate counted as one MAC or two conventional FLOPs. Not total executed hardware FLOPs. Count dense projections, expert projections for K+1 active experts, router, convolution and delta update contractions, attention QK and AV, vocab projection. Exclude scalar operations (norm, gates, decay multiplication, softmax, rotary, routing sort, residual, activation), memory traffic, masking overhead, CE/backward, optimizer, kernel padding and recomputation. Per token attending to L inclusive positions, A(L)=2qL; GDN contraction work N_mac=d(5g+2Hg)+3g*Kconv+3HgD² (error read, rank-one update, output read); separate elementwise decay omitted. Dense block D_b=G+Fd+A(L). Core pass C_b=3N_mac+Gg+A(L)+4((K+1)F+J). For m active NTP readout depths and n offset-2 readouts, forward decode MACs=(2+2m)D_b+RC_b+mVd+n(d²+Vd). Logical block invocations=2+4R+2m. For causal unpadded full prefill length T with all-position logits, sum over L=1..T, replacing sum A(L) by qT(T+1). Ordinary inference prefill may calculate only final-position vocab logits, which lowers cost; not the all-position estimate. These are analytic per-sequence counts, not latency. Forward training work for A2 uses m=R,n=min(2,R), but backward/optimizer/checkpoint work not included; never extrapolate training time from unmeasured throughput.

Mini, context 2048, ordinary target (m=1,n=0):
| R | Logical blocks | Decode MAC/token | Prefill all logits MAC |
|---:|---:|---:|---:|
| 1 | 8 | 105,931,776 | 200,850,014,208 |
| 2 | 12 | 148,948,992 | 285,729,619,968 |
| 4 | 20 | 234,983,424 | 455,488,831,488 |

Mini A2 R=4: m=4,n=2 => 26 logical blocks, 418,615,296 MAC per position at L=2048, 812,248,989,696 MAC all-position prefill, +78.1% forward contractions over ordinary R=4 all-position NTP (ratio at full prefill ~1.78). Additional vocab projections are meaningful at small width: one Mini logits projection alone 25,165,824 MAC/token; one Nano 12,582,912. Keep chunked CE and low-cost latent objective separate experiments; no assumption MTP free.

## 6. Memory budgets and state ABI
Persistent mixer payload per sequence: S_GDN=4HgD²+2(Kconv-1)*3g bytes; S_GQA=4kT bytes. Total=3R*S_GDN+(2+2m+R)*S_GQA; includes FP32 matrix and BF16 history/KV only. At Mini R=4,T=2048,m=1, 19,302,400 B (18.408 MiB); m=4 adds 12 MiB; GDN snapshot 2,525,184 B. Nano same R/T,m=1, 9,651,200 B. Raw BF16 weights 2*total; ideal 4-bit weights total/2 (not actual quantized memory). AdamW rough 16 bytes/parameter gives 2.352 GB Mini or 0.689 GB Nano before activations; implementation dependent, actual precision/optimizer to #7. Training activation size not estimated by inference cache. Use logical state ABI and rollback of matrix+raw conv history, KV and metadata; prefix equivalence, independent coda state and immutable expert residency as RFCs 0002–0005. No context capability promised beyond validated lengths.

## 7. Build and research plan
Stage 0 (#8): tokenizer and source provenance, deterministic document split and synthetic fixture. Stage 1 (#9): nonrecurrent dense GQA baseline with same tokenizer interface; validate masks, ties, CPU/GPU smoke, parameter count. Stage 2: sandwich dense recurrent all-GQA, fixed R, all-depth/MTP, exact verifier; stage 3 replace core mixers with GDN hybrid and test; stage 4 replace dense core FFNs with MoE and test, then combined candidate. Keep all-GQA and dense recurrent controls, separately match tokens, resident parameters and estimated contraction compute; controls need dedicated count configs, not pretend equal budgets. Phase 5 expert locality, low precision and adaptive depth only after correctness. Nano first, Mini after pilot and hardware budget review. New GDN variants deferred. Actual model count, forward/backward, packing and cached decode, distribution tests and benchmarks remain open. See [accounting tool](../../../tools/accounting.py) and [validation](../../../experiments/results/rfc0001-accounting.md). Issue #3 draft complete enough for review, not implemented. Next #8 data/tokenizer contract, #9 baseline.
