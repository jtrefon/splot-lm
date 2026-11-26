# RFC 0002 — Fixed recurrent depth and causal state semantics
Draft v0.1 | 2026-11-26 | Issue #4 | [Polski](pl.md)

## 1. Scope and evidence
SPLOT uses one contextualized hidden stream, deterministic initialization, a shared middle stack and distinct prelude/coda. No random latent initialization, adaptive token-depth, cross-depth state aliasing or truncated gradients in the reference. This is a design proposal, not a reproduction of Huginn, Ouro or LOOM. See [primary-source notes](sources.md). RFC 0001 defines initial dimensions; RFC 0003 will define concrete mixers; RFCs 0004/0005 govern experts and MTP. Proposed choices below supersede earlier ambiguous injection defaults only when accepted.

## 2. Forward equations and indexing
Let x in N^(B,T), E in R^(V,d), z=P(E[x]) in R^(B,T,d). P has P_n=2 independent causal blocks, shared core C has M=4 physical blocks, coda Q has Q_n=2 independent blocks, d=768 proposed. Let h_0=z, r=1..R, R in {1,2,4} for primary comparisons, configured R_cap=4. Positions and document IDs stay constant across passes. All maps are causal along sequence, never along future tokens.

At r=1 set u_1=h_0. At r>1 compare three write rules:
- I0: u_r=h_(r-1), no injection (reference).
- I1: u_r=h_(r-1)+a_r z, a_r=0.1/r (additive control).
- I2: u_r=(1-a_r)h_(r-1)+a_r z, same a_r (bounded convex injection candidate).
These coefficients are proposed pilot settings, not fitted literature constants. Do not use absolute sequence statistics or a noncausal global normalization. Optionally add per-pass learned depth vector e_r in R^d, initialized zero, before C; default off. Position RoPE remains token-based, not depth-based. Depth conditioning belongs to separate ablations.

For b=1..M, v_(r,0)=u_r and
```text
a = v_(r,b-1) + gamma_b(r) * Mixer_b(Norm1_b(v_(r,b-1)); S_(r,b))
v_(r,b) = a + gamma_b(r) * FFN_b(Norm2_b(a))
h_r = v_(r,M)
logits_r = Norm_out(Q(h_r)) E^T
```
Mixer and FFN weights and norms are shared across r; distinct b are not shared. Norm is RMSNorm with FP32 reduction, epsilon proposed 1e-5; output cast to computation dtype. Mixer state S is logically separate for every execution. FFN dense initially, routed under RFC 0004 later. Coda weights shared across readout depths, not its state. Prelude and coda use gamma=1 in the initial control. Model width stays d throughout. Logits shape [B,T,V]; use tied E, no output bias in reference.

## 3. Stability and prefix-compatible depth scaling
Compare S0 gamma=1 vs S1 gamma=1/sqrt(2*M*R_cap) vs S2 gamma=lambda/(R_cap*sqrt(M)), lambda=0.5 pilot inspired by LOOM. These are alternatives, not guarantees of stability. Reference starts S0; stability pilot S1/S2 before full training. Per-branch output norm after MoE and loop-residual EMA are additional independent interventions; do not silently import LOOM's entire recipe.

**Critical invariant:** gamma, injection, router conditioning, positions, initial state and other functions used in pass r must not depend on requested R, only on immutable checkpoint/config (including R_cap) and r. For the same prefix and same fixed config, h_r computed as part of R=4 must equal h_r computed in R=2 for r<=2. If gamma uses runtime R, this property fails, preventing exact shallow readout reuse across depths. LOOM's original H-dependent scaling is therefore not copied verbatim. A separately trained or R-conditioned drafter can still be a valid proposal for exact verification, but cannot reuse target-prefix states as if equivalent. Changing R_cap is a model/config change; invalidate caches. No assumption of useful extrapolation beyond trained R_cap.

Monitor per-pass RMS, residual-to-stream ratio, cosine similarity, relative update norms, nonfinite values, gradient norms by pass, NTP delta and expert overlap. Compare matched resident/compute/token conditions. Do not select a recipe merely for bounded norms: require held-out loss, reasoning and context-access quality. Fixed initial R=2 and R=4 as separate runs; later train mixed depth by sampling one R per batch from declared distribution, preserving fixed R per sequence/request. Evaluate all trained depths; no adaptive per-token depth yet.

## 4. Training and gradients
All reference recurrent applications participate in autograd; no detach at loop boundaries. Shared parameter theta receives sum of contributions from each occurrence in the unrolled computation graph (including intermediate losses); compare against untied clones initialized identically and summed gradients. P receives gradients through h_0 and each enabled input write. Full unroll may increase activation memory proportional to logical executions; activation checkpointing recomputes exact forward with deterministic RNG and no mutable inference cache. Train full BPTT initially. Truncation and segmented backpropagation are biased alternatives, not equivalent; defer until memory evidence justifies them. A0/A1/A2 losses and masking defined in RFC 0005. A1 at R=1 has no earlier-depth term. Disable dropout for correctness tests; training RNG policies recorded. Shared optimizer state exists once per parameter, never one copy per pass.

## 5. Causal execution and logical cache ABI
Checkpoint/config fingerprint includes topology, weights, dtype, RoPE, R_cap, routing policy and readout variant. Session owns committed prefix tokens, document IDs, position cursor and immutable fixed target R. Logical state key:
```text
(session_id, sequence_id, stage, pass_or_readout_depth, block_index)
P: pass=0; C: pass=1..R; Q: readout_depth=selected r
```
No physical-weight duplication implied. A prelude block has one cache per sequence; each core block has R independent states; each coda block has separate state for each active readout. Only final coda needed for ordinary decode. Shallow proposal with same config and available committed prefix at r may read the corresponding core states; it cannot alias writable target state. Missing shallow coda cache must be built by replay or maintained from prefill; never substitute the final coda cache. Coda weights may contain causal mixers; tokenwise auxiliary projection has no sequence cache. Immutable weight cache from RFC 0004 uses a different namespace without pass ID for shared weights.

Stage-major prefill (full sequence):
```text
z = prelude(x, causal_doc_mask)
h = z
for r in 1..R:
    h = inject(h,z,r)
    for b in 1..M: h, S[r,b] = mixer_block_b(h, empty_state)
    if r in requested_readouts: logits[r], Qstate[r] = coda(h, empty_state)
```
Each block returns state after consuming prefix; mask padding and reset at document boundaries. Incremental token-major path consumes each token through P, all C passes and selected Q readouts in the same dependency order, reading and updating only that logical invocation's previous-prefix state. For each (r,b,t), dependencies are previous stage at t and same logical invocation at positions <=t; the two schedules are topological traversals of the same causal graph if mixers implement identical state transitions. Thus prefix-extension invariance holds by induction, subject to numerical tolerance. Store pending next-token logits for committed prefix; consume each newly emitted token once. Chunk prefill uses existing states, not empty states, after first chunk. Per-sequence resets invalidate every stage/pass/readout and restart document-relative positions. Pack masks isolate docs; no cross-document labels or convolution history.

Reference snapshot transaction: before speculative suffix, snapshot every mutable activation state, positions, prefix length and pending logits. Draft and verification scratch isolated. On rejection restore snapshot and replay accepted tokens plus correction exactly once; on full acceptance commit verified states and process optional bonus token exactly once. Use RFC 0005 sampling verifier, EOS/budget rules. GQA truncation alone insufficient for GDN matrix and convolution state; snapshots/replay defined concretely in RFC 0003. Weight cache may retain immutable entries, not activation states. Snapshot must own data (clone/copy-on-write), not alias in-place mutable tensors. Cross-depth adaptive execution requires new contract, out of scope.

## 6. Acceptance tests and pilot plan
Toy oracle under tests/numerical exercises two distinct causal traversal orders, gradient summation, document reset, depth/prefix invariance and rollback using a simple linear/tanh recurrent mixer. It is not GQA/GDN or the LM. Actual implementation gates:
- CPU float64/FP32 dense reference, R=1/2/4, multiple lengths, batch/packed doc cases, tokenwise and chunked prefill, cached vs uncached logits/state, all-depth coda and shallow readout invariance, future token perturbation. Start CPU FP32 atol=1e-5, rtol=1e-4; record max abs/relative errors; dtype-specific tolerances must be established before experiments, never relaxed after observing confirmatory runs. BF16/GPU separately.
- R=1 matches untied sandwich with one core call, parameter identity across passes, optimizer sees unique tensors. Gradient equality versus untied clones with same values and aggregate objective; compare full unroll to checkpointing. Compare norm/gradient diagnostics and assert finite.
- Reset/padding/batch isolation, rollback at every rejection position, all-accepted and EOS; state tensor equality/tolerance and identical greedy outputs. Test q=p and disjoint support in #2; no stochastic identity claim.
- For R=2/4 compare I0/I1/I2 and S0/S1/S2 in a staged factorial pilot, one change at a time initially, same data and seed schedule. Depth vectors, router variants, output norm separately. Measure train FLOPs, token budget, wall clock, peak memory, intermediate and final NTP, context retrieval, math/code tasks. Register thresholds, seeds and tuning budget before full runs. No performance claim from toy oracle.

Dependencies #3/#5/#2/#6/#8/#9/#10. Next: specify exact GQA/GDN and cache tensor sizes (#5), reconcile parameter accounting (#3), then implement reference dense model (#9). Issue #4 remains open until review and real mixer integration tests; this PR advances its mathematical and toy-oracle acceptance criteria.
