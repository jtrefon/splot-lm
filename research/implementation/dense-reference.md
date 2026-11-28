# Dense reference implementation v0.1
2026-11-26 | Issue #9 | [Polski — summary](dense-reference.pl.md)

## Scope
First actual trainable reference implementation in src/splot/dense.py. Implements conventional decoder-only GQA/RoPE/RMSNorm/SwiGLU with bias-free projections, two pre-normalized residual branches, tied embeddings and output matrix. No depth recurrence, GDN, expert routing, auxiliary MTP, speculation sampler, full trainer, production tokenizer or fused kernels yet. This is the initial control for RFC 0001 v0.2. PR #19 is open at branch creation: this PR is independently based on main, uses a minimal original synthetic byte fixture with compatible ABI, and does not import or duplicate the larger data pipeline. Real-corpus training remains gated by #8.

## Config and accounting
DenseConfig immutable dimensions, heads, epsilon, RoPE base and initialization. Positive dimensions, even head size and Q/KV grouping validated. Learned matrices initialized normal std 0.02, RMSNorm scale ones, no bias. This is a provisional initialization for smoke; training recipe and tuning budget still to be preregistered.
| Control | Width | Layers | Q/KV heads | Head size | FFN width | Vocab | Parameters |
|---|---:|---:|---|---:|---:|---:|---:|
| Smoke | 32 | 2 | 4/2 | 8 | 64 | 264 | 27,040 |
| Dense-Nano | 384 | 8 | 6/2 | 64 | 1024 | 32768 | 25,172,352 |
| Dense-Mini | 768 | 8 | 12/4 | 64 | 2048 | 32768 | 75,510,528 |

Formula V*d + L*(d*(2q+2kv)+3*d*m+2d)+d, q=Q_heads*head_dim, kv=KV_heads*head_dim. Counts of all three verified against instantiated parameter inventories (larger controls on meta device only, not forward-tested). JSON configs under configs/baselines/. Dense-Nano/Mini are width/topology controls, NOT matched-resident-parameter or matched-compute controls for the full 43M/147M candidates. Add matched configurations separately before comparisons. No activation or GPU budget claimed.

## Model ABI and causal semantics
Inputs: input_ids, document_ids, position_ids (int64), valid (bool), all [B,T] on same device. Padding metadata document -1, position 0; valid document IDs nonnegative. Each row is traversed by contiguous logical document runs, ignoring invalid slots. Document position must begin at 0 and advance by one for each consumed valid token, including across cache chunks. Document switch resets attention state and position. IDs should uniquely identify segments within a training block, as in #8; repeated ID after an intervening document is a new run, not joined. All tokens (even padding) must be valid embedding indices; reserved PAD=0. Attention processes each run independently; no future tokens or cross-document keys are visible. Output shape [B,T,V], invalid logits zero. Uses explicit grouped contractions without replicating KV heads in cache, interleaved RoPE, reference FP32 (FP64 for FP64 inputs) norm/rotary arithmetic; under autocast einsum/linear kernels may execute in reduced precision, so BF16 path is exploratory. Not an optimized SDPA implementation; CPU metadata .tolist(), row loops, dynamic allocation and dense scores are intentional correctness tradeoffs.

Forward returns (logits, cache_or_none). Cache requested only in eval + no_grad/inference_mode. DenseCache is a frozen container holding owner identity, per-row last-document ID/valid length, and per-layer per-row rotated K and unrotated V [Hkv,L,D]. Tensor contents are read-only by convention; no in-place writes by model. Padding appends no KV. Earlier document states discarded on document switch. Continuing chunks concatenate keys/values and use causal offset past+local position. Snapshot is a retained old cache; branch forward returns a new one, allowing replay from prefix. Cache is bound to current model instance, batch row order, weights, device, precision and config; caller must discard after parameter/device/dtype/autocast changes, reorder or new request. Owner identity is checked but not all these invariants automatically; full fingerprint and production state manager follow RFC 0002. This cache covers dense GQA only, not future GDN or recurrent logical states.

Loss accepts pre-shifted offset-1 labels with -100 for ignore, sum CE divided by valid count clamped to 1; all-ignored labels give zero (no accidental second shift). Caller owns label correctness, skip empty-target training batches (AdamW weight decay may otherwise still update). Does not automatically infer labels or validate their document relationship. Production data fixture #19 integration test after merge.

## Validation
Six CPU float64 tests: actual parameter count/tied output, future perturbation/doc isolation, ragged batched tokenwise and chunk cached equivalence with reset, branch immutability/padding no-op and replay, finite gradients/empty masked CE, invalid input and cross-instance cache rejection. 40-step FP32 AdamW smoke on two fixed original EN/PL examples, 27,040 params, loss from 5.587563 to 2.033330; FP32 cached max absolute logit difference 1.192093e-6. Additional tiny CPU BF16 autocast forward/cache check passed (not a training or GPU result). See [evidence](../../experiments/results/dense-reference-v0.1.md).
No generalization, throughput, full-scale forward, training or production data evidence. #9 remains open for GPU smoke, #19 fixture integration and maintainer review.

## GPU smoke protocol (not executed)
From checkout at exact reviewed commit with supported CUDA PyTorch, record Python, PyTorch, CUDA runtime, driver, GPU, available VRAM, config and Git SHA. Run CPU tests first; then:
```bash
PYTHONPATH=src python tools/smoke_dense.py --config configs/baselines/dense-smoke.json --device cuda --steps 40
PYTHONPATH=src python tools/smoke_dense.py --config configs/baselines/dense-smoke.json --device cuda --bf16 --steps 40
```
Pass: finite losses/gradients, final training loss <50% initial on smoke, cache assert_close FP32 atol 1e-5 rtol 1e-4, provisional BF16 atol=rtol=.05 (record max error). These are smoke tolerances, not precision equivalence at scale. Preserve stdout and environment; no automatic tolerance widening if failed. Then separately run forward/backward on Dense-Nano short sequence with recorded memory/length; do not use tiny-example smoke gate as general learning-quality criterion. No performance extrapolation. GPU unavailable in current sandbox.

## Handover
Review this PR and pending independent #19. Once both are merged, reconcile PROJECT_STATE.md, decisions and backlog in the next work unit, add integration test using actual tools/data_fixture.py. Next fixed-depth all-GQA recurrent reference per #4, then native MTP #2 (A0/A1/A2, exact verifier). GDN #5 and MoE #6 later, no optimization until correctness. Existing shared state files intentionally untouched here to avoid collisions with pending data PR; live GitHub issue/PR comments are authoritative.

## T0021 dense NTP comparator integration (2026-11-28)
T0021 adds an explicit **`model_family: dense-ntp`** path to the existing experiment CLI without changing the default SPLOT path, config hash, identity or checkpoint semantics. Dense mode uses `DenseConfig`/`DenseLM` only and rejects recurrence/expert fields, non-A0 modes, `loops != 1`, FP8/reuse, and nonzero auxiliary loss weights. It reuses the existing tiny synthetic/verified-v2/pilot data loaders, checkpoint/resume, evaluation and semantic Muon/AdamW partition.

The dense objective is final NTP only: `Objective.total = final_ntp`, with early/offset/balance zero and `offset2_count = 0`. Dense configs have no dormant MTP head. Configs `configs/experiments/dense-ntp-512x6-q8kv4-ffn2432.json` (35,527,168 parameters) and `dense-ntp-512x6-q8kv2-ffn2176.json` (32,381,440) are planning candidates from accepted T0019, not trained or optimized results. `splot-nano-v16384-a0.json` pins the V16384 SPLOT A0 R2 comparator (36,763,620 total including dormant MTP, 36,615,780 NTP base) with explicit `balance_weight=0.01`.

`tools/accounting.py` adds `DenseShape`/`dense_account` for per-document causal contraction counts; these are analytical MACs, not training FLOPs. `tools/test_dense_harness.py` covers exact counts, semantic partition, dense objective equality, continuous 4 vs 2+2 resume, cross-family rejection, shared data ingestion and CPU BF16 capability when available. No full training, real data, test access, FP8 dense path or GPU/MPS qualification is claimed.
