# Founding research notes — 2026-11-26
Status: working record, not a paper. Captures the substantive design discussion; private conversational material is omitted.

**Historical document:** describes its original design/implementation stage, not current capability. See [current state](../../PROJECT_STATE.md) and [experimental evidence](../../experiments/results/INDEX.md) for the implemented architecture, native FP8/Muon, CUDA qualification and materialized real corpus.

## Motivation and baseline
The [llm-mini](https://github.com/jtrefon/llm-mini) educational implementation is the predecessor and possible baseline: decoder-only Transformer, RoPE, RMSNorm, SwiGLU, GQA, KV caching, streaming Wikipedia, document packing, pretraining and masked-response SFT. Its default config has 22 layers, width 768, 12 query and 4 KV heads, FFN width 2048, BF16 mixed precision, sequence length 1024, effective batch 64 sequences. Its validation split is configured as train, so document-level separation must be independently checked before use as a scientific baseline. Preserve the educational project separately.

## Architectural direction
Use a prelude/shared recurrent core/coda sandwich, initially fixed loop counts per sequence. Hybrid core: three Gated DeltaNet blocks plus one gated GQA block. Sparse FFNs with eight routed experts, top-2, and one always-active shared expert; SwiGLU and FP32 router scores, load-balancing auxiliary loss. Input injection is a candidate. Native MTP is first-generation scope; vision/audio deferred. Medusa is a comparator, not a predetermined winner. LoopMTP latent supervision and Floodgates-style shallow multi-token drafts are promising distinct techniques. Compare sequential DeepSeek MTP, EAGLE-3, LoopSpec and LoopFill (latter reference pending verification).

## Research opportunities
1. Hardware-aware expert routing across recurrent passes: train a differentiable locality proxy and compare cache hit rate, host transfers, load balance and quality. Potential prior art must be reviewed.
2. Recurrent quantization error: compare BF16, FP8 and simulated low-bit weights/state at varying depths, using matched models and quantization calibration.
3. Adaptive depth: compare fixed passes and learned halting under equivalent compute budgets; verify causal state handling first.
4. Joint MTP/routing: examine whether drafting and verification reuse experts and state without compromising exact target sampling.

## Precision and deployment
Start BF16 on RTX 4090 (24 GB), then benchmark FP8 kernels. Native NVFP4 acceleration is not available on Ada; simulated FP4/INT4 QAT is not equivalent to native training. Keep router scoring, sensitive reductions and accumulation at adequate precision; quantify master weights, gradients, optimizer states and activation memory. Initially all weights GPU-resident. For larger models investigate Strata-inspired expert placement in RAM, GPU hot cache, PCIe transfers or CPU expert execution. Do not assume offloading saves latency. Record prefill/decode and energy separately.

## Scale and data
Tentative family: Nano 30–50M, Mini ~150M, Small ~500M, Base 1–2B, Flash 4–8B, Flash XL 16B+ physical parameters (not commitments). Start with 100M–1B tokens for controlled pilots; these are not competitive fully pretrained models. AdamW illustrative 16 bytes/parameter before activations; 150M ~2.4 GB, 500M ~8 GB, 1B ~16 GB, 2B ~32 GB. At an assumed 10k tok/s, 1B tokens takes ~27.8 h excluding overhead; measure actual throughput. Initial data mix by tokens: 40% educational web, 20% licensed code, 15% math/reasoning, 10% Wikipedia, 10% technical docs, 5% multilingual incl. Polish. FineWeb-Edu and Cosmopedia are candidates, not approved sources; verify licensing, deduplication, contamination and holdouts.

## Project identity and scope
SPLOT means interweaving in Polish. English canonical; Polish translations linked to revisions. Independent open-science initiative, not a commercial product or institutional affiliate. Apache 2.0 code; proposed CC BY 4.0 original research; future weights subject to provenance review. Publish reproducible manifests, checkpoints externally, and potentially DOI-versioned releases. Amber and Compass may later provide agentic evaluations; llama.cpp integration may follow reference inference. No claim of results or novelty until validated.
