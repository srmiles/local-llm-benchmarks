# Local LLM Benchmarks (B60 Pro)

Local LLM benchmarks & configs for **Intel Arc Pro B60 (24 GB, Battlemage / Xe2)** on bare-metal Ubuntu 26.04.

All numbers below are measured on the same physical card. The stack has shifted over time — vLLM-XPU → LM Studio Vulkan → llama.cpp Vulkan → llama.cpp SYCL (current) — but the hardware is constant. Unless a row says otherwise, benchmarks were taken on [`llama.cpp:sycl-f16`](configs/images/llama.cpp-sycl-f16/README.md) — **all llama.cpp services now on b10256 (commit `6c8dcaa7a`, cutover completed 2026-08-04)**. Rollback tag `llama.cpp:sycl-f16-b10215-safe` preserved on disk.

**b10256 impact (measured 2026-08-04, isolated `/completion` probes on 5K real-workload prompts, `cache_prompt: false`, warmup preflight):**

| Model | b10215 prefill | b10256 prefill | Δ | Decode Δ | MTP acc Δ |
|---|---|---|---|---|---|
| Ornith 9B + MTP | 1,442 tps | **1,789 tps** | **+24%** | parity (66→64) | maxed (100%) |
| Gemma 4 E2B + Google MTP | 3,681 tps | 3,169 tps | ~parity | **+18%** (138→164) | **+25pp** (67→92%) |
| Gemma 4 26B QAT + MTP | 1,164 tps | **1,335 tps** | **+15%** | parity | maxed |
| Gemma 4 26B Q4_K_M + MTP | 1,180 tps | **1,475 tps** | **+25%** | parity | ~same |

Universal win across all four configs. Prefill 15-25% up on 3/4, ~parity on E2B (which was already near XMX GEMM ceiling). MTP acceptance either maxed at 100% or gained +25pp. Likely from commit series between b10215 and b10256 including PR#25852 (SYCL concat kernel parallelization) plus multiple MTP-verification path improvements.

**b10215 impact (retained for historical context):** decode + MTP acceptance unchanged from b10068 within noise (verified on decode-only synthetic bench pre-cutover), but ~2× prefill uplift discovered 2026-08-01 during brain-eval Track 2 ingest bake-off — real workload prompts hit ~3,000 tps prefill on Ornith 9B via prefix-cache reuse. Root cause: SYCL oneMKL GEMM flash attention for XMX (#25025) merged post-b10068. See [`charts/b10215_prefill_uplift.png`](charts/b10215_prefill_uplift.png).

**Methodology note (added 2026-08-04):** The b10215 "~3,000 tps prefill" figure above was measured via **real-workload wall-clock during brain-ingest** (multiple sequential prompts benefiting from prefix cache reuse across the pipeline). The b10256 comparison table uses **isolated `/completion` probes with `cache_prompt: false` + warmup preflight**, which measures single-request cold prefill only. Both numbers are legitimate but they measure different things — never mix them in a delta. Rule for future comparisons: pick one methodology and hold it constant across A/B. See finding #21 below.

> **Investigation resolved 2026-07-20:** brain-eval flagged an Ornith 9B MTP acceptance drop on b10068 (46% vs 64% baseline, small-N greedy bench). Methodology-matched A/B disproved it: (a) both b9948 and b10068 produce different hashes across repeated identical greedy runs — SYCL FP non-determinism is pre-existing, not b10068-introduced; (b) isolated prefill probe shows b10068 delivers a real but modest **~3% uplift at long context, flat at short** (not the "+42%" originally claimed). Prod acceptance range under real workload (68-76%) is the ground truth. **b10068 stays live.** See [ornith notes](models/production/ornith-1.0-9b.md#notes) for full detail.

## Current production stack

| Port | Container | Model | Purpose |
|---|---|---|---|
| 8002 | `llamacpp-sycl` | [Ornith 1.0 9B Q4_K_M + MTP drafter](models/production/ornith-1.0-9b.md) | chat + categorise + pi.dev agent |
| 8004 | `llamacpp-embed` | [EmbeddingGemma-300M QAT Q8_0](models/production/embeddinggemma-300m.md) | brain embeddings |
| 8008 | `tei-rerank` | [bge-reranker-v2-m3 fp16](models/production/bge-reranker-v2-m3.md) | rerank prod (TEI XPU-IPEX patched, no leak) |
| 8009 | `llamacpp-sycl-gemma4-e2b` | [Gemma 4 E2B QAT Q4_0 + Google MTP](models/parked/gemma-4-e2b-categorise.md) | **warm-standby**; approved for categorise cutover when 2nd B60 arrives — currently VRAM-resident but only dispatches when brain-eval routes to it (Mode B co-load, avoids parallel dispatch) |

**Retired 2026-07-19:** `llamacpp-rerank` on `:8007` (llama.cpp SYCL bge-reranker path). Kept as fallback for the first day after TEI's empty_cache patch shipped; retired after TEI proved stable at 1.4 GiB flat over 10+ hours. Launcher preserved at `/data/llm/launch/start-llamacpp-rerank.sh` for on-demand relaunch if TEI ever fails.

**Reasoning fallback:** [Gemma 4 26B-A4B Q4_K_M + MTP](models/production/gemma-4-26b-a4b.md) — launcher on disk, not running by default.

**Historical:** original `llamacpp-categorise` on `:8006` ran [Qwen3-4B-Instruct-2507](models/retired/qwen3-4b-instruct-2507.md) until quality regressed (2026-07-19).

**Categorise slot experiment 2026-07-22 — reverted same day.** Stood up [Gemma 4 E2B QAT on :8006](models/parked/gemma-4-e2b-categorise.md) as a dedicated categorise slot to relieve Ornith `:8002`. Diagnostic proved severe cross-process SYCL contention on the single B60 (Ornith 52 → 16 tps and Gemma 88 → 12 tps when both active — 3× throughput loss both sides). Reverted to Ornith-served categorise. Gemma launcher preserved at `/data/llm/launch/start-llamacpp-categorise-e2b.sh` for future use when a **second B60 is added** — then split-slot architecture becomes viable with each SYCL context owning its own GPU. See [candidate hunt](models/tested/categorise-candidates.md) for the candidate bench data.

## Chat / instruct benchmarks

Decode = steady-state single-stream tok/s. Prefill measured at the context noted. VRAM is peak observed with the config running.

| Model | Quant | Total / active params | Decode tok/s | Prefill tok/s | VRAM | Status |
|---|---|---|---|---|---|---|
| [**Gemma 4 E2B + Google MTP**](models/parked/gemma-4-e2b-categorise.md) ⭐ | QAT Q4_0 + BF16 drafter | 2B + drafter | **138.8** (67.8% MTP acc) | **3,681 @ 2K** | 4.5 GiB | benched 2026-07-31 on b10215; Google official MTP drafter ([HF](https://huggingface.co/srmiles/gemma-4-E2B-it-assistant-GGUF)); approved categorise, awaiting 2nd B60 |
| [Gemma 4 E4B + Google MTP](models/tested/gemma-4-e4b.md) | QAT Q4_0 + BF16 drafter | 4B + drafter | 114.1 (66.7% MTP acc) | 2,319 @ 2K | 7 GiB | benched 2026-07-31 on b10215; Google official MTP drafter ([HF](https://huggingface.co/srmiles/gemma-4-E4B-it-assistant-GGUF)) |
| [Gemma 4 12B + Google MTP](models/tested/gemma-4-12b-qat.md) | QAT Q4_0 + BF16 drafter | 12B dense + drafter | 70.4 (69.7% MTP acc) | 1,053 @ 2K | 8.5 GiB | benched 2026-07-31 on b10215; Google official MTP drafter ([HF](https://huggingface.co/srmiles/gemma-4-12B-it-assistant-GGUF)); 12B QAT beats 12B dense despite finding #2 (drafter shifts balance) |
| [MiniCPM5-1B](models/tested/minicpm5-1b.md) | Q4_K_M | 1.08B dense | **~187** | **4,642 @ 2K** | ~3 GB | tested 2026-07-19; fastest tested; JSON fence issue on categorise |
| [Ornith 1.0-35B MTP APEX](models/tested/ornith-1.0-35b-mtp-apex.md) | APEX I-Compact (IQ) | 35B / 3B | 35.4 | 816 @ 5K / 802 @ 12K | ~19 GB (fits co-res, 3 GiB headroom) | tested 2026-07-22; scale-up of prod Ornith 9B; **-32% decode** vs 9B; IQ-quant penalty; VLM-capable; waiting on K-quant MTP variant |
| [**Qwen 3.6-35B-A3B-MTP**](models/tested/qwen3.6-35b-a3b-mtp.md) ⭐ | UD-Q4_K_XL | 35.5B / 3B | **49.0** | 798 @ 12K cold / 974 @ 5K | **24.4 GB (won't fit prod)** | Ornith-parity speed at 4× params; need smaller quant for co-residence |
| [Qwen 3.6-35B-A3B-MTP](models/tested/qwen3.6-35b-a3b-mtp.md) | UD-Q4_K_S | 35.5B / 3B | 37.7 | 820 @ 12K cold / 985 @ 5K | 24.1 GB (still fails co-res) | tested 2026-07-19; only saves 0.3 GB vs Q4_K_XL; prefill wins but decode -23% due to MTP acceptance drop |
| [Qwen 3.6-35B-A3B-MTP](models/tested/qwen3.6-35b-a3b-mtp.md) | UD-IQ4_XS | 35.5B / 3B | 31.6 | 776 @ 12K / 918 @ 5K | 21.1 GB (fits prod with 0.5 GB headroom) | tested 2026-07-19; -36% decode + -22pp MTP acceptance vs Q4_K_XL; IQ quants underperform K quants on B60 |
| [Qwen 3.6-35B-A3B Claude 4.7 Opus Distilled](models/tested/qwen3.6-35b-a3b-claude-distilled.md) | APEX-MTP Compact | 35.5B / 3B | 36.9 | 763 @ 12K / 887 @ 5K | 19.4 GB (fits prod) | tight-reasoning distillation; only 35B-A3B that co-resides cleanly |
| [Qwen 3.6-35B-A3B Kimi K2.6 Distilled](models/tested/qwen3.6-35b-a3b-kimi-distilled.md) | IQ4_XS | 35.5B / 3B | 30.6 | **904 @ 12K cold** ⭐ | 21.4 GB (0.2 GB co-res headroom) | fastest cold prefill benched; verbose reasoning; no MTP |
| [Qwen 3.6-35B-A3B (base)](models/tested/qwen3.6-35b-a3b.md) | UD-Q3_K_M | 34.7B / 3B | 31.1 | 823 @ 2K | 20.0 GB | superseded by MTP variant above |
| [**Gemma 4 26B-A4B QAT + MTP**](models/production/gemma-4-26b-a4b.md) ⭐ | QAT Q4_0 + community MTP Q8_0 | 26B / 4B | **54.2** (100% MTP acc) | **1,164 @ 5K** (real workload, b10215) | **17.5 GiB** | **new best reasoning-fallback config 2026-08-01**; overtakes Q4_K_M+MTP on decode and VRAM despite finding #2 |
| [Gemma 4 26B-A4B (it) Q4_K_M + MTP](models/production/gemma-4-26b-a4b.md) | Q4_K_M + community MTP Q8_0 | 26B / 4B | 49.0 (96% MTP acc, b10215) | 1,180 @ 5K (b10215, +21.5% vs b10068) | 19.7 GiB | reasoning fallback; historical peak 53.0 tps on b10068 was 22.9 GiB |
| Gemma 4 26B-A4B (it) Q4_K_M (base) | Q4_K_M | 26B / 4B | 44.1 (b10068) | 632 @ 12K | 20.9 GB | original locked prod (pre-MTP) |
| Gemma 4 26B-A4B QAT (no MTP baseline) | Q4_0 | 26B / 4B | 40.1 (b10068) | 602 @ 12K | 18.2 GB | superseded by **+MTP + b10215** row above: +35% decode, +93% prefill |
| **[Ornith 1.0 9B + MTP](models/production/ornith-1.0-9b.md)** ⭐ (b10256) | Q4_K_M | 9B dense | **64** (100% MTP acc) | **1,789 @ 5K** (isolated `/completion`) / ~3,000 tps under real-workload prefix-cached ingest | 10.9 GiB (w/ MTP head) | **production chat** — b10256 delivers +24% isolated prefill over b10215 same-methodology bench; +15% decode vs b10215 baseline of 56 tps |
| [Qwen3-Coder-30B-A3B](models/tested/qwen3-coder-30b-a3b.md) | UD-Q4_K_XL | 30B / 3B | ~38 | ~700 | ~20 GB | tested; capability too poor for pi.dev |
| [Devstral Small 2 24B](models/tested/devstral-small-2-24b.md) | UD-Q4_K_XL | 24B dense | ~18 | ~340 | ~15 GB | tested; dense penalty visible |
| [Qwen3.6-27B](models/tested/qwen3.6-27b.md) | Q4_K_XL | 27B dense | ~22 | ~380 | ~17 GB | tested; bartowski build |
| [Laguna XS-2.1 (no drafter, upstream)](models/tested/2026-08-06-new-candidates-sweep.md#laguna-xs2-poolside-33b-a3b-moe) | Q4_K_M | 33B / 3B | 29.5 | 1,213 @ 5K | 22.1 GiB | tested 2026-08-06 on b10256; MoE + SWA; **DFlash drafter needs Poolside fork** (upstream framework only, no Laguna decoder contract); deferred to 2nd B60 for coding-slot bake-off |
| [gpt-oss-20b (no drafter)](models/tested/2026-08-06-new-candidates-sweep.md#gpt-oss-20b-openai) | Q4_K_M (MXFP4 native) | 20.9B / ~2.6B active (4-of-32) | 25.1 | 1,265 @ 5K | 13.4 GiB | tested 2026-08-06 on b10256; no MTP drafter released by OpenAI; potential agentic-quality upgrade — needs qualitative bake-off vs Ornith on pi.dev |
| [Mistral-Small-3.1-24B](models/tested/mistral-small-3.1-24b.md) | Q4_K_M | 24B dense | ~19 | ~350 | ~15 GB | tested (Vulkan era) |
| [Gemma 4 12B (QAT) — no MTP baseline](models/tested/gemma-4-12b-qat.md) | Q4_0 | 12B dense | 19.7 (b10068) | 167 @ 1K | ~9 GB | superseded by **+ Google MTP** row above: **+257% decode / +530% prefill** on b10215 |
| [Gemma 4 E4B (QAT) — no MTP baseline](models/tested/gemma-4-e4b.md) | Q4_0 | ~4B | 73.9 (b10068) | 376 | ~3 GB | superseded by **+ Google MTP** row above: **+54% decode / +517% prefill** on b10215 |
| [Gemma 4 E4B — no MTP baseline](models/tested/gemma-4-e4b.md) | Q4_K_M | ~4B | 68.3 (b10068) | 466 | ~3 GB | retained; K-quant vs QAT reversal @ 4B (see finding #2); superseded on speed by Google MTP variant |
| [Gemma 3 4B](models/tested/gemma-3-4b.md) | Q4_K_M | 4B | ~78 | — | ~3 GB | tested; system-role template issue |
| [Qwen3-4B-Instruct-2507](models/retired/qwen3-4b-instruct-2507.md) | Q4_K_M | 4B | ~94 (60s under 4-way) | 766 aggregate | ~1 GB | **retired** (was categorise prod) |
| [Qwen2.5-Coder-14B AWQ (vLLM-XPU)](models/tested/qwen2.5-coder-14b-awq.md) | AWQ int4 | 14B | 22.9 peak / 13–15 typical | **1,891** peak | ~11 GB | retired; cross-stack reference |
| Qwen 3.6-27B (LM Studio Vulkan) | Q4_K | 27B | 33.6 | 97 @ 12K cold | 17.6 GB | historical Vulkan baseline |

## Embed / rerank benchmarks

| Model | Server | Throughput / latency | VRAM | Notes |
|---|---|---|---|---|
| [EmbeddingGemma-300M](models/production/embeddinggemma-300m.md) | llama.cpp SYCL :8004 | **23,208 tok/s** @ 1800 tok, 253 emb/s batch 64 short | 0.5 GB | **prod embed** (dim 768, ~57 RTEB); wins single-embed by 1.85× vs TEI |
| EmbeddingGemma-300M (same-model A/B) | TEI XPU-IPEX (bench 2026-08-06) | 12,500 tok/s @ 1800 tok (-46%), **313.7 emb/s** batch 64 (+24%) | ~1 GB | TEI wins batch throughput but loses single-embed — Gemma3-encoder isn't TEI's best-tuned architecture |
| [Qwen3-Embedding-0.6B](models/tested/2026-08-06-new-candidates-sweep.md#qwen3-embedding-06b-on-both-runtimes) | TEI XPU-IPEX (bench 2026-08-06) | 12,081 tok/s @ 1800 tok, 192.8 emb/s batch 64 | ~1.2 GB | tested 2026-08-06; MTEB 64.3 (+13% quality) but 40-70% slower than EG in both regimes; 19× faster on TEI than llama.cpp (native runtime unlocked) |
| Qwen3-Embedding-0.6B (wrong-runtime) | llama.cpp SYCL (bench 2026-08-06) | 634 tok/s @ 1800 tok, 89.8 emb/s batch 64 | ~1.2 GB | wrong-runtime penalty — model is TEI-native |
| [Nemotron-3-Embed-1B (Q4_K_M)](models/tested/nemotron-3-embed-1b.md) | llama.cpp SYCL (isolated bench) | 3,361 tok/s @ 1800 tok, 29 emb/s batch 64 short | 1.0 GB | tested 2026-07-22; dim 2048, multilingual, RTEB 72.4; **6–9× slower** than EG on B60 (FA broken on ministral3 arch) |
| [LFM2.5-Embedding-350M (Q8_0)](models/tested/lfm2.5-embedding-350m.md) | llama.cpp SYCL (isolated bench) | 21,717 tok/s @ 1800 tok, 80 emb/s batch 64 short | 0.6 GB | tested 2026-07-22; dim 1024, 11 languages; parity single, EG wins batched 3.2× |
| [bge-reranker-v2-m3](models/production/bge-reranker-v2-m3.md) | **TEI XPU-IPEX :8008** | **109 ms / 25 pairs** | 1.4 GB | prod (7–9× faster than llama.cpp) |
| [LFM2.5-ColBERT-350M (Q8_0)](models/tested/lfm2.5-colbert-350m.md) | llama.cpp SYCL (isolated bench) | 293.6 ms / 25 pairs (MaxSim, `-fa off`) | 0.6 GB | tested 2026-07-22; 11 languages; **2.7× slower than bge**; per-token vector storage 21× if used as retriever |
| [bge-reranker-v2-m3](models/production/bge-reranker-v2-m3.md) | llama.cpp SYCL :8007 | 800–1,000 ms / 25 pairs | ~4 GB | retired fallback (2026-07-19) |

## VRAM co-residence budget

Isolated bench numbers are misleading — production has to fit all services simultaneously. **Non-chat steady-state stack** (as of 2026-07-19, after llama.cpp rerank retirement):

| Container | VRAM steady |
|---|---|
| llamacpp-embed (EmbeddingGemma-300M) | 0.5 GiB |
| tei-rerank (patched image, no leak) | 1.4 GiB |
| **Non-chat total** | **~1.9 GiB** |
| **Available for chat model** | **~22.1 GiB / 24 GiB** |

Note: reverted to the pre-2026-07-22 budget after the categorise split experiment showed cross-process SYCL contention on single B60. When a second B60 is added, the dedicated categorise slot (~2 GiB, Gemma 4 E2B QAT) will move to that card, keeping the primary B60 chat-model budget at 22.1 GiB.

Any candidate that reports isolated VRAM > 22.1 GiB either won't fit alongside prod, or needs a smaller quant / context / eviction trade-off. See individual model pages for per-candidate co-residence analysis.

**Historical (pre-2026-07-19)**: budget was 21.6 GiB while `llamacpp-rerank` fallback ran on `:8007`. Freed 0.5 GiB steady when retired; nothing above the ceiling suddenly fits, but 35B-A3B candidates have marginally more room, and Qwen 3.6-35B-A3B Claude APEX-MTP Compact now co-resides with **2.7 GiB headroom** instead of 2.2.

## Working Gemma 4 MTP drafter GGUFs on HF

The community GGUFs of Google's Gemma 4 MTP assistants use a broken architecture string (`gemma4_assistant` underscore vs upstream `gemma4-assistant` hyphen) and fail to load on any modern `llama.cpp` build. During this benching effort I converted Google's official BF16 safetensors from scratch with `llama.cpp`'s own `convert_hf_to_gguf.py` (b10215) and uploaded the working GGUFs:

- [`srmiles/gemma-4-E2B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-E2B-it-assistant-GGUF) — 170 MB, pairs with `google/gemma-4-E2B-it`
- [`srmiles/gemma-4-E4B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-E4B-it-assistant-GGUF) — 172 MB, pairs with `google/gemma-4-E4B-it`
- [`srmiles/gemma-4-12B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-12B-it-assistant-GGUF) — 862 MB, pairs with `google/gemma-4-12B-it`

All Apache 2.0. See [`models/hf-uploads/gemma-4-assistant-drafters.md`](models/hf-uploads/gemma-4-assistant-drafters.md) for full detail (conversion recipe, bench numbers, usage).

## Track 2 quality bake-off: Ornith 9B vs Gemma 4 E2B (2026-08-01)

Head-to-head brain-eval on real Tier A corpus, grammar-constrained (JSON schema server-side), sampler pinned, 131K context matched across arms, 3-repeat ingests to control variance. Judge: DeepSeek-V3.2 with retry-with-backoff.

| Metric | Ornith 9B + MTP | Gemma 4 E2B + Google MTP | Delta |
|---|---|---|---|
| Pass rate (3-repeat mean) | 0.556 | 0.422 | quality noise-band overlap |
| Mean score (LLM-judge) | 0.784 | 0.787 | ~tie |
| Classify median latency | 19,403 ms | 6,286 ms | **3.1× faster** |
| VRAM steady | 10.9 GiB | 5.8 GiB | **1.9× smaller** |
| Full-ingest wall-clock | 2,370s | 1,145s | **2.1× faster** |

**Interpretation: quality is indistinguishable inside ingest-run variance** (individual ingests swung 0.400–0.600 across all four runs regardless of model — ingest variance dominates model gap). E2B wins latency, VRAM, and wall-clock. **Decision: E2B is approved to move into the categorise slot when a 2nd B60 arrives** — cross-process SYCL contention on a single card (finding #15) still forbids concurrent dispatch today, so E2B stays warm-standby on `:8009`. See [`charts/track2_quality_vs_speed.png`](charts/track2_quality_vs_speed.png), [`charts/gemma_google_mtp.png`](charts/gemma_google_mtp.png), and [`models/parked/gemma-4-e2b-categorise.md`](models/parked/gemma-4-e2b-categorise.md) for full methodology and caveats (Tier A only; Tier C confirmation pending; Track 1 chat/agent eval not run).

## Key findings

1. **MoE beats dense on Battlemage.** 26B-4B-active decodes ~2× faster than 12B dense at similar quality.
2. **Post-training K-quant beats QAT Q4_0 at ≥26B — with build-specific caveat that has since reversed.** Reverses at 4B (finding was model-size-dependent). Also briefly reversed at 26B-A4B on b10215+MTP where QAT edged out Q4_K_M by ~10% decode. **On b10256 (2026-08-04) the ordering restored to K-quant winning:** Q4_K_M + MTP prefill 1,475 tps vs QAT + MTP 1,335 tps (+10% for K-quant), decode ~parity. QAT still saves 2.3 GiB VRAM at 26B-A4B, so it's the right pick when VRAM headroom matters more than absolute speed. **Bottom line by build:** on b10068 K-quant wins (original finding); on b10215+MTP QAT briefly won; on b10256+MTP K-quant wins again but by a smaller margin. On any recent build: K-quant at ≥26B, QAT at ≤4B — the exception window was one build.
3. **MTP drafters are worth +5–15%** when a purpose-built head exists (Gemma 4 official, Ornith community).
4. **`-ub 2048` is the SYCL sweet spot.** 4096 regresses on this card; monotonic climb from 16 → 2048 then plateau.
5. **`-fa on` is mandatory** — turns 36s "warm" re-prefills into 0.55s cache hits.
6. **TEI XPU-IPEX crushes llama.cpp for rerank** — 7–9× on 25-pair batches. Requires periodic restart (weekly) to reclaim VRAM growth.
7. **`--jinja` is mandatory for tool-calling reliability** — the built-in template handler doesn't emit Gemma 4's tool delimiters.
8. **XMX+oneDNN FA (llama.cpp b10068) is a modest win on dense-GQA models.** Initial cold-vs-warm comparison overstated it as "+42% throughput / -47% wall time" — methodology-matched isolated probe on Ornith 9B shows the real uplift is **~3% at 8K/12K, flat at short context**. Effect on Gemma 4 MoE is similar magnitude — the FA vec kernel path helps but not dramatically. **Lesson: never trust a first-look uplift claim that pairs a cold-start baseline against a warmed candidate.** Always re-probe under identical conditions.
9. **b10068 also carries a silent Q4_K get_rows correctness fix** — the older build had a subtle bug in Q4_K row gather that affected Ornith and MiniCPM5 decodes. No perceptible quality change post-swap, but it's closed regardless.
10. **MTP variants matter more than base model choice for A3B MoEs.** Qwen 3.6-35B-A3B base was 31 tok/s decode; same architecture with MTP head enabled (via the `-MTP-GGUF` sibling repo) jumps to 49 tok/s — **+58% purely from picking the right repo.** Always check for `-MTP-GGUF` variants of MoE candidates.
11. **b10068's XMX FA win doesn't scale to dense-27B.** Ornith dense-9B GQA got +42% cold prefill from b10068. Qwen 3.6-27B dense-27B got flat-to-slightly-negative. b10068's XMX FA optimises FA vec kernels — small models can afford the launch overhead, larger dense models are still bandwidth-bound.
12. **Co-residence budget dominates viability.** A 24 GiB isolated bench that beats Ornith is meaningless if it leaves 0 GiB for embed + rerank + TEI. Always subtract ~2.4 GiB of non-chat services before deciding if a candidate can actually ship. See the [VRAM co-residence budget](#vram-co-residence-budget) section.
13. **MTP acceptance is quant-sensitive** — same architecture, same base weights, same MTP head, but changing from Q4_K_XL → Q4_K_S drops MTP acceptance from 77.8% → 71%, and IQ4_XS drops it further to 60.8%. The drafter head's calibration against the target degrades faster than raw quant math would suggest. Meaningful lesson for anyone hoping "just quantise smaller" is a free move on MTP models.
14. **Smaller K-quant can be FASTER on prefill.** Counter-intuitive but measured: Q4_K_S beats Q4_K_XL on cold 12K prefill (820 vs 798 tok/s) and 5K prefill (985 vs 974). The smaller weights let more of the model stay in cache during prefill's memory-bound phase. Decode reverses this — larger K-quant wins because MTP acceptance recovers.
15. **Two SYCL processes on a single B60 contend severely.** Split-slot experiment 2026-07-22: running `llamacpp-sycl` (Ornith) and `llamacpp-categorise` (Gemma 4 E2B) concurrently on the same B60 drops both from isolated speed to ~30% (Ornith 52 → 16 tps, Gemma 88 → 12 tps). Verified by stopping the second container mid-session — the first immediately recovered from 17 to 55 tps decode. Combined throughput of the split (28 tps) is *worse* than a single process alternating (52 tps FIFO). Root cause is Level Zero context-switch overhead + shared kernel dispatch queue; not a llama.cpp-fixable issue. **Split-slot architecture on B60 requires a dedicated GPU per SYCL process** — future direction: add a second B60 and pin the categorise slot to `level_zero:1`.
16. **Google QAT Q4_0 beats post-hoc K-quants at 2-4B on B60.** Gemma 4 E2B QAT hit 88 tps decode vs Qwen3-4B and Agents-A1-4B post-training K-quants at 79 tps. QAT preserves output distribution better than post-training quants at small sizes, AND Q4_0 layout dispatches faster than Q4_K_M on Battlemage. Reverses at ≥26B where K-quants pull ahead again (already documented in finding #2).
17. **Reasoning-tuned models (Gemma 4, Agents-A1) route output to `reasoning_content` by default** — `content` is empty, breaking any structured-JSON workflow. Fix: pass `--reasoning off` to `llama-server`. Without it, 0/10 JSON parseability. With it, 10/10. Small (~5%) decode-speed cost.
18. **Prefill/decode asymmetry matters more than raw decode tps for model choice.** Same B60 (456 GB/s bandwidth), same workload shape (5K prompt + 200 gen), measured under real skill_server load:

    | Model | Prefill tps | Decode tps | Decode % of BW ceiling | Notes |
    |---|---|---|---|---|
    | Gemma 4 E2B QAT Q4_0 (3.35 GB) | **1,600** | 30 | 22% | fast prefill, no MTP |
    | Ornith 9B + MTP Q4_K_M (5.0 GB) | 1,000 | **50** | 55% | slower prefill, +2× MTP boost |

    The disparity comes from two things Ornith has and Gemma doesn't: (a) protoLabsAI's MTP drafter accelerates decode ~2× via speculative decoding at 68-80% acceptance; (b) K-quant super-blocks amortize dequantization better than Q4_0's simpler layout for the single-token latency of decode. Q4_0 wins prefill (large batched matmul), Q4_K_M wins decode.

    **Workload-shape total-time math** — the categorise/summarise/chat winner flips depending on output length:

    | Workload | Prompt | Gen | Gemma total | Ornith total | Winner |
    |---|---|---|---|---|---|
    | Categorise (short JSON) | 5K | 200 | 3.1 + 6.7 = 9.8s | 5.0 + 4.0 = 9.0s | ~tie |
    | Summarise | 5K | 500 | 3.1 + 16.7 = 19.8s | 5.0 + 10 = 15.0s | Ornith |
    | Chat/agent | 3K | 800 | 1.9 + 26.7 = 28.6s | 3.0 + 16 = 19.0s | Ornith |

    **Lesson: never pick a small model purely on prefill/isolated-decode headline numbers.** Compute total-time for the actual workload shape. For anything with >200 output tokens on this stack, a 9B + MTP beats a 2B without MTP even when the 2B is "faster" on both isolated axes. This is also why the split-slot experiment for categorise didn't help much — brain routes both categorise (Gemma-favorable shape) AND summarise (Ornith-favorable shape) to the same categorise endpoint. Model choice should follow workload shape, not the "small model for cheap tasks" instinct.

    **Track prefill/decode ratio in future benches**, not just headline decode tps. Add a column to the model comparison table with the workload-time calculation.

19. **b10215 delivers ~2× prefill via SYCL oneMKL GEMM XMX FA (#25025), invisible to short-prompt/decode-only benches.** Discovered 2026-08-01 during brain-eval Track 2 ingest arm — both Ornith 9B and Gemma 4 E2B independently hit ~3,000 tps prefill on real 2-5K token brain-ingest prompts vs historical ~1,600 tps on b10068. The pre-cutover synthetic decode bench (300-token `ignore_eos` generation, 60-token prompts) showed "no meaningful change" and would have missed this entirely — real workload is what surfaced the win. Not Gemma-specific; benefits every arch that uses standard attention, only above ~1K prompt tokens (below that, launch overhead dominates and the XMX GEMM path isn't reached). **Meta-lesson: post-upgrade validation needs to include production-shape prompts, not just decode microbenches.** Ties into #8 — the b10068 "+42% prefill" claim we retracted was cold-vs-warm methodology; the b10215 "+2× prefill" claim IS the real methodology-matched result under real workload.

20. **Isolated `/completion` probe vs real-workload prefix-cached wall-clock measures different things. Never mix them in a delta.** 2026-08-04 lesson: comparing Ornith 9B on b10215 vs b10256 initially showed a -40% "regression" on b10256 that was pure methodology artifact. The b10215 "baseline" number (3,000 tps) came from Track 2 brain-ingest wall-clock, where sequential prompts share prefix cache and effective prefill is amortized. The b10256 "regression" number (1,789 tps) came from a fresh `/completion` probe with `cache_prompt: false` and warmup — pure cold prefill on a single request. When re-run with matched methodology (both isolated `/completion` cold probes on the same 5K prompt), b10215 delivered 1,442 tps and b10256 delivered 1,789 tps — a **+24% real improvement**, not a regression. **Rule:** pick one methodology (isolated cold, warm-cached, real-workload aggregate, etc.) and hold it constant across every arm of an A/B; document which one you used inline with the number so future comparisons don't get confused. This applies retroactively to finding #19 — the "~2× prefill" claim there is real-workload-with-cache; the isolated-probe delta for b10068→b10215 hasn't been re-measured cleanly.

22. **TEI's advantage over llama.cpp SYCL is architecture-specific, not universal for encoder-only models.** Discovered 2026-08-06 during embed-runtime A/B benches (finding-in-response-to finding #6 which called out TEI's 7-9× rerank win). **BERT-family cross-encoders** (bge-reranker-v2-m3, older embed models): TEI wins big (7-9× rerank, 19× on Qwen3-Embedding-0.6B). **Gemma3-encoder** (EmbeddingGemma-300M): llama.cpp SYCL wins single-embed by 1.85× (23,208 vs 12,500 tps), TEI wins batch throughput by 24% (313.7 vs 253 emb/s). Reason: TEI's fused encoder-attention kernel is very well tuned for BERT-family architectures; llama.cpp SYCL happens to have excellent Gemma-family kernels (the same code path that gives you all your Gemma 4 chat wins). **Rule:** always A/B the same model on both runtimes before assuming an "encoder = TEI" advantage generalizes. Also TEI's win regime is batch throughput; llama.cpp SYCL's win regime is single-request latency. **See [`models/tested/2026-08-06-new-candidates-sweep.md`](models/tested/2026-08-06-new-candidates-sweep.md) for the full A/B matrix.**

23. **B60 draws ~100W actual vs 220W TDP under sustained LLM load — compute engines pinned but half the die is idle.** Measured 2026-08-01 during brain-eval Track 2 arm 3 (Gemma 4 E4B + Google MTP, 2-5K token prompts, structured JSON output) via `xpu-smi stats`:

    | Metric | Value | Interpretation |
    |---|---|---|
    | Power | **97-98W** of 220W cap | 44% of design TDP |
    | GPU frequency | **2,400 MHz** (RP0 boost, max) | Card is at max clock, not throttled |
    | Core temp | 53°C | Cold — no thermal headroom concern |
    | Memory temp | 54°C | Cold |
    | Compute engine util | **99.99%** | Pinned — every XMX + Xe-core doing math |
    | Copy engine util | 90% | Active |
    | Media engine util | 0% | Idle (no video work) |
    | 3D render util | 0% | Idle (no graphics) |
    | Memory BW util | 20% | 91 of 456 GB/s |

    The 55% TDP headroom is unused because the die area for **media encoders, 3D pipeline, ray tracing units, and display controllers** sits idle under LLM inference — those transistors aren't LLM-relevant. What matters (XMX + Xe-cores) is already pinned at 99.99%.

    **The interesting number is 20% memory BW + 100% compute** — the current SYCL kernels are compute-limited, not memory-limited. That means future llama.cpp kernel improvements (like the #25025 oneMKL win for prefill) can still lift decode without any hardware change. Theoretical decode ceiling for Ornith 9B Q4_K_M on 456 GB/s: 91 tps; we're at 56 tps (61% of memory ceiling), which means the kernel path has ~50% headroom to grow into.

    **Practical implication for other B60 users:** B60 runs cool and quiet under LLM load (thermal and acoustic headroom is not the constraint) and cards can be packed tighter than TDP suggests for multi-card setups. Don't undersize your PSU based on TDP × N; sizing based on measured LLM load × N + spike margin is the more accurate approach. Also don't confuse "GPU util 21%" reported by xpu-smi with under-utilization — that metric averages the 100%-pinned compute engine with the 0%-idle media/3D/display engines, which is misleading for LLM workloads. Look at `ENGINE_GROUP_COMPUTE_ALL_UTILIZATION` (99.99%) instead.

## Journey summary

| Stage | Decode | Cold 12K prefill | Warm follow-up |
|---|---|---|---|
| LM Studio Vulkan (start) | 33.6 tok/s | 127s @ 97 tok/s | 36s (cache broken) |
| SYCL out-of-box | 38.4 tok/s | 37s @ 388 tok/s | 36s |
| + FA on, `-ub 2048` (Config D) | 38.6 tok/s | 30s @ 477 tok/s | 0.66s |
| + `GGML_SYCL_F16=ON` rebuild | 40.1 tok/s | 24s @ 602 tok/s | 0.61s |
| + Q4_K_M post-training | 44.1 tok/s | 22.8s @ 632 tok/s | 0.55s |
| + Config C + MTP (bare-metal, b9948) | 50.0 tok/s | ~13.7s @ ~938 tok/s | ~0.55s |
| + b10068 rebuild (XMX+oneDNN FA, Ornith prod) | 51.8 tok/s | ~13.3s @ 969 tok/s (+3%) | ~0.55s |
| + b10215 rebuild (oneMKL GEMM XMX FA #25025, Ornith prod, real workload) | 56 tok/s | ~4.3s @ ~3,000 tok/s (real workload w/ prefix cache) — or ~8.3s @ 1,442 tok/s isolated `/completion` | ~0.55s |
| + **b10256 rebuild** (SYCL concat parallelization #25852 + MTP verification improvements, cutover 2026-08-04) | **64 tok/s** (+15%) | **~6.7s @ 1,789 tok/s isolated `/completion` (+24% vs b10215 same methodology)** | ~0.55s |

**Overall vs LM Studio start (isolated `/completion` methodology, 5K prompt cold):** LM Studio Vulkan 97 tps prefill → b10256 SYCL 1,789 tps prefill = **~18× cold prefill**. Warm-path cache hits: 36s → 0.55s = **~65×**. Decode: 33.6 → 64 tok/s = **+90%**. If measured under real-workload with prefix cache reuse (Track 2 style), b10215 hit ~3,000 tps — b10256 hasn't been re-measured under that methodology yet, expect similar or better. See finding #20 for the methodology-vs-methodology explanation and finding #19 for the b10215 workload-level jump.

(b9948 12K prefill row revised 2026-07-20 from isolated probe — earlier "22.8s @ 632" figure was cold-start against a bandwidth-contended stack; methodology-matched probe gives 14.2s @ 938 tok/s. This means most of the prefill journey landed with SYCL + FA + `-ub 2048` + F16 rebuild + Q4_K, then b10215's XMX GEMM FA more than doubled it again under real load.)

## Repo layout

```
├── README.md                       ← this file
├── models/
│   ├── production/                 ← currently running
│   │   ├── ornith-1.0-9b.md
│   │   ├── gemma-4-26b-a4b.md      (reasoning fallback)
│   │   ├── embeddinggemma-300m.md
│   │   └── bge-reranker-v2-m3.md
│   ├── retired/                    ← no longer receiving traffic
│   │   └── qwen3-4b-instruct-2507.md
│   └── tested/                     ← benched, not adopted
│       ├── qwen3.6-35b-a3b.md
│       ├── qwen3.6-35b-a3b-mtp.md         (2026-07-19 retest with MTP)
│       ├── qwen3.6-35b-a3b-claude-distilled.md
│       ├── qwen3.6-35b-a3b-kimi-distilled.md
│       ├── minicpm5-1b.md
│       ├── qwen3-coder-30b-a3b.md
│       ├── devstral-small-2-24b.md
│       ├── qwen3.6-27b.md
│       ├── mistral-small-3.1-24b.md
│       ├── gemma-4-12b-qat.md
│       ├── gemma-4-e4b.md
│       ├── gemma-3-4b.md
│       └── qwen2.5-coder-14b-awq.md
├── configs/images/                ← Docker image build docs + patches
│   ├── llama.cpp-sycl-f16/         (build.sh, README.md, flags used)
│   └── tei-xpu-ipex-nomemleak/     (VRAM leak patch)
└── configs/launchers/              ← docker run scripts (mirror of /data/llm/launch/ on llm.local)
    ├── start-llamacpp-sycl-ornith.sh
    ├── start-llamacpp-sycl-gemma4-mtp.sh
    ├── start-llamacpp-embed.sh
    ├── start-llamacpp-rerank.sh
    ├── start-tei-rerank.sh
    ├── start-llamacpp-minicpm5.sh          (candidate on :8009)
    └── start-llamacpp-sycl-categorise.sh   (retired)
```

Each per-model file includes: HF link, specs, benchmark numbers, launcher link (where applicable), and verdict.
