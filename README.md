# Local LLM Benchmarks (B60 Pro)

Local LLM benchmarks & configs for **Intel Arc Pro B60 (24 GB, Battlemage / Xe2)** on bare-metal Ubuntu 26.04.

All numbers below are measured on the same physical card. The stack has shifted over time — vLLM-XPU → LM Studio Vulkan → llama.cpp Vulkan → llama.cpp SYCL (current) — but the hardware is constant. Unless a row says otherwise, benchmarks were taken on [`llama.cpp:sycl-f16`](configs/images/llama.cpp-sycl-f16/README.md) (custom build **b10068**, `GGML_SYCL_F16=ON`, oneAPI 2025.3.3 base, `-DCMAKE_BUILD_TYPE=Release`, includes XMX+oneDNN FA + `fattn_vec_nthreads=256` Battlemage tuning + fused top-k MoE + Q4_K get_rows correctness fix) with `-fa on`, KV Q8, `-ub 2048 -b 2048`, `--parallel 1`, `--jinja`.

## Current production stack

| Port | Container | Model | Purpose |
|---|---|---|---|
| 8002 | `llamacpp-sycl` | [Ornith 1.0 9B Q4_K_M + MTP drafter](models/production/ornith-1.0-9b.md) | chat + categorise + pi.dev agent (dual-role) |
| 8004 | `llamacpp-embed` | [EmbeddingGemma-300M QAT Q8_0](models/production/embeddinggemma-300m.md) | brain embeddings |
| 8007 | `llamacpp-rerank` | [bge-reranker-v2-m3 Q8_0](models/production/bge-reranker-v2-m3.md) | rerank fallback (llama.cpp path) |
| 8008 | `tei-rerank` | [bge-reranker-v2-m3 fp16](models/production/bge-reranker-v2-m3.md) | rerank prod (TEI XPU-IPEX, 7–9× faster) |

**Reasoning fallback:** [Gemma 4 26B-A4B Q4_K_M + MTP](models/production/gemma-4-26b-a4b.md) — launcher on disk, not running by default.

**Retired:** [`llamacpp-categorise` (Qwen3-4B on :8006)](models/retired/qwen3-4b-instruct-2507.md) — brain env vars now point to `:8002`; the container is still up with 6+ days uptime but receives no traffic. Candidate for shutdown.

## Chat / instruct benchmarks

Decode = steady-state single-stream tok/s. Prefill measured at the context noted. VRAM is peak observed with the config running.

| Model | Quant | Total / active params | Decode tok/s | Prefill tok/s | VRAM | Status |
|---|---|---|---|---|---|---|
| [MiniCPM5-1B](models/tested/minicpm5-1b.md) | Q4_K_M | 1.08B dense | **~187** | **4,642 @ 2K** | ~3 GB | tested 2026-07-19; fastest tested; JSON fence issue on categorise |
| [**Qwen 3.6-35B-A3B-MTP**](models/tested/qwen3.6-35b-a3b-mtp.md) ⭐ | UD-Q4_K_XL | 35.5B / 3B | **49.0** | 798 @ 12K cold / 974 @ 5K | **24.4 GB (won't fit prod)** | Ornith-parity speed at 4× params; need smaller quant for co-residence |
| [Qwen 3.6-35B-A3B-MTP](models/tested/qwen3.6-35b-a3b-mtp.md) | UD-Q4_K_S | 35.5B / 3B | 37.7 | 820 @ 12K cold / 985 @ 5K | 24.1 GB (still fails co-res) | tested 2026-07-19; only saves 0.3 GB vs Q4_K_XL; prefill wins but decode -23% due to MTP acceptance drop |
| [Qwen 3.6-35B-A3B-MTP](models/tested/qwen3.6-35b-a3b-mtp.md) | UD-IQ4_XS | 35.5B / 3B | 31.6 | 776 @ 12K / 918 @ 5K | 21.1 GB (fits prod with 0.5 GB headroom) | tested 2026-07-19; -36% decode + -22pp MTP acceptance vs Q4_K_XL; IQ quants underperform K quants on B60 |
| [Qwen 3.6-35B-A3B Claude 4.7 Opus Distilled](models/tested/qwen3.6-35b-a3b-claude-distilled.md) | APEX-MTP Compact | 35.5B / 3B | 36.9 | 763 @ 12K / 887 @ 5K | 19.4 GB (fits prod) | tight-reasoning distillation; only 35B-A3B that co-resides cleanly |
| [Qwen 3.6-35B-A3B Kimi K2.6 Distilled](models/tested/qwen3.6-35b-a3b-kimi-distilled.md) | IQ4_XS | 35.5B / 3B | 30.6 | **904 @ 12K cold** ⭐ | 21.4 GB (0.2 GB co-res headroom) | fastest cold prefill benched; verbose reasoning; no MTP |
| [Qwen 3.6-35B-A3B (base)](models/tested/qwen3.6-35b-a3b.md) | UD-Q3_K_M | 34.7B / 3B | 31.1 | 823 @ 2K | 20.0 GB | superseded by MTP variant above |
| [Gemma 4 26B-A4B (it) Q4_K_M + MTP](models/production/gemma-4-26b-a4b.md) | Q4_K_M | 26B / 4B | **53.0** (peak) | **971 @ 5K / 650 @ 12K cold** | 22.9 GB | reasoning fallback; b10068 refresh |
| Gemma 4 26B-A4B (it) Q4_K_M (base) | Q4_K_M | 26B / 4B | 44.1 | 632 @ 12K | 20.9 GB | original locked prod (pre-MTP) |
| Gemma 4 26B-A4B QAT | Q4_0 | 26B / 4B | 40.1 | 602 @ 12K | 18.2 GB | beaten by K-quant on Battlemage |
| **[Ornith 1.0 9B](models/production/ornith-1.0-9b.md)** | Q4_K_M | 9B dense | **~50** (65–70 w/MTP est.) | 1,310 @ 6.7K | 4.1 GB | **production chat** |
| [Qwen3-Coder-30B-A3B](models/tested/qwen3-coder-30b-a3b.md) | UD-Q4_K_XL | 30B / 3B | ~38 | ~700 | ~20 GB | tested; capability too poor for pi.dev |
| [Devstral Small 2 24B](models/tested/devstral-small-2-24b.md) | UD-Q4_K_XL | 24B dense | ~18 | ~340 | ~15 GB | tested; dense penalty visible |
| [Qwen3.6-27B](models/tested/qwen3.6-27b.md) | Q4_K_XL | 27B dense | ~22 | ~380 | ~17 GB | tested; bartowski build |
| [Mistral-Small-3.1-24B](models/tested/mistral-small-3.1-24b.md) | Q4_K_M | 24B dense | ~19 | ~350 | ~15 GB | tested (Vulkan era) |
| [Gemma 4 12B (QAT)](models/tested/gemma-4-12b-qat.md) | Q4_0 | 12B dense | 19.7 | 167 @ 1K | ~9 GB | tested; 12B dense < 26B-4B MoE |
| [Gemma 4 E4B (QAT)](models/tested/gemma-4-e4b.md) | Q4_0 | ~4B | 73.9 | 376 | ~3 GB | tested; QAT wins decode at small size |
| [Gemma 4 E4B](models/tested/gemma-4-e4b.md) | Q4_K_M | ~4B | 68.3 | 466 | ~3 GB | tested; QAT-vs-K-quant reversal @ 4B |
| [Gemma 3 4B](models/tested/gemma-3-4b.md) | Q4_K_M | 4B | ~78 | — | ~3 GB | tested; system-role template issue |
| [Qwen3-4B-Instruct-2507](models/retired/qwen3-4b-instruct-2507.md) | Q4_K_M | 4B | ~94 (60s under 4-way) | 766 aggregate | ~1 GB | **retired** (was categorise prod) |
| [Qwen2.5-Coder-14B AWQ (vLLM-XPU)](models/tested/qwen2.5-coder-14b-awq.md) | AWQ int4 | 14B | 22.9 peak / 13–15 typical | **1,891** peak | ~11 GB | retired; cross-stack reference |
| Qwen 3.6-27B (LM Studio Vulkan) | Q4_K | 27B | 33.6 | 97 @ 12K cold | 17.6 GB | historical Vulkan baseline |

## Embed / rerank benchmarks

| Model | Server | Throughput / latency | VRAM | Notes |
|---|---|---|---|---|
| [EmbeddingGemma-300M](models/production/embeddinggemma-300m.md) | llama.cpp SYCL :8004 | ~5,000 tok/s, batch 2048 | ~2 GB | prod embed |
| [bge-reranker-v2-m3](models/production/bge-reranker-v2-m3.md) | **TEI XPU-IPEX :8008** | **109 ms / 25 pairs** | 2.5 GB | prod (7–9× faster than llama.cpp) |
| [bge-reranker-v2-m3](models/production/bge-reranker-v2-m3.md) | llama.cpp SYCL :8007 | 800–1,000 ms / 25 pairs | ~4 GB | fallback |

## VRAM co-residence budget

Isolated bench numbers are misleading — production has to fit all services simultaneously. **Non-chat steady-state stack**:

| Container | VRAM steady |
|---|---|
| llamacpp-embed (EmbeddingGemma-300M) | 0.5 GiB |
| llamacpp-rerank (bge-reranker fallback) | 0.5 GiB idle (3-4 active) |
| tei-rerank (patched image) | 1.4 GiB |
| **Non-chat total** | **~2.4 GiB** |
| **Available for chat model** | **~21.6 GiB / 24 GiB** |

Any candidate that reports isolated VRAM > 21.6 GiB either won't fit alongside prod, or needs a smaller quant / context / rerank-eviction trade-off. See individual model pages for per-candidate co-residence analysis.

## Key findings

1. **MoE beats dense on Battlemage.** 26B-4B-active decodes ~2× faster than 12B dense at similar quality.
2. **Post-training K-quant beats QAT Q4_0 at ≥26B.** Reverses at 4B — the winner is model-size-dependent.
3. **MTP drafters are worth +5–15%** when a purpose-built head exists (Gemma 4 official, Ornith community).
4. **`-ub 2048` is the SYCL sweet spot.** 4096 regresses on this card; monotonic climb from 16 → 2048 then plateau.
5. **`-fa on` is mandatory** — turns 36s "warm" re-prefills into 0.55s cache hits.
6. **TEI XPU-IPEX crushes llama.cpp for rerank** — 7–9× on 25-pair batches. Requires periodic restart (weekly) to reclaim VRAM growth.
7. **`--jinja` is mandatory for tool-calling reliability** — the built-in template handler doesn't emit Gemma 4's tool delimiters.
8. **XMX+oneDNN FA (llama.cpp b10068) is a big win on dense-GQA models** — Ornith 9B cold 12K prefill dropped from 22.8s to 12.1s (-47% wall time, +42% throughput). Effect on Gemma 4 MoE is much smaller (~3-17% depending on prompt) — the MoE path was already well-optimized.
9. **b10068 also carries a silent Q4_K get_rows correctness fix** — the older build had a subtle bug in Q4_K row gather that affected Ornith and MiniCPM5 decodes. No perceptible quality change post-swap, but it's closed regardless.
10. **MTP variants matter more than base model choice for A3B MoEs.** Qwen 3.6-35B-A3B base was 31 tok/s decode; same architecture with MTP head enabled (via the `-MTP-GGUF` sibling repo) jumps to 49 tok/s — **+58% purely from picking the right repo.** Always check for `-MTP-GGUF` variants of MoE candidates.
11. **b10068's XMX FA win doesn't scale to dense-27B.** Ornith dense-9B GQA got +42% cold prefill from b10068. Qwen 3.6-27B dense-27B got flat-to-slightly-negative. b10068's XMX FA optimises FA vec kernels — small models can afford the launch overhead, larger dense models are still bandwidth-bound.
12. **Co-residence budget dominates viability.** A 24 GiB isolated bench that beats Ornith is meaningless if it leaves 0 GiB for embed + rerank + TEI. Always subtract ~2.4 GiB of non-chat services before deciding if a candidate can actually ship. See the [VRAM co-residence budget](#vram-co-residence-budget) section.
13. **MTP acceptance is quant-sensitive** — same architecture, same base weights, same MTP head, but changing from Q4_K_XL → Q4_K_S drops MTP acceptance from 77.8% → 71%, and IQ4_XS drops it further to 60.8%. The drafter head's calibration against the target degrades faster than raw quant math would suggest. Meaningful lesson for anyone hoping "just quantise smaller" is a free move on MTP models.
14. **Smaller K-quant can be FASTER on prefill.** Counter-intuitive but measured: Q4_K_S beats Q4_K_XL on cold 12K prefill (820 vs 798 tok/s) and 5K prefill (985 vs 974). The smaller weights let more of the model stay in cache during prefill's memory-bound phase. Decode reverses this — larger K-quant wins because MTP acceptance recovers.

## Journey summary

| Stage | Decode | Cold 12K prefill | Warm follow-up |
|---|---|---|---|
| LM Studio Vulkan (start) | 33.6 tok/s | 127s @ 97 tok/s | 36s (cache broken) |
| SYCL out-of-box | 38.4 tok/s | 37s @ 388 tok/s | 36s |
| + FA on, `-ub 2048` (Config D) | 38.6 tok/s | 30s @ 477 tok/s | 0.66s |
| + `GGML_SYCL_F16=ON` rebuild | 40.1 tok/s | 24s @ 602 tok/s | 0.61s |
| + Q4_K_M post-training | 44.1 tok/s | 22.8s @ 632 tok/s | 0.55s |
| + Config C + MTP (bare-metal, b9948) | 50.0 tok/s | ~21.5s @ ~655 tok/s | ~0.55s |
| + b10068 rebuild (XMX+oneDNN FA, Ornith prod) | **51.8 tok/s** | **12.1s @ 896 tok/s** | ~0.55s |

**Overall vs LM Studio start: 10.5× cold prefill, ~65× warm-path, +54% decode.**

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
