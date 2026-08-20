# Local LLM Benchmarks (B60 Pro)

Local LLM benchmarks & configs for **2× Intel Arc Pro B60 (24 GB each, Battlemage / Xe2)** on bare-metal Ubuntu 26.04, 64 GiB RAM.

All numbers below are measured on the same physical hardware. Unless a row says otherwise, benchmarks were taken on [`llama.cpp:sycl-f16`](configs/images/llama.cpp-sycl-f16/README.md) at the current build tag.

**Current llama.cpp build:** `b10433` (commit `9b05354ec`, cutover 2026-08-14). Rollback tags `sycl-f16-b10256-safe` and `sycl-f16-b10215-safe` preserved on disk.

**Build history + per-release impact tables →** [`docs/build-history.md`](docs/build-history.md)
**Key findings (numbered #1-#24) →** [`docs/findings.md`](docs/findings.md)

## Current production stack

Traefik consolidates all endpoints under `https://llm.levirge.com/v1/*` (path-based routing, internal LAN only, split-DNS to manager.local Traefik). LB round-robins between paired card-1 / card-2 backends for each service. Both cards mirror the full 4-service stack (Ornith chat + embed + rerank + E2B categorise).

| Port | Container | Card | Model | Purpose |
|---|---|---|---|---|
| 8002 | `llamacpp-sycl` | 1 | [Ornith 1.5 9B + MTP](models/production/ornith-1.5-9b.md) ⭐ | chat + pi.dev agent (mirror pair) |
| 8010 | `llamacpp-sycl-c2` | 2 | Ornith 1.5 9B + MTP (mirror) | chat + pi.dev agent |
| 8004 | `llamacpp-embed` | 1 | [EmbeddingGemma-300M QAT Q8_0](models/production/embeddinggemma-300m.md) | brain embeddings |
| 8012 | `llamacpp-embed-c2` | 2 | EmbeddingGemma-300M (mirror) | embeddings |
| 8008 | `tei-rerank` | 1 | [bge-reranker-v2-m3 fp16](models/production/bge-reranker-v2-m3.md) | rerank |
| 8013 | `tei-rerank-c2` | 2 | bge-reranker-v2-m3 (mirror) | rerank |
| 8006 | `llamacpp-categorise-c1` | 1 | [Gemma 4 E2B QAT + Google MTP](models/production/gemma-4-e2b-categorise.md) | categorise (mirror) |
| 8009 | `llamacpp-categorise` | 2 | Gemma 4 E2B QAT + Google MTP ⭐ | categorise (primary, watchdog-monitored) |

**Reasoning fallback (not running by default):** [Gemma 4 26B-A4B QAT + Google MTP](models/production/gemma-4-26b-a4b.md) — 62.8 tok/s decode, 97.2% MTP acceptance on b10433.

**Traefik routes:**
- `/v1/completions` and `/v1/chat/completions` → Ornith pair (:8002 + :8010)
- `/v1/embeddings` → embed pair (:8004 + :8012)
- `/v1/rerank` → TEI pair (:8008 + :8013), path rewritten to `/rerank`
- `/v1/categorise` → E2B pair (:8006 + :8009), path rewritten to `/v1/chat/completions`, inFlightReq=2
- `/` → monitor dashboard :8005 (HTTPS only)

## Recent stack changes

- **2026-08-21** — Ornith 1.0-9B → **1.5-9B cutover on both cards** (+23% decode, MTP acc 76% → 82.7%). Sampling aligned to Ornith 1.5 coding recipe. [Bench and rationale.](models/production/ornith-1.5-9b.md)
- **2026-08-21** — Gemma 4 26B-A4B QAT re-bench: **QAT decode regression from 2026-08-14 is gone.** 62.8 tps + 97.2% MTP acceptance on b10433 — QAT now leads Q4_K_M by a wide margin, reverses finding #2 for this build. [Details in gemma-4-26b-a4b.md.](models/production/gemma-4-26b-a4b.md)
- **2026-08-21** — Ornith 1.5-35B-A3B benched on both bartowski IQ4_XS and mudler APEX-MTP-Compact. Both fail on MTP acceptance under compression (32.5% and 26.2% respectively vs 82.7% for 1.5-9B). See finding #24. Single-card 35B not viable on this stack. [Bench 1](models/tested/ornith-1.5-35b-a3b-single-card.md) · [Bench 2](models/tested/ornith-1.5-35b-a3b-apex-mtp.md).
- **2026-08-22** — RAM upgrade 30 → 64 GiB; **dual-load mirror pattern live** on both cards (Ornith + embed + rerank + E2B all mirrored). Traefik LB round-robin.
- **2026-08-15** — 2nd B60 install; Gemma 4 E2B moved to card 2 `:8009` (dedicated categorise slot, physical GPU isolation). Delivered 4.7× wall-clock vs Ornith on categorise workload. See [`models/production/gemma-4-e2b-categorise.md`](models/production/gemma-4-e2b-categorise.md).
- **2026-08-14** — llama.cpp `b10256` → **`b10433`** cutover (all services). See [`docs/build-history.md`](docs/build-history.md) for per-model deltas.

**Next architectural step (task #144):** B580 12GB new-host migration — offload embed + rerank + E2B to a fresh B580-based node, freeing both B60s for **Ornith 1.5-35B-A3B tensor-split** (task #142) or vLLM XPU migration.

## Chat / instruct benchmarks

Decode = steady-state single-stream tok/s. Prefill measured at the context noted. VRAM is peak observed with the config running. Same-build rows are directly comparable; historical rows kept for reference.

### Ornith family (Qwen 3.5 fine-tunes)

| Model | Quant | Params | Decode | Prefill | VRAM | MTP acc | Status |
|---|---|---|---|---|---|---|---|
| [**Ornith 1.5-9B + MTP**](models/production/ornith-1.5-9b.md) ⭐ (b10433) | Q4_K_M + protoLabsAI Q8_0 head | 9B dense | **65.44** | **2,040 @ 4K** | 20.7 GiB | **82.7%** | **production chat (both cards)** — cutover 2026-08-21 |
| [Ornith 1.0-9B + MTP](models/tested/ornith-1.0-9b.md) (b10433) | Q4_K_M + Q8_0 head | 9B dense | 67.9 (100% acc reported) | 1,911 @ 5K (isolated) / ~3,000 real | 10.9 GiB | 68-76% real workload | superseded 2026-08-21 |
| [Ornith 1.5-35B-A3B (bartowski)](models/tested/ornith-1.5-35b-a3b-single-card.md) (b10433) | IQ4_XS + embedded MTP Q4_0 | 35B / 3B | 32.6 | 1,360 @ 4K | 22.5 GiB | 32.5% | **not viable single-card** — MTP head at Q4_0 crushes acceptance |
| [Ornith 1.5-35B-A3B (mudler APEX)](models/tested/ornith-1.5-35b-a3b-apex-mtp.md) (b10433) | APEX-MTP-Compact 17.4 GB + embedded MTP Q8_0 | 35B / 3B | 25.6 | 1,211 @ 4K | 21.0 GiB | 26.2% | **worse than bartowski** — APEX compression breaks target/drafter alignment (finding #24) |
| [Ornith 1.0-35B MTP APEX](models/tested/ornith-1.0-35b-mtp-apex.md) | APEX I-Compact (IQ) | 35B / 3B | 35.4 | 816 @ 5K / 802 @ 12K | ~19 GB | — | tested 2026-07-22; VLM-capable |

### Gemma 4 family (Google, all with Google's official MTP drafters)

| Model | Quant | Params | Decode | Prefill | VRAM | MTP acc | Status |
|---|---|---|---|---|---|---|---|
| [**Gemma 4 26B-A4B QAT + MTP**](models/production/gemma-4-26b-a4b.md) ⭐ (b10433 re-bench 2026-08-21) | QAT Q4_0 + Google MTP Q8_0 | 26B / 4B | **62.84** | **1,592 @ 4K** | 19.9 GiB @ 131K | **97.2%** | **reasoning fallback** — QAT regression cleared; 2.90 accepted/draft (highest ever) |
| [Gemma 4 26B-A4B Q4_K_M + MTP](models/production/gemma-4-26b-a4b.md) (b10433) | Q4_K_M + Google MTP Q8_0 | 26B / 4B | 47.7 | 1,473 @ 5K | 19.7 GiB | 96.1% | prior bench; QAT now preferred after 2026-08-21 |
| [**Gemma 4 E2B + Google MTP**](models/production/gemma-4-e2b-categorise.md) ⭐ | QAT Q4_0 + BF16 drafter | 2B + drafter | 138.8 isolated / 71.5 prod | 3,681 @ 2K isolated / 3,040 prod | 4.5 GiB | 67.8% isolated / 25% prod | **production categorise slot** on card 2 `:8009`; 4.7× wall-clock vs Ornith on categorise |
| [Gemma 4 E4B + Google MTP](models/tested/gemma-4-e4b.md) (b10215) | QAT Q4_0 + BF16 drafter | 4B + drafter | 114.1 | 2,319 @ 2K | 7 GiB | 66.7% | benched 2026-07-31 |
| [Gemma 4 12B + Google MTP](models/tested/gemma-4-12b-qat.md) (b10215) | QAT Q4_0 + BF16 drafter | 12B dense + drafter | 70.4 | 1,053 @ 2K | 8.5 GiB | 69.7% | benched 2026-07-31; QAT beats dense despite finding #2 |
| Gemma 4 E4B (no MTP baseline) | Q4_K_M | ~4B | 68.3 (b10068) | 466 | ~3 GB | — | K-quant vs QAT reversal @ 4B (finding #2) |
| Gemma 4 26B-A4B (it) baseline | Q4_K_M | 26B / 4B | 44.1 (b10068) | 632 @ 12K | 20.9 GB | — | original locked prod (pre-MTP) |
| [Gemma 3 4B](models/tested/gemma-3-4b.md) | Q4_K_M | 4B | ~78 | — | ~3 GB | — | system-role template issue |

### Qwen 3.x family

| Model | Quant | Params | Decode | Prefill | VRAM | MTP acc | Status |
|---|---|---|---|---|---|---|---|
| [Qwen 3.6-35B-A3B-MTP](models/tested/qwen3.6-35b-a3b-mtp.md) ⭐ | UD-Q4_K_XL | 35.5B / 3B | **49.0** | 798 @ 12K / 974 @ 5K | **24.4 GB (won't fit prod)** | 77.8% | Ornith-parity speed at 4× params; needs smaller quant for co-res |
| Qwen 3.6-35B-A3B-MTP | UD-Q4_K_S | 35.5B / 3B | 37.7 | 820 @ 12K / 985 @ 5K | 24.1 GB (fails co-res) | 71% | -23% decode from MTP acceptance drop (finding #13) |
| Qwen 3.6-35B-A3B-MTP | UD-IQ4_XS | 35.5B / 3B | 31.6 | 776 @ 12K / 918 @ 5K | 21.1 GB (fits, 0.5 GB) | 60.8% | -36% decode; IQ quants underperform K quants on B60 |
| [Qwen 3.6-35B-A3B Claude distilled](models/tested/qwen3.6-35b-a3b-claude-distilled.md) | APEX-MTP Compact | 35.5B / 3B | 36.9 | 763 @ 12K / 887 @ 5K | 19.4 GB (fits prod) | — | tight-reasoning distillation; only 35B-A3B that co-res cleanly |
| [Qwen 3.6-35B-A3B Kimi distilled](models/tested/qwen3.6-35b-a3b-kimi-distilled.md) | IQ4_XS | 35.5B / 3B | 30.6 | **904 @ 12K cold** ⭐ | 21.4 GB | — | fastest cold prefill benched; verbose reasoning; no MTP |
| [Qwen 3.6-35B-A3B base](models/tested/qwen3.6-35b-a3b.md) | UD-Q3_K_M | 34.7B / 3B | 31.1 | 823 @ 2K | 20.0 GB | — | superseded by MTP variant |
| [Qwen 3.8-27B + native MTP + vision](models/tested/qwen-3.8-27b.md) (tested, b10433) | Q4_K_M + Q4_0 MTP + Q8_0 mmproj | 27B dense hybrid (48 SSM + 16 attn) | 23.0 | 333.5 | 22.3 GiB | 57.9% | **parked** 2026-08-15; ~½ Gemma 4 26B-A4B; revisit when SYCL SSM gets XMX GEMM |
| [Qwen 3.6-27B](models/tested/qwen3.6-27b.md) | Q4_K_XL | 27B dense | ~22 | ~380 | ~17 GB | — | tested; bartowski build |
| [Qwen3-Coder-30B-A3B](models/tested/qwen3-coder-30b-a3b.md) | UD-Q4_K_XL | 30B / 3B | ~38 | ~700 | ~20 GB | — | tested; capability too poor for pi.dev |
| [Qwen2.5-Coder-14B AWQ](models/tested/qwen2.5-coder-14b-awq.md) | AWQ int4 (vLLM-XPU) | 14B | 22.9 peak / 13-15 typical | **1,891 peak** | ~11 GB | — | retired; cross-stack reference |
| Qwen 3.6-27B (LM Studio Vulkan) | Q4_K | 27B | 33.6 | 97 @ 12K cold | 17.6 GB | — | historical Vulkan baseline |
| [Qwen3-4B-Instruct-2507](models/retired/qwen3-4b-instruct-2507.md) | Q4_K_M | 4B | ~94 (60s under 4-way) | 766 aggregate | ~1 GB | — | **retired** (was categorise prod) |

### Others (Meta, Poolside, OpenAI, Mistral, MiniCPM, LFM)

| Model | Quant | Params | Decode | Prefill | VRAM | Notes |
|---|---|---|---|---|---|---|
| [**Muse Glimmer-30B + DFlash**](models/tested/muse-glimmer-30b.md) | K-Quant-17GB (Meta official) | 29.6B dense + 1.8B ViT-G/14 | 25.3 (100% DFlash acc) | 682 @ 5K | 21.9 GiB | Meta 2026-08 drop; multimodal; DFlash drafter; **bandwidth-bound at ~24 tps ceiling on B60**; only vision-capable option in tested lineup |
| [Laguna XS-2.1](models/tested/2026-08-06-new-candidates-sweep.md#laguna-xs2-poolside-33b-a3b-moe) | Q4_K_M | 33B / 3B | 29.5 | 1,213 @ 5K | 22.1 GiB | MoE + SWA; DFlash drafter needs Poolside fork; deferred |
| [gpt-oss-20b](models/tested/2026-08-06-new-candidates-sweep.md#gpt-oss-20b-openai) | Q4_K_M (MXFP4 native) | 20.9B / ~2.6B active | 25.1 | 1,265 @ 5K | 13.4 GiB | no MTP; potential Ornith alternative pending qualitative bake-off |
| [Devstral Small 2 24B](models/tested/devstral-small-2-24b.md) | UD-Q4_K_XL | 24B dense | ~18 | ~340 | ~15 GB | dense penalty visible |
| [Mistral-Small-3.1-24B](models/tested/mistral-small-3.1-24b.md) | Q4_K_M | 24B dense | ~19 | ~350 | ~15 GB | tested Vulkan era |
| [MiniCPM5-1B](models/tested/minicpm5-1b.md) | Q4_K_M | 1.08B dense | **~187** | **4,642 @ 2K** | ~3 GB | fastest tested; JSON fence issue on categorise |

## Embed / rerank benchmarks

| Model | Server | Throughput / latency | VRAM | Notes |
|---|---|---|---|---|
| [**EmbeddingGemma-300M**](models/production/embeddinggemma-300m.md) ⭐ | llama.cpp SYCL :8004 + :8012 | **23,208 tok/s** @ 1800 tok, 253 emb/s batch 64 | 0.5 GB | **prod embed** (dim 768, ~57 RTEB); wins single-embed by 1.85× vs TEI |
| EmbeddingGemma-300M (A/B) | TEI XPU-IPEX | 12,500 tok/s @ 1800 tok, **313.7 emb/s** batch 64 | ~1 GB | TEI wins batch throughput but loses single-embed (finding #22) |
| [Qwen3-Embedding-0.6B](models/tested/2026-08-06-new-candidates-sweep.md#qwen3-embedding-06b-on-both-runtimes) | TEI XPU-IPEX | 12,081 tok/s @ 1800 tok, 192.8 emb/s batch 64 | ~1.2 GB | MTEB 64.3 (+13% quality) but 40-70% slower than EG |
| [Nemotron-3-Embed-1B](models/tested/nemotron-3-embed-1b.md) | llama.cpp SYCL | 3,361 tok/s @ 1800 tok, 29 emb/s batch 64 | 1.0 GB | dim 2048, multilingual, RTEB 72.4; **6-9× slower** than EG (FA broken on ministral3 arch) |
| [LFM2.5-Embedding-350M](models/tested/lfm2.5-embedding-350m.md) | llama.cpp SYCL | 21,717 tok/s @ 1800 tok, 80 emb/s batch 64 | 0.6 GB | dim 1024, 11 languages; parity single, EG wins batched 3.2× |
| [**bge-reranker-v2-m3**](models/production/bge-reranker-v2-m3.md) ⭐ | **TEI XPU-IPEX :8008 + :8013** | **109 ms / 25 pairs** | 1.4 GB | prod (7-9× faster than llama.cpp) |
| [LFM2.5-ColBERT-350M](models/tested/lfm2.5-colbert-350m.md) | llama.cpp SYCL | 293.6 ms / 25 pairs (MaxSim) | 0.6 GB | 2.7× slower than bge; per-token vector storage 21× if used as retriever |
| bge-reranker-v2-m3 (fallback) | llama.cpp SYCL :8007 | 800-1,000 ms / 25 pairs | ~4 GB | retired 2026-07-19 |

## VRAM co-residence budget

Isolated bench numbers are misleading — production has to fit all services simultaneously. With the dual-card mirror pattern (2026-08-22), each card carries:

| Container per card | VRAM steady |
|---|---|
| Ornith 1.5-9B + MTP | ~20.7 GiB |
| llamacpp-embed (EmbeddingGemma-300M) | 0.5 GiB |
| tei-rerank (patched image) | 1.4 GiB |
| llamacpp-categorise (E2B QAT + MTP) | 3.4 GiB |
| **Total per card** | **~26 GiB** — over the 24 GiB limit if all four co-loaded on ONE card |

**How this actually fits:** each service loads its VRAM only for the card it's assigned to via `ONEAPI_DEVICE_SELECTOR=level_zero:{0|1}`. So card 1 gets `llamacpp-sycl` + `llamacpp-embed` + `tei-rerank` + `llamacpp-categorise-c1`, and card 2 gets the `-c2` variants. Each card lands at ~20-22 GiB — within budget with ~2 GiB headroom.

**Historical:** pre-2026-08-15 the stack ran on a single B60, so co-residence math was hard-capped. Ornith 22.1 GiB budget, forcing every candidate below that. See finding #12.

## Working Gemma 4 MTP drafter GGUFs on HF

The community GGUFs of Google's Gemma 4 MTP assistants use a broken architecture string (`gemma4_assistant` underscore vs upstream `gemma4-assistant` hyphen) and fail to load on any modern `llama.cpp` build. During this benching effort I converted Google's official BF16 safetensors from scratch with `llama.cpp`'s own `convert_hf_to_gguf.py` (b10215) and uploaded the working GGUFs:

- [`srmiles/gemma-4-E2B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-E2B-it-assistant-GGUF) — 170 MB, pairs with `google/gemma-4-E2B-it`
- [`srmiles/gemma-4-E4B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-E4B-it-assistant-GGUF) — 172 MB, pairs with `google/gemma-4-E4B-it`
- [`srmiles/gemma-4-12B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-12B-it-assistant-GGUF) — 862 MB, pairs with `google/gemma-4-12B-it`

All Apache 2.0. See [`models/hf-uploads/gemma-4-assistant-drafters.md`](models/hf-uploads/gemma-4-assistant-drafters.md) for full detail.

## Deep-dive docs

- **[`docs/findings.md`](docs/findings.md)** — 24 numbered findings from the build-out (quantization, MTP, SYCL kernels, workload-shape math, power draw, etc.)
- **[`docs/build-history.md`](docs/build-history.md)** — llama.cpp SYCL build history + per-release impact tables (b10068 → b10433) + journey summary
- **[`docs/track2-quality-bakeoff.md`](docs/track2-quality-bakeoff.md)** — Ornith 9B vs Gemma 4 E2B quality bake-off (2026-08-01)
- **[`docs/2nd-b60-arrival-playbook.md`](docs/2nd-b60-arrival-playbook.md)** — 2nd B60 day-1 to day-7 test sequence (includes Sergio Barrientos' vLLM XPU + MTP finding)
- **[`docs/research/ornith-1.5-upgrade.md`](docs/research/ornith-1.5-upgrade.md)** — Ornith 1.5 family research + upgrade plan
- **[`docs/research/lmcache-evaluation.md`](docs/research/lmcache-evaluation.md)** — LMCache evaluation for categorise workload
- **[`docs/handoff-2026-08-15-e2b-wedge.md`](docs/handoff-2026-08-15-e2b-wedge.md)** — E2B wedge investigation handoff
- **[`docs/monitoring-dashboard-scope.md`](docs/monitoring-dashboard-scope.md)** — Monitoring dashboard scope

## Repo layout

```
├── README.md                       ← this file
├── docs/                            ← deep-dive documents (see above)
├── models/
│   ├── production/                 ← currently running
│   │   ├── ornith-1.5-9b.md        (chat + pi.dev, both cards)
│   │   ├── gemma-4-e2b-categorise.md (categorise, both cards)
│   │   ├── gemma-4-26b-a4b.md      (reasoning fallback)
│   │   ├── embeddinggemma-300m.md  (embed)
│   │   └── bge-reranker-v2-m3.md   (rerank)
│   ├── tested/                     ← benched, not in prod
│   │   ├── ornith-1.0-9b.md        (prior prod, superseded 2026-08-21)
│   │   ├── ornith-1.5-35b-a3b-*.md (single-card MoE benches)
│   │   ├── ornith-1.5-9b-first-bench.md
│   │   ├── ornith-1.0-35b-mtp-apex.md
│   │   ├── qwen3.6-35b-a3b*.md, qwen-3.8-27b.md, qwen3-coder-30b-a3b.md
│   │   ├── gemma-4-12b-qat.md, gemma-4-e4b.md, gemma-3-4b.md
│   │   ├── muse-glimmer-30b.md
│   │   ├── minicpm5-1b.md, mistral-small-3.1-24b.md, devstral-small-2-24b.md
│   │   ├── lfm2.5-*.md, nemotron-3-embed-1b.md
│   │   ├── qwen3.6-27b.md, qwen2.5-coder-14b-awq.md
│   │   ├── categorise-candidates.md, 2026-08-06-new-candidates-sweep.md
│   ├── retired/                    ← no longer receiving traffic
│   │   └── qwen3-4b-instruct-2507.md
│   └── hf-uploads/                 ← GGUFs I've uploaded to HF
│       └── gemma-4-assistant-drafters.md
├── configs/images/                 ← Docker image build docs + patches
│   ├── llama.cpp-sycl-f16/         (build.sh, README.md)
│   └── tei-xpu-ipex-nomemleak/     (VRAM leak patch)
├── configs/launchers/              ← docker run scripts (mirror of /data/llm/launch/ on llm.local)
│   ├── start-llamacpp-sycl-ornith.sh                  (card 1 Ornith 1.5)
│   ├── start-llamacpp-sycl-ornith-1.5-c2.sh           (card 2 Ornith 1.5)
│   ├── start-llamacpp-sycl-gemma4-mtp.sh              (reasoning fallback)
│   ├── start-llamacpp-sycl-categorise-card1.sh        (E2B card 1)
│   ├── start-llamacpp-sycl-categorise-card2.sh        (E2B card 2)
│   ├── start-llamacpp-embed.sh                        (card 1)
│   ├── start-tei-rerank.sh                            (card 1)
│   └── start-llamacpp-rerank.sh                       (retired fallback)
└── charts/                         ← generated benchmark charts
```

Each per-model file includes: HF link, specs, benchmark numbers, launcher link (where applicable), and verdict.
