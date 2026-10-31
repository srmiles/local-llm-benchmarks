# Local LLM Benchmarks (B60 Pro)

Local LLM benchmarks & configs for **2× Intel Arc Pro B60 (24 GB each, Battlemage / Xe2)** on bare-metal Ubuntu 26.04, 64 GiB RAM — plus **`llm2.local`, an Intel Arc B580 12 GB node** carrying the small services since 2026-08-22.

All numbers below are measured on the same physical hardware. Unless a row says otherwise, benchmarks were taken on [`llama.cpp:sycl-f16`](configs/images/llama.cpp-sycl-f16/README.md) at the current build tag.

## Builds

| | Image | Notes |
|---|---|---|
| **Production** (verified on host 2026-10-24) | `llama.cpp:sycl-f16-9d1878a43-patched-def76ec9a` | all four llama.cpp containers on llm.local (`:8002` `:8004` `:8006` `:8011`) |
| **Bench, not cut over** | `llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e` (b11850) | carries upstream [#29864](https://github.com/ggml-org/llama.cpp/pull/29864) XMX multi-column MMVQ: **Ornith +30% decode** — `:8002` should move to it ([A/B](models/tested/2026-10-24-b11850-xmx-ab.md)) |

Build lineage, rebase notes and every past cutover: [`docs/updates.md`](docs/updates.md) · per-release tables: [`docs/build-history.md`](docs/build-history.md) · local `ggml-sycl` patches: [`configs/patches/`](configs/patches/README.md).

## Production stack

| Host | Port | Container | GPU | Model | Purpose |
|---|---|---|---|---|---|
| llm.local | 8002 | `llamacpp-sycl` | B60 card 1 | [Ornith 1.5 9B + MTP](models/production/ornith-1.5-9b.md) ⭐ | chat + pi.dev agent — **sole backend** |
| llm.local | **8011** | `llamacpp-nemotron` *(on `sycl-f16-9d1878a43-patched-def76ec9a`)* | **B60 card 2 (whole card)** | [**Nemotron 3.5 Lightning 30B-A3B + MTP**](models/tested/nemotron-3.5-lightning-30b-a3b.md) ⭐ | **agent testing — deliberately NOT in any LB pool** |
| llm.local | 8004 | `llamacpp-embed` | B60 card 1 | [EmbeddingGemma-300M QAT Q8_0](models/production/embeddinggemma-300m.md) | brain embeddings |
| llm.local | 8008 | `tei-rerank` | B60 card 1 | [bge-reranker-v2-m3 fp16](models/production/bge-reranker-v2-m3.md) | rerank |
| llm.local | 8006 | `llamacpp-categorise-c1` | B60 card 1 | [Gemma 4 E2B QAT + Google MTP](models/production/gemma-4-e2b-categorise.md) | categorise |
| **llm2.local** | 8004 | `llamacpp-embed` | B580 | EmbeddingGemma-300M | embeddings (pair) |
| **llm2.local** | 8008 | `tei-rerank` | B580 | bge-reranker-v2-m3 | rerank (pair) |
| **llm2.local** | 8009 | `llamacpp-categorise` | B580 | Gemma 4 E2B QAT + Google MTP | categorise (pair) |

Card 1 carries four co-resident services (~20.5 GiB); card 2 runs Nemotron alone (~22.8 GiB); the B580 ~5.8 of 12 GiB. Reasoning fallback (not running): [Gemma 4 26B-A4B QAT + MTP](models/production/gemma-4-26b-a4b.md).

**Traefik** (`https://llm.levirge.com/v1/*`, LAN only, plus per-service `*.srmiles.com` hostnames):
- `/v1/completions` and `/v1/chat/completions` → Ornith, `llm.local:8002` only
- `/v1/embeddings` → `llm.local:8004` + `llm2.local:8004`
- `/v1/rerank` → `llm.local:8008` + `llm2.local:8008`, path rewritten to `/rerank`
- `/v1/categorise` → `llm.local:8006` + `llm2.local:8009`, path rewritten to `/v1/chat/completions`, inFlightReq=2
- `/` → monitor dashboard :8005 (HTTPS only)

`:8011` (Nemotron) has no Traefik route — agent-test traffic cannot reach a production pool.

## Latest

- **2026-10-31** — llama.cpp **b12017**: Gemma 4 E2B **+36%**, 26B +16%, Ornith +5% / prefill +8%, Nemotron flat. The gain is upstream: the stock control gets it too (#64–#65). **Not promoted:** two card-2 CCS resets in the candidate bench that followed (#66). **LFM3-1.7B prefills 9,410 tok/s @ 12K**, about 1.9× the repo record (#67). Mellum 2.1 MTP, Flint and Spark-X2.6 don't displace an incumbent.
- **2026-10-24** — #29864 XMX kernel: Ornith +30%, Nemotron flat (#58). NVIDIA Bolt E4B / 74B and Nemotron-3.5-Embed / Rerank benched — none displaces an incumbent; Embed-8B wins prose at ~16× the cost (#61). **The prod reranker lowers NDCG on EmbeddingGemma's top 50** (#62). A bench-script race cost a card-2 reboot (#59).
- **2026-09-09** — MiniCPM5-2B + DSpark: 302.65 tok/s, repo decode record (#55–#57). Spark-X2.5-4B beats Ornith undrafted (#52–#54).

Full dated log: **[`docs/updates.md`](docs/updates.md)** · numbered findings: **[`docs/findings.md`](docs/findings.md)** (#1–#67).

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
| [**Gemma 4 26B-A4B QAT + MTP**](models/production/gemma-4-26b-a4b.md) ⭐ (b10433 re-bench 2026-08-21) | QAT Q4_0 + Google MTP Q8_0 | 26B / 4B | **62.84** | **1,592 @ 4K** | 19.9 GiB @ 131K | **97.2%** | **reasoning fallback** — QAT regression cleared; 2.90 accepted/draft. **n-max swept 2026-08-21 at both harness and production sampling: peak is `--spec-draft-n-max 5`, +9.9% / +10.2%** (findings #30, #32); 8 falls off the same cliff Nemotron does |
| [Gemma 4 26B-A4B Q4_K_M + MTP](models/production/gemma-4-26b-a4b.md) (b10433) | Q4_K_M + Google MTP Q8_0 | 26B / 4B | 47.7 | 1,473 @ 5K | 19.7 GiB | 96.1% | prior bench; QAT now preferred after 2026-08-21 |
| [**Gemma 4 E2B + Google MTP**](models/production/gemma-4-e2b-categorise.md) ⭐ | QAT Q4_0 + BF16 drafter | 2B + drafter | 138.8 isolated / 71.5 prod | 3,681 @ 2K isolated / 3,040 prod | 4.5 GiB | 67.8% isolated / 25% prod | **production categorise slot** on card 2 `:8009`; 4.7× wall-clock vs Ornith on categorise |
| [Gemma 4 E4B + Google MTP](models/tested/gemma-4-e4b.md) (b10215) | QAT Q4_0 + BF16 drafter | 4B + drafter | 114.1 | 2,319 @ 2K | 7 GiB | 66.7% | benched 2026-07-31 |
| [Gemma 4 12B + Google MTP](models/tested/gemma-4-12b-qat.md) (b10215) | QAT Q4_0 + BF16 drafter | 12B dense + drafter | 70.4 | 1,053 @ 2K | 8.5 GiB | 69.7% | benched 2026-07-31; QAT beats dense despite finding #2 |
| Gemma 4 E4B (no MTP baseline) | Q4_K_M | ~4B | 68.3 (b10068) | 466 | ~3 GB | — | K-quant vs QAT reversal @ 4B (finding #2) |
| Gemma 4 26B-A4B (it) baseline | Q4_K_M | 26B / 4B | 44.1 (b10068) | 632 @ 12K | 20.9 GB | — | original locked prod (pre-MTP) |
| [Gemma 3 4B](models/tested/gemma-3-4b.md) | Q4_K_M | 4B | ~78 | — | ~3 GB | — | system-role template issue |
| [**MiniCPM5-2B + DSpark**](models/tested/minicpm5-2b.md) ⭐ (b10867) | Q4_K_M + DSpark Q8_0, n-max 7 / p-min 0.6 | 2.5B dense | **302.65** (σ 1.87) | 3,614 @ 12K | 8.33 GiB | **100.0%** | **benched 2026-09-09** — fastest decode in this repo; VRAM likely blocks it on the 12 GiB B580 |
| [MiniCPM5-2B (solo)](models/tested/minicpm5-2b.md) (b10867) | Q4_K_M, no drafter | 2.5B dense | 137.14 (σ 1.27) | **5,029 @ 12K · 5,989 @ 2K** | 5.24 GiB | n/a | **highest prefill in this repo** — the config to bake off against E2B for categorise |
| [Spark-X2.5-4B + 1.7B draft](models/tested/spark-x2.5-4b.md) (b10867) | Q4_K_M + 1.7B Q4_K_M, `draft-simple` | 4.1B + 1.7B | 93.79 | 2,583 @ 12K | 10.31 GiB | 99.7% | rejected — +6.8% decode for −29% prefill and +60% VRAM (finding #55) |
| [**Spark-X2.5-4B**](models/tested/spark-x2.5-4b.md) ⭐ (b10867) | Q4_K_M, **no drafter** | 4.1B dense | **87.74** (σ 0.35) | **3,643 @ 12K** | **6.44 GiB** | n/a — no MTP exists | **benched 2026-09-09** — beats Ornith 1.5-9B+MTP on decode, prefill and VRAM; needs a quality bake-off before any cutover |
| [Spark-X2.5-4B](models/tested/spark-x2.5-4b.md) (b10867) | Q8_0, no drafter | 4.1B dense | 56.25 | 3,105 @ 12K | 8.09 GiB | n/a | rejected — −35.9% decode vs Q4_K_M for +1.65 GiB |

### Qwen 3.x family

| Model | Quant | Params | Decode | Prefill | VRAM | MTP acc | Status |
|---|---|---|---|---|---|---|---|
| [Qwen 3.6-35B-A3B-MTP](models/tested/qwen3.6-35b-a3b-mtp.md) ⭐ | UD-Q4_K_XL | 35.5B / 3B | **49.0** | 798 @ 12K / 974 @ 5K | **24.4 GB (won't fit prod)** | 77.8% | Ornith-parity speed at 4× params; needs smaller quant for co-res |
| Qwen 3.6-35B-A3B-MTP | UD-Q4_K_S | 35.5B / 3B | 37.7 | 820 @ 12K / 985 @ 5K | 24.1 GB (fails co-res) | 71% | -23% decode from MTP acceptance drop (finding #13) |
| Qwen 3.6-35B-A3B-MTP | UD-IQ4_XS | 35.5B / 3B | 31.6 | 776 @ 12K / 918 @ 5K | 21.1 GB (fits, 0.5 GB) | 60.8% | -36% decode; IQ quants underperform K quants on B60 |
| [Qwen 3.6-35B-A3B Claude distilled](models/tested/qwen3.6-35b-a3b-claude-distilled.md) | APEX-MTP Compact | 35.5B / 3B | 36.9 | 763 @ 12K / 887 @ 5K | 19.4 GB (fits prod) | — | tight-reasoning distillation; only 35B-A3B that co-res cleanly |
| [Qwen 3.6-35B-A3B Kimi distilled](models/tested/qwen3.6-35b-a3b-kimi-distilled.md) | IQ4_XS | 35.5B / 3B | 30.6 | **904 @ 12K cold** ⭐ | 21.4 GB | — | fastest cold prefill benched; verbose reasoning; no MTP |
| [Qwen 3.6-35B-A3B base](models/tested/qwen3.6-35b-a3b.md) | UD-Q3_K_M | 34.7B / 3B | 31.1 | 823 @ 2K | 20.0 GB | — | superseded by MTP variant |
| [**Qwen3.8-9B-Distill + MTP**](models/tested/qwen3.8-9b-distill.md) (b10566) | Q4_K_M + self-converted Q8_0 head | 9.65B dense hybrid | **73.97** (56.69 unassisted) | 1,957 @ 5K / 2,020 @ 12K | 14.76 GiB | 81.4% | **+13.5% decode over Ornith 1.5-9B** at equal VRAM/prefill; arch-identical so the prod launcher works unchanged. Head is worth **+30.5% decode for −14.6% prefill and +3.85 GiB** (finding #31). Needs a qualitative bake-off before any cutover |
| [Qwen 3.8-27B + native MTP + vision](models/tested/qwen-3.8-27b.md) (tested, b10433) | Q4_K_M + Q4_0 MTP + Q8_0 mmproj | 27B dense hybrid (48 SSM + 16 attn) | 23.0 | 333.5 | 22.3 GiB | 57.9%* | **parked** 2026-08-15. ~~revisit when SYCL SSM gets XMX GEMM~~ — **that diagnosis was wrong** (finding #25): this is the dense-27B bandwidth wall, not the SSM layers. Nothing upstream will fix it. *\*acceptance is prompt-dominated on this model — 53.5-55.8% on real prompts, 84.6-96.9% on the filler prompt; see the 08-27 note* |
| [Qwen 3.8-27B — unsloth **UD 2.0**](models/tested/2026-08-27-unsloth-dynamic-3-vs-2.md) (`sycl-f16-allfixes`) | UD-Q4_K_XL 17.92 GB + unsloth MTP head | 27B dense hybrid | **18.91** | **711 @ 12K** | 20.36 GiB | 55.2% | fastest of the four arms — **prefer 2.0 over 3.0 when you want tok/s** (finding #44). Pooled over 16 interleaved runs. Not directly comparable to the 23.0 row above: different prompt, sampling and `ignore_eos` |
| [Qwen 3.8-27B — unsloth **UD 3.0**](models/tested/2026-08-27-unsloth-dynamic-3-vs-2.md) (`sycl-f16-allfixes`) | UD-Q4_K_XL 17.56 GB + unsloth MTP head | 27B dense hybrid | 17.11 | 685 @ 12K | 20.02 GiB | 53.5% | **most faithful** to Q8_0 of the four (mean KLD 0.00453) but 9.5% slower to decode — more block types, mechanism unproven |
| Qwen 3.8-27B — unsloth UD 2.0 | UD-Q3_K_XL 13.44 GB | 27B dense hybrid | 7.89 | 588 @ 12K | 16.39 GiB | 55.8% | **40% of Q4_K_XL's decode while 25% smaller** — the ladder stops paying below Q4_K (finding #45) |
| Qwen 3.8-27B — unsloth UD 3.0 | UD-Q3_K_XL 13.15 GB | 27B dense hybrid | ⚠ **wedges** | 587 @ 12K | 16.13 GiB | — | **do not deploy** — token `-1` reaches the batch and the slot 500s permanently once a drafter is attached; `/health` stays 200 (finding #46) |
| [Qwen 3.6-27B](models/tested/qwen3.6-27b.md) | Q4_K_XL | 27B dense | ~22 | ~380 | ~17 GB | — | tested; bartowski build |
| [Qwen3-Coder-30B-A3B](models/tested/qwen3-coder-30b-a3b.md) | UD-Q4_K_XL | 30B / 3B | ~38 | ~700 | ~20 GB | — | tested; capability too poor for pi.dev |
| [Qwen2.5-Coder-14B AWQ](models/tested/qwen2.5-coder-14b-awq.md) | AWQ int4 (vLLM-XPU) | 14B | 22.9 peak / 13-15 typical | **1,891 peak** | ~11 GB | — | retired; cross-stack reference |
| Qwen 3.6-27B (LM Studio Vulkan) | Q4_K | 27B | 33.6 | 97 @ 12K cold | 17.6 GB | — | historical Vulkan baseline |
| [Qwen3-4B-Instruct-2507](models/retired/qwen3-4b-instruct-2507.md) | Q4_K_M | 4B | ~94 (60s under 4-way) | 766 aggregate | ~1 GB | — | **retired** (was categorise prod) |

### Others (Meta, Poolside, OpenAI, Mistral, MiniCPM, LFM)

| Model | Quant | Params | Decode | Prefill | VRAM | Notes |
|---|---|---|---|---|---|---|
| [**Nemotron 3.5 Lightning 30B-A3B + MTP**](models/tested/nemotron-3.5-lightning-30b-a3b.md) ⭐ **DEPLOYED :8011** (b10566) | Q4_0 + MTP Q8_0 | 31.6B / 3B (`nemotron_h` Mamba2 hybrid) | **91.91** @ n-max 7 · 79.05 @ n-max 3 | 1,760 @ 12K | 22.18 GiB | **99.5-100% acceptance at every n-max — best on this stack.** Beats Gemma 4 26B-A4B on every axis. **Run it at `--spec-draft-n-max 7`** (+16.3% free; 8 falls off a 34% cliff — finding #30). Needs the whole card (no co-residence). `nemotron_h` barely quantises: ladder floors at 17.5 GiB, Q4_K_M is 23.73 GiB and **cannot fit** — Q4_0 is the only sensible pick (finding #26) |
| [**LFM2.5-8B-A1B + DSpark**](models/tested/lfm2.5-8b-a1b-dspark.md) (b10566) | Q4_K_M + DSpark Q8_0 | 8B / 1B | **168.25** | 3,156 @ 5K / **3,665 @ 12K** | 8.63 GiB | Fastest categorise candidate — vs Gemma 4 E2B's 138.8 and 3,681 @ *2K*. DSpark drafts **wide**: 5.51 accepted/draft at n-max 7 vs MTP's ~3 (finding #29). Blocked on the 3.4 GiB categorise budget, not on merit |
| [Nemotron 3.5 Bolt E4B + DFlash](models/tested/nemotron-3.5-bolt.md) (b11850) | Q4_K_M + DFlash BF16, n-max 7 | ~4B (`gemma4`) + 360M drafter | 98.0 / 111.8 median (bimodal, 69–269) · 66.3 solo | 2,630 @ 12K · **3,399 solo** | 9.5 GiB · 4.68 solo | 32–44% DFlash acceptance at `--reasoning off`; no case vs Gemma 4 E2B on speed |
| [Nemotron 3.5 Bolt 74B-A9B-24GB](models/tested/nemotron-3.5-bolt.md) (b11850) | native 2-bit: IQ2_XXS experts + NVFP4 SSM | 74B / 9B (`nemotron_h_moe`) | 15.8 (σ 0.03) | 494 @ 12K | 21.7 GiB | fits one B60; slow, cause not established. DFlash drafter **crashes** on an ggml-sycl `supports_op` gap (finding #60) |
| [Ling-3.0-tiny](models/tested/ling-3.0-tiny.md) (b10566) | Q4_K_M | 7.9B / 0.8B (`bailingmoe3`) | 91.86 | 1,811 @ 5K / **1,218 @ 12K** | 5.72 GiB | **Rejected** — prefill *decreases* with context (2,293 @ 2K → 1,218 @ 12K), backwards vs every other model, and categorise is prefill-heavy. No drafter available. Needed the b10566 build to load at all |
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
| [Nemotron-3.5-Embed-8B](models/tested/nemotron-3.5-embed-rerank.md) | llama.cpp SYCL (b11850) | 5.6–7.9 docs/s, p50 28–39 ms | ~9 GB (Q8_0) | dim 4096; **beats EG on prose** (scifact 0.837 vs 0.791, nfcorpus 0.423 vs 0.386 NDCG@10), loses code; ~16× slower |
| [Nemotron-3.5-Embed-1B](models/tested/nemotron-3.5-embed-rerank.md) | llama.cpp SYCL (b11850) | 29–47 docs/s | 1.2–2.3 GB | dim 2048; **loses to EG on all three BEIR sets**; exact parity with NVIDIA's reference |
| [LFM2.5-Embedding-350M](models/tested/lfm2.5-embedding-350m.md) | llama.cpp SYCL | 21,717 tok/s @ 1800 tok, 80 emb/s batch 64 | 0.6 GB | dim 1024, 11 languages; parity single, EG wins batched 3.2× |
| [**bge-reranker-v2-m3**](models/production/bge-reranker-v2-m3.md) ⭐ | **TEI XPU-IPEX :8008 + :8013** | **109 ms / 25 pairs** | 1.4 GB | prod (7-9× faster than llama.cpp) |
| [Nemotron-3.5-Rerank-8B](models/tested/nemotron-3.5-embed-rerank.md) | none works | — | 17.4 GiB bf16 | not loadable in TEI 1.9.3 or llama.cpp; PyTorch-XPU GPU-faulted mid-run (finding #63). Also: **bge-reranker-v2-m3 lowers NDCG@10 on EmbeddingGemma's top 50** (finding #62) |
| [LFM2.5-ColBERT-350M](models/tested/lfm2.5-colbert-350m.md) | llama.cpp SYCL | 293.6 ms / 25 pairs (MaxSim) | 0.6 GB | 2.7× slower than bge; per-token vector storage 21× if used as retriever |
| bge-reranker-v2-m3 (fallback) | llama.cpp SYCL :8007 | 800-1,000 ms / 25 pairs | ~4 GB | retired 2026-07-19 |

## Published to Hugging Face

- [`srmiles/Qwen3.8-9B-Distill-MTP-GGUF`](https://huggingface.co/srmiles/Qwen3.8-9B-Distill-MTP-GGUF) — MTP head (BF16 + Q8_0) for `empero-ai/Qwen3.8-9B-Distill` · [notes](models/hf-uploads/qwen3.8-9b-distill-mtp.md)
- Working Gemma 4 MTP assistant GGUFs (community ones use a broken arch string): [E2B](https://huggingface.co/srmiles/gemma-4-E2B-it-assistant-GGUF) · [E4B](https://huggingface.co/srmiles/gemma-4-E4B-it-assistant-GGUF) · [12B](https://huggingface.co/srmiles/gemma-4-12B-it-assistant-GGUF) · [notes](models/hf-uploads/gemma-4-assistant-drafters.md)

## Docs

- [`docs/updates.md`](docs/updates.md) — dated stack changes and build lineage
- [`docs/findings.md`](docs/findings.md) — numbered findings #1–#63
- [`docs/build-history.md`](docs/build-history.md) — llama.cpp SYCL build history + per-release impact
- [`docs/track2-quality-bakeoff.md`](docs/track2-quality-bakeoff.md) — Ornith 9B vs Gemma 4 E2B quality bake-off
- [`docs/research/`](docs/research/) — runtime/hardware alternatives, OpenVINO, Ornith 1.5, LMCache
- Incident/handoff notes: [`docs/handoff-2026-08-15-e2b-wedge.md`](docs/handoff-2026-08-15-e2b-wedge.md), [`docs/session-2026-08-15-e2b-wedge-resolution.md`](docs/session-2026-08-15-e2b-wedge-resolution.md)

## Repo layout

```
docs/                 findings, updates log, build history, research
models/production/    what is serving now (one page per model)
models/tested/        everything benched, with verdicts
models/retired/       no longer receiving traffic
models/hf-uploads/    GGUFs published to Hugging Face
configs/launchers/    docker run scripts (mirror of /data/llm/launch/ on llm.local)
configs/benchmarks/   raw results + runner scripts, one folder per bench run
configs/patches/      local ggml-sycl patch series
configs/images/       image build docs
charts/               generated charts
```
