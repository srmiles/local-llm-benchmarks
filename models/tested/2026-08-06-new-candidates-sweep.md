# 2026-08-06 New-candidate sweep on b10256 (single B60)

Four fresh benches in one day: gpt-oss-20b, Qwen3-Embedding-0.6B (both runtimes), EmbeddingGemma-300M on TEI (same-model A/B), and Laguna XS.2. All run on the current llama.cpp:sycl-f16 → b10256 alias (commit `6c8dcaa7a`) unless noted. Methodology: same 3-prompt suite as prior chat benches (short/1.3K/5K real-workload), isolated `/completion` probes with `cache_prompt: false` and warmup preflight; embed benches use single 1800-tok + batch 64 short.

Motivating context: second B60 arrives next week. Some verdicts here shift meaningfully once dual-card is live — noted per-model.

---

## Laguna XS.2 (Poolside 33B-A3B MoE) — [`poolside/Laguna-XS-2.1-GGUF`](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF)

**HF:** `poolside/Laguna-XS-2.1-GGUF` (Q4_K_M 20.3 GB, BF16 67 GB). **License:** OpenMDW-1.1. **Arch:** `laguna-xs-2.1` (upstream support via [PR#25165](https://github.com/ggml-org/llama.cpp/pull/25165), merged 2026-07-22 — in b10256).

### Spec
- 33B total / 3B active MoE
- Sliding-window + global attention (30/40 layers SWA, per-head gating)
- 256K context (benched at 32K to fit `cache_type q8_0` KV budget)
- Coding-focused (agentic training), companion DFlash draft models exist

### Bench (b10256 SYCL, isolated `/completion`, no drafter)

| Prompt | Prefill tps | Decode tps | VRAM |
|---|---|---|---|
| Warmup (100 tok, 50 gen) | 4.7 (cold overhead) | 33.5 | — |
| 1.3K prompt, 200 gen | 431 | 42.3 | — |
| **5K real-workload, 100 gen** | **1,213** | **29.5** | **22.1 GiB** |

Power: 31 W under load (idle-ish between requests, low sustained draw).

### Peer comparison (all 5K real-workload, b10256)

| Model | Prefill | Decode | VRAM | Notes |
|---|---|---|---|---|
| Ornith 9B + MTP | 1,789 | **64** | 12.5 GiB | prod, drafter helps a lot |
| Gemma 4 26B-A4B QAT + MTP | 1,335 | 54 | 17.5 GiB | best-quant-per-VRAM at 26B |
| Gemma 4 26B-A4B Q4_K_M + MTP | 1,475 | 48 | 19.7 GiB | |
| Qwen 3.6-35B-A3B-MTP Q4_K_XL | 766 | 43 | 23.1 GiB | 100% MTP but slow prefill |
| **Laguna XS-2.1 Q4_K_M (no drafter)** | **1,213** | **29.5** | **22.1 GiB** | |
| gpt-oss-20b Q4_K_M (no drafter) | 1,265 | 25 | 13.4 GiB | Also no MTP |

### DFlash drafter status (blocker for Laguna prod use)

- Poolside publishes DFlash draft models for the S 2.1 (118B) sibling, but **not** for XS 2.1 as of 2026-08-06.
- Even with DFlash: **upstream llama.cpp has the "generic DFlash framework" but not the Laguna decoder contract.** Steve's b10256 is upstream, so DFlash won't run there.
- To use DFlash on Laguna you need Poolside's llama.cpp fork on branch `laguna`.
- Reddit user testing DFlash on Laguna S 2.1 (2× RTX 5090) reported **DFlash made it 2.5× slower** — the drafter isn't a slam dunk even on the intended fork. Model spilled experts to CPU on that specific setup; may not generalize but a warning sign.

### Verdict (single B60 today)

**Not a compelling replacement.** Ornith beats Laguna on decode by 2.2×; Gemma 4 26B-A4B beats it by 1.6-1.8× at 2-4 GiB less VRAM. Even gpt-oss-20b uses 9 GiB less. The Laguna value proposition — coding-focused 33B MoE with DFlash — isn't available on upstream today, and Laguna without DFlash has no compensating decode advantage.

### Verdict (2nd B60, next week)

**Worth revisiting for the coding/agentic slot.**

- 22.1 GiB fits comfortably on dedicated card 2 (fits with 2 GiB headroom, no co-res compromise like today)
- Could co-exist with Ornith on card 1 for chat, Laguna on card 2 for pi.dev / coding-heavy agent workload
- Still, **29.5 tps decode is user-visibly slow for agent chains** — before promoting, worth:
  1. Waiting for upstream DFlash decoder-contract support (or biting the bullet and building Poolside's fork)
  2. Bake-off vs Ornith on actual pi.dev agent quality (Laguna is coding-tuned, Ornith is general — Laguna could win on quality tokens/sec even while losing on raw tokens/sec)
  3. Testing at higher-context (128K+) since Laguna's SWA+global architecture is designed for long-context coding sessions where Ornith's dense attention would slow down

**Deferred to 2nd-B60 arrival playbook.** No prod change today.

---

## gpt-oss-20b (OpenAI) — [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) via [`unsloth/gpt-oss-20b-GGUF`](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)

**Spec:** 20.9B total / 4-of-32 experts active (~2.6B active), MXFP4 native, 131K context, Apache 2.0. Arch `gpt_oss` — well-supported in llama.cpp since Aug 2025.

### Bench (Q4_K_M via unsloth, b10256 SYCL, no drafter)

| Prompt | Prefill tps | Decode tps |
|---|---|---|
| Warmup (100 tok, 50 gen) | 29 | 20.3 |
| 1.3K prompt, 200 gen | 1,010 | 28.4 |
| **5K real-workload, 100 gen** | **1,265** | **25.1** |

VRAM: **13.4 GiB** (@ 32K context Q8 KV). Power: 32 W.

### Verdict

**Not a compelling chat upgrade.** Decode is half of Ornith's despite similar effective compute (2.6B active vs Ornith 9B dense) — the killer is the missing MTP drafter (OpenAI hasn't released one).

**Where gpt-oss-20b could still be interesting:**
- **Agentic tool-calling quality** — designed for that; would need a qualitative bake-off vs Ornith on pi.dev tasks to see if quality justifies the 2.5× slower tokens/sec.
- **Reasoning-heavy tasks** — 25 tps is acceptable for occasional reasoning if the quality profile differs from Gemma 4 26B-A4B (48-54 tps decode with MTP) in useful ways.
- **When 2nd B60 lands** — 13.4 GiB fits comfortably alongside a chat model on the other card; could co-reside as a dedicated agentic slot.

Not promoted. Model kept on disk at `/data/llm/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf`.

---

## Qwen3-Embedding-0.6B on both runtimes — [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

**Spec:** 596M params, dim 1024, MTEB avg 64.3 (+13% vs EmbeddingGemma's ~57), Apache 2.0. Explicitly tagged `text-embeddings-inference` — designed for TEI runtime.

### Bench (isolated, single 1800-tok + batch 64 short)

| Runtime | Single tok/s | Batch emb/s | Notes |
|---|---|---|---|
| llama.cpp SYCL (b10256, Q8_0 GGUF) | 634 | 89.8 | wrong-runtime penalty |
| **TEI XPU-IPEX (float16 safetensors)** | **12,081** (+1,806%) | **192.8** (+115%) | native runtime |

The `tei:xpu-ipex-nomemleak` image built for rerank Just Works for embedding models too — `--auto-truncate` picks up the model card's task from its config.

### Verdict

**Confirmed hypothesis: TEI's fused encoder kernel wins big when the model was designed for TEI.** The 19× single-embed jump (634 → 12,081 tps) mirrors the class of win seen on bge-reranker (7-9× on rerank).

**But Qwen3-Embedding on TEI is still slower than EmbeddingGemma on either runtime:**
- Single: EG@llama.cpp 23,208 > EG@TEI 12,500 > **Qwen3@TEI 12,081** > Qwen3@llama.cpp 634
- Batch: EG@TEI 313.7 > EG@llama.cpp 253 > **Qwen3@TEI 192.8** > Qwen3@llama.cpp 89.8

**+13% MTEB quality upgrade costs 40-70% throughput even in Qwen3's best config.** Not a clear win unless retrieval quality is actually the bottleneck.

---

## EmbeddingGemma-300M on TEI (same-model A/B) — [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)

**Spec:** 300M params, dim 768, MTEB ~57, Gemma3-encoder architecture, Gemma license (gated on HF).

### Bench (isolated)

| Runtime | Single 1800-tok tok/s | Batch 64 short emb/s |
|---|---|---|
| **llama.cpp SYCL (b10256, Q8_0 GGUF)** ⭐ | **23,208** | 253 |
| TEI XPU-IPEX (float16 safetensors) | 12,500 (-46%) | **313.7** (+24%) |

### Verdict — the surprise

**The rerank finding does NOT generalize the way I predicted.** For EmbeddingGemma specifically:
- **llama.cpp SYCL wins single-embed latency by 1.85×** — the opposite of the rerank result
- **TEI wins batch throughput by 24%** — smaller than the rerank win but consistent direction

**Why the divergence with rerank:**
1. **Model architecture matters.** bge-reranker-v2-m3 is BERT-family cross-encoder — TEI's fused kernels are very well tuned for BERT. EmbeddingGemma is Gemma3-encoder — llama.cpp SYCL has excellent Gemma-family kernels (all your Gemma 4 wins live here). TEI's kernel isn't specialized for Gemma3-encoder the same way.
2. **Sequence length regime.** TEI's dynamic batching scales better than llama.cpp's slot-based `--parallel N` model at larger batches. But llama.cpp SYCL wins on single-request latency for longer sequences because the Gemma3 kernel path is quicker per-request even if it can't fan out as wide.

**Broader lesson (worth its own README finding):**
- **TEI's advantage over llama.cpp SYCL is architecture-specific, not universal for encoder-only models.** BERT-family (BGE, reranker, older embed): TEI wins big. Gemma3-encoder: llama.cpp wins single, TEI wins batch. Always A/B same-model on both runtimes before assuming.

### Practical implication for prod

**Keep EmbeddingGemma on llama.cpp SYCL — no change today.** For your workload profile (brain retrieval = single embed dominant), llama.cpp SYCL is 1.85× faster. If brain workload ever shifts batch-dominant (bulk reingest sweeps), migrate to TEI for the 24% batch win.

---

## Summary decisions (single B60 today)

| Candidate | Verdict | Rationale |
|---|---|---|
| Laguna XS.2 | **Defer to 2nd B60** | 22.1 GiB co-res-hostile today, no DFlash on upstream, decode 29.5 without drafter |
| gpt-oss-20b | **Not promoted** | 25 tps decode, no MTP; potential agentic quality upgrade needs bake-off |
| Qwen3-Embedding-0.6B on TEI | **Not promoted** | +13% MTEB quality costs 40-70% throughput vs EmbeddingGemma |
| EmbeddingGemma on TEI | **Not promoted** | Loses single-embed latency (-46%); prod use case is single-embed dominant |

## Deferred to 2nd-B60 arrival playbook

- **Laguna XS.2** on dedicated card 2 — bake-off vs Ornith on pi.dev agent workload for coding quality
- **gpt-oss-20b** on dedicated agentic slot — quality bake-off vs Ornith on tool-calling tasks
- **Qwen 3.6-35B-A3B-MTP** revisit — no longer VRAM-blocked, could displace Gemma 4 26B-A4B as reasoning fallback
- **DFlash decoder-contract** — watch upstream llama.cpp for merge; if it lands before 2nd B60, Laguna XS.2 becomes much more interesting