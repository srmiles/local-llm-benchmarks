# HF candidate sweep — 2026-08-21

Sweep of Hugging Face for models worth a bench slot on **2× Intel Arc Pro B60 (24 GB, Battlemage)** under `llama.cpp:sycl-f16` **b10433**. Covers releases roughly 2026-08-05 → 2026-08-21 (previous sweep: [`2026-08-06-new-candidates-sweep.md`](2026-08-06-new-candidates-sweep.md)).

## Gates applied

| Gate | Threshold | Source |
|---|---|---|
| VRAM | ≤ ~20.7 GiB for the chat slot, ≤ ~3.5 GiB for the categorise slot | co-residence budget, README |
| Sparsity | MoE with ≤ ~4B active strongly preferred; dense > 12B is bandwidth-bound at ~24 tps | finding #12, Muse Glimmer ceiling |
| Quant family | K-quants only — IQ costs 28-32% decode and 15-22 pp MTP acceptance on Battlemage | finding #13 |
| Drafter | MTP/DFlash head must exist **and** be quantised ≥ Q8_0 | finding #24 (Q4_0 MTP head → 32.5% acceptance) |
| Runtime | must load on mainline llama.cpp SYCL — treat a `ggml-org` or `bartowski` GGUF as the support signal | — |

---

## Tier 1 — bench these

### 1. `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` ⭐ top pick

| | |
|---|---|
| Arch | `nemotron_h` (Mamba2 hybrid), 52 layers, hidden 2688 |
| Params | 31.6B total, 128 routed experts / **6 active (~3B)** |
| Released | 2026-08-01 (BF16), 2026-08-04 (NVFP4) |
| GGUFs | [`ggml-org`](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) ⭐, [`bartowski`](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF), [`unsloth`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF), `lmstudio-community` |
| Drafter | MTP present upstream; a community [`MTP-GGUF`](https://huggingface.co/h1st0ry3D/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MTP-GGUF) exists — **verify head quant is Q8_0, not Q4_0** |
| Predicted VRAM | ~17.5 GB weights @ Q4_K_M; Mamba2 layers carry no KV, so long-context VRAM should behave like Qwen 3.8-27B did (19.9 GiB @ 131K class) |

**Why it matters:** this is the 30B-A3B-class model that *fits*, which is exactly what Ornith 1.5-35B-A3B failed to be. Both Ornith 35B-A3B attempts died on MTP acceptance under compression (32.5% / 26.2%, finding #24), not on VRAM. Nemotron ships at 31.6B/A3B with a first-party MTP and a `ggml-org` GGUF, so it dodges both failure modes.

**Known risk:** the Mamba2/SSM path. Qwen 3.8-27B was parked at **23.0 tps** with the note *"revisit when SYCL SSM gets XMX GEMM"*. That model was **dense** 27B, so SSM cost and bandwidth cost were confounded. Nemotron isolates the variable: if a 3B-active SSM hybrid still lands near 23 tps, the SYCL SSM kernel is the ceiling and the whole hybrid class is off the table until upstream fixes it. If it lands at 45-60 tps, you have a new reasoning-fallback contender against Gemma 4 26B-A4B (62.84 tps / 97.2% acceptance).

Either result is worth the bench slot — it settles a question that currently blocks a whole architecture family.

### 2. `empero-ai/Qwen3.8-9B-Distill`

| | |
|---|---|
| Arch | `qwen3_5` / `Qwen3_5ForConditionalGeneration` — **byte-for-byte the same shape as Ornith 1.5-9B** |
| Layers | 32 total, 24 `linear_attention` + 8 `full_attention` (3:1), hidden 4096, head_dim 256, `mtp_num_hidden_layers: 1`, 262k ctx |
| Params | 9.65B dense, multimodal (vision + video preprocessors present) |
| Base | `Qwen/Qwen3.5-9B`, distilled from Qwen 3.8 |
| Released | 2026-08-15 |
| GGUF | [`empero-ai/Qwen3.8-9B-Distill-GGUF`](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF) — BF16 / Q4_K_M / Q5_K_M / Q6_K / Q8_0 |

**Why it matters:** zero integration risk. Same arch, same layer pattern, same MTP layout, same context length as the model in the production chat slot. Launcher flags carry over unchanged, so this is a pure capability A/B against Ornith 1.5-9B's 65.44 tps / 82.7% acceptance rather than a plumbing exercise.

**⚠️ Gotcha — the GGUF repo ships no MTP head.** Only the five main-model quants are published. Benched out of the box it runs unassisted and will land ~20-25% below the Ornith number for reasons that have nothing to do with the model. The BF16 safetensors *does* carry `mtp_num_hidden_layers: 1`, so convert the head yourself with `convert_hf_to_gguf.py` at Q8_0 — the same play as the [`gemma-4-*-assistant-GGUF`](../hf-uploads/gemma-4-assistant-drafters.md) uploads. Do that **before** benching, or the result is meaningless. It is also another plausible HF upload.

---

## Tier 2 — cheap benches, categorise slot

### 3. `LiquidAI/LFM2.5-8B-A1B` + official DSpark drafter

8B total / **1B active**, `lfm2_moe`, ~5 GB at Q4. The news is not the model (May 2026) but the drafter: Liquid published [`LFM2.5-8B-A1B-DSpark-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF) on **2026-08-19** — their first llama.cpp-format speculative head, alongside [`LFM2.5-2.6B-DSpark-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF) for the dense 2.6B.

DFlash-family drafters have been the best-behaved on this hardware (Muse Glimmer hit 100% acceptance on b10433), and a 1B-active MoE at ~5 GB is squarely in categorise-slot territory. Target to beat: Gemma 4 E2B at 138.8 tps isolated / 71.5 prod, 3.4 GiB. LFM2.5 will be bigger; it has to win on quality or acceptance to justify the extra ~2 GB.

### 4. `inclusionAI/Ling-3.0-tiny`

7.9B total, 128 experts / **8 active (~0.8B)**, `bailing_hybrid` (BailingMoeV3). Released 2026-08-10; [`bartowski` GGUF](https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF) landed 2026-08-18, which is the mainline-support signal. Sub-5 GB at Q4, no drafter available.

Second categorise candidate. Lowest active-param count in the sweep — if the SYCL MoE path scales the way it did on Gemma 4 E2B, this should be very fast. Bench it in the same session as LFM2.5 to share the harness.

---

## Tier 3 — revisit, not new

### `Qwen3.8-27B` + DFlash2 drafters

Two DFlash2 heads with GGUFs landed on 2026-08-18: [`z-lab`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) and [`incoai`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) (block-diffusion drafters). Qwen 3.8-27B is parked at 23.0 tps / 57.9% MTP acceptance, and a DFlash-class head is the one lever that could move it.

**Still low priority.** Even a perfect drafter leaves prefill at 333.5 tok/s — 4.8× worse than Gemma 4 26B-A4B — and the model is dense 27B against a ~24 tps bandwidth ceiling. Fold this into the Nemotron bench only if the SSM path turns out to be fast; if Nemotron confirms the SSM kernel is the bottleneck, this stays parked.

---

## Explicitly skipped

| Model | Why |
|---|---|
| `Qwen/Qwen3.8-2.4T-A95B`, `ornith-ai/Ornith-1.5-397B`, `inclusionAI/Ling-3.0-flash` (127B), `deepseek-ai/DeepSeek-V4-Pro-0813` | orders of magnitude past 24 GB |
| The Qwen3.8-27B abliterated swarm — `JonathanColetti/*-Uncensored`, `OBLITERATUS/*`, `AEON-7/*`, `0bserverx/*-Heretic`, `outsourc-e/*-Unleashed` | dominates the trending list on downloads, but it is all the same parked 27B arch. No performance implication. |
| `*-NVFP4`, `*-ROCmFP4`, `*-MTPLX`, MLX and vLLM/sglang-only drafters (`RadixArk/*-DSpark`, `LiquidAI/*-DSpark` safetensors) | wrong runtime or wrong vendor's FP4 |
| `hyrelabs/Homura-30B-GGUF` | Muse Glimmer finetune; base already benched at the 25.3 tps bandwidth ceiling |
| `Kwaipilot/KAT-Coder-V2.5-Dev` | `qwen3_5_moe` coding agent, but no GGUF and no published size that fits |

## Embed / rerank

**Nothing actionable.** Trending on both `feature-extraction` and `text-ranking` is unchanged since the 2026-08-06 sweep — no new release since then outranks the incumbents. `EmbeddingGemma-300M` (23,208 tok/s) and `bge-reranker-v2-m3` (109 ms / 25 pairs) keep their slots. The only new entrant of note, `amgix/static-retrieval-multilingual-69m-v1` (2026-08-16), is a static-embedding model in a different quality class.

---

## Suggested bench order

1. **Nemotron-3.5-Lightning-30B-A3B** Q4_K_M + MTP head @ Q8_0 — settles the SYCL-SSM question and either yields a reasoning-fallback contender or closes off the hybrid class.
2. **Convert the Qwen3.8-9B-Distill MTP head to Q8_0**, then A/B against Ornith 1.5-9B on the prod chat harness.
3. **LFM2.5-8B-A1B + DSpark** and **Ling-3.0-tiny** in one categorise session against the Gemma 4 E2B baseline.
