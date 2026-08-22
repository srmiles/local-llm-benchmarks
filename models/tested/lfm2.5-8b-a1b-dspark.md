# LFM2.5-8B-A1B + DSpark (Liquid AI) — Tested 2026-08-21

**Status:** Benched. **Fastest categorise candidate measured** — 168.25 tok/s decode and 3,665 tok/s prefill @ 12K, against Gemma 4 E2B's 138.8 and 3,681 @ *2K*. **Not promoted: 8.63 GiB against E2B's 3.4 GiB in the co-residence budget** — +5.2 GiB per card, which the current four-service layout has nowhere to put. Blocked on the card budget, not on merit. Revisit with task #144.

**HF:** [`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) · [GGUF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF) · [**DSpark drafter GGUF**](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF)
**License:** LFM Open License v1.0 (`license:other`)
**Publisher:** Liquid AI
**Released:** 2026-05-28 (model) · **2026-08-19 (DSpark GGUF drafter — the reason this got benched)**
**Arch:** `lfm2moe` — short-conv / attention hybrid MoE. Present in both b10433 and b10566.
**Drafter:** `LFM2.5-8B-A1B-DSpark-Q8_0.gguf`, first-party, block-diffusion. Requires `--spec-type draft-dspark`.

## Why this was interesting

The model is from May; **the news is the drafter.** On 2026-08-19 Liquid published their first llama.cpp-format speculative head — `LFM2.5-8B-A1B-DSpark-GGUF`, alongside `LFM2.5-2.6B-DSpark-GGUF` for the dense 2.6B. Until then LFM2.5's DSpark heads were sglang/safetensors only.

DFlash-family drafters have been the best-behaved on this hardware (Muse Glimmer-30B hit 100% DFlash acceptance on b10433), and a **1B-active** MoE at ~5 GB sits squarely in categorise-slot territory. The target to beat: Gemma 4 E2B at 138.8 tok/s isolated / 71.5 prod, 3,681 prefill @ 2K, 3.4 GiB in the co-residence budget.

## Specs

| | |
|---|---|
| Total parameters | 8B |
| Active per token | ~1B (32 experts, **4 active**; 2 dense layers) |
| Architecture | `lfm2_moe` — short-convolution / attention hybrid |
| Layers | 24 — 18 `conv` + 6 `full_attention` (roughly 3:1) |
| Hidden dimension | 2,048 |
| Conv cache | `conv_L_cache: 3` |
| Attention heads | 32 Q / 8 KV (GQA 4:1) |
| Expert FFN | intermediate 1,792 |
| Vocabulary | 128,000 |
| Context | 128,000 |
| Languages | en, ar, zh, fr, de, ja, ko, es, pt, it |
| Modalities | text → text |

**DSpark drafter:** 0.36 GB at Q8_0, **block size 9**, `n_extract 5`, `mask_token_id 125017`. Confirmed in the server log line `common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dspark'`.

## Setup (llama.cpp SYCL on Intel Arc Pro B60)

```bash
docker run -d --name llamacpp-lfm25 \
  --memory=14g --memory-swap=14g --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/LFM2.5-8B-A1B-GGUF:/models:ro \
  -p 0.0.0.0:8020:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  llama.cpp:sycl-f16-next-bb4caa754 \
  -m /models/LFM2.5-8B-A1B-Q4_K_M.gguf \
  --model-draft /models/LFM2.5-8B-A1B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  -ngl 99 -ngld 99 \
  -c 32768 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --top-k 20 --min-p 0.0 \
  --host 0.0.0.0 --port 8000 --metrics
```

**Critical flags:**
- `--spec-type draft-dspark` — **not** `draft-mtp` and not `draft-dflash`. DSpark is its own implementation (it shares the dflash code path but carries an extra Markov head in its sidecar).
- `--spec-draft-n-max 7` — the drafter's block size is 9, but a block of 9 needs verify batch 10, which is past this card's cliff (finding #30). **7 is the effective ceiling regardless of what the drafter can produce.**
- A benign warning appears during memory fitting and can be ignored: `dflash requires ctx_other to be set (this warning is normal during memory fitting)`.

## Benchmarks (b10566, isolated card 2, 20 runs × 300 tok @ temp 0.6 / top-p 0.95 / top-k 20, n-max 7)

| Metric | Value |
|---|---|
| Decode | **168.25 tok/s** median · 161.40 mean · σ 35.26 (min 73.74, max 198.25) |
| DSpark acceptance | 79.8% — 5,060 accepted of 6,337 draft tokens |
| **Accepted per draft** | **5.51** (918 drafts) |
| Prefill | 1,985 @ 500 · 2,667 @ 2K · 3,156 @ 5K · **3,665 @ 12K** |
| Peak VRAM | 8.63 GiB |

Prefill *rises* with context out to 12K, which almost nothing else in the lineup does — Ling-3.0-tiny falls from 2,293 @ 2K to 1,218 @ 12K over the same range.

## Comparison to the categorise incumbent

| | LFM2.5-8B-A1B + DSpark | Gemma 4 E2B QAT + MTP (prod) |
|---|---|---|
| Decode | **168.25** | 138.8 isolated / 71.5 prod |
| Prefill | **3,665 @ 12K** | 3,681 @ **2K** |
| Acceptance | 79.8% · **5.51/draft** | 67.8% isolated / 25% prod · ~2.03/draft |
| Active params | 1B | ~2B dense |
| **VRAM** | **8.63 GiB** | **3.4 GiB** ✅ |

## Verdict

**Wins on throughput, loses on the only constraint that matters right now.**

**1. It is genuinely faster.** +21% decode over E2B's isolated figure, and it holds full prefill throughput out to 12K where E2B was only ever characterised at 2K. It is also an 8B-class model rather than a 2B, so the quality ceiling is higher.

**2. DSpark drafts wide where MTP drafts deep (finding #29).** 5.51 accepted tokens per draft against MTP's 2.4–3.0, at a *lower* acceptance rate (79.8% vs Ornith's 84.7%). That is the whole point of a block-diffusion drafter: it proposes a block rather than a chain, so the throughput-relevant quantity is accepted-tokens-per-draft, not the headline percentage. **Ranking drafters by acceptance % picks the wrong model.**

**3. But 8.63 GiB versus 3.4 GiB kills it for now.** The categorise slot's budget is set by co-residence with chat + embed + rerank on the same card. +5.2 GiB per card has nowhere to go. Same shape of answer as Nemotron — excellent model, blocked on the card budget — and the same conclusion: this is a **task #144** payload.

**4. High decode variance.** σ 35.26 with a 73.74 → 198.25 range. Block drafters are all-or-nothing per block, so a block miss costs proportionally more than a chain miss. Median is the number to quote.

## Watch items

- **`--spec-draft-n-max` sweep at 3 / 5 / 7** — not run for this model. Its acceptance is 79.8%, below the p ≈ 0.914 break-even for pushing to 7, so the optimum may well be lower. Two runs would settle it.
- **`LFM2.5-2.6B-DSpark-GGUF`** (also 2026-08-19) pairs with the dense 2.6B and would land far closer to the E2B VRAM budget. Untested and arguably the more relevant candidate for the categorise slot as it stands.
- **`convert_hf_to_gguf.py --dspark`** can build DSpark heads locally (finding #28), so a head for a different LFM2.5 variant is no longer gated on Liquid publishing one.
- Quality has not been assessed at all — this is a throughput bench only. The categorise workload has a JSON-fence failure mode that bit MiniCPM5-1B; that needs checking before any promotion.

## Files on disk

```
/data/llm/LFM2.5-8B-A1B-GGUF/
├── LFM2.5-8B-A1B-Q4_K_M.gguf          5.16 GB — main model
└── LFM2.5-8B-A1B-DSpark-Q8_0.gguf     0.36 GB — DSpark block drafter
```

Raw bench results: `/data/llm/benchmarks/20260821/lfm2.5-8b-a1b-dspark.json`.

## References

- [Model card — LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [DSpark drafter GGUF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF)
- [LFM2 architecture paper (2511.23404)](https://huggingface.co/papers/2511.23404)
- [llama.cpp `docs/speculative.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) — `draft-dspark` usage and block-size clamping
- [Full bench write-up + findings #25-#32](2026-08-21-tier1-tier2-bench.md)
