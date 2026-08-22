# Ling-3.0-tiny (inclusionAI) — Tested 2026-08-21

**Status:** Benched. **Rejected.** Prefill *decreases* with context — 2,293 tok/s @ 2K falling to 1,218 @ 12K — which is backwards versus every other model in the lineup, and the categorise workload it would target is prefill-heavy. No drafter available. 91.86 tok/s decode at 5.72 GiB is respectable but does not compensate. Not worth a follow-up.

**HF:** [`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny) · [bartowski GGUF](https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF) (used here) · [bloomer010 GGUF](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF)
**License:** MIT
**Publisher:** inclusionAI (Ant Group)
**Released:** 2026-08-10 · bartowski GGUF 2026-08-18
**Arch:** `bailing_hybrid` / `BailingMoeV3ForCausalLM` → GGUF arch **`bailingmoe3`**
**Drafter:** none published in any format.

## Why this was interesting

**Lowest active-parameter count in the whole sweep** — 7.9B total with only ~0.8B active per token (128 experts, 8 routed + 1 shared). If the SYCL MoE path scaled the way it does on Gemma 4 E2B, sub-1B active at under 5 GB should have been very fast, and the categorise slot is exactly where that would pay.

It was also **the one candidate that forced a build.** `bailingmoe3` is absent from b10433 — bartowski's GGUF landed 2026-08-18, four days after the b10433 cutover — so the model would not load at all. That is what prompted rebuilding at master `bb4caa754` (**b10566**), which every other row in this round then also ran on.

## Specs

| | |
|---|---|
| Total parameters | 7.9B |
| Active per token | ~0.8B (128 routed experts, **8 active**, + 1 shared expert) |
| Architecture | `bailing_hybrid` (BailingMoe V3) |
| Layers | 24 (`first_k_dense_replace: 1` — first layer dense) |
| Hidden dimension | 1,536 |
| Attention heads | 16 Q / 16 KV (MHA, no GQA), QK-norm enabled |
| Expert FFN | intermediate 512 |
| Vocabulary | 157,184 |
| Context | 131,072 |
| Thinking | `detailed thinking on/off` via system prompt; `<think>` block in template |
| Modalities | text → text |

## Setup (llama.cpp SYCL on Intel Arc Pro B60)

**Build requirement: b10566 or later.** `bailingmoe3` is not in b10433. Verify before wasting a download:

```bash
docker run --rm --entrypoint sh <image> -c \
  'grep -aq bailingmoe3 /app/libllama.so.0.1.0 && echo YES || echo NO'
```

Note the arch strings live in `libllama.so`, **not** in the `llama-server` binary — grepping the binary returns false negatives for every architecture including ones that demonstrably work.

```bash
docker run -d --name llamacpp-ling \
  --memory=14g --memory-swap=14g --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/Ling-3.0-tiny-GGUF:/models:ro \
  -p 0.0.0.0:8020:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  llama.cpp:sycl-f16-next-bb4caa754 \
  -m /models/Ling-3.0-tiny-Q4_K_M.gguf \
  -ngl 99 -c 32768 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --top-k 20 --min-p 0.0 \
  --host 0.0.0.0 --port 8000 --metrics
```

No `--model-draft` — nothing exists to point it at.

## Benchmarks (b10566, isolated card 2, 20 runs × 300 tok @ temp 0.6 / top-p 0.95 / top-k 20)

| Metric | Value |
|---|---|
| Decode | 91.86 tok/s median · 90.28 mean · σ 3.75 (min 76.55, max 92.80) |
| Speculative | — (no drafter) |
| Peak VRAM | **5.72 GiB** |

### Prefill — the disqualifier

| Prompt | 500 | 2K | 5K | 12K |
|---|---|---|---|---|
| tok/s | 1,587 | **2,293** | 1,811 | **1,218** |

Throughput peaks at 2K and then **falls by 47% out to 12K.** For comparison, over the same range on the same build and card:

| Model | 2K | 5K | 12K | Shape |
|---|---|---|---|---|
| LFM2.5-8B-A1B | 2,667 | 3,156 | **3,665** | rises |
| Qwen3.8-9B-Distill | 1,914 | 1,957 | **2,020** | rises |
| Ornith 1.5-9B | 1,910 | 1,946 | **1,987** | rises |
| Nemotron 30B-A3B | 1,536 | 1,658 | **1,772** | rises |
| **Ling-3.0-tiny** | **2,293** | 1,811 | **1,218** | **falls** |

## Verdict

**Rejected, and the reason is the shape, not the level.**

**1. Inverse prefill scaling is disqualifying for the slot it targets.** The categorise workload is prefill-heavy with short outputs — models with 1,000+ tok/s prefill can win short-output tasks even with mediocre decode. A model whose prefill *degrades* as prompts get longer is the wrong shape for that workload, and 1,218 @ 12K puts it last in the lineup at exactly the context length that matters most. Every other model here rises or holds. Whatever the `bailingmoe3` SYCL path is doing at longer contexts, it is not something to build a production slot on.

**2. No drafter, in any format.** Every other candidate in this round had one. At 91.86 tok/s unassisted it is not slow, but Gemma 4 E2B reaches 138.8 *with* a drafter at 3.4 GiB, and LFM2.5-8B-A1B reaches 168.25 at 8.63 GiB. Ling sits in between on VRAM and last on throughput.

**3. Decode is stable and the model is small** — σ 3.75 and 5.72 GiB are both fine. If the prefill curve were flat this would be a live candidate. It is not.

**4. The build it forced was still worth it.** b10566 is now on disk carrying `bailingmoe3`, `nemotron_h_moe` and `qwen3next`, and the Ornith 1.5-9B reference re-run on that build reproduced the repo's recorded numbers to within noise — so the bump cost nothing and unblocks the next `bailingmoe3` model that shows up.

## Watch items

- **Re-check prefill scaling after a llama.cpp bump.** The arch landed upstream days before this bench; a young SYCL path is exactly where an inverse-scaling bug would live, and Muse Glimmer showed the same "kernels merged, optimisation hasn't caught up" pattern. Worth one cheap re-probe at 2K/12K on a future build rather than a full re-bench.
- **`Ling-3.0-flash`** is the same family at 127B total — far past this card.
- No drafter has been published; `convert_hf_to_gguf.py` has no `bailing` MTP export path either, so building one locally is not currently an option.

## Files on disk

```
/data/llm/Ling-3.0-tiny-GGUF/
└── Ling-3.0-tiny-Q4_K_M.gguf    4.92 GB
```

Raw bench results: `/data/llm/benchmarks/20260821/ling-3.0-tiny.json`.

## References

- [Model card — inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
- [bartowski GGUF](https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF) — the mainline-support signal for a new arch
- [Full bench write-up + findings #25-#32](2026-08-21-tier1-tier2-bench.md)
- [Candidate sweep that shortlisted it](2026-08-21-new-candidates-sweep.md)
