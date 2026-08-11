# Muse Glimmer-30B (Meta) — Tested 2026-08-10

**Status:** Benched. **Not promoted.** Slower than Ornith / Gemma 4 26B / Qwen 3.6-35B on B60 despite 100% DFlash acceptance. Kernel path for `muse-glimmer` arch just merged upstream today ([PR #26841](https://github.com/ggml-org/llama.cpp/pull/26841)) — expect kernel optimizations over the coming weeks. Retain on disk, revisit in ~1 month.

**HF:** [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) · [GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) · [assistant (safetensors)](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) · [ExecuTorch PTE](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE)
**License:** Apache 2.0
**Publisher:** Meta Superintelligence Lab
**Released:** 2026-08-10 (~2 hours before bench)
**Arch:** `muse-glimmer` (upstream llama.cpp support via PR #26841, master HEAD `62bf73d25`)
**Drafter:** `dflash-kquant.gguf` (Meta-published GGUF, block-diffusion; requires `--spec-type draft-dflash`)

## Why this was interesting

Meta's first model in this family — explicitly designed for local deployment on 24 / 32 GB consumer hardware. Distilled from Muse Spark (Meta's larger flagship). Multimodal (text + image → text). Agent-tuned end-to-end. Ships with:
- 4-bit quantization variants validated by Meta at 1% degradation for 24 GB target
- DFlash speculative-decoding drafter (block size 16)
- Perception encoder (ViT-G/14, 1.8B params) for vision input
- ExecuTorch PTE for mobile/edge deployment
- Official GGUFs from Meta themselves (rare)

Meta claims **3.1× DFlash speedup on RTX 5090** (74.9 → 233 tps decode). This bench measures the reality on B60.

## Specs

| | |
|---|---|
| Total parameters | 29.6B (28.2B LM + 1.8B ViT-G/14 vision encoder) |
| Architecture | Dense causal transformer + Perception Encoder |
| Layers | 52 (LM) + 50 (ViT) |
| Hidden dimension | 6,656 |
| Attention pattern | `[Local, Local, Local, Global]` repeating |
| Sliding window | 2,048 |
| Attention heads | 32 Q / 2 KV (GQA 16:1) |
| Head dimension | 128 |
| FFN | SwiGLU, intermediate 19,968 |
| Position encoding | RoPE (θ = 500,000), local layers only |
| Vocabulary | 202,048 |
| Context (trained) | 131,072+ |
| Modalities | Input: text + image; Output: text |
| Knowledge cutoff | 2026-01-04 |
| Reasoning strength | user-controllable via system prompt: `low` / `medium` / `high` / `xhigh` |

## Quantizations shipped by Meta

| Variant | Size | VRAM target | Meta-measured degradation |
|---|---|---|---|
| BF16 (full precision) | ~60 GB | 64 GB | baseline |
| K-Quant-Dynamic | 19.7 GB | 32 GB | 0.2% |
| **K-Quant-17GB** ⭐ | **16.8 GB** | **24 GB** | **1.0%** |
| DFlash drafter (K-quant) | 1.6 GB | — | — |
| mmproj vision projector (K-quant) | 1.4 GB | — | — |

## Setup (llama.cpp SYCL on Intel Arc Pro B60)

**Build requirement:** `muse-glimmer` arch was only added to llama.cpp master today (commit `62bf73d25`, no tag cut yet). Neither b10256 nor b10308 include it. Build from master until a tag is cut.

```bash
cd /data/llm/build/llama.cpp
git checkout master && git pull origin master
./build.sh muse   # or use systemd-run pattern in configs/images/llama.cpp-sycl-f16/build.sh
```

**Launcher (single-slot, no vision, DFlash drafter):**

```bash
docker run -d --name muse-bench \
  --restart no --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  --memory 22g --memory-swap 22g \
  -v /data/llm/muse-glimmer-30b-GGUF:/models:ro \
  -p 0.0.0.0:8002:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16-muse \
  -m /models/muse-glimmer-30B-kquant-17gb.gguf \
  --model-draft /models/dflash-kquant.gguf \
  --spec-type draft-dflash \
  -ngl 99 -ngld 99 \
  -c 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --top-p 0.95 --top-k 20 --min-p 0.0
```

**Critical flag:** `--spec-type draft-dflash` (not `draft-mtp` — DFlash is a distinct spec method). Look for `common_speculative_impl_draft_dflash: - block_size=16, mask_token_id=201818, n_extract=5` in server logs to confirm DFlash is active.

For vision input, add `--mmproj /models/mmproj-kquant.gguf`. Bench above was text-only.

## Benchmarks (b10256+PR#26841 master, single B60, isolated `/completion`, `cache_prompt: false`, warmup preflight, 2026-08-10)

| Prompt | Prefill tps | Decode tps | DFlash acceptance | VRAM |
|---|---|---|---|---|
| Warmup (201 tok, 50 gen) | 121 | 24.3 | 36/36 (100%) | 21.9 GiB |
| **5K real workload, 100 gen (warm)** | **682** | **25.3** | **74/74 (100%)** | 21.9 GiB |
| 1.3K prompt, 200 gen | 611 | 26.0 | 149/149 (100%) | 21.9 GiB |

**Power draw under load:** 32 W (light — model spends most of the time waiting on memory-bandwidth-bound decodes rather than doing compute).

## Comparison to peers on b10256+ (5K real workload)

| Model | Prefill | Decode | Spec acc | VRAM | Drafter |
|---|---|---|---|---|---|
| Ornith 9B + MTP | 1,789 | **64** | 100% | 12.5 GiB | community MTP |
| Gemma 4 26B-A4B QAT + MTP | 1,335 | 54 | 100% | 17.5 GiB | community MTP |
| Gemma 4 26B-A4B Q4_K_M + MTP | 1,475 | 48 | 96% | 19.7 GiB | community MTP |
| Qwen 3.6-35B-A3B-MTP Q4_K_XL | 766 | 43 | 100% | 23.1 GiB | fused-MTP |
| Laguna XS.2 (no drafter) | 1,213 | 29.5 | — | 22.1 GiB | ❌ (needs Poolside fork) |
| gpt-oss-20b (no drafter) | 1,265 | 25 | — | 13.4 GiB | ❌ (no drafter released) |
| **Muse Glimmer-30B + DFlash** | **682** ⚠ | **25.3** | **100%** | **21.9 GiB** | Meta DFlash ✅ |

## Verdict

**Not promotable to any slot on B60 today.** Ornith is 2.5× faster on decode, Gemma 4 26B is 2× faster on prefill at less VRAM. Even gpt-oss-20b matches decode with 8 GiB less VRAM.

**Why the gap vs Meta's RTX 5090 claim (233 tps):**
1. **Dense 29.6B is bandwidth-bound at ~19 GB/token.** B60's 456 GB/s ÷ 19 GB = ~24 tps ceiling — matches measured 25.3 tps. Ornith 9B at ~5 GB/token has a ~90 tps bandwidth ceiling in comparison.
2. **RTX 5090 has ~1,792 GB/s bandwidth** — 4× higher, hence Meta's 233 tps. Bandwidth ratio predicts the delta almost exactly.
3. **`muse-glimmer` SYCL kernels were merged today** (PR #26841). Prefill at 682 tps is well below what Battlemage's XMX oneMKL GEMM FA path delivers on other archs (Ornith gets 1,789 on same-size prompts). Kernel optimization for this specific arch has not yet had time to catch up with generic SYCL improvements like #25025.

**Where Muse Glimmer could still matter:**
- **Vision input** — the ONLY vision-capable model in the tested lineup. If a specific brain workload needs screenshot / chart / document understanding, this is the only local option. 25 tps decode is livable for occasional-use vision tasks.
- **Agent quality** — Meta's benchmarks show Muse Glimmer winning on MCP Atlas (75.5 vs Gemma 4 31B's 54.2, Qwen 3.6-27B's 62.5), DeepSearch QA, 𝛕3-Banking, WildClawBench. If pi.dev tool-calling quality is bottlenecked more than throughput, could be worth a qualitative bake-off vs Ornith on real agent workloads.
- **Long-context** — 131K+ native context via Local/Global attention pattern with 2K sliding window. Comparable to Qwen 3.6-35B on context capacity.

## Watch items

- **SYCL kernel optimization for `muse-glimmer` arch** — history shows kernel-path improvements arrive over the 1-3 months after arch merge. Track PRs touching `ggml-sycl` that reference `muse` or `glimmer`.
- **Additional quantizations** — Meta may release smaller K-quant variants (e.g. K-Quant-14GB targeting 20 GiB VRAM) that would ease co-residence pressure.
- **DFlash + Battlemage-specific tuning** — DFlash's block-diffusion pattern is different from MTP's per-token draft. XMX-tuned DFlash kernels may improve significantly on B60.

## Files on disk

```
/data/llm/muse-glimmer-30b-GGUF/
├── muse-glimmer-30B-kquant-17gb.gguf   16.8 GB — main model
└── dflash-kquant.gguf                    1.6 GB — DFlash drafter
```

Not downloaded (skipped for text-only bench):
- `muse-glimmer-30B-kquant-dynamic.gguf` (19.7 GB, 32 GB VRAM target, 0.2% degradation)
- `mmproj-kquant.gguf` (1.4 GB — needed only for vision input)

## References

- [Model card — Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [DFlash paper (2602.06036)](https://huggingface.co/papers/2602.06036) — block-diffusion speculative decoding method
- [Perception Encoder paper (2504.13181)](https://huggingface.co/papers/2504.13181) — ViT-G/14 vision encoder
- [llama.cpp PR #26841 — model: Muse Glimmer Support](https://github.com/ggml-org/llama.cpp/pull/26841) — arch merge, commit `62bf73d25`
- [Meta's Muse Spark Safety & Preparedness Report](https://ai.meta.com/static-resource/muse-spark-safety-and-preparedness-report/)
