# Runtime + hardware alternatives — signal from external benches

**Status:** Research seed, not yet acted on. Two independent 2026-08 data points from r/IntelArc that reframe two open questions on our stack: **which runtime should the reasoning slot use?** and **is the B70 the right upgrade from the B60?**
**Trigger:** Steve dropped both links 2026-08-27 during Nemotron soak.
**Priority:** Medium-high — either finding, if it replicates on our hardware, would materially change our roadmap.

## Signal 1 — llama.cpp SYCL is dispatch-bound on MoE, OpenVINO is ~3× faster

**Source:** u/marfrit, [r/IntelArc "Arc A770 27B MoE model — 14 tok/s on llama.cpp, 43 tok/s on OpenVINO"](https://www.reddit.com/r/IntelArc/comments/1vxc2l8/)

### Their setup
- **Card:** Intel Arc A770 16GB (Alchemist / Xe1 — one gen older than our Battlemage / Xe2, but same SYCL kernel path)
- **Model:** Qwen 3.6-27B-A3B-Coder (same 27B/A3B MoE architecture family we tested at 49 tps on B60 llama.cpp SYCL)
- **CPU:** AMD 5700x, 8 cores (baseline reference)

### Their numbers

| Stack | Format | Decode |
|---|---|---|
| llama.cpp SYCL, all layers on GPU | Q4_K_M (~4.85 bpw) | **14.4 tok/s** |
| llama.cpp CPU only (8 cores) | same GGUF | 15.5 tok/s |
| OpenVINO GenAI, same card | int4 group-64 (~4.3 bpw) | **~43 tok/s** |

**GPU loses to CPU under llama.cpp on this workload** — that's the smoking gun.

### Their root cause (traced via `SYCL_UR_TRACE`)

llama.cpp dissolves the hybrid-MoE graph into **~2,500 kernel launches per token**. The GPU spends its life waiting for dispatches, not computing. OpenVINO compiles the same math into a fused, near-gap-free graph at **~24 ms device time/token** — dispatch-bound, not bandwidth-bound.

### How it reconciles with our own findings

- **Finding #39 corrected 2026-08-27:** the dequant path is running at 61% of fp16 peak — kernels are fine. Framework overhead is real but "not the dominant term" at prefill.
- **Finding #36:** MoE reorder win is +32% end-to-end at batch 1, invisible at n≥4. Dispatch cost is exactly where an SoA reorder helps.
- **Finding #43:** oneDNN with 4-bit weight decompression measures 2.12× the current path in a standalone probe — a 2× speedup on the same silicon, waiting for integration effort. Two independent 2× wins (theirs on the runtime side, ours on the kernel side) plausibly compose to the observed 3×.

So the marfrit result is not an outlier — it aligns with what we've measured on our own kernels, just measured end-to-end against a runtime that already avoids the dispatch tax.

### Their traps and process notes (worth preserving)

1. **Load with `VLMPipeline`, not `LLMPipeline`** for this architecture (needs `--task image-text-to-text` on export).
2. **`transformers==5.2.0` exactly** — newer versions break the export two different ways.
3. **Mixed-precision ratios (`--ratio 0.8`) produce IRs the GPU MoE fusion pass rejects.** Every official Intel IR uses ratio 1.0.
4. **Do not set `ov::cache_dir`:** compiled-blob cache round-trip loses MoE expert weights → `"expert weight provider not initialized"` on second start.
5. **`enable_prefix_caching` switches to paged-attention with different numerics** — cost them 2 greedy points on their 10-point code-gen harness. Off.
6. **AWQ + Scale Estimation calibration is a RAM monster** — >250 GB working set for a 27B on real code samples. They rented a 494 GB Graviton box for ~$10 total. Image-dataset calibration fits in far less.

### Their power measurement (indirect validation)

Wall-power under OpenVINO load: 233 W total system → **0.23 tok/s/W**, ~3.5× the efficiency of the SYCL path (216 W for a third of the speed). The gap isn't a corner case — it's persistent across the whole run.

### Their published artifact
Model + full reproduction recipe on HF: [`marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov`](https://huggingface.co/marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov). We can pull this and bench it on B60 directly — no conversion needed for a first read.

### Implications for our stack

Our best current MoE numbers on B60 SYCL (b10567 + MoE reorder):
- **Nemotron 30B-A3B:** 74 tps under real workload (task #147 monitoring)
- **Gemma 4 26B-A4B QAT:** 62.8 tps isolated
- **Ornith 1.5-35B-A3B:** 32.6 tps (failed on MTP acceptance under compression, finding #24)

If the 3× reproduces on B60 for the same architecture family:
- Nemotron could reach **~220 tps**
- Gemma 26B could reach **~180 tps**
- Ornith 1.5-35B-A3B on OpenVINO would sidestep the entire MTP-under-compression problem (finding #24), because OpenVINO doesn't use draft-MTP the way llama.cpp does — it uses its own decode graph. The 32 tps floor would likely lift substantially.

**This is the third independent piece of evidence that llama.cpp SYCL on Arc is well below the achievable ceiling on this silicon:**
- Sergio Barrientos (vLLM XPU + MTP): +5.2× prefill / +1.8× decode over llama.cpp SYCL on B60 for MoE (referenced in `docs/2nd-b60-arrival-playbook.md`)
- IPEX-LLM stack: consistently 1.5-3× faster than llama.cpp for models it supports
- marfrit (OpenVINO): 3× on A770 for Qwen 3.6-27B-A3B

### Suggested probe (minimal effort, high-signal)

1. Pull `marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov` to `/data/llm/` on llm.local
2. Install `openvino-genai` in a venv on llm.local (2026 build supports Battlemage)
3. Run the same harness we use for finding #34's `p-min` sweep (opencode-style code prompt at 20K context) against both:
   - llama.cpp SYCL b10567+moereorder on Nemotron 30B-A3B (current prod)
   - OpenVINO GenAI on Qwen 3.6-27B-A3B-Coder
4. If OpenVINO is anywhere near 2× on comparable workloads, kick off task #142's re-frame: OpenVINO becomes the reasoning-slot runtime, not vLLM XPU.

## Signal 2 — B70 vs RTX 5070: it's about VRAM, not compute

**Source:** [r/LocalLLM "RTX 5070 12GB vs Intel Arc Pro B70 32GB llama-bench result"](https://www.reddit.com/r/IntelArc/comments/1vxzuh2/), [video](https://youtu.be/jTosl0KIFw4)

**Format:** `llama-bench` `pp512 / pp8192 / tg256 / tg4096` (tokens per second)

| Model | 5070 12GB (CUDA) | B70 32GB (Vulkan) | Read |
|---|---|---|---|
| Ornith 1.5-9B Q6_K | 3266 / 3190 / 81 / 80 | 2299 / 1621 / 61 / 56 | 5070 wins clean when fits |
| Gemma 4 12B Q4_K_M | 2643 / 2324 / 69 / 69 | 1650 / 715 / 50 / 38 | 5070 wins clean, B70 tg falls off with context |
| **Qwen 3.6-27B Q4_K_M** | **59 / 44 / 1.63 / —** | **765 / 750 / 25 / 25** | **B70 wins by 13-15×** |

### What the 27B row is showing

The 5070 with 12 GB VRAM **cannot fit the 27B Q4_K_M model** (~16 GB weights). It's paging or spilling to CPU, dropping to 1.63 tps decode and falling out entirely at tg4096. The B70's 32 GB lets the whole model live on the GPU — decode is 25 tps, sustained.

### The uncomfortable truth about Vulkan

The B70 result above is **Vulkan**, not SYCL. Our own bench on Vulkan (LM Studio era, journey summary line 1 in `docs/build-history.md`) was 33.6 tps decode for the same architecture family; SYCL got us to 65+ tps for the same model. So the B70 numbers here are *undersampled* — a SYCL-based B70 bench would very plausibly be 1.5-2× the Vulkan numbers.

Which means:
- 9B on B70 SYCL: probably 90-120 tps (vs current B60 65.44)
- 12B Gemma on B70 SYCL: probably 70-100 tps
- 27B MoE on B70 SYCL: probably 40-60 tps at long context, plus room for Q5_K_M or Q6_K quants that don't fit on B60

### The strategic read

**5070 vs B60/B70 is the wrong frame.** The right question is "when does raw compute win, when does VRAM headroom win?":

- **≤9B models:** 5070 dominates on speed (2×), and 12 GB is enough. If your entire workload is a 9B chat model, buy the 5070.
- **12-24B models:** 5070 still wins on speed *if* the model + KV fits in 12 GB. Q4_K_M 12B needs 8-10 GB, so 12B fits but 24B doesn't.
- **≥27B models (including all MoE we care about):** B70's 32 GB is decisive. Not because it's faster silicon — because the 5070 can't run these models at all.

### Implications for our upgrade path

Our current pain points:
- Ornith 1.5-35B-A3B Q4_K_M (21.7 GB) — doesn't fit on B60 with KV headroom
- Nemotron 30B-A3B — pinned to Q4_0 (17.75 GB) for VRAM, would rather run Q5_K_M (~21 GB) or Q6_K (~24 GB) for quality
- Any Q5+ quant of anything ≥30B — off the table on B60

B70's 32 GB fits:
- Ornith 1.5-35B-A3B Q4_K_M at 262K context (finally)
- Nemotron 30B-A3B at Q6_K
- Qwen 3.8-32B-A3B at Q4_K_M (this doesn't exist yet but Qwen usually publishes A3B variants)
- Any 32B dense model at Q4_K_M

The B70 upgrade path isn't "faster silicon per token" — it's "unlocks a whole class of model quants and sizes."

### Combined with Signal 1

**If OpenVINO gets us 3× decode on the same silicon (Signal 1), then a B70 running OpenVINO is potentially 6× a B60 running llama.cpp SYCL on the same MoE workload** — from 74 tps to 400+ tps on Nemotron. The two effects compose.

**But** — OpenVINO on Battlemage is less proven than on Alchemist. marfrit's data is A770, one generation older. We should establish the OpenVINO baseline on B60 first before staking anything on OpenVINO+B70.

## Next steps (proposed order)

1. **Bench marfrit's Qwen 3.6-27B-A3B-Coder-int4-awq-se-ov on our B60 via OpenVINO GenAI** — no conversion needed, ~1 day of work. Establishes whether the 3× replicates on Battlemage.
2. If replicates: convert Nemotron 3.5 Lightning to OpenVINO IR (int4 group-64, ratio 1.0, `VLMPipeline` if arch needs it) — validates that our current reasoning slot can migrate.
3. If Nemotron on OpenVINO ≥ 100 tps: reframe task #142 (Ornith 1.5-35B tensor-split) as an OpenVINO port rather than a SYCL tensor-split.
4. **Independently:** monitor B70 SYCL benches. When someone publishes B70 SYCL numbers on the same models 5070 doesn't fit, we have real "what's the ceiling for B70" data.

## Sources

- Signal 1: [r/IntelArc post by u/marfrit](https://www.reddit.com/r/IntelArc/comments/1vxc2l8/) · [HF model](https://huggingface.co/marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov)
- Signal 2: [r/LocalLLM post](https://www.reddit.com/r/IntelArc/comments/1vxzuh2/) · [YouTube video](https://youtu.be/jTosl0KIFw4)
- Sergio Barrientos vLLM XPU + MTP: `docs/2nd-b60-arrival-playbook.md`
- MoE reorder finding: `docs/findings.md` #36
- Dequant path finding: `docs/findings.md` #39, #43
