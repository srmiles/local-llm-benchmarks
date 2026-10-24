# Nemotron 3.5 Bolt — E4B and 74B-A9B-24GB (NVIDIA) — Tested 2026-10-24

**Status:** Benched on b11850 (`sycl-f16-9ae73e68f-patched-3bd0de68e`), isolated card 2, two rounds each. **Neither displaces an incumbent.**

- **Bolt E4B** decodes **66.3 tok/s** solo and **98–112 tok/s median** with its DFlash drafter; prefill **3,399 @ 12K**. Against the categorise incumbent (Gemma 4 E2B + MTP: 188 decode, 3,328 prefill) it is ~2× the size for similar prefill and slower decode.
- **Bolt 74B-A9B-24GB** runs at **15.8 tok/s** decode and **494 @ 12K** prefill in 21.7 GiB — native-2-bit 74B on one 24 GB card works, but slowly, and the cause is **not established**.
- **74B + DFlash crashes** on a llama.cpp SYCL bug (`outprod.cpp:12`), patched locally but not rebuilt — finding **#60**.

Raw results + runner: [`configs/benchmarks/20261024-xmx-bolt-embed/bolt/`](../../configs/benchmarks/20261024-xmx-bolt-embed/bolt/).

**HF:** [`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) · [E4B DFlash drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash) · [`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) · [74B LiLiCorr DFlash drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF)
**License:** OpenMDW 1.1 · **Released:** 2026-10-05 (E4B) / 2026-10-07 (74B GGUF) / 2026-10-12 (LiLiCorr drafter)

"Bolt" is the October family; there were **no new Lightning checkpoints** after August.

## Specs

| | Bolt E4B | Bolt 74B-A9B-24GB |
|---|---|---|
| GGUF arch | `gemma4` (post-trained from Gemma 4 E4B IT) | `nemotron_h_moe` (same arch as Lightning) |
| File | `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`, 5.76 GB | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`, 20.51 GB (+ 1.61 GB mmproj, not loaded) |
| Weight types | Q4_K_M | **experts IQ2_XXS 15.88 GiB · SSM NVFP4 2.30 GiB · other Q8_0 1.67 GiB** (read with `GGUFReader`) |
| Drafter | DFlash, 360M params, 5 Qwen3-style layers, BF16 0.69 GB, shares target embeddings/head | DFlash "LiLiCorr", 0.93 GB |
| NVIDIA recipe | `-c 65536`, `--spec-draft-n-max 7` | `-c 98304` on 28 GB, `--spec-draft-n-max 6`, temp 1.0 / top-p 0.95 |
| Min llama.cpp | b11808 (drafter card) | — |

## Setup

```bash
# E4B + DFlash (the 74B is identical in shape: MODEL_DIR=/data/llm/nemotron-3.5-bolt-74b-a9b-GGUF, MEM=28g, n-max 6)
MODEL_DIR=/data/llm/nemotron-3.5-bolt-e4b-GGUF MODEL=Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf \
  DRAFT=Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf \
  SPEC_ARGS="--spec-type draft-dflash --spec-draft-n-max 7" \
  IMAGE=llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e CTX=65536 MEM=16g \
  /data/llm/launch/bench-slot.sh
```

`-c 65536` for the 74B too, not NVIDIA's 98K: 20.5 GB of weights on a 24 GB card, mmproj left out.

## Benchmarks (b11850-patched, card 2, `bench-candidate.py` 20 × 300 tok @ temp 0.6 / top-p 0.95 / top-k 20, `--reasoning off`)

| Config | Decode median r1 / r2 | Decode mean r1 / r2 | Decode range | Acceptance (acc/draft) | Prefill 2K | Prefill 12K | Peak VRAM |
|---|---|---|---|---|---|---|---|
| **E4B solo** | 66.2 / 66.4 | 66.2 / 66.5 | 65.6–67.1 (σ ≈0.4) | — | 3,321 / 3,344 | **3,399** | **4.68 GiB** |
| **E4B + DFlash n-max 7** | 98.0 / 111.8 | 126.0 / 153.0 | **68.8–269.3** | 32.3% (2.26) / 43.6% (3.04) | 2,463 / 2,604 | 2,623 / 2,631 | 9.51 GiB |
| **74B solo** | 15.7 / 15.8 | 15.7 / 15.8 | 15.6–15.8 (σ 0.03) | — | 408 / 410 | 494 | 21.7 GiB |
| 74B + DFlash n-max 6 | **crash** | | | | | | |

### E4B

- **The drafter buys +48–68% median decode** for −23% prefill and **+4.8 GiB** — far more than the 0.69 GB file, presumably the drafter's own context and compute buffers (not confirmed: this build no longer prints the memory breakdown at default verbosity).
- **It is bimodal and prompt-sensitive**: 69–269 tok/s across the 20 runs, and acceptance moved 32% → 44% between rounds on the same harness. Medians and means disagree by 30+ tok/s, so neither is a stable single number.
- **The acceptance is probably understated.** The drafter card says it was trained on mixed thinking/non-thinking data and measured with thinking **on**; this harness runs `--reasoning off` on a filler prompt. A real-traffic acceptance probe is the next measurement if E4B is ever a candidate.
- **Vs Gemma 4 E2B + MTP (categorise prod):** E2B 188.06 decode / 3,328 prefill / 3.96 GiB. Bolt E4B solo is 66 / 3,399 / 4.68. No case on speed; only a quality bake-off could make one.

### 74B-A9B-24GB

- **15.8 tok/s, extremely stable (σ 0.03).** Prefill climbs with context (299 → 494), the opposite of a launch-bound model.
- **Why it is slow is not established.** The obvious hypothesis — NVFP4 SSM projections falling back to CPU — **is falsified by the source**: ggml-sycl has an NVFP4 MMVQ kernel (`mmvq.cpp:3116`, `:3286`) and NVFP4 dequant (`convert.cpp:757`). Neither IQ2_XXS nor NVFP4 is reordered or covered by #29864's XMX kernel, so both run the generic dp4a paths, but that is a description, not a measured cause. A per-op profile (`GGML_SYCL_DEBUG` / unitrace) is the way to find it.
- **For scale:** Nemotron Lightning 30B-A3B (3B active) is ~99 tok/s drafted on the same card; Bolt activates **9B** per token through 2-bit experts.

### 74B + DFlash: `GGML_ASSERT(ggml_is_contiguous(src0))` — a SYCL `supports_op` gap

Both rounds abort on the first draft pass, before any token, with **zero kernel faults**:

```
/app/ggml/src/ggml-sycl/outprod.cpp:12: GGML_ASSERT(ggml_is_contiguous(src0)) failed
  ggml_sycl_op_out_prod → ggml_backend_sched_graph_compute_async → llama_context::decode
  → common_speculative_impl_draft_dflash::draft
```

`ggml_sycl_op_out_prod` asserts contiguous `src0` and `dst` (`outprod.cpp:12-13`), but `supports_op` for `GGML_OP_OUT_PROD` (`ggml-sycl.cpp:7492-7497`) checks only types, so the scheduler hands the LiLiCorr drafter's non-contiguous OUT_PROD to SYCL instead of falling back. The E4B drafter (a different architecture) never emits one. **Local fix:** report both contiguity conditions in `supports_op` — [`configs/patches/0006`](../../configs/patches/0006-sycl-report-OUT_PROD-unsupported-for-non-contiguous.patch), committed on the host branch as `ba6c3f9fc`, **not yet built or measured**. Upstream-reportable once reproduced on a clean build (finding #60).

## The void first attempt

The first E4B run on 2026-10-24 crashed the card into `UR_RESULT_ERROR_DEVICE_LOST` and took a host reboot to recover. **That was a scheduling bug in the bench chain, not this model**: `:8011` was restarted onto card 2 underneath the bench, so two tenants shared the card (23.9 GiB peak). Every number above is from the post-reboot rerun with a card-empty guard before each launch; the void results are not committed. Full account: finding **#59**.
