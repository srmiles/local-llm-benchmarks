# Inkling 2 Mini (Thinking Machines) — Tested 2026-11-22

**Status: not viable on SYCL yet.** It loads on one B60 and gives correct output, but two ops per layer pass have no SYCL kernel and run on the CPU (finding #69). Decode is 14.4 tok/s, against 74.3 for Nemotron 3.5 Lightning on the same binary. pp4096 crashes.

Raw results, scripts and the scheduler summary: [`configs/benchmarks/20261122-inkling2-mini/`](../../configs/benchmarks/20261122-inkling2-mini/).

## Model

| | |
|---|---|
| Source | [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) · [docs](https://unsloth.ai/docs/models/inkling-2-mini) |
| Licence | Apache 2.0 |
| Shape | 26.4B total / 5.4B active MoE, 128 experts / 8 active, no shared expert. 48 layers, and core layers 6–17 run **3×** per token (`inkling2.num_loops = 3`) |
| Quant | UD-Q4_K_XL, 16.41 GiB. Experts are NVFP4 with a per-expert scale (40 tensors); the rest is Q8_0 / BF16 / F32 |
| Extras | vision projector (`mmproj-*.gguf`), DSpark drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.40 GB) |
| Vendor claims | SWEBench Verified 79.2 %, SWEBench Pro v2 (Hard) 77.9 %, BrowseComp 79.0 %. Vendor-run, not verified here |

## Runtime: what actually reads these files

The docs page names llama.cpp PR #32187. **That is out of date.** The PR does not read the published files correctly.

| GGUF set | Runtime |
|---|---|
| Root files (Unsloth looped layout) | Unsloth fork [`b12588-mix-c0c3de8`](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8) or later. Also carries DSpark. **No SYCL prebuilt**, so build from the source tarball |
| `Mainline_Support/*.gguf_file` (rename to `.gguf`) | mainline master ≥ `690b87d88` (#32694 + #32704, both merged). Before #32704, the 2nd and 3rd loop passes lose their NVFP4 scales (PPL ~130 instead of 7.65). No DSpark yet (#32218 open) |

Image used: `llama.cpp:sycl-f16-unsloth-b12588-c0c3de8-full`, unpatched, `GGML_SYCL_F16=ON`, `--target full` for `llama-bench`. The fork's `intel.Dockerfile` does not copy `ui/lib/voice` into the web stage, so the UI build fails. The one-line fix is in [`intel-bench.Dockerfile.diff`](../../configs/benchmarks/20261122-inkling2-mini/intel-bench.Dockerfile.diff).

## Method

- card 2 (L0 index 1), `:8011` Nemotron stopped for the window (00:49–01:01 UTC), card empty (46 MiB) before the first load
- `llama-bench -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 -r 3`
- **Same binary for both models.** The Nemotron row is an unpatched, undrafted `llama-bench` control. It is not comparable to the production `:8011` numbers (MTP, patched series)

## Results

| Test | Inkling 2 Mini UD-Q4_K_XL | Nemotron 3.5 Lightning Q4_0 (control) | Ratio |
|---|---|---|---|
| pp512 | 93.6 ± 0.5 | 1,073.5 ± 5.6 | 0.09× |
| pp2048 | 55.6 (r = 1) | – | |
| pp4096 | **crash**: `Error OP GET_ROWS`, kernel built with `-fsycl-id-queries-range=int` and range overflows int | 1,882.6 ± 3.4 | |
| tg128 | 14.4 ± 0.05 | 74.3 ± 0.06 | 0.19× |

Smoke test (`llama-completion`, 24-token prompt, greedy): load 38.9 s, prompt 13.0 tok/s, decode 13.6 tok/s. **The output is coherent**, so NVFP4 dequant on SYCL is correct.

VRAM: 16.3 GiB weights on SYCL0, 469 MiB token embeddings on CPU (mapped), 0.8 GiB compute buffer at `-c 4096`.

Prefill gets slower as the prompt gets longer (93.6 → 55.6 from pp512 to pp2048). This matches CPU flash attention.

## Not tested

- DSpark drafter: no point while the target is CPU-bound
- quality and agent evals, context sweep, vision

## Revisit when

- ggml-sycl gets flash attention with the banded / relative-position mask and per-token sinks, and the router `SCALE` variant; or
- upstream adds those SYCL kernels for `inkling2_loop`

Then re-run on mainline (`Mainline_Support` files) and on the patched series.
