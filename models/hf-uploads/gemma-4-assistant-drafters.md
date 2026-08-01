# Gemma 4 Assistant (MTP Drafter) GGUFs — HF Uploads

Correctly-converted BF16 GGUFs of Google's official Gemma 4 MTP drafters, uploaded to Hugging Face on 2026-08-01. These are the first working GGUFs of the Google-official drafters for upstream `llama.cpp` — community versions on HF use a broken architecture string (`gemma4_assistant` underscore vs upstream `gemma4-assistant` hyphen) and fail to load.

## Uploaded repos

| Drafter | HF repo | Size | Params | Pairs with |
|---|---|---|---|---|
| Gemma 4 E2B assistant | [`srmiles/gemma-4-E2B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-E2B-it-assistant-GGUF) | 170 MB | 77.2M | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| Gemma 4 E4B assistant | [`srmiles/gemma-4-E4B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-E4B-it-assistant-GGUF) | 172 MB | 78M | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| Gemma 4 12B assistant | [`srmiles/gemma-4-12B-it-assistant-GGUF`](https://huggingface.co/srmiles/gemma-4-12B-it-assistant-GGUF) | 862 MB | 0.4B | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |

All under **Apache 2.0**, same as the upstream Gemma 4 assistant weights.

## Why these exist

While benching Gemma 4 + Google's MTP drafters on B60 in July 2026, the community GGUFs floating around HF turned out to be unloadable: they used `gemma4_assistant` (underscore) as the architecture string, while upstream `llama.cpp` had merged support under `gemma4-assistant` (hyphen). Byte-patching the arch string surfaced the next problem — every metadata key was namespaced under the wrong architecture prefix and had to be patched individually.

Solution: convert Google's official BF16 safetensors from scratch with `llama.cpp`'s own `convert_hf_to_gguf.py` on b10215 (commit `eb41d503b`). Every metadata key lands in the correct namespace and the GGUFs load cleanly on upstream `llama.cpp` SYCL, CUDA, Metal, Vulkan, and CPU backends.

## Bench results (Intel Arc Pro B60, llama.cpp SYCL b10215)

| Drafter | Decode | MTP acc | Prefill @ 2K | VRAM | Δ vs no-drafter |
|---|---|---|---|---|---|
| E2B + drafter | 138.8 tps | 67.8% | 3,681 tps | 4.5 GiB | +58% decode |
| E4B + drafter | 114.1 tps | 66.7% | 2,319 tps | 7 GiB | +54% decode, +517% prefill |
| 12B + drafter | 70.4 tps | 69.7% | 1,053 tps | 8.5 GiB | +257% decode, +530% prefill |

Full bench methodology and comparison against Ornith 9B production baseline: see main [README.md](../../README.md) and [gemma-4-e2b-categorise.md](../parked/gemma-4-e2b-categorise.md).

## Reproducing the conversion

```bash
# Prereqs: llama.cpp b10215+, torch installed
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b10215
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Pull Google's official assistant, convert to BF16 GGUF
hf download google/gemma-4-E2B-it-assistant --local-dir gemma-4-E2B-it-assistant-hf
python convert_hf_to_gguf.py gemma-4-E2B-it-assistant-hf/ \
  --outfile gemma-4-E2B-it-assistant-official.bf16.gguf \
  --outtype bf16
```

Same recipe applies for E4B and 12B — just substitute the model ID.

BF16 was kept (no quantization) because drafters are small — 170-862 MB unquantized costs negligible VRAM vs the target model, and quantizing a drafter risks MTP acceptance regression for no meaningful footprint saving.

## Usage in llama.cpp

```bash
llama-server \
  -m gemma-4-E2B_q4_0-it.gguf \
  --model-draft gemma-4-E2B-it-assistant-official.bf16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  -ngl 99 -c 8192 -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --host 0.0.0.0 --port 8000
```

Key flags:
- `--spec-type draft-mtp` — MTP-native speculative decode path (not classic n-gram)
- `--spec-draft-n-max 3` — Google's recommended draft length for Gemma 4 assistants
- `--reasoning off` — mandatory for structured-JSON workflows (otherwise output routes to `reasoning_content`)
