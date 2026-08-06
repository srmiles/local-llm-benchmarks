---
license: apache-2.0
base_model: google/gemma-4-E2B-it-assistant
language:
- en
tags:
- gguf
- llama.cpp
- mtp
- speculative-decoding
- gemma
- gemma-4
- draft-model
- intel-arc
- battlemage
- xpu
library_name: gguf
pipeline_tag: text-generation
---

# Gemma 4 E2B-it Assistant (MTP Drafter) — GGUF (BF16)

Correctly-converted **BF16 GGUF** of Google's official [`google/gemma-4-E2B-it-assistant`](https://huggingface.co/google/gemma-4-E2B-it-assistant) MTP (Multi-Token Prediction) drafter, for use with [llama.cpp](https://github.com/ggml-org/llama.cpp) speculative decoding.

Pair this drafter with the [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) target model to get **~2× decode throughput** with mathematically identical output quality.

## Why this GGUF exists

The community GGUFs for Google's Gemma 4 assistants use an architecture string mismatch — `gemma4_assistant` (underscore) instead of upstream llama.cpp's `gemma4-assistant` (hyphen) — which makes them fail to load on any modern llama.cpp build. Byte-patching the arch string only surfaces the next problem: metadata keys are all namespaced under the wrong architecture prefix.

This GGUF was converted directly from Google's official BF16 safetensors with **llama.cpp's own `convert_hf_to_gguf.py`** (b10215 / commit `eb41d503b`), so every metadata key is correctly namespaced and it loads cleanly on upstream llama.cpp.

## Verified working

- **llama.cpp SYCL b10215+** on Intel Arc Pro B60 (Battlemage / Xe2, 24 GB)
- Should also work on any llama.cpp backend (CUDA, Metal, Vulkan, CPU) at b10215 or newer, since the `gemma4-assistant` architecture was already merged upstream by that build

## Usage (llama.cpp)

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
- `--spec-type draft-mtp` — use the MTP-native speculative decode path (not the classic n-gram draft)
- `--spec-draft-n-max 3` — Google's recommended draft length for Gemma 4 assistants
- `--reasoning off` — required for structured-JSON workflows on Gemma 4 (otherwise output routes to `reasoning_content`)

## Benchmark (Intel Arc Pro B60, llama.cpp SYCL b10215)

Measured on real workload (2-5K token prompts, structured JSON output):

| Metric | Value |
|---|---|
| Decode | **138.8 tok/s** |
| MTP acceptance rate | 67.8% |
| Prefill @ 2K tokens | **3,681 tok/s** |
| VRAM (target + drafter, Q4_0 target + BF16 drafter) | 4.5 GiB |

Compared to E2B without a drafter: **+58% decode throughput**. Compared to the community (broken) GGUF: **N/A — the community version doesn't load.**

Full bench methodology, hardware notes, and comparison to the Ornith 9B production model: [github.com/srmiles/local-llm-benchmarks](https://github.com/srmiles/local-llm-benchmarks).

## Conversion recipe (reproducible)

```bash
# Prerequisites: llama.cpp b10215 or newer, torch installed
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b10215
pip install --index-url https://download.pytorch.org/whl/cpu torch

hf download google/gemma-4-E2B-it-assistant --local-dir gemma-4-E2B-it-assistant-hf
python convert_hf_to_gguf.py gemma-4-E2B-it-assistant-hf/ \
  --outfile gemma-4-E2B-it-assistant-official.bf16.gguf \
  --outtype bf16
```

BF16 was kept (no quantization) because the drafter is small — 170 MB unquantized adds negligible VRAM vs the target model, and quantizing the drafter would risk MTP acceptance regression for no meaningful footprint saving.

## Files

- `gemma-4-E2B-it-assistant-official.bf16.gguf` — 170 MB, BF16

## Credits

- **Model weights:** Google — [`google/gemma-4-E2B-it-assistant`](https://huggingface.co/google/gemma-4-E2B-it-assistant)
- **MTP research:** DeepMind (Google AI) — [Gemma 4 MTP documentation](https://ai.google.dev/gemma/docs/mtp/mtp)
- **Conversion tool:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `convert_hf_to_gguf.py` at commit `eb41d503b` (b10215)
- **Conversion & bench:** [srmiles/local-llm-benchmarks](https://github.com/srmiles/local-llm-benchmarks)

## License

Apache 2.0 — same as the upstream Gemma 4 assistant weights.
