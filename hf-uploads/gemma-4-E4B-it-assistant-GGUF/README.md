---
license: apache-2.0
base_model: google/gemma-4-E4B-it-assistant
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

# Gemma 4 E4B-it Assistant (MTP Drafter) — GGUF (BF16)

Correctly-converted **BF16 GGUF** of Google's official [`google/gemma-4-E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant) MTP (Multi-Token Prediction) drafter, for use with [llama.cpp](https://github.com/ggml-org/llama.cpp) speculative decoding.

Pair this drafter with the [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) target model for ~1.5× decode throughput with mathematically identical output quality.

## Why this GGUF exists

The community GGUFs for Google's Gemma 4 assistants use an architecture string mismatch — `gemma4_assistant` (underscore) instead of upstream llama.cpp's `gemma4-assistant` (hyphen) — which makes them fail to load on any modern llama.cpp build.

This GGUF was converted directly from Google's official BF16 safetensors with **llama.cpp's own `convert_hf_to_gguf.py`** (b10215 / commit `eb41d503b`), so every metadata key is correctly namespaced and it loads cleanly on upstream llama.cpp.

## Verified working

- **llama.cpp SYCL b10215+** on Intel Arc Pro B60 (Battlemage / Xe2, 24 GB)
- Should work on any llama.cpp backend at b10215 or newer.

## Usage (llama.cpp)

```bash
llama-server \
  -m gemma-4-E4B_q4_0-it.gguf \
  --model-draft gemma-4-E4B-it-assistant-official.bf16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  -ngl 99 -c 8192 -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --host 0.0.0.0 --port 8000
```

## Benchmark (Intel Arc Pro B60, llama.cpp SYCL b10215)

Measured on real workload (2-5K token prompts, structured JSON output):

| Metric | Value |
|---|---|
| Decode | **114.1 tok/s** |
| MTP acceptance rate | 66.7% |
| Prefill @ 2K tokens | **2,319 tok/s** |
| VRAM (target + drafter, Q4_0 target + BF16 drafter) | 7 GiB |

Compared to E4B without a drafter: **+54% decode, +517% prefill** on b10215.

Full bench methodology + comparison to Ornith 9B production model: [github.com/srmiles/local-llm-benchmarks](https://github.com/srmiles/local-llm-benchmarks).

## Conversion recipe (reproducible)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b10215
pip install --index-url https://download.pytorch.org/whl/cpu torch

hf download google/gemma-4-E4B-it-assistant --local-dir gemma-4-E4B-it-assistant-hf
python convert_hf_to_gguf.py gemma-4-E4B-it-assistant-hf/ \
  --outfile gemma-4-E4B-it-assistant-official.bf16.gguf \
  --outtype bf16
```

## Files

- `gemma-4-E4B-it-assistant-official.bf16.gguf` — 172 MB, BF16

## Credits

- **Model weights:** Google — [`google/gemma-4-E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant)
- **MTP research:** DeepMind — [Gemma 4 MTP documentation](https://ai.google.dev/gemma/docs/mtp/mtp)
- **Conversion tool:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `convert_hf_to_gguf.py` at commit `eb41d503b` (b10215)
- **Conversion & bench:** [srmiles/local-llm-benchmarks](https://github.com/srmiles/local-llm-benchmarks)

## License

Apache 2.0.
