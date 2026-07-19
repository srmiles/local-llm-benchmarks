# Qwen 3.6-27B — Tested, not promoted

**Status:** Benched via LM Studio Vulkan (early) then llama.cpp SYCL. Dense-27B throughput uncompetitive vs MoE.
**HF:** `bartowski/Qwen3.6-27B-GGUF` (Q4_K_XL)

## Specs

| | |
|---|---|
| Parameters | 27B dense |
| Quant | Q4_K_XL |
| File size | ~16 GB |
| Context (deployed) | 128,000 |

## Benchmarks

| Metric | Value |
|---|---|
| Decode | ~22 tok/s |
| Prefill @ 12K | ~380 tok/s |
| VRAM (loaded) | ~17 GB |

## Verdict

Roughly half Gemma 4 26B-A4B's decode throughput at similar VRAM budget. Confirms the dense-vs-MoE pattern already established with Devstral 24B and Gemma 4 12B dense. Retired after single bench pass.
