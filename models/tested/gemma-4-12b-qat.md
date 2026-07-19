# Gemma 4 12B (QAT) — Tested as target and drafter

**Status:** Tested standalone and as MTP drafter for Gemma 4 26B-A4B; superseded by Google's official 26B assistant drafter.
**HF:** `google/gemma-4-12b-it-qat-q4_0-gguf`

## Specs

| | |
|---|---|
| Parameters | 12B dense |
| Quant | QAT Q4_0 |
| File size | ~7 GB |

## Benchmarks (standalone, b9777 SYCL)

| Metric | Value |
|---|---|
| Decode | 19.7 tok/s |
| Prefill @ 1K | 167 tok/s |
| Prefill @ 4K | 126 tok/s |
| VRAM (loaded) | ~9 GB |

## Comparison: 12B dense vs 26B-A4B MoE (same hardware, same config)

| Model | Prefill 1K | Prefill 4K | Decode |
|---|---|---|---|
| 12B dense | 167 tok/s | 126 tok/s | 19.7 tok/s |
| **26B-A4B MoE** | **672 tok/s** | **834 tok/s** | **38.6 tok/s** |

The **26B-4B-active MoE is ~2× faster decode and 4–6× faster prefill** than the 12B dense on the same B60. This is Finding 5 in the blog, evidenced here directly.

## As MTP drafter for 26B-A4B

Tried using 12B QAT as spec-decoding drafter for the 26B-A4B target. Acceptance rate was poor (12B and 26B-A4B have different tokenization behavior on Gemma 4's SWA layers) — only ~30% draft acceptance vs 78% with Google's purpose-built assistant model. **Drafter must be architecturally aligned with target** for spec decoding to pay off. Replaced immediately when Google published the official 26B-A4B assistant drafter.

## Verdict

Useful comparison point for the MoE-vs-dense pattern. Not viable as a standalone production model. Not viable as a drafter. Retired.
