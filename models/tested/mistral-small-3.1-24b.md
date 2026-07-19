# Mistral-Small-3.1-24B — Tested for tool reliability

**Status:** Tested during Vulkan era (LM Studio) for tool-call reliability; not promoted.
**HF:** `mistralai/Mistral-Small-3.1-24B-Instruct-2503`

## Specs

| | |
|---|---|
| Parameters | 24B dense |
| Quant | Q4_K_M |
| Backend at time of test | llama.cpp Vulkan (LM Studio-hosted) |

## Benchmarks

| Metric | Value |
|---|---|
| Decode | ~19 tok/s |
| Prefill @ 12K | ~350 tok/s |
| VRAM (loaded) | ~15 GB |

## Why tested

Recruited specifically to replicate the edit-tool reliability test that broke Qwen 2.5-Coder-14B — Mistral had a reputation for cleaner tool-call formatting. Ran the test suite; passed marginally better than Qwen 14B but still trailed Gemma 4 26B-A4B on end-to-end pi.dev session throughput.

## Verdict

Same dense-24B throughput ceiling as Devstral. No unique capability advantage. Retired.

## Notes

- Was reloaded once with `TTL=0` after LM Studio JIT auto-reloaded it with wrong defaults after idle
- The TTL=0 pattern became a standard defensive setting during the LM Studio period
