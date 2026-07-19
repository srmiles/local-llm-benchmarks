# Qwen3-Coder-30B-A3B-Instruct — Tested, not promoted

**Status:** Tested extensively; capability too poor for pi.dev workload despite good throughput.
**HF:** [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) · unsloth Q4_K_XL GGUF

## Specs

| | |
|---|---|
| Parameters | 30B total / **3B active** (MoE) |
| Quant tested | UD-Q4_K_XL, AutoRound int4-mixed, AWQ |
| File size (Q4_K_XL) | ~17 GB |
| Context (deployed) | 128,000 |

## Benchmarks (b9777 SYCL, Q4_K_XL)

| Metric | Value |
|---|---|
| Decode | ~38 tok/s |
| Prefill @ 12K | ~700 tok/s |
| VRAM (loaded @ 128K KV Q8) | ~20 GB |

## Verdict

- **Speed:** Good (~38 tok/s decode, competitive with Gemma 4 26B-A4B on similar batch)
- **Capability:** Failed the tool-loop and edit-tool reliability check that Gemma 4 passed; pi.dev sessions produced malformed tool calls at long context
- **Verdict:** Pinned briefly as production; switched back to Gemma 4 26B-A4B Q4_K_M within a week

## Historical stack path

Cycled through Hermes tool-call parser (needed regex shim), then Qwen's native tool template. Neither fixed the base capability issue — the model would emit fenced JSON tool blocks that streamed correctly but the model itself would enter loops after 3-4 tool calls in a session.

## Notes

- vLLM-XPU path: AutoRound int4-mixed OOMed at ~23 GB (FusedMoE handler not wired up in intel/llm-scaler-vllm)
- AWQ path: needed `_C.get_cuda_view_from_cpu_tensor` CUDA-only kernel; unusable
- llama.cpp SYCL runs cleanly at both 32K and 128K with Q8_0 KV
- `--cache-ram 0` was required at one point to avoid Gemma-adjacent SWA bug; unclear if still needed
