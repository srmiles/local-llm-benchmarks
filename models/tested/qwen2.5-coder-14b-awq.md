# Qwen 2.5-Coder-14B-Instruct AWQ — Historical vLLM prod

**Status:** Retired. Cross-stack reference for the vLLM-XPU peak-prefill vs llama.cpp-SYCL agent-wall-time tradeoff.
**HF:** `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`
**Backend at time of use:** `intel/llm-scaler-vllm:1.4`

## Specs

| | |
|---|---|
| Parameters | 14B dense |
| Quant | AWQ int4 |
| File size | ~10 GB |
| Context (deployed at time) | 32,768 |

## Benchmarks (recovered from vLLM engine logs)

| Metric | Value |
|---|---|
| Prompt prefill (peak) | **1,891 tok/s** |
| Prompt prefill (typical) | ~1,857 tok/s |
| Generation/decode (peak) | 22.9 tok/s |
| Generation/decode (typical) | 13–15 tok/s |
| VRAM (loaded) | ~11 GB |

## What killed vLLM as the production path

- **Coverage broke on newer MoE models.** Qwen3-Coder-30B-A3B AutoRound OOMed at ~23 GB (FusedMoE not wired). AWQ needed `_C.get_cuda_view_from_cpu_tensor` (CUDA-only).
- **No tool-calling parser** for Qwen's Hermes format — required an external tool-shim container to rewrite `<function>` tags into OpenAI tool_calls
- **Decode was slower than current llama.cpp SYCL** — 22.9 tok/s peak vs 44.1 tok/s (Gemma 4) / ~50 tok/s (Ornith) on same hardware
- **RAM ceiling.** Host has 27 GB RAM; vLLM-XPU's memory footprint left insufficient headroom for larger MoE models

## Verdict

vLLM's peak prefill number is real — 1,891 tok/s is the fastest single-batch prefill measured on this B60 across any stack. But agentic workloads (pi.dev / brain ingest) are decode-bound and prefix-cache-reliant, not first-token-latency-bound. **Peak prefill is the wrong benchmark for this shape of workload.**

## Historical stack diagram

```
pi.dev → tool-shim (:8003, regex rewriter) → lsv-container (:8001, vLLM-XPU, Qwen 14B AWQ)
```

Retired when the shim + Qwen 14B combination failed the multi-turn edit-tool reliability check and the migration path to newer MoE architectures required either substantial vLLM patching or a stack swap.
