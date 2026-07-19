# Qwen 3.6-35B-A3B Kimi K2.6 Reasoning Distilled — Tested 2026-07-19

**Status:** Benched. Fastest cold prefill of any model tested on this stack. No MTP, verbose reasoning. VRAM headroom marginal for co-residence.
**HF:** [`lordx64/Qwen3.6-35B-A3B-Kimi-K2.6-Reasoning-Distilled-GGUF`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Kimi-K2.6-Reasoning-Distilled-GGUF)
**Base:** Qwen 3.6-35B-A3B + LoRA-distilled Kimi K2.6 reasoning chains
**Quant:** IQ4_XS (author-recommended for 24 GB single GPU)

## Specs

| | |
|---|---|
| Parameters | 35.5B total / 3B active (MoE) |
| Quant | IQ4_XS |
| File size | 17.64 GB |
| Context (bench) | 32,768 |
| MTP | **No** — plain Qwen 3.6 base without MTP head |
| Reasoning style | Verbose (Kimi K2.6-distilled — 2,933 tok mean, 9,764 p95) |

## Benchmarks (b10068 SYCL, isolated)

| Metric | Value | Notes |
|---|---|---|
| **Cold 12K prefill** | **904 tok/s (14.4s)** | ⭐ **Fastest cold prefill benched, even beats Ornith (896)** |
| Decode (12K cold) | 19.9 tok/s | No MTP |
| Decode (5K + 300 gen) | 30.6 tok/s | Warm, no MTP |
| 5K prefill | 1,018 tok/s | Also beats Ornith (1,213 was mixed with MTP overhead) |
| VRAM (32K, KV Q8) | 21.4 GiB | |
| Correctness (chat + tool call) | ✓ | |

## Co-residence

**21.4 + 2.4 = 23.8 GiB** with production embed + rerank + TEI stack — fits with only **0.2 GiB headroom**. **Dangerously tight** — one KV growth spike during a long generation and we OOM. Would need either a smaller context (e.g. 16K) or to swap one of the rerank containers out during heavy chat.

## Config

```bash
docker run -d --name bench-sycl \
  --memory=20g --memory-swap=20g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/Qwen3.6-35B-A3B-Kimi-Distilled:/models:ro \
  -p 0.0.0.0:8019:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/Qwen3.6-35B-A3B-Kimi-K2.6-Reasoning-Distilled.IQ4_XS.gguf \
  -ngl 99 -c 32768 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja --predict 300 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning off
```

## Verdict

**Prefill king.** IQ4_XS quant is measurably faster on prefill than Q4_K_M / Q4_K_XL on this workload — beats even Ornith 9B at 12K cold prefill despite being ~4× the parameters. That's remarkable.

**Decode loses** to MTP-equipped variants (30.6 vs 49 tok/s) because no MTP head. IQ4 quantization has a bigger decode cost than K-quants too.

**Verbose reasoning tradeoff:** mean 2,933 tokens per response, p95 9,764. At 30 tok/s that's **~100 seconds for a full reasoning response** and **5.4 minutes for a p95 response**. Great for offline deep_review of hard problems. Bad for agentic tool loops where each step needs to be sub-30-second.

## Where this fits

Not a chat replacement. But **potentially interesting as a specialized "hard reasoning" endpoint** users hit explicitly for math, code walkthroughs, or `/deep_review` — accepting the latency in exchange for careful chain-of-thought. If we ever stand up a second model slot, this is a candidate for that role.

Also worth an eval-runner check to see if Kimi-distilled reasoning actually solves problems Ornith fails on — that's the whole reason to accept the latency.
