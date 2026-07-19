# Qwen 3.6-35B-A3B Claude 4.7 Opus Reasoning Distilled — Tested 2026-07-19

**Status:** Benched. Fits comfortably in the production stack co-residence budget but decode slower than the plain MTP variant.
**HF (safetensors):** [`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled)
**HF (GGUF):** [`mudler/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-GGUF`](https://huggingface.co/mudler/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-GGUF)
**Base:** Qwen 3.6-35B-A3B + LoRA-distilled Claude Opus 4.7 reasoning chains
**Quant:** APEX (mudler's MoE-aware quantization) with MTP head embedded

## Specs

| | |
|---|---|
| Parameters | 35.5B total / 3B active (MoE) |
| Quant | APEX-MTP Compact |
| File size | 16.14 GB |
| Context (bench) | 32,768 |
| MTP | Embedded (community APEX-MTP variant) |
| Reasoning style | Tight (Claude Opus 4.7-distilled — ~849 tok mean, 2404 p95) |

## Benchmarks (b10068 SYCL, isolated)

| Metric | Value |
|---|---|
| Cold 12K prefill | 763 tok/s (17.0s) |
| Decode (12K cold) | 35.5 tok/s |
| Decode (5K + 300 gen) | 36.9 tok/s |
| 5K prefill | 887 tok/s |
| MTP acceptance | 61–100% (sample 63.5%, mean 2.23 accepted) |
| VRAM (32K, KV Q8) | **19.4 GiB** |
| Correctness (chat + tool call) | ✓ |

## Co-residence

**19.4 + 2.4 = 21.8 GiB with production embed + rerank + TEI stack** — fits with **2.2 GiB headroom**. Only 35B-A3B variant tested that clears production coexistence without config compromise.

## Config

```bash
docker run -d --name bench-sycl \
  --memory=20g --memory-swap=20g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/Qwen3.6-35B-A3B-Claude-Distilled:/models:ro \
  -p 0.0.0.0:8019:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-Compact.gguf \
  -ngl 99 -c 32768 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning off
```

## Verdict

Speed: 36.9 tok/s decode is 71% of Ornith's 51.8. Loss vs the plain MTP-Q4_K_XL variant is ~25% decode (smaller quant + lower MTP acceptance). Prefill also weaker (763 vs 798 cold).

Trade-off: **Claude-style tight reasoning** vs the Kimi variant's verbose chains. Better fit for tool-heavy agentic loops where long thinking gets in the way.

**Only 35B-A3B variant that co-exists with the current stack out-of-box.** If we want the 35B capacity in prod without re-engineering the KV budget or evicting rerank containers, this is the honest option.

## Next steps if considered

Same KB dual-eval + pi.dev win rate bake-off as any promotion candidate. Plus: sanity check that the Claude-style tight reasoning holds vs Ornith on multi-step tool-heavy tasks — that's the specific advantage this variant promises.
