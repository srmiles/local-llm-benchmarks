# Ornith 1.0 9B — Production chat

**Status:** Production chat + categorise + pi.dev agent (`:8002`, single-slot FIFO shared)
**HF:** [`deepreinforce-ai/Ornith-1.0-9B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) · GGUF from `unsloth/Ornith-1.0-9B-GGUF`
**MTP drafter:** [`protoLabsAI/Ornith-1.0-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF)
**Base:** Qwen 3.5 9B fine-tune
**Launcher:** [`configs/launchers/start-llamacpp-sycl-ornith.sh`](../../configs/launchers/start-llamacpp-sycl-ornith.sh)

## Specs

| | |
|---|---|
| Parameters | 9B dense |
| Quant | Q4_K_M (post-training) |
| File size | ~5 GB |
| Context (trained) | 262K |
| Context (deployed) | 131,072 |
| MTP drafter | Q8_0, 2.4 GB |

## Benchmarks

### On b10068 build (2026-07-19, isolated, single-user)

| Metric | Value | vs b9948 |
|---|---|---|
| **Cold 12K prefill** | **12.1 s @ 896 tok/s** | **-47% wall time / +42% throughput** |
| **Decode (12K cold, MTP)** | **51.8 tok/s** | +4% |
| **5K prefill** | **1,213 tok/s** | +46% |
| Warm follow-up (cache hit) | ~0.55 s | flat |
| VRAM (loaded @ 128K KV Q8) | 10.86 GiB (target + drafter + KV) | flat |
| Correctness (chat + tool call) | ✓ | identical |

The XMX+oneDNN FA path (llama.cpp #25222) and `fattn_vec_nthreads=256` Battlemage tuning (#25205) land squarely on Ornith's dense-9B GQA attention. This is the single biggest cold-prefill win since the original 6.5× journey.

### Historical baselines (kept for reference)

| Metric | Value |
|---|---|
| Decode (single-stream, b9948) | ~50 tok/s (base), 65–70 tok/s est. w/ MTP |
| Prefill @ 6.7K (b9948) | 1,310 tok/s |
| VRAM (loaded, older builds) | 4.1 GB (base) + ~2.5 GB drafter |
| KB dual-eval score | .80 (36/45) — best local |
| pi.dev win rate (finalized) | 66.7% |

## Config (`start-llamacpp-sycl-ornith.sh`)

```bash
docker run -d --name llamacpp-sycl \
  --restart unless-stopped \
  --memory=18g --memory-swap=18g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s \
  -v /data/llm/Ornith-1.0-9B-GGUF:/models:ro \
  -p 0.0.0.0:8002:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/ornith-1.0-9b-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 \
  --cache-ram 8192 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --model-draft /models/mtp-Ornith-1.0-9B-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --predict 2048 \
  --repeat-penalty 1.05 --repeat-last-n 256 \
  --min-p 0.0 \
  --reasoning off
```

## Notes

- Won both eval lanes (KB dual-eval + pi.dev bake-off) against Gemma 4 26B-A4B in July 2026 despite being 3× smaller
- MTP drafter is a purpose-built KL-distilled head published by protoLabsAI after Ornith release; requires llama.cpp ≥ b9616
- `--reasoning off` avoids PEG parser edge cases; Ornith uses inline reasoning tags via Qwen 3.5 template
- Dual-role: serves both categorise and agent chat; FIFO queue (`--parallel 1`) means brain ingest can briefly stall pi.dev on cold prefill
- `--cache-ram 8192` enables in-RAM prefix cache (was 0 during early SWA hang bug, now safe on Ornith)
- **b10068 upgrade (2026-07-19)** — silent Q4_K get_rows correctness fix (#25656) is in this build, closing a subtle bug in Q4_K row gather that affected Ornith decodes in earlier builds. No perceptible quality change post-swap, but it's closed regardless.
