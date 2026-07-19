# Qwen3-4B-Instruct-2507 — Retired categorise (container still running)

**Status:** **Retired** — replaced by Ornith 9B on `:8002` (July 2026 dual-eval flip).
Container `llamacpp-categorise` on `:8006` is still up but no client calls it — brain's `CATEGORISE_MODEL_CHEAP` and `CATEGORISE_MODEL_STRONG` both point at `:8002`. Candidate for shutdown to reclaim ~1 GB VRAM.
**HF:** [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
**Launcher (kept for reference):** [`configs/launchers/start-llamacpp-sycl-categorise.sh`](../../configs/launchers/start-llamacpp-sycl-categorise.sh)

## Specs

| | |
|---|---|
| Parameters | 4B dense |
| Quant | Q4_K_M |
| File size | ~2.5 GB |
| Context (deployed) | 32,768 (8K/slot × 4 slots) |
| Slots | 4 |

## Benchmarks

| Metric | Value |
|---|---|
| Decode (single-request, warm) | ~94 tok/s |
| Decode (multi-slot cache-evicted) | 34.2 tok/s (staircase pattern) |
| Prefill aggregate (from /metrics) | 766 tok/s |
| VRAM (loaded) | ~1 GB (minimal) |
| JSON validity | 3/3 on categorise bake-off |

## Historical bake-off vs baselines (3-request categorise workload)

| Model | Decode tok/s | JSON | Notes |
|---|---|---|---|
| Ornith 1.0 9B Q4_K_M | ~47 | 3/3 | Baseline — 2× too slow |
| Gemma 3 4B Q4_K_M | ~78 | 3/3 | System-role incompatibility |
| **Qwen3-4B-Instruct-2507** | **~94** | 3/3 | Winner (at the time) |

## Config (`start-llamacpp-sycl-categorise.sh`)

```bash
docker run -d --name llamacpp-categorise \
  --restart unless-stopped \
  --memory=7g --memory-swap=7g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s \
  -v /data/llm/candidates:/models:ro \
  -p 0.0.0.0:8006:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 \
  -c 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 2048 \
  --parallel 4 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  -fa on -ub 1024 -b 1024 \
  --jinja \
  --predict 1024 \
  --min-p 0.0 \
  --reasoning off
```

## Why retired

- Ornith 1.0 9B won the July 2026 dual-eval bake-off and the pi.dev win rate check
- Brain flipped `CATEGORISE_MODEL_*` env vars to `:8002` (Ornith); Qwen3-4B stopped receiving traffic without an explicit flip
- Single-slot Ornith on `:8002` FIFO-shares between categorise + agent chat — works because pi.dev sessions are bursty
- `:8006` retained as "burst overflow" in July 12 notes but never re-activated

## Cleanup TODO

```bash
docker rm -f llamacpp-categorise      # free VRAM + memory limit
# optional: rm start-llamacpp-sycl-categorise.sh from systemd auto-start
```
