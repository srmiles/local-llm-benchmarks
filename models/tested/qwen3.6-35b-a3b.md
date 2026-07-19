# Qwen 3.6-35B-A3B — Tested, kept on disk

**Status:** Benched 2026-07-19. Not promoted — half the decode throughput of Ornith 9B. Model retained on disk for future re-testing (e.g. if an MTP drafter appears, or a llama.cpp build improves `qwen35moe` kernels).
**HF:** [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
**Arch:** `qwen35moe` (Qwen 3.5 MoE variant)
**Bench config:** inline below (no persistent launcher — model kept on disk for re-testing)

## Specs

| | |
|---|---|
| Parameters | 34.66B total / **3B active** (MoE) |
| Quant | UD-Q3_K_M |
| File size | 16.6 GB |
| Context (trained) | 262,144 |
| Context (bench) | 32,768 |
| MTP head | **Not present** in base GGUF (no `nextn` metadata) |

## Benchmarks

| Metric | Value |
|---|---|
| Decode (3-run avg) | **31.1 tok/s** (±0.2) |
| Prefill @ 2K | 823 tok/s |
| VRAM (loaded) | 20.0 GB / 24 GB |
| Correctness: chat | ✓ ("12×7 = eighty four") |
| Correctness: tool call | ✓ (structured `get_weather({"city":"Perth"})`) |

## Bench config

```bash
docker run -d --name llamacpp-sycl \
  --restart unless-stopped \
  --memory=20g --memory-swap=20g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 180s \
  -v /data/llm/Qwen3.6-35B-A3B-GGUF:/models:ro \
  -p 0.0.0.0:8002:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf \
  -ngl 99 \
  -c 32768 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning off
```

## Verdict

Fits (20 GB / 24 GB, 4 GB headroom). Works. Tool-calls correctly. But at 31 tok/s decode it's ~half Ornith 9B's throughput (~50 tok/s base, 65–70 est. w/ MTP). Not an obvious upgrade until either:

- Unsloth (or another packager) ships an MTP drafter variant
- A future llama.cpp build lands qwen35moe-specific SYCL kernel improvements

## Download notes

- Initial curl download stalled at 2.08 GB — HF's xet CAS backend deprioritizes plain HTTP clients
- `huggingface_hub` Python library (with `hf-xet` extra) hit ~20 MB/s vs ~3 MB/s for curl — **10× faster**
- Total download time via xet: ~14 min for 16.6 GB
- Install needed `--break-system-packages --ignore-installed` due to stale click package
