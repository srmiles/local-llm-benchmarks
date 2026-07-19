# EmbeddingGemma-300M — Production embeddings

**Status:** Production (`:8004`, consumed by brain MCP for knowledge indexing)
**HF:** [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) (QAT-Q8_0 variant)
**Launcher:** [`configs/launchers/start-llamacpp-embed.sh`](../../configs/launchers/start-llamacpp-embed.sh)

## Specs

| | |
|---|---|
| Parameters | 300M |
| Quant | QAT Q8_0 |
| File size | ~330 MB |
| Context | 8,192 |
| Pooling | mean |
| Slots | 4 (`--parallel 4`) |

## Benchmarks

| Metric | Value |
|---|---|
| Throughput | ~5,000 tok/s |
| Batch size | 2,048 (full context per batch) |
| VRAM (loaded) | ~2 GB |
| Warm-standby | Byte-identical GGUF on brain host CPU llama.cpp as failover |

## Config (`start-llamacpp-embed.sh`)

```bash
docker run -d --name llamacpp-embed \
  --device=/dev/dri \
  -v /data/llm/embeddinggemma:/models:ro \
  -p 8004:8000 \
  --restart unless-stopped \
  -e ZES_ENABLE_SYSMAN=1 \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s --health-retries 3 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/embeddinggemma-300m-qat-Q8_0.gguf \
  -ngl 99 \
  -c 8192 \
  --parallel 4 \
  --host 0.0.0.0 --port 8000 \
  --embeddings --pooling mean \
  -b 2048 -ub 2048 \
  --no-mmap \
  --metrics
```

## Notes

- Batch size 2048 chosen to fit full 8K context in a single embed call
- `--no-mmap` set because embed workload is short-lived + repetitive; mmap adds page-fault overhead
- Warm-standby on brain host (CPU llama.cpp, byte-identical GGUF) covers B60 outages without re-indexing
- Endpoints: Tailscale `http://100.76.185.66:8004/v1/embeddings`, LAN `http://192.168.1.90:8004`
- Container is stateless — safe to restart anytime; brain will re-poll health
