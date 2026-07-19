#!/usr/bin/env bash
# EmbeddingGemma-300m QAT Q8_0 on SYCL F16 image, port 8004
# Used by: Brain MCP for knowledge indexing
# Endpoint: http://100.76.185.66:8004/v1/embeddings (Tailscale) / 192.168.1.90:8004 (LAN)
set -euo pipefail

NAME=llamacpp-embed
IMAGE=llama.cpp:sycl-f16
MODEL_PATH=/data/llm/embeddinggemma
PORT=8004

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --device=/dev/dri \
  -v "$MODEL_PATH":/models:ro \
  -p "${PORT}:8000" \
  --restart unless-stopped \
  -e ZES_ENABLE_SYSMAN=1 \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s --health-retries 3 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  "$IMAGE" \
  -m /models/embeddinggemma-300m-qat-Q8_0.gguf \
  -ngl 99 \
  -c 8192 \
  --parallel 4 \
  --host 0.0.0.0 --port 8000 \
  --embeddings \
  --pooling mean \
  -b 2048 -ub 2048 \
  --no-mmap \
  --metrics

# Wait for ready
for i in $(seq 1 30); do
  if curl -s -m 2 "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
    # Warmup: kernel autotune + paging on first contact
    for w in 1 2 3 4 5; do
      curl -sS -m 10 -X POST "http://localhost:$PORT/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"model":"embeddinggemma","input":"warmup"}' >/dev/null 2>&1
    done
    echo "READY on http://0.0.0.0:$PORT/v1/embeddings (warmed)"
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT waiting for health endpoint"
exit 1
