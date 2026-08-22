#!/usr/bin/env bash
# llm2.local (B580 12GB) — EmbeddingGemma-300M QAT Q8_0 on SYCL F16, port 8004
# Same launcher as llm.local card-1 embed. level_zero:0 = B580 (only compute-capable device
# on this host — the iGPU UHD 630 is not enumerated by L0 for compute).
set -euo pipefail

NAME=llamacpp-embed
IMAGE=llama.cpp:sycl-f16
MODEL_PATH=/data/llm/embeddinggemma
PORT=8004

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --device=/dev/dri --memory=4g --memory-swap=4g \
  -v "$MODEL_PATH":/models:ro \
  -p "${PORT}:8000" \
  --restart unless-stopped \
  -e ZES_ENABLE_SYSMAN=1 \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s --health-retries 3 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  "$IMAGE" \
  -m /models/embeddinggemma-300m-qat-Q8_0.gguf \
  --alias embeddinggemma-300m \
  -ngl 99 \
  -c 8192 \
  --parallel 4 --cache-ram 0 \
  --host 0.0.0.0 --port 8000 \
  --embeddings \
  --pooling mean \
  -b 2048 -ub 2048 \
  --no-mmap \
  --metrics

# Warmup on first ready
for i in $(seq 1 30); do
  if curl -s -m 2 "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
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
