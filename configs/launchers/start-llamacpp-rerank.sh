#!/usr/bin/env bash
# bge-reranker-v2-m3 Q8_0 on SYCL F16 image, port 8007 (fallback path)
# TEI XPU-IPEX on :8008 is production; llama.cpp path retained for failover
# Used by: brain (skills-server) /api/search rerank stage
# Endpoint: http://100.70.193.48:8007/v1/rerank (Tailscale) / 192.168.1.253:8007 (LAN)
set -euo pipefail

NAME=llamacpp-rerank
IMAGE=llama.cpp:sycl-f16
MODEL_PATH=/data/llm/bge-reranker-v2-m3
PORT=8007

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
  -m /models/bge-reranker-v2-m3-Q8_0.gguf \
  --reranking \
  -ngl 99 \
  -c 65536 \
  --parallel 8 \
  --host 0.0.0.0 --port 8000 \
  -b 2048 -ub 2048 \
  --no-mmap \
  --metrics

for i in $(seq 1 30); do
  if curl -s -m 2 "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
    echo "READY on http://0.0.0.0:$PORT/v1/rerank"
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT waiting for health endpoint"
exit 1
