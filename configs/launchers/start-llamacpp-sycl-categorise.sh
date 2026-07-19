#!/usr/bin/env bash
# RETIRED — brain now points CATEGORISE_MODEL_CHEAP + _STRONG at :8002 (Ornith)
# Kept in-repo for reference / possible re-activation as burst overflow
# Container may still be running with 6+ days uptime but receives no traffic
set -euo pipefail

NAME=llamacpp-categorise
IMAGE=llama.cpp:sycl-f16
MODEL_PATH=/data/llm/candidates
PORT=8006

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=7g --memory-swap=7g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s \
  -v "$MODEL_PATH":/models:ro \
  -p "0.0.0.0:$PORT:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  "$IMAGE" \
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
echo "Qwen3-4B categorise container started on :$PORT (--parallel 4, -c 32768 [8K/slot], memory=7g, KV=Q8, cache-ram=2G, mmap enabled)"
echo "NOTE: this container is retired — brain routes categorise to :8002 (Ornith)"
