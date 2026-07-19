#!/usr/bin/env bash
# Production chat: Ornith 1.0 9B for pi.dev / agent workload
# Single-slot, large context, prefix-cache friendly
set -euo pipefail

NAME=llamacpp-sycl
IMAGE=llama.cpp:sycl-f16
MODEL_PATH=/data/llm/Ornith-1.0-9B-GGUF
PORT=8002

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=18g --memory-swap=18g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s \
  -v "$MODEL_PATH":/models:ro \
  -p "0.0.0.0:$PORT:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  "$IMAGE" \
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
echo "Ornith 9B agent container started on :$PORT (--parallel 1, -c 131072, KV=Q8_0, memory=18g, cache-ram=8g, MTP drafter, mmap enabled)"
