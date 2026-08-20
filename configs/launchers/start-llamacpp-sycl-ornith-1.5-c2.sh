#!/usr/bin/env bash
# Ornith 1.5-9B + MTP on card 2 (level_zero:1). Bench replacement for :8010.
# Card 1 :8002 continues serving prod Ornith 1.0 during eval.
# Pull card 2 from Traefik LB before running this; rejoin after passing gates.
set -euo pipefail

NAME=llamacpp-sycl-c2
IMAGE=llama.cpp:sycl-f16
MODEL_DIR=/data/llm/Ornith-1.5-9B-GGUF
MODEL=Ornith-1.5-9B-Q4_K_M.gguf
DRAFT=mtp-Ornith-1.5-9B-head-Q8_0.gguf
PORT=8010

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=14g --memory-swap=16g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s --health-retries 3 \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -v "$MODEL_DIR":/models:ro \
  -p "0.0.0.0:${PORT}:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  -e LLAMA_ARG_HOST=0.0.0.0 \
  "$IMAGE" \
  -m "/models/${MODEL}" \
  --model-draft "/models/${DRAFT}" \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --alias ornith-1.5-9b \
  -ngl 99 -ngld 99 \
  -c 262144 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 \
  --top-k 20 --min-p 0.0

echo "llamacpp-sycl-c2 (Ornith 1.5-9B on card 2) starting on :${PORT}"
