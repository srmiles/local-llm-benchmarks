#!/usr/bin/env bash
# Categorise slot: Gemma 4 E2B (Google official QAT Q4_0) on :8006
# Frees Ornith :8002 from categorise contention; brain env CATEGORISE_URL -> :8006
set -euo pipefail

docker rm -f llamacpp-categorise 2>/dev/null || true

docker run -d --name llamacpp-categorise \
  --restart unless-stopped \
  --memory=6g --memory-swap=6g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 60s \
  -v /data/llm/gemma-4-E2B-it-GGUF:/models:ro \
  -p 0.0.0.0:8006:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/gemma-4-E2B_q4_0-it.gguf \
  -ngl 99 \
  -c 8192 \
  --parallel 2 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --reasoning off \
  --temp 0.0 --top-p 0.95 --min-p 0.0
