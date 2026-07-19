#!/usr/bin/env bash
# Candidate small-model slot: MiniCPM5-1B Q4_K_M on port 8009
# Runs alongside Ornith / embed / rerank as ad-hoc test target
# Not yet wired to any client — bench and evaluate before promoting
set -euo pipefail

NAME=llamacpp-minicpm5
IMAGE=llama.cpp:sycl-f16
MODEL_PATH=/data/llm/MiniCPM5-1B-GGUF
PORT=8009

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=4g --memory-swap=4g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 90s \
  -v "$MODEL_PATH":/models:ro \
  -p "0.0.0.0:$PORT:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  "$IMAGE" \
  -m /models/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --temp 0.6 --top-p 0.95 --min-p 0.0 \
  --reasoning off
echo "MiniCPM5-1B container started on :$PORT (--parallel 1, -c 131072, KV=Q8, memory=4g)"
