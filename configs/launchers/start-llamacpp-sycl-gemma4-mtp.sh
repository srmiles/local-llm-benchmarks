#!/usr/bin/env bash
# Reasoning-fallback chat: Gemma 4 26B-A4B Q4_K_M + Google MTP drafter (Config C + MTP)
# Post-OOM-crash safe: mmap on, memory-limited, KV Q8
set -euo pipefail

NAME=llamacpp-sycl
IMAGE=llama.cpp:sycl-f16
MODEL_PATH=/data/llm/lmstudio-community/gemma-4-26B-A4B-it-GGUF
DRAFT_PATH=/data/llm/Gemma-4-Assistant
TEMPLATE_PATH=/data/llm/templates
PORT=8002

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=20g --memory-swap=20g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 180s \
  -v "$MODEL_PATH":/models:ro \
  -v "$DRAFT_PATH":/draft:ro \
  -v "$TEMPLATE_PATH":/templates:ro \
  -p "0.0.0.0:$PORT:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  "$IMAGE" \
  -m /models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --model-draft /draft/gemma-4-26B-A4B-it-qat-assistant-MTP-Q8_0.gguf \
  --spec-type draft-mtp \
  -ngl 99 -ngld 99 \
  -c 131072 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 3072 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  -fa on -ub 2048 -b 2048 \
  --predict 2048 \
  --min-p 0.0 \
  --temp 1.0 \
  --top-k 64 \
  --chat-template-file /templates/gemma-4-official-current.jinja \
  --jinja \
  --reasoning off
echo "Gemma 4 26B-A4B + MTP chat container started on :$PORT (Config C, KV=Q8, mmap on, memory=20g)"
