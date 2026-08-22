#!/usr/bin/env bash
# llm2.local (B580 12GB) — bge-reranker-v2-m3 via TEI XPU-IPEX (empty_cache patch), port 8008
# Same launcher as llm.local card-1 rerank.
set -euo pipefail

NAME=tei-rerank
IMAGE=tei:xpu-ipex-nomemleak
MODEL_PATH=/data/llm/bge-reranker-v2-m3-hf
PORT=8008

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped --memory=6g --memory-swap=6g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --shm-size=4g --ipc=host \
  -v "$MODEL_PATH":/data:ro \
  -p "0.0.0.0:$PORT:80" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e PYTORCH_XPU_ALLOC_CONF=max_split_size_mb:256 \
  "$IMAGE" \
  --model-id /data --port 80 --dtype float16 --auto-truncate \
  --max-client-batch-size 64 \
  --max-batch-tokens 32768 \
  --max-concurrent-requests 512

echo "TEI rerank on :$PORT (patched empty_cache image, max-concurrent 512)"
