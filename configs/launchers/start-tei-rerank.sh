#!/usr/bin/env bash
# Production rerank: bge-reranker-v2-m3 via TEI XPU-IPEX on port 8008
# 7-9x faster than llama.cpp SYCL rerank on the same model
# VRAM caps required — TEI leaks without max_split_size_mb
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
  --model-id /data --served-model-name bge-reranker-v2-m3 \
  --port 80 --dtype float16 --auto-truncate \
  --max-client-batch-size 64 \
  --max-batch-tokens 32768 \
  --max-concurrent-requests 128

echo "TEI rerank on :$PORT (float16, batch=64, tokens=32768, IPEX allocator capped, 6g mem)"
echo "Restart cadence: weekly, or when VRAM > 5 GB"
# 2026-08-15: bumped --memory 2g -> 6g after OOM loop on b10433 restore.
#              Python backend RSS ~2.4 GiB + Rust router + --shm-size=4g pushes past 4 GiB.
#              Post-fix stable at 4.9 GiB / 6 GiB (81%).
