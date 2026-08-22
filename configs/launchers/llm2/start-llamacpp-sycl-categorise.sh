#!/usr/bin/env bash
# llm2.local (B580 12GB) — Gemma 4 E2B QAT Q4_0 + Google MTP, port 8009
# Adapted from llm.local start-llamacpp-sycl-categorise-card1.sh with two changes:
#   1. --device /dev/dri (full passthrough) — was --device /dev/dri/card1 --device /dev/dri/renderD128
#      That llm.local combo hands the container B580 card + iGPU render on llm2 =
#      "No device of requested type" SYCL error. Same signature as the earlier
#      E2B wedge investigation on B60. Full /dev/dri fixes it.
#   2. NAME=llamacpp-categorise (not -c1) and PORT=8009 — primary categorise slot on this host.
set -euo pipefail

NAME=llamacpp-categorise
IMAGE=llama.cpp:sycl-f16
MODEL_DIR=/data/llm/gemma-4-E2B-it-GGUF
DRAFT_DIR=/data/llm/gemma-4-E2B-it-assistant-GGUF
PORT=8009

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=10g --memory-swap=12g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s --health-retries 3 \
  -v "$MODEL_DIR":/models:ro \
  -v "$DRAFT_DIR":/drafter:ro \
  -p "0.0.0.0:${PORT}:8000" \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -e NEO_CACHE_PERSISTENT=1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e LLAMA_ARG_HOST=0.0.0.0 \
  "$IMAGE" \
  -m /models/gemma-4-E2B_q4_0-it.gguf \
  --model-draft /drafter/gemma-4-E2B-it-assistant-official.bf16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --alias gemma-4-E2B-it \
  -ngl 99 \
  -c 131072 --parallel 1 --cache-ram 0 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --top-p 0.95 --top-k 20 --min-p 0.0

echo "llamacpp-categorise on host :${PORT} (B580, level_zero:0)"
