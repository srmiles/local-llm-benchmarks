#!/usr/bin/env bash
# Gemma 4 E2B QAT Q4_0 + Google MTP drafter — second categorise instance on card 1
# Paired with primary E2B on card 2 :8009 for dual-load throughput.
# Container name llamacpp-categorise-c1 to avoid collision with the card-2 instance.
#
# Port :8006 (adjacent to card-1 range; originally the categorise slot from
# the 2026-07-22 same-card experiment).
#
# NOTE: co-loads with Ornith on the SAME physical card (level_zero:0).
# Prior split-slot finding (2026-07-22) showed severe cross-process SYCL
# contention when two llama-server processes share a B60. If Ornith decode
# drops meaningfully under real workload, we have to revert or defer this
# instance to post-RAM-upgrade dual-load.
set -euo pipefail

NAME=llamacpp-categorise-c1
IMAGE=llama.cpp:sycl-f16
MODEL_DIR=/data/llm/gemma-4-E2B-it-GGUF
DRAFT_DIR=/data/llm/gemma-4-E2B-it-assistant-GGUF
PORT=8006

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
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e LLAMA_ARG_HOST=0.0.0.0 \
  "$IMAGE" \
  -m /models/gemma-4-E2B_q4_0-it.gguf \
  --model-draft /drafter/gemma-4-E2B-it-assistant-official.bf16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --alias gemma-4-E2B-it \
  -ngl 99 -ngld 99 \
  -c 131072 --parallel 2 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --top-p 0.95 --top-k 20 --min-p 0.0

echo "llamacpp-categorise-c1 E2B+MTP on :${PORT} card 1 (level_zero:0) starting"
