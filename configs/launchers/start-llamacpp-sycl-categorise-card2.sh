#!/usr/bin/env bash
# Gemma 4 E2B QAT Q4_0 + Google MTP drafter — production categorise slot on card 2
# Serves the 99% of workload that's brain categorisation.
#
# Cutover 2026-08-15: moved from bench-eval :8009 (card 1, shared with Ornith)
# to llamacpp-categorise :8009 (card 2, physically isolated).
# 4.7x wall-clock speedup on 1.5K token categorise prompt; near-100% card 2 util under load.
#
# Card 2 pin: ONEAPI_DEVICE_SELECTOR=level_zero:1
# Port :8009 (card 2 categorise); :8010 reserved for future card-2 Ornith replica when RAM upgrade lands
set -euo pipefail

NAME=llamacpp-categorise
IMAGE=llama.cpp:sycl-f16
MODEL_DIR=/data/llm/gemma-4-E2B-it-GGUF
DRAFT_DIR=/data/llm/gemma-4-E2B-it-assistant-GGUF
PORT=8010

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
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e LLAMA_ARG_HOST=0.0.0.0 \
  "$IMAGE" \
  -m /models/gemma-4-E2B_q4_0-it.gguf \
  --alias gemma-4-E2B-it \
  --model-draft /drafter/gemma-4-E2B-it-assistant-official.bf16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -ngld 99 \
  -c 131072 --parallel 2 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --top-p 0.95 --top-k 20 --min-p 0.0

echo "llamacpp-categorise E2B+MTP on :${PORT} card 2 (level_zero:1) starting"
echo "Model load ~30-90s; check: docker logs -f llamacpp-categorise"
