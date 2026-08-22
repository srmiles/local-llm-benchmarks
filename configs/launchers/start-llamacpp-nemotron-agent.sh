#!/usr/bin/env bash
# Nemotron 3.5 Lightning 30B-A3B + MTP — agent-testing slot, card 2 (level_zero:1).
#
# Card 2 was freed 2026-08-22: embed/rerank/categorise moved to llm2.local (B580)
# and the Ornith mirror :8010 was retired. This model needs the WHOLE card
# (~22 GiB) so nothing else may be scheduled on level_zero:1.
#
# Deliberately on :8011 with NO Traefik route — isolated from every LB pool so
# agent-test traffic can never land in a production route or vice versa.
#
# Bench basis: models/tested/nemotron-3.5-lightning-30b-a3b.md
#   91.91 tok/s decode @ n-max 7, 99.5% MTP acceptance, 1,760 prefill @ 12K.
set -euo pipefail

NAME=llamacpp-nemotron
IMAGE=${IMAGE:-llama.cpp:sycl-f16-next-bb4caa754}     # b10566 — has nemotron_h_moe
MODEL_DIR=/data/llm/nemotron-3.5-lightning-30b-a3b-GGUF
MODEL=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
DRAFT=mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf
PORT=${PORT:-8011}
CTX=${CTX:-131072}   # 262144 was measured and REJECTED: idle 23.38 GiB, and a
                     # 105K-token request took the card to 24,450 of 24,576 MiB
                     # (126 MiB free) with decode collapsing to 20 tok/s.

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=24g --memory-swap=24g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 180s --health-retries 3 \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -v "$MODEL_DIR":/models:ro \
  -p "0.0.0.0:${PORT}:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  -e LLAMA_ARG_HOST=0.0.0.0 \
  "$IMAGE" \
  -m "/models/${MODEL}" \
  --model-draft "/models/${DRAFT}" \
  --spec-type draft-mtp --spec-draft-n-max 7 \
  --alias nemotron-3.5-lightning-30b-a3b \
  -ngl 99 -ngld 99 \
  -c "$CTX" --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 \
  --top-k 20 --min-p 0.0 \
  --host 0.0.0.0 --port 8000 --metrics

echo "$NAME starting on :${PORT} (card 2, ctx ${CTX}, --spec-draft-n-max 7, image ${IMAGE})"
