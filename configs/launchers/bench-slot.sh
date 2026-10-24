#!/usr/bin/env bash
# Generic isolated bench slot on card 2 (level_zero:1), port 8020.
# Deliberately NOT :8010 so Traefik's LB never routes real traffic into a bench model.
#
#   MODEL_DIR=/data/llm/x-GGUF MODEL=x.gguf [DRAFT=mtp-x.gguf] [SPEC_ARGS="..."] \
#   [IMAGE=llama.cpp:sycl-f16-next-bb4caa754] [CTX=131072] [ALIAS=x] bench-slot.sh
set -euo pipefail

NAME=${NAME:-llamacpp-bench}
IMAGE=${IMAGE:-llama.cpp:sycl-f16-next-bb4caa754}
PORT=${PORT:-8020}
CTX=${CTX:-131072}
ALIAS=${ALIAS:-bench}
MEM=${MEM:-24g}
: "${MODEL_DIR:?set MODEL_DIR}"
: "${MODEL:?set MODEL}"

/data/llm/launch/gpu-teardown.sh "$NAME" 30 || true

ARGS=( -m "/models/${MODEL}" --alias "$ALIAS" )
if [[ -n "${DRAFT:-}" ]]; then
  ARGS+=( --model-draft "/models/${DRAFT}" -ngld 99 )
  # shellcheck disable=SC2206
  ARGS+=( ${SPEC_ARGS:---spec-type draft-mtp --spec-draft-n-max 3} )
fi
ARGS+=( -ngl 99 -c "$CTX" --parallel 1
        --host 0.0.0.0 --port 8000 --metrics
        --cache-type-k q8_0 --cache-type-v q8_0
        -fa on -ub 2048 -b 2048
        --jinja --reasoning off
        --predict 2048 --top-k 20 --min-p 0.0 )
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=( ${EXTRA_ARGS} )

docker run -d --name "$NAME" \
  --memory="$MEM" --memory-swap="$MEM" \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 15s --health-timeout 5s --health-start-period 60s --health-retries 40 \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -v "$MODEL_DIR":/models:ro \
  -p "0.0.0.0:${PORT}:8000" \
  -e ZE_AFFINITY_MASK=1 \
  -e GGML_SYCL_MMVQ_MAX_COLS=${MMVQ_MAX_COLS:-8} \
  -e GGML_SYCL_ENABLE_XMX_ESIMD=${XMX_ESIMD:-1} \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e NEO_CACHE_PERSISTENT=1 \
  -e LLAMA_ARG_HOST=0.0.0.0 \
  "$IMAGE" "${ARGS[@]}" >/dev/null

echo "$NAME ($ALIAS) starting on :${PORT} card2 image=$IMAGE"
