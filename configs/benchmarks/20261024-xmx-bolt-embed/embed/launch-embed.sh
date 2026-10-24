#!/usr/bin/env bash
# usage: launch-embed.sh <model_dir> <model_file> [extra llama-server args...]
# Mirrors prod start-llamacpp-embed.sh flags, but on card 2 (GPU1) at :8020 (never :8010).
set -euo pipefail
MODEL_DIR=$1; MODEL=$2; shift 2
NAME=llamacpp-embbench; PORT=${PORT:-8020}
IMAGE=${IMAGE:-llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e}
/data/llm/launch/gpu-teardown.sh "$NAME" 30 || true
docker run -d --name "$NAME" --device /dev/dri --memory=${MEM:-6g} --memory-swap=${MEM:-6g} \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  -v "$MODEL_DIR":/models:ro -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -p "0.0.0.0:${PORT}:8000" \
  -e ZES_ENABLE_SYSMAN=1 -e ZE_AFFINITY_MASK=1 -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e NEO_CACHE_PERSISTENT=1 \
  "$IMAGE" -m "/models/$MODEL" --alias bench -ngl 99 -c "${CTX:-8192}" --parallel "${PAR:-4}" --cache-ram 0 \
  --host 0.0.0.0 --port 8000 --embeddings -b "${UB:-2048}" -ub "${UB:-2048}" --load-mode none --metrics "$@" >/dev/null
for i in $(seq 1 90); do
  curl -s -m 2 "http://localhost:$PORT/health" 2>/dev/null | grep -q ok && { echo "READY $MODEL after $((i*2))s"; exit 0; }
  docker ps -q -f name=$NAME | grep -q . || { echo "CONTAINER EXITED"; docker logs --tail 30 $NAME 2>&1; exit 1; }
  sleep 2
done
echo "TIMEOUT"; docker logs --tail 30 $NAME 2>&1; exit 1
