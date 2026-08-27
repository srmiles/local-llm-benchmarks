#!/usr/bin/env bash
# One perplexity/KLD run in a throwaway container on card 2 (level_zero:1).
#   MODEL=/abs/path.gguf OUT=/abs/out.log [NGL=99] [CHUNKS=40] [KLDBASE=/abs/base.dat] [MAKEBASE=1]
set -uo pipefail
: "${MODEL:?}"; : "${OUT:?}"
IMAGE=${IMAGE:-llama.cpp:sycl-f16-allfixes}
NGL=${NGL:-99}; CHUNKS=${CHUNKS:-40}; CTX=${CTX:-512}
MDIR=$(dirname "$MODEL"); MFILE=$(basename "$MODEL")
BDIR=/data/llm/benchmarks/20260826-ud3vs2
ARGS=( -m "/models/$MFILE" -f /bench/wiki.test.raw -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" --seed 1234 )
if [ "${MAKEBASE:-0}" = 1 ]; then
  ARGS+=( --kl-divergence-base "/bench/$(basename "${KLDBASE:?}")" )
elif [ -n "${KLDBASE:-}" ]; then
  ARGS+=( --kl-divergence --kl-divergence-base "/bench/$(basename "$KLDBASE")" )
fi
docker run --rm --name llamacpp-ppl \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -v "$MDIR":/models:ro -v "$BDIR":/bench \
  -e ZE_AFFINITY_MASK=1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e NEO_CACHE_PERSISTENT=1 \
  --entrypoint /app/llama "$IMAGE" perplexity "${ARGS[@]}" > "$OUT" 2>&1
echo "exit $? -> $OUT"
