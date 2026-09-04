#!/usr/bin/env bash
# A/B bench driver: same four models, one image per invocation.
#   IMAGE=llama.cpp:sycl-f16-b10742-patched-a82c13531 LABEL=old ./run-b10809-ab.sh
#   IMAGE=llama.cpp:sycl-f16-b10809-patched-f937ca544 LABEL=new ./run-b10809-ab.sh
# Runs on card 2 (ZE_AFFINITY_MASK=1) via bench-slot.sh on :8020. Prod on card 1 untouched.
set -uo pipefail
: "${IMAGE:?set IMAGE}"; : "${LABEL:?set LABEL}"
OUT=/data/llm/benchmarks/20260904-b10809
MODELS=${MODELS:-"ornith e2b gemma26b nemotron"}
mkdir -p "$OUT"

run_one() {
  local alias=$1; shift
  echo "=== [$LABEL] $alias : $IMAGE ==="
  if ! env IMAGE="$IMAGE" NAME=llamacpp-bench ALIAS="$alias" "$@" /data/llm/launch/bench-slot.sh; then
     echo "!! [$LABEL] $alias failed to launch"; return 1
  fi
  python3 /data/llm/benchmarks/bench-candidate.py --port 8020 --card 1 \
      --name "${alias}-${LABEL}" --out "${OUT}/${alias}-${LABEL}.json" \
      2>&1 | tail -45
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "!! [$LABEL] $alias bench rc=$rc — server log tail:"; docker logs --tail 25 llamacpp-bench 2>&1
  fi
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true
  return $rc
}

for m in $MODELS; do
case $m in
  ornith)
    run_one ornith-1.5-9b \
      MODEL_DIR=/data/llm/Ornith-1.5-9B-GGUF \
      MODEL=Ornith-1.5-9B-Q4_K_M.gguf \
      DRAFT=mtp-Ornith-1.5-9B-head-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" \
      CTX=131072 MEM=14g ;;
  e2b)
    run_one gemma-4-E2B \
      MODEL_DIR=/data/llm/benchmarks/bench-models/gemma-4-E2B \
      MODEL=gemma-4-E2B_q4_0-it.gguf \
      DRAFT=gemma-4-E2B-it-assistant-official.bf16.gguf \
      SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" \
      CTX=131072 MEM=12g MMVQ_MAX_COLS=64 ;;
  gemma26b)
    run_one gemma-4-26B-A4B \
      MODEL_DIR=/data/llm/benchmarks/bench-models/gemma-4-26B \
      MODEL=gemma-4-26B-A4B-it-Q4_K_M.gguf \
      DRAFT=gemma-4-26B-A4B-it-qat-assistant-MTP-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 5" \
      CTX=131072 MEM=22g ;;
  nemotron)
    run_one nemotron-3.5-lightning \
      MODEL_DIR=/data/llm/nemotron-3.5-lightning-30b-a3b-GGUF \
      MODEL=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
      DRAFT=mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-p-min 0.6" \
      CTX=131072 MEM=24g ;;
esac
done
echo "=== [$LABEL] done ==="
