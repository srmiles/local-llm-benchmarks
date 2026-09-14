#!/usr/bin/env bash
# Spark-X2.5-4B bench + incumbent anchors on llama.cpp b10867-patched.
#
# Spark-X2.5 needs upstream PR #27868 (merged 2026-09-06, after b10809), so the
# incumbents are re-run on the SAME new image to separate "the model" from
# "the 4-day master bump". No A/B is possible for Spark itself: it cannot load
# on any earlier build.
#
#   ./run-spark-x25.sh              # all four
#   MODELS="spark_q4 spark_q8" ./run-spark-x25.sh
set -uo pipefail
IMAGE=${IMAGE:-llama.cpp:sycl-f16-b10867-patched-b3ce098f7}
LABEL=${LABEL:-b10867}
OUT=/data/llm/benchmarks/20260909-spark-x25
MODELS=${MODELS:-"ornith e2b spark_q4 spark_q8"}
mkdir -p "$OUT"

run_one() {
  local alias=$1; shift
  echo "=== [$LABEL] $alias : $IMAGE ==="
  if ! env IMAGE="$IMAGE" NAME=llamacpp-bench ALIAS="$alias" "$@" /data/llm/launch/bench-slot.sh; then
     echo "!! [$LABEL] $alias failed to launch"; return 1
  fi
  # finding #49: after a B60 engine reset the server stays "healthy" and simply never
  # returns, so bench-candidate.py burns its full 900s timeout on every remaining probe.
  /data/llm/benchmarks/hangwatch.sh 8020 180 5400 "${OUT}/${alias}-${LABEL}.hangwatch" &
  local hw=$!
  ( while kill -0 $hw 2>/dev/null; do sleep 5; done
    wait $hw 2>/dev/null; [[ $? -eq 42 ]] && pkill -f "bench-candidate.py --port 8020" ) &
  local hwk=$!

  python3 /data/llm/benchmarks/bench-candidate.py --port 8020 --card 1 \
      --name "${alias}-${LABEL}" --out "${OUT}/${alias}-${LABEL}.json" \
      2>&1 | tee "${OUT}/${alias}-${LABEL}.log" | tail -45
  local rc=${PIPESTATUS[0]}
  kill $hw $hwk 2>/dev/null; wait $hw $hwk 2>/dev/null
  if [[ $rc -ne 0 ]]; then
    echo "!! [$LABEL] $alias bench rc=$rc — server log tail:"
    docker logs --tail 40 llamacpp-bench 2>&1 | tee "${OUT}/${alias}-${LABEL}.serverlog"
  else
    docker logs --tail 200 llamacpp-bench > "${OUT}/${alias}-${LABEL}.serverlog" 2>&1
  fi
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true
  return $rc
}

for m in $MODELS; do
case $m in
  # --- anchors: identical config to run-b10809-ab.sh, new image ---
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
  # --- candidate: dense 4B, no MTP head published, runs unassisted ---
  spark_q4)
    run_one spark-x2.5-4b-q4_k_m \
      MODEL_DIR=/data/llm/Spark-X2.5-4B-GGUF \
      MODEL=Spark-X2.5-4B-Q4_K_M.gguf \
      CTX=131072 MEM=12g ;;
  spark_q8)
    run_one spark-x2.5-4b-q8_0 \
      MODEL_DIR=/data/llm/Spark-X2.5-4B-GGUF \
      MODEL=Spark-X2.5-4B-Q8_0.gguf \
      CTX=131072 MEM=12g ;;
esac
done
echo "=== [$LABEL] done ==="
