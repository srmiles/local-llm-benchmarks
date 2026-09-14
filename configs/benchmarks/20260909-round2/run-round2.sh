#!/usr/bin/env bash
# Round 2, 2026-09-09, on b10867-patched, isolated card 2.
#
#   A. Spark-X2.5-4B + Spark-X2.5-1.7B as a CLASSIC draft model (--spec-type
#      draft-simple). Spark has no MTP head and none can be converted (290
#      tensors, no spare layers), but the 1.7B sibling is vocab-identical
#      (131,072, same spark2_5 arch, same head_dim/sliding_window), so the
#      draft-simple path is open. 2.4x size ratio is poor for speculative
#      decoding - this measures whether that matters on Battlemage.
#   B. MiniCPM5-2B, unassisted and with the official DSpark drafter at two
#      quants. Categorise-slot candidate against Gemma 4 E2B (188.06 tok/s,
#      3.96 GiB on this same image).
set -uo pipefail
IMAGE=${IMAGE:-llama.cpp:sycl-f16-b10867-patched-b3ce098f7}
LABEL=${LABEL:-b10867}
OUT=/data/llm/benchmarks/20260909-round2
MODELS=${MODELS:-"spark_draft3 spark_draft5 mcp_solo mcp_dspark_q8 mcp_dspark_f16"}
mkdir -p "$OUT"

run_one() {
  local alias=$1; shift
  echo "=== [$LABEL] $alias : $IMAGE ==="
  if ! env IMAGE="$IMAGE" NAME=llamacpp-bench ALIAS="$alias" "$@" /data/llm/launch/bench-slot.sh; then
     echo "!! [$LABEL] $alias failed to launch"; return 1
  fi
  /data/llm/benchmarks/hangwatch.sh 8020 180 5400 "${OUT}/${alias}.hangwatch" &
  local hw=$!
  python3 /data/llm/benchmarks/bench-candidate.py --port 8020 --card 1 \
      --name "${alias}" --out "${OUT}/${alias}.json" \
      2>&1 | tee "${OUT}/${alias}.log" | tail -45
  local rc=${PIPESTATUS[0]}
  kill $hw 2>/dev/null; wait $hw 2>/dev/null
  [[ $rc -ne 0 ]] && { echo "!! rc=$rc"; docker logs --tail 40 llamacpp-bench 2>&1; }
  docker logs --tail 200 llamacpp-bench > "${OUT}/${alias}.serverlog" 2>&1
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true
  return $rc
}

for m in $MODELS; do
case $m in
  spark_draft3)
    run_one spark-4b-draft1.7b-nmax3 \
      MODEL_DIR=/data/llm/Spark-X2.5-4B-GGUF \
      MODEL=Spark-X2.5-4B-Q4_K_M.gguf \
      DRAFT=Spark-X2.5-1.7B-Q4_K_M.gguf \
      SPEC_ARGS="--spec-type draft-simple --spec-draft-n-max 3" \
      CTX=131072 MEM=14g ;;
  spark_draft5)
    run_one spark-4b-draft1.7b-nmax5 \
      MODEL_DIR=/data/llm/Spark-X2.5-4B-GGUF \
      MODEL=Spark-X2.5-4B-Q4_K_M.gguf \
      DRAFT=Spark-X2.5-1.7B-Q4_K_M.gguf \
      SPEC_ARGS="--spec-type draft-simple --spec-draft-n-max 5 --spec-draft-p-min 0.6" \
      CTX=131072 MEM=14g ;;
  mcp_solo)
    run_one minicpm5-2b-solo \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      CTX=131072 MEM=12g ;;
  mcp_dspark_q8)
    run_one minicpm5-2b-dspark-q8 \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 3" \
      CTX=131072 MEM=12g ;;
  mcp_dspark_f16)
    run_one minicpm5-2b-dspark-f16 \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-F16.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 3" \
      CTX=131072 MEM=12g ;;
  mcp_nmax5)
    run_one minicpm5-2b-dspark-q8-nmax5 \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 5" \
      CTX=131072 MEM=12g ;;
  mcp_nmax7)
    run_one minicpm5-2b-dspark-q8-nmax7 \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min 0.6" \
      CTX=131072 MEM=12g ;;
  mcp_nmax9)
    run_one minicpm5-2b-dspark-q8-nmax9 \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 9 --spec-draft-p-min 0.6" \
      CTX=131072 MEM=12g ;;
  mcp_nmax11)
    run_one minicpm5-2b-dspark-q8-nmax11 \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 11 --spec-draft-p-min 0.6" \
      CTX=131072 MEM=12g ;;
  mcp_nmax9_nopmin)
    run_one minicpm5-2b-dspark-q8-nmax9-nopmin \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 9" \
      CTX=131072 MEM=12g ;;
  mcp_nmax7_nopmin)
    run_one minicpm5-2b-dspark-q8-nmax7-nopmin \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q4_K_M.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 7" \
      CTX=131072 MEM=12g ;;
  mcp_q8)
    run_one minicpm5-2b-q8-dspark \
      MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF \
      MODEL=MiniCPM5-2B-Q8_0.gguf \
      DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
      SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 3" \
      CTX=131072 MEM=12g ;;
esac
done
echo "=== round2 done ==="
