#!/usr/bin/env bash
# 1) redo q4_k_xl-v2 (first probe ran before the acceptance metric names were fixed)
# 2) q3_k_xl-v3 WITHOUT the drafter - isolates whether the token[0]=-1 wedge is
#    the MTP head or the model file itself
# 3) q3_k_xl-v3 WITH the drafter again - confirm the wedge reproduces
set -uo pipefail
B=/data/llm/benchmarks/20260826-ud3vs2
IMG=llama.cpp:sycl-f16-allfixes
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "restoring nemotron"; $B/restore-nemotron.sh' EXIT
run_arm(){ # tag dir file draftargs
  local T=$1 D=$2 F=$3 USEDRAFT=$4
  log "=== $T (drafter=$USEDRAFT)"
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1; sleep 3
  local DR="" SA=""
  [ "$USEDRAFT" = yes ] && { DR=mtp-Qwen3.8-27B-Q4_0.gguf; SA="--spec-type draft-mtp --spec-draft-n-max 3"; }
  IMAGE=$IMG MMVQ_MAX_COLS=8 CTX=32768 MEM=24g ALIAS="$T" \
  MODEL_DIR="$D" MODEL="$F" DRAFT="$DR" SPEC_ARGS="$SA" \
    /data/llm/launch/bench-slot.sh || { log "$T LAUNCH FAILED"; return 1; }
  for i in $(seq 1 40); do curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 && break; sleep 10; done
  python3 $B/probe-decode.py --port 8020 --name "$T" --runs 10 --tokens 300 --out "$B/probe-$T.json" > "$B/probe-$T.log" 2>&1
  docker logs llamacpp-bench 2>&1 | tail -60 > "$B/probe-$T.serverlog"
  grep -cE "^  run" "$B/probe-$T.log"; grep -c ERROR "$B/probe-$T.log"
  grep -E "invalid token|Invalid input batch" "$B/probe-$T.serverlog" | head -3
}
log "freeing card 2"
/data/llm/launch/gpu-teardown.sh llamacpp-nemotron 30 >/dev/null 2>&1; sleep 3
rm -f $B/probe-q4_k_xl-v2.json
run_arm q4_k_xl-v2      /data/llm/qwen3.8-27b-unsloth-v2 Qwen3.8-27B-UD-Q4_K_XL.gguf yes
run_arm q3_k_xl-v3-nodraft /data/llm/qwen3.8-27b-unsloth Qwen3.8-27B-UD-Q3_K_XL.gguf no
run_arm q3_k_xl-v3-repro   /data/llm/qwen3.8-27b-unsloth Qwen3.8-27B-UD-Q3_K_XL.gguf yes
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1
log "FINAL PASS COMPLETE"
