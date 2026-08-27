#!/usr/bin/env bash
# Controlled decode pass: ignore_eos so all four arms do identical work per run.
set -uo pipefail
B=/data/llm/benchmarks/20260826-ud3vs2
IMG=llama.cpp:sycl-f16-allfixes
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "restoring nemotron"; $B/restore-nemotron.sh' EXIT
ROWS=(
"q4_k_xl-v2|/data/llm/qwen3.8-27b-unsloth-v2|Qwen3.8-27B-UD-Q4_K_XL.gguf"
"q4_k_xl-v3|/data/llm/qwen3.8-27b-unsloth|Qwen3.8-27B-UD-Q4_K_XL.gguf"
"q3_k_xl-v2|/data/llm/qwen3.8-27b-unsloth-v2|Qwen3.8-27B-UD-Q3_K_XL.gguf"
"q3_k_xl-v3|/data/llm/qwen3.8-27b-unsloth|Qwen3.8-27B-UD-Q3_K_XL.gguf"
)
log "freeing card 2"
/data/llm/launch/gpu-teardown.sh llamacpp-nemotron 30 >/dev/null 2>&1; sleep 3
for R in "${ROWS[@]}"; do
  IFS='|' read -r N D F <<< "$R"
  [ -s "$B/probe-$N.json" ] && { log "probe-$N done, skip"; continue; }
  log "=== probe $N"
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1; sleep 3
  IMAGE=$IMG MMVQ_MAX_COLS=8 CTX=32768 MEM=24g ALIAS="$N" \
  MODEL_DIR="$D" MODEL="$F" DRAFT="mtp-Qwen3.8-27B-Q4_0.gguf" \
  SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" \
    /data/llm/launch/bench-slot.sh || { log "$N LAUNCH FAILED"; continue; }
  for i in $(seq 1 40); do curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 && break; sleep 10; done
  curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 || { log "$N NEVER CAME UP"; continue; }
  python3 $B/probe-decode.py --port 8020 --name "$N" --runs 10 --tokens 300 \
    --out "$B/probe-$N.json" > "$B/probe-$N.log" 2>&1
  docker logs llamacpp-bench 2>&1 | tail -80 > "$B/probe-$N.serverlog"
  tail -14 "$B/probe-$N.log"
done
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1
log "PROBE PASS COMPLETE"
