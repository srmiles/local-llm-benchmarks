#!/usr/bin/env bash
# Reproduce the q3_k_xl-v3 500s with the server log kept.
set -uo pipefail
B=/data/llm/benchmarks/20260826-ud3vs2
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
log "freeing card 2"
/data/llm/launch/gpu-teardown.sh llamacpp-nemotron 30 >/dev/null 2>&1; sleep 3
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1; sleep 2
IMAGE=llama.cpp:sycl-f16-allfixes MMVQ_MAX_COLS=8 CTX=32768 MEM=24g ALIAS=q3v3diag \
MODEL_DIR=/data/llm/qwen3.8-27b-unsloth MODEL=Qwen3.8-27B-UD-Q3_K_XL.gguf \
DRAFT=mtp-Qwen3.8-27B-Q4_0.gguf SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" \
  /data/llm/launch/bench-slot.sh || { log "LAUNCH FAILED"; exit 1; }
for i in $(seq 1 40); do curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 && break; sleep 10; done
log "up; firing 4 sampled completions"
for i in 1 2 3 4; do
  code=$(curl -s -o "$B/diag-resp-$i.json" -w '%{http_code}' -m 300 \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"[run $i] Write a detailed technical explanation.\",\"n_predict\":300,\"ignore_eos\":true,\"cache_prompt\":false,\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}" \
    http://localhost:8020/completion)
  log "run $i -> HTTP $code"
  [ "$code" != 200 ] && head -c 400 "$B/diag-resp-$i.json" && echo
done
log "=== server log tail ==="
docker logs llamacpp-bench 2>&1 | tail -60
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1
$B/restore-nemotron.sh
log "DIAG DONE"
