#!/usr/bin/env bash
# Settle the Q4_K_XL v2.0-vs-v3.0 decode gap with a same-session interleaved A/B.
#
# The first answer was a CROSS-PASS comparison: v2.0 was re-run ~17 min after v3.0
# (to recover acceptance counters lost to wrong metric names) and gave 19.68 against
# the first pass's 18.82, while v3.0's 16.95 came from the first pass only. Findings
# #20/#33/#40 all say that is not resolvable below ~10%.
#
# A-B-B-A block order so linear drift over the run cancels rather than loading onto
# one arm. Two independent measurements per arm; report the spread, not just a median.
set -uo pipefail
B=/data/llm/benchmarks/20260826-ud3vs2
IMG=llama.cpp:sycl-f16-allfixes
V2=/data/llm/qwen3.8-27b-unsloth-v2
V3=/data/llm/qwen3.8-27b-unsloth
F=Qwen3.8-27B-UD-Q4_K_XL.gguf
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "restoring nemotron"; $B/restore-nemotron.sh' EXIT

block(){ # tag dir
  local T=$1 D=$2
  log "=== block $T"
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1; sleep 3
  IMAGE=$IMG MMVQ_MAX_COLS=8 CTX=32768 MEM=24g ALIAS="$T" \
  MODEL_DIR="$D" MODEL="$F" DRAFT="mtp-Qwen3.8-27B-Q4_0.gguf" \
  SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" \
    /data/llm/launch/bench-slot.sh || { log "$T LAUNCH FAILED"; return 1; }
  for i in $(seq 1 40); do curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 && break; sleep 10; done
  curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 || { log "$T NEVER CAME UP"; return 1; }
  python3 $B/probe-decode.py --port 8020 --name "$T" --runs 8 --tokens 300 \
    --out "$B/il-$T.json" > "$B/il-$T.log" 2>&1
  python3 -c "
import json;d=json.load(open('$B/il-$T.json'))
print('   $T', d['decode']['median'],'sd',d['decode']['stdev'],'acc',d['spec']['acceptance_pct'],'ok',d['ok_runs'])" 2>&1
}

log "freeing card 2"
/data/llm/launch/gpu-teardown.sh llamacpp-nemotron 30 >/dev/null 2>&1; sleep 3
block v2a "$V2"; block v3a "$V3"; block v3b "$V3"; block v2b "$V2"
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1
log "INTERLEAVED COMPLETE"
