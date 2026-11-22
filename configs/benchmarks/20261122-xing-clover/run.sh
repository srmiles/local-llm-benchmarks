#!/usr/bin/env bash
# 2026-11-22: Xing4.0-29B-A4B + Clover-35B-A3B-Preview (bartowski Q4_0) on card 2, vs Nemotron control.
# Same binary for all: Unsloth fork b12588-mix-c0c3de8, unpatched (server + full targets).
set -uo pipefail
OUT=/data/llm/benchmarks/xing-clover-20261122
IMG=llama.cpp:sycl-f16-unsloth-b12588-c0c3de8
FULL=llama.cpp:sycl-f16-unsloth-b12588-c0c3de8-full
L=/data/llm/repos/local-llm-benchmarks/configs/launchers
BC=/data/llm/benchmarks/bench-candidate.py
XD=/data/llm/xing4.0-29b-a4b-GGUF;  XM=XingChen-AGI_Xing4.0-29B-A4B-Q4_0.gguf
CD=/data/llm/clover-35b-a3b-GGUF;   CM=Diffuspace_Clover-35B-A3B-Preview-Q4_0.gguf
ND=/data/llm/nemotron-3.5-lightning-30b-a3b-GGUF; NM=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf; NDR=mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf
SPLIT_GATE=150
log(){ echo "== $(date -u +%FT%TZ) $*"; }
restore(){ log "teardown + restore nemotron"; /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1 || docker rm -f llamacpp-bench >/dev/null 2>&1
  docker start llamacpp-nemotron >/dev/null; for i in $(seq 1 40); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8011/health)" = 200 ] && { log "nemotron health 200"; return; }; sleep 5; done; log "!! nemotron NOT healthy"; }
trap restore EXIT

log "waiting for downloads"; while systemctl --user is-active -q dl-xing || systemctl --user is-active -q dl-clover; do sleep 10; done
ls -l $XD/$XM $CD/$CM || exit 1
DEV=$(xpu-smi discovery 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && /Device Name/ {id=$1} /PCI BDF Address: 0000:0b:00.0/{print id; exit}'); log "card2 xpu-smi dev=$DEV"
docker stop llamacpp-nemotron >/dev/null; log "STOPPED nemotron"
$L/card2-empty.sh 120 || exit 1

slot(){ # name ctx env...
  local name=$1 ctx=$2; shift 2
  log "slot $name ctx=$ctx"
  env "$@" IMAGE=$IMG NAME=llamacpp-bench CTX=$ctx ALIAS=$name $L/bench-slot.sh
  local st=""; for i in $(seq 1 60); do st=$(docker inspect -f '{{.State.Health.Status}}' llamacpp-bench 2>/dev/null || echo gone); [ "$st" = healthy ] && break; [ "$st" = gone ] && break; docker inspect -f '{{.State.Running}}' llamacpp-bench 2>/dev/null | grep -q false && { st=exited; break; }; sleep 10; done
  docker logs llamacpp-bench > $OUT/$name.server.log 2>&1
  log "$name health=$st"; [ "$st" = healthy ] || return 1
  local sp=$(grep -oE 'splits = [0-9]+' $OUT/$name.server.log | head -1 | grep -oE '[0-9]+'); grep -E 'graph: nodes' $OUT/$name.server.log | head -2 > $OUT/$name.splits; log "$name splits=${sp:-?}"
  if [ -n "$sp" ] && [ "$sp" -gt $SPLIT_GATE ]; then log "$name SPLIT GATE (>$SPLIT_GATE): skip harness"; return 2; fi
  timeout 1800 python3 $BC --port 8020 --name $name --card $DEV --out $OUT/$name.json > $OUT/$name.log 2>&1; log "$name harness rc=$?"
  docker logs llamacpp-bench > $OUT/$name.server.log 2>&1
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1 || docker rm -f llamacpp-bench >/dev/null; $L/card2-empty.sh 120 >/dev/null || true
}
fallback(){ local name=$1; shift; slot "$name" 131072 "$@"; local rc=$?; if [ $rc = 1 ]; then log "$name retry ctx 65536"; docker rm -f llamacpp-bench >/dev/null 2>&1; $L/card2-empty.sh 120 >/dev/null; slot "$name-c64k" 65536 "$@"; fi; docker rm -f llamacpp-bench >/dev/null 2>&1; $L/card2-empty.sh 120 >/dev/null || true; }

fallback xing4.0-29b-a4b-solo   MODEL_DIR=$XD MODEL=$XM
fallback xing4.0-29b-a4b-mtp3   MODEL_DIR=$XD MODEL=$XM SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3"
fallback clover-35b-a3b-solo    MODEL_DIR=$CD MODEL=$CM
fallback nemotron-control-mtp7  MODEL_DIR=$ND MODEL=$NM DRAFT=$NDR SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-p-min 0.6"

R="docker run --rm --device /dev/dri -e ZE_AFFINITY_MASK=1 -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -v /data/llm:/data/llm -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache --entrypoint /app/llama-bench $FULL"
B="-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 -p 512,4096 -n 128 -r 3 -o md"
for m in "$XD/$XM xing" "$CD/$CM clover"; do set -- $m; log "llama-bench $2"; timeout 900 $R -m $1 $B > $OUT/llama-bench-$2.md 2> $OUT/llama-bench-$2.err; log "llama-bench $2 rc=$?"; done
log DONE
