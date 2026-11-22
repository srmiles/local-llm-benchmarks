#!/usr/bin/env bash
# 2026-11-22 retry window: Clover-35B-A3B-Preview Q4_0 only, after the 01:47:55Z CCS reset in the Xing window.
# Guards: split check before harness (#69); watcher aborts on new ccs reset in dmesg or n_decode_total frozen 60 s with a request in flight (#66b).
set -uo pipefail
OUT=/data/llm/benchmarks/xing-clover-20261122
IMG=llama.cpp:sycl-f16-unsloth-b12588-c0c3de8; FULL=$IMG-full
L=/data/llm/repos/local-llm-benchmarks/configs/launchers; BC=/data/llm/benchmarks/bench-candidate.py
CD=/data/llm/clover-35b-a3b-GGUF; CM=Diffuspace_Clover-35B-A3B-Preview-Q4_0.gguf; N=clover-35b-a3b-solo
log(){ echo "== $(date -u +%FT%TZ) $*"; }
restore(){ log "teardown + restore nemotron"; [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; [ -n "${HPID:-}" ] && kill $HPID 2>/dev/null
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1; docker rm -f llamacpp-bench >/dev/null 2>&1
  docker start llamacpp-nemotron >/dev/null; for i in $(seq 1 40); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8011/health)" = 200 ] && { log "nemotron health 200"; return; }; sleep 5; done; log "!! nemotron NOT healthy"; }
trap restore EXIT
RESETS0=$(sudo -n dmesg | grep -c 'engine_class=ccs')
DEV=$(xpu-smi discovery 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && /Device Name/ {id=$1} /PCI BDF Address: 0000:0b:00.0/{print id; exit}')
docker stop llamacpp-nemotron >/dev/null; log "STOPPED nemotron"; $L/card2-empty.sh 120 || exit 1

log "split check"
timeout 150 docker run --rm --device /dev/dri -e ZE_AFFINITY_MASK=1 -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -v /data/llm:/data/llm -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache --entrypoint /app/llama-completion $FULL -m $CD/$CM -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 4096 -n 4 --temp 0 -no-cnv -lv 4 -p "Hello" > $OUT/$N.splitcheck.out 2> $OUT/$N.splitcheck.err
SP=$(grep -oE 'splits = [0-9]+' $OUT/$N.splitcheck.err | head -1 | grep -oE '[0-9]+'); grep -E 'graph: nodes|n_layer  ' $OUT/$N.splitcheck.err | head -2; log "splits=${SP:-?}"
[ -n "$SP" ] && [ "$SP" -gt 150 ] && { log "SPLIT GATE: skip harness"; exit 0; }
$L/card2-empty.sh 120 || exit 1

start(){ env MODEL_DIR=$CD MODEL=$CM IMAGE=$IMG NAME=llamacpp-bench CTX=$1 ALIAS=$N $L/bench-slot.sh
  for i in $(seq 1 60); do st=$(docker inspect -f '{{.State.Health.Status}}/{{.State.Running}}' llamacpp-bench 2>/dev/null || echo gone); case $st in healthy/*) return 0;; */false|gone) return 1;; esac; sleep 10; done; return 1; }
CTX=131072; start $CTX || { docker logs llamacpp-bench > $OUT/$N.server-c128k-fail.log 2>&1; log "128K failed, retry 64K"; docker rm -f llamacpp-bench >/dev/null; $L/card2-empty.sh 120 >/dev/null; CTX=65536; start $CTX || { log "64K failed"; docker logs llamacpp-bench > $OUT/$N.server.log 2>&1; exit 1; }; }
log "healthy ctx=$CTX"

watch(){ local last=-1 since=$(date +%s)
  while sleep 5; do
    if [ "$(sudo -n dmesg | grep -c 'engine_class=ccs')" -gt "$RESETS0" ]; then log "!! WATCHER: new ccs reset"; kill $HPID 2>/dev/null; return; fi
    m=$(curl -s -m 4 localhost:8020/metrics); d=$(echo "$m" | awk '/^llamacpp:n_decode_total/{print $2}'); r=$(echo "$m" | awk '/^llamacpp:requests_processing/{print int($2)}')
    if [ "${r:-0}" -gt 0 ] && [ "$d" = "$last" ]; then [ $(( $(date +%s) - since )) -ge 60 ] && { log "!! WATCHER: n_decode_total frozen 60s"; kill $HPID 2>/dev/null; return; }; else last=$d; since=$(date +%s); fi
  done; }
python3 $BC --port 8020 --name $N --card $DEV --out $OUT/$N.json > $OUT/$N.log 2>&1 & HPID=$!
watch & WPID=$!
wait $HPID; log "harness rc=$?"; kill $WPID 2>/dev/null
docker logs llamacpp-bench > $OUT/$N.server.log 2>&1
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1; docker rm -f llamacpp-bench >/dev/null 2>&1; $L/card2-empty.sh 120 >/dev/null
log "llama-bench"; timeout 900 docker run --rm --device /dev/dri -e ZE_AFFINITY_MASK=1 -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -v /data/llm:/data/llm -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache --entrypoint /app/llama-bench $FULL -m $CD/$CM -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 -p 512,4096 -n 128 -r 3 -o md > $OUT/llama-bench-clover.md 2> $OUT/llama-bench-clover.err; log "llama-bench rc=$?"
[ "$(sudo -n dmesg | grep -c 'engine_class=ccs')" -gt "$RESETS0" ] && log "!! ccs reset(s) during window"
log DONE
