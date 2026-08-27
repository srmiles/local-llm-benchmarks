#!/usr/bin/env bash
# Serialise the whole run: wait for the KLD base, reclaim the referee's disk,
# fetch the v2.0 arm, then bench. Disk is the binding constraint (102 GB free at
# start vs 91 GB of models), so Q8_0 is deleted the moment its logits are on disk.
set -uo pipefail
B=/data/llm/benchmarks/20260826-ud3vs2
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "waiting for KLD base gen"
while pgrep -f "llamacpp-ppl" >/dev/null 2>&1 || docker ps --format '{{.Names}}' | grep -qx llamacpp-ppl; do sleep 20; done
grep -q "Final estimate" $B/kld-base.log || { log "BASE GEN FAILED - aborting"; tail -20 $B/kld-base.log; exit 1; }
log "base ready: $(ls -la $B/kld-base.dat | awk '{print $5}') bytes"

log "waiting for v3.0 fetch"
while ! grep -q DONE $B/fetch-v3.log; do sleep 20; done

log "deleting Q8_0 referee (logits captured; blob is re-downloadable)"
rm -f /data/llm/qwen3.8-27b-unsloth/Qwen3.8-27B-Q8_0.gguf
rm -rf /data/llm/qwen3.8-27b-unsloth/.cache
df -h / | tail -1

log "fetching v2.0 arm"
$B/fetch-quants.sh v2
mkdir -p /data/llm/qwen3.8-27b-unsloth-v2
# same drafter for both arms - hardlink, same filesystem, no extra disk
ln -f /data/llm/qwen3.8-27b-unsloth/MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
      /data/llm/qwen3.8-27b-unsloth-v2/mtp-Qwen3.8-27B-Q4_0.gguf 2>/dev/null
ln -f /data/llm/qwen3.8-27b-unsloth/MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
      /data/llm/qwen3.8-27b-unsloth/mtp-Qwen3.8-27B-Q4_0.gguf 2>/dev/null
ls -la /data/llm/qwen3.8-27b-unsloth/*.gguf /data/llm/qwen3.8-27b-unsloth-v2/*.gguf

log "starting bench"
$B/run-ud3vs2.sh
log "CHAIN COMPLETE"
