#!/usr/bin/env bash
# Unsloth Dynamic 3.0 vs 2.0 on Qwen3.8-27B, matched quant names.
#
#   v3.0 = unsloth/Qwen3.8-27B-GGUF @ main         (repo re-quantized in place 2026-08-19)
#   v2.0 = same repo @ f1bfb127c64f                (2026-08-15, last commit before the re-quant)
#
# v3.0 is SMALLER at every matched name, so throughput alone would flatter it for
# reasons that have nothing to do with the accuracy claim. Quality is therefore
# measured against a Q8_0 referee whose blob is byte-identical in both revisions
# (the 08-19 re-quant never touched Q8_0), so neither version was tuned to it.
#
# Held constant across all four arms: build (sycl-f16-allfixes), card 2, ctx 32768,
# the same MTP drafter, n-max 3, seed 1234, 120 chunks @ n_ctx 512.
set -uo pipefail
B=/data/llm/benchmarks/20260826-ud3vs2
IMG=llama.cpp:sycl-f16-allfixes
CHUNKS=120
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "restoring nemotron"; $B/restore-nemotron.sh' EXIT

# name|dir|file
ROWS=(
"q4_k_xl-v2|/data/llm/qwen3.8-27b-unsloth-v2|Qwen3.8-27B-UD-Q4_K_XL.gguf"
"q4_k_xl-v3|/data/llm/qwen3.8-27b-unsloth|Qwen3.8-27B-UD-Q4_K_XL.gguf"
"q3_k_xl-v2|/data/llm/qwen3.8-27b-unsloth-v2|Qwen3.8-27B-UD-Q3_K_XL.gguf"
"q3_k_xl-v3|/data/llm/qwen3.8-27b-unsloth|Qwen3.8-27B-UD-Q3_K_XL.gguf"
)

quality(){ # name dir file
  local N=$1 D=$2 F=$3
  [ -s "$B/ppl-$N.log" ] && grep -q "Final estimate" "$B/ppl-$N.log" && { log "ppl-$N done, skip"; return 0; }
  [ -f "$D/$F" ] || { log "MISSING $D/$F"; return 1; }
  log "=== quality $N (PPL + KLD vs Q8_0 referee)"
  /data/llm/launch/gpu-teardown.sh llamacpp-ppl 20 >/dev/null 2>&1
  MODEL="$D/$F" OUT="$B/ppl-$N.log" NGL=99 CHUNKS=$CHUNKS \
    KLDBASE=$B/kld-base.dat IMAGE=$IMG $B/ppl-run.sh
}

speed(){ # name dir file
  local N=$1 D=$2 F=$3
  [ -s "$B/bench-$N.json" ] && { log "bench-$N done, skip"; return 0; }
  [ -f "$D/$F" ] || { log "MISSING $D/$F"; return 1; }
  log "=== throughput $N"
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1; sleep 3
  IMAGE=$IMG MMVQ_MAX_COLS=8 CTX=32768 MEM=24g ALIAS="$N" \
  MODEL_DIR="$D" MODEL="$F" DRAFT="mtp-Qwen3.8-27B-Q4_0.gguf" \
  SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" \
    /data/llm/launch/bench-slot.sh || { log "$N LAUNCH FAILED"; return 1; }
  for i in $(seq 1 40); do
    docker ps --format '{{.Names}}' | grep -qx llamacpp-bench || { log "$N container exited"; docker logs llamacpp-bench 2>&1 | tail -25 > "$B/bench-$N.fail"; return 1; }
    curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 && break; sleep 10
  done
  curl -fsS -m 5 http://localhost:8020/health >/dev/null 2>&1 || {
    log "$N NEVER CAME UP"; docker logs llamacpp-bench 2>&1 | tail -25 > "$B/bench-$N.fail"; return 1; }
  python3 /data/llm/benchmarks/bench-candidate.py --port 8020 --name "$N" --card 1 \
    --decode-runs 20 --temp 0.6 --top-p 0.95 --top-k 20 --out "$B/bench-$N.json" > "$B/bench-$N.log" 2>&1
  python3 -c "
import json;d=json.load(open('$B/bench-$N.json'))
s=d.get('spec') or {};print('   $N dec',d['decode']['median'],'sd',d['decode']['stdev'],'| acc',s.get('acceptance_pct'),'| vram',d['peak_vram_gib'])" 2>&1
}

log "PHASE 1 - quality"
for R in "${ROWS[@]}"; do IFS='|' read -r n d f <<< "$R"; quality "$n" "$d" "$f"; done
/data/llm/launch/gpu-teardown.sh llamacpp-ppl 20 >/dev/null 2>&1
log "PHASE 2 - throughput"
for R in "${ROWS[@]}"; do IFS='|' read -r n d f <<< "$R"; speed "$n" "$d" "$f"; done
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1
log "UD3VS2 COMPLETE"
