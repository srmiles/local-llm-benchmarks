#!/usr/bin/env bash
# Matched-pair fetch: same filenames, two revisions of unsloth/Qwen3.8-27B-GGUF.
#   main          = Dynamic 3.0 (repo re-quantized in place 2026-08-19)
#   f1bfb127c64f  = Dynamic 2.0 (last commit before the re-quant, 2026-08-15)
# Filenames collide, so v2.0 lands in a separate directory.
set -uo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
R=unsloth/Qwen3.8-27B-GGUF
V2REV=f1bfb127c64f
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
ARM=${1:-all}
if [ "$ARM" = v3 ] || [ "$ARM" = all ]; then
  for F in Qwen3.8-27B-UD-Q4_K_XL.gguf Qwen3.8-27B-UD-Q3_K_XL.gguf; do
    log "v3.0 $F"; hf download "$R" "$F" --local-dir /data/llm/qwen3.8-27b-unsloth || log "FAIL v3 $F"
  done
fi
if [ "$ARM" = v2 ] || [ "$ARM" = all ]; then
  for F in Qwen3.8-27B-UD-Q4_K_XL.gguf Qwen3.8-27B-UD-Q3_K_XL.gguf; do
    log "v2.0 $F @ $V2REV"
    hf download "$R" "$F" --revision "$V2REV" --local-dir /data/llm/qwen3.8-27b-unsloth-v2 || log "FAIL v2 $F"
  done
fi
log "DONE"; df -h / | tail -1
