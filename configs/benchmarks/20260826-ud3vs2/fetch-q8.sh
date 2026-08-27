#!/usr/bin/env bash
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
log "Q8_0 referee (29.05 GB) - identical blob in both revisions"
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf \
  --local-dir /data/llm/qwen3.8-27b-unsloth || log "FAIL q8"
log "MTP head (constant drafter for all throughput arms)"
hf download unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  --local-dir /data/llm/qwen3.8-27b-unsloth || log "FAIL mtp"
log "DONE"; df -h / | tail -1
