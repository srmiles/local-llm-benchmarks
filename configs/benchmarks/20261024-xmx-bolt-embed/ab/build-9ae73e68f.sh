#!/usr/bin/env bash
# 2026-10-24: rebase of local SYCL series onto master 9ae73e68f (has #29864 XMX multi-col MMVQ, #30775 q8 fast dequant).
# Dropped 749b075e9 (MMVQ chunk) + b7d8ce69b (reorder gate on mmvq_max_cols): both inert at prod MMVQ_MAX_COLS=8, conflict with #29864's gate.
set -euo pipefail
P=llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e
echo "== $(date -u +%FT%TZ) build patched $P"
cd /data/llm/build/llama.cpp
docker build --target server -f .devops/intel.Dockerfile --build-arg GGML_SYCL_F16=ON -t "$P" .
echo "BUILT $P $(date -u +%FT%TZ)"
echo ALLDONE
