#!/usr/bin/env bash
# 2026-10-31: rebase of local SYCL series onto master 5047748f1 (tag b12017).
# Brings #31125 DPAS FA on quantized KV, 5320c87 chunked reorder, 5cd62ff MMVQ templating, SYCL graphs (#28725).
# Our Q4_0/Q3_K/Q2_K MoE reorder kernels ported to upstream's 32 MiB chunked form. Includes ba6c3f9fc OUT_PROD guard (now c9bf2e9a6).
set -euo pipefail
P=llama.cpp:sycl-f16-5047748f1-patched-c9bf2e9a6
C=llama.cpp:sycl-f16-5047748f1-clean
echo "== $(date -u +%FT%TZ) build patched $P"
cd /data/llm/build/llama.cpp
docker build --target server -f .devops/intel.Dockerfile --build-arg GGML_SYCL_F16=ON -t "$P" .
echo "BUILT $P $(date -u +%FT%TZ)"
echo "== build clean $C"
cd /data/llm/build/llama.cpp-master-clean
docker build --target server -f .devops/intel.Dockerfile --build-arg GGML_SYCL_F16=ON -t "$C" .
echo "BUILT $C $(date -u +%FT%TZ)"
echo ALLDONE
