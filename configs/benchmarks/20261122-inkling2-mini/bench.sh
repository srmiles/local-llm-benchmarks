#!/usr/bin/env bash
# Inkling 2 Mini vs Nemotron 3.5 Lightning, card 2 (L0 idx 1), same unpatched Unsloth b12588 SYCL binary.
set -uo pipefail
IMG=llama.cpp:sycl-f16-unsloth-b12588-c0c3de8-full
OUT=/data/llm/benchmarks/inkling2-mini-20261122
RUN="docker run --rm --device /dev/dri -e ZE_AFFINITY_MASK=1 -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -v /data/llm:/data/llm -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache"
INK=/data/llm/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf
NEM=/data/llm/nemotron-3.5-lightning-30b-a3b-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
B="-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 -p 512,4096 -n 128 -r 3 -o md"
echo "== $(date -u +%FT%TZ) inkling"
$RUN --entrypoint /app/llama-bench $IMG -m $INK $B > $OUT/llama-bench-inkling.md 2> $OUT/llama-bench-inkling.err; echo rc=$?
echo "== $(date -u +%FT%TZ) nemotron"
$RUN --entrypoint /app/llama-bench $IMG -m $NEM $B > $OUT/llama-bench-nemotron.md 2> $OUT/llama-bench-nemotron.err; echo rc=$?
echo "== $(date -u +%FT%TZ) DONE"
