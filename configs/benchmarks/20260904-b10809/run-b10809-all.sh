#!/usr/bin/env bash
# Interleaved A/B: for each model, bench old image then new image back to back.
set -uo pipefail
OLD=llama.cpp:sycl-f16-b10742-patched-a82c13531
NEW=llama.cpp:sycl-f16-b10809-patched-f937ca544
for m in ornith e2b gemma26b nemotron; do
  MODELS="$m" IMAGE="$OLD" LABEL=old /data/llm/benchmarks/run-b10809-ab.sh
  MODELS="$m" IMAGE="$NEW" LABEL=new /data/llm/benchmarks/run-b10809-ab.sh
done
echo "ALLDONE"
