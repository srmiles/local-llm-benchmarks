#!/usr/bin/env bash
set -euo pipefail
cd /data/llm/build/llama.cpp
TAG=llama.cpp:sycl-f16-b10809-patched-f937ca544
docker build --target server -f .devops/intel.Dockerfile \
  --build-arg GGML_SYCL_F16=ON \
  -t "$TAG" .
echo "BUILT $TAG"
