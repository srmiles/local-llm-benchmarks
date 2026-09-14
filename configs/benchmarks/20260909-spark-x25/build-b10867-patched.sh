#!/usr/bin/env bash
set -euo pipefail
cd /data/llm/build/llama.cpp
TAG=llama.cpp:sycl-f16-b10867-patched-b3ce098f7
docker build --target server -f .devops/intel.Dockerfile \
  --build-arg GGML_SYCL_F16=ON \
  -t "$TAG" .
echo "BUILT $TAG"
