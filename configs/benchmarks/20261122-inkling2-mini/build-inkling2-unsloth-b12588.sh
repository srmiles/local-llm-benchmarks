#!/usr/bin/env bash
# 2026-11-22: Inkling 2 Mini bench. Unsloth fork b12588-mix-c0c3de8 (source tarball), UNPATCHED.
# Fix: fork Dockerfile omits ui/lib/voice in web stage -> intel-bench.Dockerfile adds it.
# Reads root (Unsloth looped layout) GGUFs + DSpark drafter. Mainline alt: master >= 690b87d88 (#32694+#32704) w/ Mainline_Support files.
set -euo pipefail
T=llama.cpp:sycl-f16-unsloth-b12588-c0c3de8-full
echo "== $(date -u +%FT%TZ) build $T"
cd /data/llm/build/unsloth-b12588/llama.cpp-b12588-mix-c0c3de8
docker build --target full -f .devops/intel-bench.Dockerfile --build-arg GGML_SYCL_F16=ON -t "$T" .
echo "BUILT $T $(date -u +%FT%TZ)"
echo ALLDONE
