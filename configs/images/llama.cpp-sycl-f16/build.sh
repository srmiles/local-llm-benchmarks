#!/usr/bin/env bash
# Build llama.cpp:sycl-f16 for Intel Arc Pro B60 (Battlemage).
# Runs on llm.local against /data/llm/build/llama.cpp checkout.
#
# Usage:   ./build.sh <tag>
# Example: ./build.sh b10068
#
# Runs the build in the background via systemd-run so it survives SSH.
# Poll progress with:  tail -f /tmp/llama-build.log | grep -E '^#[0-9]+ (\[|DONE)'
set -euo pipefail

TAG="${1:?usage: $0 <llama.cpp-tag>   e.g. $0 b10068}"
CHECKOUT="/data/llm/build/llama.cpp"
IMAGE="llama.cpp:sycl-f16-${TAG}"
LOGFILE="/tmp/llama-build.log"

echo "=== fetch + checkout ${TAG} ==="
cd "$CHECKOUT"
git fetch origin --tags --prune
git checkout "$TAG"
git log -1 --oneline

echo ""
echo "=== verify hot commits are in the tree ==="
missing=0
for sha in 32b741c c1063ac efb3036 d3fba0c 956973c; do
  if git log --oneline HEAD | grep -q "^$sha"; then
    echo "  ✓ $sha"
  else
    echo "  ✗ $sha MISSING"
    missing=1
  fi
done
if [ "$missing" -eq 1 ]; then
  echo ""
  echo "WARNING: some target commits are missing from ${TAG}. Continuing anyway."
fi

echo ""
echo "=== launch background build → ${IMAGE} ==="
sudo systemctl reset-failed llama-build 2>/dev/null || true
sudo systemd-run --unit=llama-build \
  --property=StandardOutput=file:"$LOGFILE" \
  --property=StandardError=append:"$LOGFILE" \
  --property=WorkingDirectory="$CHECKOUT" \
  /usr/bin/docker build \
    --build-arg GGML_SYCL_F16=ON \
    --target server \
    -f .devops/intel.Dockerfile \
    -t "$IMAGE" \
    .

echo ""
echo "Build running as systemd unit 'llama-build'."
echo "Log: $LOGFILE"
echo ""
echo "Watch progress:"
echo "  tail -f $LOGFILE | grep -E '^#[0-9]+ (\\[|DONE)'"
echo ""
echo "Check status:"
echo "  systemctl is-active llama-build"
echo ""
echo "When done, promote to prod:"
echo "  docker tag llama.cpp:sycl-f16 llama.cpp:sycl-f16-prev"
echo "  docker tag $IMAGE llama.cpp:sycl-f16"
echo "  sudo /data/llm/launch/start-llamacpp-sycl-ornith.sh"
echo "  sudo /data/llm/launch/start-llamacpp-embed.sh"
echo "  sudo /data/llm/launch/start-llamacpp-rerank.sh"
