#!/usr/bin/env bash
# Qwen 3.8-27B (dense 27B hybrid Mamba+Attention, native MTP, vision-language)
# on Intel Arc Pro B60 card #2 (level_zero:1), port :8010
#
# Reference: https://huggingface.co/Qwen/Qwen3.8-27B
# GGUFs from ggml-org/Qwen3.8-27B-GGUF (llama.cpp team's official conversion)
#
# Layout (Q4_K_M + Q4_0 MTP + Q8_0 mmproj = ~22.3 GiB VRAM):
#   /models/Qwen3.8-27B-Q4_K_M.gguf         19.0 GB  base
#   /models/mtp-Qwen3.8-27B-Q4_0.gguf        1.7 GB  MTP drafter
#   /models/mmproj-Qwen3.8-27B-Q8_0.gguf     629 MB  vision projector (image+video)
#
# Card 2 pin: ONEAPI_DEVICE_SELECTOR=level_zero:1 (card 1 services stay on level_zero:0)
# Port 8010 (NOT 8003 — headroom-proxy claims :8003 after reboot)
# Sampling per Qwen 3.8 model card (thinking mode)
set -euo pipefail

NAME=llamacpp-sycl-qwen38
IMAGE=llama.cpp:sycl-f16-qwen38
MODEL_DIR=/data/llm/qwen3.8-27b-GGUF
PORT=8010

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=22g --memory-swap=24g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 180s --health-retries 3 \
  -v "$MODEL_DIR":/models:ro \
  -p "0.0.0.0:${PORT}:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e LLAMA_ARG_HOST=0.0.0.0 \
  "$IMAGE" \
  -m /models/Qwen3.8-27B-Q4_K_M.gguf \
  --model-draft /models/mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --mmproj /models/mmproj-Qwen3.8-27B-Q8_0.gguf \
  -ngl 99 -ngld 99 \
  -c 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --repeat-penalty 1.0

echo "Qwen 3.8-27B + MTP + vision on :${PORT} card 2 (level_zero:1) — status starting"
echo ""
echo "  --reasoning off is set for structured-JSON compat (matches your stack pattern)"
echo "  To enable Qwen 3.8 native thinking mode, remove '--reasoning off' and clients"
echo "  can pass enable_thinking=true/false via chat_template_kwargs"
echo ""
echo "  Context 32K to start (Qwen 3.8 supports 262K native / 1M via YaRN)"
echo "  To bump to 262K: change -c 32768 -> -c 262144 (KV cache Q8 grows accordingly)"
echo ""
echo "  Vision loaded (mmproj). Send image URLs via OpenAI chat_completions format."
echo "  For text-only, delete --mmproj line to save 629 MB VRAM"
