# MiniCPM5-1B — Tested 2026-07-19, promising

**Status:** **Stopped** 2026-07-19. Ran briefly on `:8009` for pi.dev experimentation. Container removed to reclaim ~3 GiB VRAM; launcher script kept for on-demand restart via `sudo /data/llm/launch/start-llamacpp-minicpm5.sh`.
**HF:** [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) · GGUF: [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)
**Arch:** Standard `LlamaForCausalLM` — llama.cpp loads it directly, no fork
**Launcher:** [`configs/launchers/start-llamacpp-minicpm5.sh`](../../configs/launchers/start-llamacpp-minicpm5.sh)

## Specs

| | |
|---|---|
| Parameters | 1.08B (679M non-embedding) |
| Layers | 24 |
| Attention | GQA (16 Q heads / 2 KV) |
| Vocab | 130,560 |
| Quant | Q4_K_M |
| File size | 688 MB |
| Context (trained) | 131,072 |
| Modes | Think / No-Think dual on same checkpoint |

## Benchmarks (bare-metal B60, `llama.cpp:sycl-f16`, port 8009 alongside Ornith)

| Metric | Value |
|---|---|
| **Decode (steady, 3-run avg)** | **~187 tok/s** (181–191) |
| **Prefill @ 2K** | **4,642 tok/s** |
| Prefill @ ~1 tok warm | 165 tok/s (single-token startup cost) |
| Load time | ~20s cold |
| VRAM delta on load @ 32K ctx, KV Q8 | ~3 GB |
| Correctness: math chat | ✓ ("17×23 = 391") |
| Correctness: structured tool call | ✓ (clean OpenAI tool_calls) |
| Correctness: JSON categorise | ⚠ content correct, wrapped in ```json fences |

## Comparison against current small-model slot candidates

| Model | Decode tok/s | Prefill @ 2K | VRAM | JSON |
|---|---|---|---|---|
| **MiniCPM5-1B Q4_K_M** | **~187** | **4,642** | ~3 GB | ⚠ needs fence strip |
| Qwen3-4B-Instruct-2507 Q4_K_M | ~94 | 766 aggregate | ~1 GB | ✓ clean |
| Gemma 4 E4B QAT Q4_0 | 73.9 | 376 | ~3 GB | ✓ |
| Gemma 4 E4B Q4_K_M | 68.3 | 466 | ~3 GB | ✓ |
| Gemma 3 4B Q4_K_M | ~78 | — | ~3 GB | ✓ (needs system-role fix) |

**MiniCPM5-1B is ~2× decode and ~6× prefill vs the fastest 4B model tested.** Same VRAM class.

## Bench command

```bash
docker run -d --name llamacpp-minicpm5 \
  --memory=4g --memory-swap=4g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 90s \
  -v /data/llm/MiniCPM5-1B-GGUF:/models:ro \
  -p 0.0.0.0:8009:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 \
  -c 32768 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --temp 0.6 --top-p 0.95 --min-p 0.0 \
  --reasoning off
```

## Verdict

**Best raw speed of any model tested.** 187 tok/s decode + 4.6K tok/s prefill on a 1B model on B60 is exceptional — this is the sub-second-tool-call territory that Ornith 9B can't reach.

**Blockers before promotion:**
1. **JSON fence wrapping** — model wraps categorise output in ```json … ``` blocks. Fix options:
   - Strip fences in the categorise client
   - Use llama.cpp's `--grammar` with a JSON grammar
   - Prompt-engineer the system message to forbid fences
2. **Capability check needed** — 1B parameter count may hurt entity extraction quality vs 4B. Would need to run the same 3-request bake-off (JSON validity + entity accuracy) that pinned Qwen3-4B originally.
3. **Long-context behaviour** — tested at 32K only; the model advertises 128K but this stack hasn't verified prefill/decode at that length.

**Recommended next step:** re-run the categorise bake-off (Ornith 9B baseline + Qwen3-4B + MiniCPM5-1B with fence-stripping in the client) using real brain doc batches. If MiniCPM5 clears the entity/atom quality bar, it's the new categorise standby / burst-overflow — and possibly a candidate for MTP drafter experiments given how fast prefill runs.

## Download

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='openbmb/MiniCPM5-1B-GGUF',
    filename='MiniCPM5-1B-Q4_K_M.gguf',
    local_dir='/data/llm/MiniCPM5-1B-GGUF',
)"
```
~2s download at HF xet speeds; file kept at `/data/llm/MiniCPM5-1B-GGUF/`.
