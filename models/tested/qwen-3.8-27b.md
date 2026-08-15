# Qwen 3.8-27B — Hybrid Mamba+Attention, vision-capable, benched then parked

**Status:** **Tested 2026-08-14/15, then stopped.** Ran briefly on card 2 as the first tenant of the 2nd B60. Decode came in at 23 tps on 27B dense hybrid — roughly half of Gemma 4 26B-A4B Q4_K_M (47.7 tps) at the same VRAM class. Vision niche is already covered by Muse Glimmer-30B at similar speed with a working DFlash drafter. Stopped 2026-08-15 to free card 2 for the dedicated Gemma 4 E2B categorise slot (99% of real workload). Launcher preserved on disk for on-demand relaunch.

**Revisit trigger:** either (a) upstream SYCL Gated DeltaNet / SSM_SCAN / SSM_CONV kernels gain proper XMX GEMM path (currently functional but not XMX-optimised — decode ceiling would roughly double when they do), or (b) a use case appears that specifically needs Qwen 3.8's long-context or vision behaviour that Muse Glimmer can't cover. Otherwise recheck in ~1 month.

**HF (base):** [`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) — llama.cpp team's official conversion. Weights on disk at `/data/llm/qwen3.8-27b-GGUF/`.
**HF (upstream):** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — original PyTorch checkpoint from Alibaba.
**Architecture family:** `qwen3_5` in llama.cpp (support merged in [#19468](https://github.com/ggerganov/llama.cpp/pull/19468); MTP for the arch merged in [#24025](https://github.com/ggerganov/llama.cpp/pull/24025)).
**Chat template:** built into GGUF (`--jinja` mode). Handles vision content-parts natively via image_count / video_count namespace macros.
**Launcher:** [`configs/launchers/start-llamacpp-sycl-qwen38-27b.sh`](../../configs/launchers/start-llamacpp-sycl-qwen38-27b.sh) → deployed at `/data/llm/launch/start-llamacpp-sycl-qwen38-27b.sh` on `llm.local`.

## Specs

| | |
|---|---|
| Parameters | **27B dense hybrid** — 48 Gated DeltaNet (SSM) layers + 16 Gated Attention layers (3:1 ratio) |
| Quant deployed | Q4_K_M base + Q4_0 MTP drafter + Q8_0 vision mmproj |
| File sizes | base 18.97 GB · MTP drafter 1.68 GB · mmproj 629 MB |
| Context (trained) | 262,144 |
| Context (deployed) | 32,768 (`--cache-type-k/v q8_0`) |
| MTP drafter | Native to Qwen 3.5-family release, ships in the ggml-org GGUF bundle |
| Vision | ViT via mmproj Q8_0, single-image content-parts through OpenAI-style `image_url` |
| Target card | Card 2 (`ONEAPI_DEVICE_SELECTOR=level_zero:1`) — 0b:00.0 |

## Smoke test on b10433 (2026-08-15, isolated `/v1/chat/completions`)

Three probes covering reasoning, longer real-workload prompts, and vision.

| Probe | Prefill | Decode | MTP acceptance | Notes |
|---|---|---|---|---|
| Short reasoning (snail well, 58 tok prompt) | 13.6 tps* | 24.1 tps | 56/90 = **62.2%** | *warm-cache prompt_n=4; disregard |
| Longer real prompt (489 tok, hybrid-arch summary) | **333.5 tps** | **23.0 tps** | 179/309 = **57.9%** | Real prefill baseline; 4-bullet summary quality good |
| Vision (256×256 PNG, shape identification) | image-encode dominated | 30.9 tps | 9/9 = **100%** | Correctly identified blue square + red circle + green triangle |

**VRAM under load (isolated):** 22,304 → 22,781 MiB (idle → post-vision). Vision encode adds ~500 MiB transient.
**Power under load:** peaks at 41 W idle, no sustained load probe yet (single-shot bench).
**Steady-state gap vs. same-size models:** Decode 23 tps on 27B dense hybrid is roughly **half** of Gemma 4 26B-A4B Q4_K_M (47.7 tps). Two reasons stacked: (a) Gemma is 4B active (MoE), Qwen 3.8 is 27B dense; (b) SYCL kernels for Gated DeltaNet / SSM_SCAN / SSM_CONV are functional but not yet XMX-optimized to the extent standard attention is. Expect this gap to narrow with future SYCL kernel PRs.

## Config (`start-llamacpp-sycl-qwen38-27b.sh`)

```bash
NAME=llamacpp-sycl-qwen38
IMAGE=llama.cpp:sycl-f16-qwen38
MODEL_DIR=/data/llm/qwen3.8-27b-GGUF
PORT=8010   # NOT 8003 — headroom-proxy (Anthropic upstream) owns 8003

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --memory=22g --memory-swap=24g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  -v "$MODEL_DIR":/models:ro \
  -p "0.0.0.0:${PORT}:8000" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
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
```

**Image alias:** `llama.cpp:sycl-f16-qwen38` currently points at the same b10433 build image as `llama.cpp:sycl-f16`. Kept separate so a future rebuild for a Qwen 3.8-specific kernel PR can be pinned without disturbing card 1's services.

## Notes

- **Port conflict watch:** initial launcher used `PORT=8003`; `headroom-proxy` (Steve's external Anthropic proxy) claims :8003 after reboot. Moved Qwen 3.8 to :8010 to keep both services co-resident. Any future card-2 service should also avoid :8003.
- **Vision content-parts must be OpenAI-format** (`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`) — the built-in jinja template macros expect that shape. Bare `{"type":"image","image":"..."}` won't render.
- **MTP acceptance runs lower than Ornith/Gemma** (58–62% vs. 96–100%). Consistent with the general observation that MTP drafters trained on attention-heavy targets transfer imperfectly to hybrid SSM+attention targets; the SSM-dominant middle layers see 5–10 pp lower acceptance than a pure-attention target of the same size. Not a bug — an arch-family cost. Still positive on tps because the drafter itself is small (1.68 GB Q4_0) and even 58% acceptance beats the no-drafter baseline.
- **Context deployed at 32K, not 262K.** Trained context is 262,144 tokens but K/V memory scales linearly; 32K keeps the working set comfortable on one B60 alongside the model weights and mmproj. Bump to 64K–128K viable if the drafter is dropped, or after the RAM upgrade (2026-08-22) if we want to spill KV to host with `--cache-ram`.
- **`--reasoning off`** matches the rest of the stack (Gemma 4 26B fallback config) — enables consistent behaviour when the reasoning-fallback slot toggles between models. Qwen 3.8 supports reasoning mode; can be re-enabled per-call if a client sets `reasoning=high` in the future.
- **Card 2 is otherwise unused** — no other services co-resident on `level_zero:1` yet. Full 24 GiB minus 22.8 GiB in-use = ~1.2 GiB headroom, not enough to co-load another meaningful model. Future card-2 tenants will need Qwen 3.8 stopped or moved to a smaller quant first.
- **Rollback:** if a llama.cpp build regresses Qwen 3.8 support, image tag `llama.cpp:sycl-f16-b10256-safe` and `-b10215-safe` are on disk. Note that b10215 and earlier **do not** have the SYCL Mamba PRs (#26612, #26643) — Qwen 3.8 loads but will be visibly slower. Real rollback target for this model is b10433 → newer, not b10433 → older.
- **Not currently routed to by any client.** Available for direct-hit testing at `http://llm.local:8010/v1/chat/completions`. Formal routing (e.g. via headroom-proxy or a Litellm layer) is a separate decision.
