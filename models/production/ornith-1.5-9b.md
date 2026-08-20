# Ornith 1.5 9B — Production chat (both B60s)

**Status:** Production chat + pi.dev agent on **both cards** in Traefik round-robin LB (`:8002` card 1, `:8010` card 2). Cutover 2026-08-21 from Ornith 1.0-9B.
**HF (base):** [`ornith-ai/Ornith-1.5-9B-GGUF`](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) · Q4_K_M 5.63 GB
**HF (MTP drafter):** [`protoLabsAI/Ornith-1.5-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-MTP-GGUF) · Q8_0 2.43 GB (same publisher as 1.0's drafter — direct successor, purpose-trained against the full-precision base)
**Base:** Qwen 3.5 9B fine-tune (arch `qwen3_5`) — recognized cleanly by llama.cpp SYCL b10433
**Launcher (card 1):** [`configs/launchers/start-llamacpp-sycl-ornith.sh`](../../configs/launchers/start-llamacpp-sycl-ornith.sh)
**Launcher (card 2):** [`configs/launchers/start-llamacpp-sycl-ornith-1.5-c2.sh`](../../configs/launchers/start-llamacpp-sycl-ornith-1.5-c2.sh)
**Frontend:** `https://llm.levirge.com/v1/completions` and `/v1/chat/completions` (Traefik path-based routing, round-robin between :8002 and :8010)

## Specs

| | |
|---|---|
| Parameters | 9B dense |
| Quant | Q4_K_M (post-training) |
| File size | ~5.63 GB |
| Context (trained) | 262K |
| Context (deployed) | 262,144 (both cards) |
| MTP drafter | protoLabsAI 0.5B head Q8_0 |
| Reasoning | Model emits `<think>` blocks; currently launched `--reasoning off` for inline behavior (matches 1.0 pattern) |

## Benchmarks

### On b10433 build (2026-08-21, card 2 isolated bench during cutover — LB pulled)

3 samples per prefill size, `cache_prompt=false`, UUID prefix, greedy. Standard bench methodology (matches `models/production/gemma-4-26b-a4b.md`).

| Metric | Value | vs Ornith 1.0-9B (b10068 baseline) |
|---|---|---|
| Prefill @ ~500 tok | **1,133 tok/s** | +22% (927) |
| Prefill @ ~2K tok | **1,930 tok/s** | +54% (1,254) |
| Prefill @ ~4K tok | **2,040 tok/s** | +84% vs 1.0 @ 8K (1,106) |
| Prefill @ ~4K tok | **2,044 tok/s** | +111% vs 1.0 @ 12K (969) |
| Decode + MTP (512 tok, greedy) | **65.44 tok/s** | +23% (50-53 under load) |
| MTP acceptance rate | **82.7%** (364/440 draft tokens) | +6-14pp vs 1.0 (68.9-76.6%) |
| Mean accepted per draft round | 2.48 | -0.6 (1.0 was 3.07-3.28) |
| VRAM @ 262K KV Q8 | **20.7 GiB** | +1.9 (1.0 was 18.8 GiB) |
| Load time cold | 12 s | +2 s |
| Engine resets during bench | 0 | ✅ |

Caveat: 1.0 baselines are from b10068 (July 2026); build gains since then account for some of the prefill uplift. Decode + MTP acceptance deltas are pure model wins.

### Sampling — aligned to Ornith 1.5 coding recipe

Ornith's HF card documents different sampling for coding vs general tasks. Our workload is dominantly agentic coding (pi.dev + brain-agents), so the launcher matches the coding recipe:

| Param | Ornith 1.5 general | Ornith 1.5 coding | Our launcher | Match? |
|---|---|---|---|---|
| temperature | 1.0 | 0.6 | client-set | client-controlled |
| top_p | 0.95 | 0.95 | server default | ≈ |
| top_k | 20 | 20 | **20** (was 40) | ✅ |
| min_p | 0.0 | 0.0 | 0.0 | ✅ |
| presence_penalty | 1.5 | 0.0 | not set (0.0) | ✅ coding |
| repetition_penalty | 1.0 | 1.0 | **1.0** (was 1.05) | ✅ |

Changes from the 1.0 launcher: added `--top-k 20`, dropped `--repeat-penalty 1.05 --repeat-last-n 256` (Ornith 1.5 was trained without repetition penalty). Clients can still override per-request; general-chat requests can pass `presence_penalty=1.5` explicitly.

### Head-to-head with other benches on b10433 (this session)

| Model | Params (active) | Prefill @ 4K | Decode + MTP | MTP accept | VRAM |
|---|---|---|---|---|---|
| **Ornith 1.5-9B (this)** | 9B dense | 2,040 tok/s | **65.44 tok/s** | 82.7% | 20.7 GiB |
| Gemma 4 26B-A4B QAT + MTP | 26B / 4B | 1,592 | 62.84 | **97.2%** | 19.9 GiB |
| Ornith 1.5-35B-A3B (bartowski IQ4_XS) | 35B / 3B | 1,360 | 32.6 | 32.5% | 22.5 GiB |
| Ornith 1.5-35B-A3B (mudler APEX Compact) | 35B / 3B | 1,211 | 25.6 | 26.2% | 21.0 GiB |

Ornith 1.5-9B wins single-card decode at same-build against every other candidate. Gemma 26B-A4B is a very close second at 62.8 tok/s despite carrying 3× the params — thanks to its 97.2% MTP acceptance (Google publishes a purpose-trained drafter matched to QAT base).

## Cutover ops notes

- **Card 2 tested first** (Sat 2026-08-21) — pulled from Traefik LB, bench-in-isolation, four success gates all passed decisively (prefill ±5%, decode ≥48, MTP ≥55%, no engine resets)
- **Card 1 cutover** — swapped launcher paths + alias while card 2 was live in LB; card 1 restart was invisible to clients (LB covered)
- **Sampling alignment** applied to both cards after cutover — 2× ~15s container restarts, LB rotation covered
- Sanity via `https://llm.levirge.com/v1/chat/completions` returned `model: "ornith-1.5-9b"` — confirmed round-robin working

## Config (`start-llamacpp-sycl-ornith.sh` — card 1)

```bash
docker run -d --name llamacpp-sycl \
  --restart unless-stopped \
  --memory=12g --memory-swap=14g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s \
  -v /data/llm/Ornith-1.5-9B-GGUF:/models:ro \
  -p 0.0.0.0:8002:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/Ornith-1.5-9B-Q4_K_M.gguf \
  --model-draft /models/mtp-Ornith-1.5-9B-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --alias ornith-1.5-9b \
  -ngl 99 -c 262144 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 \
  --top-k 20 --min-p 0.0
```

Card 2 (`-c2` variant) is the same with `ONEAPI_DEVICE_SELECTOR=level_zero:1`, `-p 8010:8000`, and NEO cache mount for faster cold restart.

## Notes

- **Direct drop-in from 1.0** — same qwen3.x family, same MTP drafter architecture, same wiring pattern. Only paths + alias + sampling changed.
- **Uniform benchmark improvements** claimed by Ornith on the HF card: Terminal-Bench +3-6, SWE-bench +1-5, HLE +3-4, GPQA +4, BrowseComp +12, Toolathlon +8 (per 5-run averages). We haven't run agentic-quality evals internally yet; the raw decode/MTP wins in the bench above are what justified the cutover.
- **Multimodal** — the 1.5-9B non-GGUF card advertises image+text. llama.cpp GGUF path is text-only; if we ever want image input, vLLM XPU is the migration.
- **Reasoning parser** — model supports `--reasoning qwen3` for separate `reasoning_content` field. Left `off` to match 1.0 behavior; flip if downstream tooling wants structured `<think>` extraction.
- **Both cards mirrored** — Traefik `passHostHeader: false` round-robin. If either card fails health check, LB removes it automatically.
- **Prior 1.0-9B doc** archived at [`models/tested/ornith-1.0-9b.md`](../tested/ornith-1.0-9b.md) — preserves historical baselines and the b10068 MTP acceptance investigation.

## Historical context

- 1.0-9B was production chat from July 2026 through Sat 2026-08-21 (7 weeks)
- The 1.0-9B build history — through Vulkan → b9948 → b10068 → b10215 → b10256 → b10433 — is preserved in the archived [tested/ornith-1.0-9b.md](../tested/ornith-1.0-9b.md)
- pi.dev win rate 66.7% (1.0-9B, July 2026). Re-measure pending on 1.5-9B.
- KB dual-eval score .80 (36/45, 1.0-9B). Re-measure pending on 1.5-9B.
