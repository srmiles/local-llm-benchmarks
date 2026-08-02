# Gemma 4 26B-A4B (it) — Reasoning fallback / historical prod

**Status:** Reserved for reasoning-heavy queries; launcher `start-llamacpp-sycl-gemma4-mtp.sh` on disk. **QAT Q4_0 + MTP on b10215 is the new best config** (see 2026-08-01 bench below) — 54.2 tps decode / 1,164 tps prefill / **17.5 GiB VRAM** (2.2 GiB less than Q4_K_M). Launcher still points at Q4_K_M for historical continuity; swap to QAT for reasoning-fallback deployment if VRAM headroom matters.
**HF (base):** [`lmstudio-community/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF) (Q4_K_M) · [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) → QAT Q4_0 on disk at `/data/llm/Gemma-4-QAT/gemma-4-26B-A4B-it-QAT-Q4_0.gguf`
**HF (drafter):** Google's official 26B-A4B assistant, community-packaged by Janvitos as MTP Q8_0. **This community drafter uses the correct `gemma4-assistant` arch string** (verified via `strings` on GGUF metadata) — unlike the E2B/E4B/12B community versions which had the underscore bug. So no re-conversion needed for this size.
**Chat template:** Google's updated official Jinja (strip_thinking macro, OpenAI tool response handling)
**Launcher:** [`configs/launchers/start-llamacpp-sycl-gemma4-mtp.sh`](../../configs/launchers/start-llamacpp-sycl-gemma4-mtp.sh)

## Specs

| | |
|---|---|
| Parameters | 26B total / **4B active** (MoE) |
| Quant options | Q4_K_M (post-training) · **QAT Q4_0 (recommended on b10215+MTP)** |
| File size | 15 GB (Q4_K_M) · ~13 GB (QAT Q4_0) |
| Context (trained) | 256K |
| Context (deployed) | 131,072 |
| MTP drafter | Google official assistant, Q8_0, 441 MiB (community-packaged, correct arch) |

## Benchmarks

### On b10215 build (2026-08-01, isolated, real-workload prompts, `--cache-type-k/v q8_0`, `-c 8192`)

| Metric | **QAT Q4_0 + MTP** ⭐ | Q4_K_M + MTP |
|---|---|---|
| Prefill @ 5K real prompt | **1,164 tok/s** | 1,180 tok/s (~parity) |
| Prefill @ 1.3K real prompt | 987 tok/s | 1,099 tok/s |
| Decode (5K + 100 gen) | **54.2 tok/s** | 49.0 tok/s |
| Decode (1.3K + 150 gen) | 53.8 tok/s | 45.2 tok/s |
| MTP acceptance (5K bench) | **100%** (74/74) | 96.1% (73/76) |
| MTP acceptance (1.3K bench) | 84.9% (107/126) | 70.6% (101/143) |
| VRAM (loaded, 8K ctx KV Q8) | **17.5 GiB** | 19.7 GiB |
| Power draw under load | 42 W (of 220 W TDP) | 45 W |
| Correctness (chat + tool call) | ✓ (inherits Q4_K_M behaviour) | ✓ |

**QAT Q4_0 vs Q4_K_M on b10215+MTP:** QAT wins decode by +10.6% (54.2 vs 49.0), essentially ties on prefill, and saves **2.2 GiB VRAM**. This is a genuine surprise — historical finding #2 said "K-quant beats QAT Q4_0 at ≥26B" but that was measured on b10068 without MTP. Under b10215+MTP the ordering reverses at this size, most likely because Q4_0's simpler layout matches the new SYCL oneMKL XMX GEMM path (PR#25025) better than K-quant's super-block structure.

### On b10068 build (2026-07-19, isolated)

| Metric | Q4_K_M + MTP (Config C) |
|---|---|
| Cold 12K prefill | **20.0 s @ 650 tok/s** (was 21.5s @ 655 on b9948) |
| 5K prefill | **971 tok/s** (was ~830 on b9948, +17%) |
| Decode (peak, MTP-accepted) | **53.0 tok/s** (was ~50 on b9948, +6%) |
| MTP draft acceptance | 37–89% (highly prompt-dependent) |
| VRAM (loaded, 128K KV Q8 + drafter) | 22.9 GiB |
| Correctness (chat + tool call) | ✓ |

### b10068 → b10215 delta (Q4_K_M + MTP, like-for-like)

| Metric | b10068 | b10215 | Δ |
|---|---|---|---|
| 5K prefill | 971 tok/s | 1,180 tok/s | **+21.5%** (XMX FA PR#25025) |
| Decode | 53.0 tok/s peak | 49.0 tok/s | ~parity within noise |
| VRAM | 22.9 GiB (128K ctx) | 19.7 GiB (8K ctx) | not like-for-like (context differs) |

Prefill uplift on 26B-A4B (+21.5%) is smaller than seen on Ornith 9B (+87%) or Gemma 4 E2B (~2×) because MoE-26B is more compute-heavy per token — proportionally less benefit from the XMX GEMM FA optimization than a smaller/denser model gets. Still meaningful.

### Historical baselines (kept for reference)

| Metric | Q4_K_M base | + MTP (Config C) |
|---|---|---|
| Decode (steady-state) | 44.1 tok/s | 50.0 tok/s (+15.6%) |
| Cold 12K prefill | 22.8s @ 632 tok/s | ~21.5s @ 655 tok/s |
| Warm follow-up | 0.55s | 0.55s |
| Concurrent 2×4K | 13.9s | ~13.9s |
| VRAM (loaded) | 20.9 GB | 22.8 GB |
| MTP acceptance | — | 78%, 3.30 mean accepted tok/draft |
| KB dual-eval (OpenRouter fp8) | .76 | — |
| KB dual-eval (local Q4_K_M) | .73–.76 | — |
| pi.dev win rate | 63.3% | 63.3% |

## Config (`start-llamacpp-sycl-gemma4-mtp.sh`)

```bash
docker run -d --name llamacpp-sycl \
  --restart unless-stopped \
  --memory=20g --memory-swap=20g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 180s \
  -v /data/llm/lmstudio-community/gemma-4-26B-A4B-it-GGUF:/models:ro \
  -v /data/llm/Gemma-4-Assistant:/draft:ro \
  -v /data/llm/templates:/templates:ro \
  -p 0.0.0.0:8002:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --model-draft /draft/gemma-4-26B-A4B-it-qat-assistant-MTP-Q8_0.gguf \
  --spec-type draft-mtp \
  -ngl 99 -ngld 99 \
  -c 131072 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 3072 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  -fa on -ub 2048 -b 2048 \
  --predict 2048 \
  --min-p 0.0 \
  --temp 1.0 --top-k 64 \
  --chat-template-file /templates/gemma-4-official-current.jinja \
  --jinja \
  --reasoning off
```

## Notes

- Was locked production for months; superseded by Ornith 9B after July 2026 dual-eval bake-off
- **Historical Finding #2 no longer holds at 26B on b10215+MTP:** Q4_K_M beat QAT Q4_0 by +10% decode on b10068 without MTP. On b10215 with MTP, QAT wins decode by +10.6% AND saves 2.2 GiB VRAM. Q4_0's simpler layout matches SYCL oneMKL XMX GEMM (PR#25025) better than K-quant super-blocks. Finding #2 in main README still holds at other sizes and non-MTP configs, but has this 26B+MTP+b10215 caveat.
- **Community MTP drafter for 26B is correctly built** (uses `gemma4-assistant` hyphen arch string) — unlike the E2B/E4B/12B community versions which had `gemma4_assistant` underscore bug and had to be re-converted. So no fresh HF upload needed for the 26B drafter; the existing Janvitos Q8_0 packaging works with upstream llama.cpp b10215+.
- `--jinja` **mandatory** for tool calls; without it the built-in template drops Gemma 4's `<|tool>`/`<tool|>` delimiters and the agent loops
- Config C = vLLM-fixed → then Google-official template. Fixes tool-loop drift Config B had at long context
- `--reasoning off` currently used because PEG parser 500s; Google official template makes `--reasoning on` viable but hasn't been re-locked
- MTP drafter absolutely worth it: +15.6% decode on b10068, and on b10215 the MTP acceptance rate hits 96-100% on real prompts vs 37-89% previously — the newer build's kernel path apparently interacts better with the draft/target coherence
- 256K context loads but leaves zero margin; 128K–192K is the safe range
- **VRAM co-residence:** QAT Q4_0 + MTP at 17.5 GiB fits alongside current prod stack (embed 2.65 + TEI 0.9 = 3.55 GiB non-chat) with 3 GiB headroom. Q4_K_M at 19.7 GiB is also workable with 0.8 GiB headroom. Both are much more co-res friendly than the historical 22.9 GiB Q4_K_M b10068 configuration.
