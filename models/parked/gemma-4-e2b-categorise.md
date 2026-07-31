# Gemma 4 E2B QAT Q4_0 — Parked categorise (awaiting 2nd B60)

**Status:** **Parked, not currently running.** Deployed 2026-07-22 as dedicated categorise on `:8006`, reverted same day after cross-process SYCL contention on single B60 made both slots slower. Launcher preserved at `/data/llm/launch/start-llamacpp-categorise-e2b.sh` — will re-deploy when a second B60 GPU is added to eliminate cross-process contention.
**HF:** [`google/gemma-4-E2B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
**Base:** Gemma 4 E2B (2B effective, ~5B total), Google-official QAT (Quantization-Aware Training) Q4_0
**Launcher:** [`configs/launchers/start-llamacpp-categorise-e2b.sh`](../../configs/launchers/start-llamacpp-categorise-e2b.sh)

## Specs

| | |
|---|---|
| Parameters | 2B effective / ~5B total |
| Architecture | Gemma 4 (dense) |
| Quant | Q4_0 (Google QAT — quality-aware training, not post-hoc) |
| File size | 3.35 GB |
| Context | 8,192 (deployed; native supports 128K+) |
| VRAM (loaded, KV Q8) | ~2 GiB |
| License | Apache-2.0 (Gemma terms) |

## Deployed benchmarks (2026-07-22)

Isolated bench on `:8009` before promotion — 10-task categorise workload (short-context, JSON structured output, taxonomy classification):

| Metric | Value | vs Ornith 9B (prod baseline) | vs Qwen3-4B-2507 (retired) |
|---|---|---|---|
| Decode | **88 tok/s** | +54% (57) | +11% (79) |
| Wall median | **689 ms** | -42% (1193) | +19% (581) |
| JSON parseable | 10/10 | tie | tie |
| Schema-valid | 10/10 | tie | tie |
| Categories used | 4/5 (business, other, personal, technical) | tie | +1 (retired only used 3/5) |
| Match to Ornith on same 10 tasks | 9/10 | — | Qwen was 8/10 |

Sample outputs match Ornith's categorisations except for one defensible edge case (bare sourdough recipe → `other` here, `personal` on Ornith).

## Config (`start-llamacpp-categorise-e2b.sh`)

```bash
docker run -d --name llamacpp-categorise \
  --restart unless-stopped \
  --memory=6g --memory-swap=6g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video |cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 60s \
  -v /data/llm/gemma-4-E2B-it-GGUF:/models:ro \
  -p 0.0.0.0:8006:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/gemma-4-E2B_q4_0-it.gguf \
  -ngl 99 -c 8192 --parallel 2 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --reasoning off \
  --temp 0.0 --top-p 0.95 --min-p 0.0
```

**`--reasoning off` is mandatory** — Gemma 4 routes thinking tokens to OpenAI `reasoning_content` by default, leaving `content` empty. Without this flag, all JSON parsing fails.

**`--parallel 2`** — categorise workload can burst from brain-ingest; 2 slots absorb the burst without meaningful VRAM cost (~500 MiB extra KV).

## Endpoints

- Local: `http://127.0.0.1:8006/v1/chat/completions`
- LAN: `http://192.168.1.253:8006/v1/chat/completions`
- Tailscale: `http://100.70.193.48:8006/v1/chat/completions`

## Brain integration

Point brain's categorise env at `:8006` instead of `:8002`. Depending on brain config layout:

```bash
# in brain env / compose file
CATEGORISE_URL=http://192.168.1.253:8006/v1/chat/completions
# or via Tailscale
CATEGORISE_URL=http://100.70.193.48:8006/v1/chat/completions
```

Restart brain-categorise consumer after the change.

## Timeline

- **2026-07-22 21:31 local:** Deployed on single B60 as dedicated categorise slot.
- **2026-07-22 21:56:** First HTTP 400s from brain summarise (5215-token prompts exceeded 4K/slot when `--parallel 2` split 8K context). Restarted with `-c 32768`.
- **2026-07-22 22:00:** Timeouts under mixed load. Diagnostic confirmed cross-process SYCL contention: Ornith dropped from 52 → 16 tps and categorise dropped from 88 → 12 tps when both processes were active on the shared B60. Verified by stopping categorise container mid-session → Ornith immediately jumped from 17 to 55 tps.
- **2026-07-22 22:10:** Reverted. brain `CATEGORISE_URL` back to `:8002`; container stopped; VRAM freed to 11.4 GiB.

## Findings from the split experiment

**Cross-process SYCL contention on single B60 is severe.** Two `llama-server` processes both driving the B60 through the Level Zero driver:
- Each process drops to ~30% of isolated throughput
- Combined output (Ornith 16 + categorise 12 = 28 tps) is *worse* than a single process at 52 tps
- The bottleneck we were trying to relieve (queue wait) turns into a throughput bottleneck (per-token slowdown) that's larger than the wait savings
- Root cause is likely Level Zero context switching and shared kernel dispatch queue between the two processes; not a software fix that llama.cpp itself can address

**Split-slot architecture only works when each slot has its own GPU.**

## When to re-deploy

- **Second B60 added to llm.local:** relaunch with `ONEAPI_DEVICE_SELECTOR=level_zero:1` to bind to the second card; Ornith stays on `:0`. No cross-process contention because different physical GPUs = different SYCL contexts.
- Bench config already validated (see [candidate hunt](../tested/categorise-candidates.md)). Model + quant + reasoning-off flag ready to go.

## Google MTP drafter now available (2026-07-31)

Google released official MTP drafters for Gemma 4 E2B/E4B on 2026-05-05. We built `llama.cpp:sycl-f16-next-eb41d503b` (b10215) which supports the `gemma4-assistant` architecture, then converted Google's HF safetensors directly with the new build's `convert_hf_to_gguf.py`:

- Source: [`google/gemma-4-E2B-it-assistant`](https://huggingface.co/google/gemma-4-E2B-it-assistant) (BF16 safetensors, 158 MB)
- Converted GGUF: `/data/llm/gemma-4-E2B-it-assistant-GGUF/gemma-4-E2B-it-assistant-official.bf16.gguf` (170 MB)
- Community GGUFs (AtomicChat etc.) use the wrong `gemma4_assistant` underscore arch name — incompatible with upstream. **Only Google-official-converted GGUF works.**

### Bench on b10215 (single B60, isolated, categorise + prefill probe)

Extended 2026-07-31 bench to all three Google MTP-drafter combos:

| Model + drafter | Params | Decode | MTP acc | Wall/task | Prefill 500t | Prefill 2Kt | VRAM |
|---|---|---|---|---|---|---|---|
| **Gemma 4 E2B + Google MTP** | 2B eff | **138.8 tps** | 67.8% | 0.72s | 1,862 tps | **3,681 tps** | ~4.5 GiB |
| **Gemma 4 E4B + Google MTP** | 4B eff | 114.1 tps | 66.7% | **0.69s** | 1,396 tps | 2,319 tps | ~7 GiB |
| **Gemma 4 12B + Google MTP** | 12B dense | 70.4 tps | 69.7% | 2.18s | 428 tps | 1,053 tps | ~8 GiB |
| Gemma 4 E2B alone (no MTP) | 2B eff | 88 tps | — | ~1.15s | — | — | ~4.5 GiB |
| Ornith 9B + MTP (current prod) | 9B dense | 56 tps | 76.3% | 1.25s | ~1,600 tps* | — | ~10.9 GiB |

*Ornith prefill from live-brain observation, not measured in this session

**Observations:**
- MTP acceptance stays flat at 67-70% across all three sizes — Google's drafters are well-calibrated per size, no quality degradation as target scales
- **12B + MTP hits 70 tps decode** — matches Ornith 9B decode at similar VRAM class. Was previously written off as too slow; MTP resurrected it as a viable Ornith competitor.
- **E2B is the prefill king** — 3,681 tps @ 2K is ~50% of theoretical B60 FP16 memory-bandwidth ceiling for a 3.35 GB model
- Per-workload winners by output length:
  - Short JSON output (~200 tok): E4B narrowly beats E2B (0.69s vs 0.72s wall)
  - Medium generations (200-500 tok): E2B best (decode dominates)
  - Long chat/agent (800+ tok): Ornith 9B still best (higher MTP acceptance, better on high-entropy long output)
  - Reasoning-heavy needing 12B+ params: 12B + MTP now viable at ~2s per task

### Redeployment plan when 2nd B60 arrives

```bash
docker run -d --name llamacpp-categorise \
  --restart unless-stopped --memory=6g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video |cut -d: -f3)" \
  -v /data/llm/gemma-4-E2B-it-GGUF:/models:ro \
  -v /data/llm/gemma-4-E2B-it-assistant-GGUF:/drafter:ro \
  -p 0.0.0.0:8006:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  llama.cpp:sycl-f16 \
  -m /models/gemma-4-E2B_q4_0-it.gguf \
  --model-draft /drafter/gemma-4-E2B-it-assistant-official.bf16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -c 32768 --parallel 2 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off --temp 0.0
```

Then brain env `CATEGORISE_URL=http://192.168.1.253:8006/v1/chat/completions`.

## Prefill/decode asymmetry finding (2026-07-22, under real brain load)

The split-slot experiment surfaced a subtle-but-important observation about model-choice trade-offs:

| Slot / model | Prefill tps | Decode tps | Notes |
|---|---|---|---|
| :8006 Gemma 4 E2B QAT Q4_0 | **1,600** | 30 | fast prefill, no MTP drafter |
| :8002 Ornith 9B + MTP Q4_K_M | 1,000 | **50** | MTP boost + K-quant decode-friendly layout |

**Why Ornith wins decode:** MTP drafter (~2× speculative acceleration at 70%+ acceptance) + Q4_K_M super-block dequant path favors single-token latency.

**Total-task-time math** shows Gemma only wins for very short outputs:

| Workload | Prompt | Gen | Gemma | Ornith | Winner |
|---|---|---|---|---|---|
| Categorise (JSON) | 5K | 200 | 9.8s | 9.0s | ~tie |
| Summarise | 5K | 500 | 19.8s | 15.0s | Ornith |
| Chat/agent | 3K | 800 | 28.6s | 19.0s | Ornith |

**Implication for the 2nd-B60 re-deployment plan:**

Splitting on "small model for cheap tasks" intuition is wrong when brain routes both categorise AND summarise to the categorise endpoint. Better options once card 1 exists:

1. **Two identical Ornith slots** (same model on both cards, brain routes by queue depth) — same VRAM cost, wins every workload shape, no per-workload routing logic needed.
2. **Wait for Gemma 4 E2B MTP drafter** — if a community drafter lands (unsloth/protoLabsAI/SC117/huihui-ai all publish MTP heads for popular models), Gemma decode jumps 60-90 tps and the model-mix argument becomes real.
3. **Route by workload shape** in brain — categorise → Gemma card, summarise/chat → Ornith card. Requires brain to split its CATEGORISE_MODEL_CHEAP (short, Gemma) from CATEGORISE_MODEL_STRONG (long, Ornith) into different URLs, which the env vars already support.

Recommended: Option 1 unless Gemma MTP drafter appears. Option 3 if you want to preserve model diversity for capability reasons.

## Original bench numbers (kept for reference)

| Metric | Value |
|---|---|
| Params | 2B effective / ~5B total |
| Architecture | Gemma 4 (dense) |
| Quant | Q4_0 (Google QAT) |
| File size | 3.35 GB |
| Context deployed | 32,768 (`--parallel 2` = 16K/slot) |
| VRAM loaded | ~2 GiB |
| **Isolated decode** | **88 tps** (bench :8009, single-tenant B60) |
| **Under contention decode** | 12 tps (with Ornith active on same B60) |
| Categorise quality vs Ornith 9B on 10-task bench | matches 9/10 tasks |
| JSON validity | 10/10 |
