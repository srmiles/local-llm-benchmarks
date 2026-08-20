# Ornith 1.5 9B — First bench on B60

**Status:** Tested on card 2 (:8010), pulled from Traefik LB during bench. Card 1 (:8002) remains on 1.0-9B as prod.
**HF base:** [`ornith-ai/Ornith-1.5-9B-GGUF`](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) · Q4_K_M, 5.63 GB
**MTP drafter:** [`protoLabsAI/Ornith-1.5-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-MTP-GGUF) · Q8_0, 2.43 GB
**Base:** Qwen 3.5 9B fine-tune (arch `qwen3_5`) — recognized cleanly by llama.cpp SYCL b10433
**Launcher:** `/data/llm/launch/start-llamacpp-sycl-ornith-1.5-c2.sh` (clone of `-c2.sh`, new paths + `--alias ornith-1.5-9b`, all other flags identical)
**Build:** `llama.cpp:sycl-f16` (b10433)
**Card:** Intel Arc Pro B60 #2 (level_zero:1)
**Bench date:** 2026-08-21

## Benchmarks (2026-08-21, isolated, LB pulled)

### Prefill (3 samples per size, `cache_prompt=false`, UUID prefix, greedy)

| Size (tok, actual) | Mean prefill tok/s | 1.0-9B baseline (b10068) | Δ |
|---|---|---|---|
| ~500 (472)   | **1,133** | 927 | **+22%** |
| ~2K  (1805)  | **1,930** | 1,254 | **+54%** |
| ~4K  (4040)* | **2,040** | 1,106 (@8K) | **+84%** |
| ~4K  (4039)* | **2,044** | 969 (@12K)  | **+111%** |

\* filler pattern tokenizer-collapsed the 8K/12K prompts down to ~4K each — need to widen the filler set for larger samples. Still: 4K-context prefill is well above the 1.0 baseline at all comparable sizes.

Caveat: some of the uplift is build-related (b10068 → b10433 shipped further SYCL improvements). Not a pure model-to-model comparison. Ballpark: 1.5-9B prefill is at minimum comparable, likely genuinely faster on Battlemage.

### Decode + MTP (512 tokens, temp=0, greedy, cache off)

| Metric | 1.5-9B | 1.0-9B baseline | Δ |
|---|---|---|---|
| Decode tok/s | **65.44** | 50.4–53.4 (under load) | **+23%** |
| MTP acceptance rate | **82.7%** (364/440 draft tokens) | 68.9–76.6% | **+6 to +14pp** |
| Mean accepted per draft round | 2.48 | 3.07–3.28 | -0.6 (fewer per round, but rounds are tighter) |
| `predicted_ms` for 512 tok | 7.81 s | ~9.6 s | -19% wall |

Draft ratio interpretation: 1.5 accepts a higher fraction of proposed draft tokens (82.7%) but the drafter sometimes commits to shorter draft chains — net result is faster decode. Very promising for real-workload throughput.

### VRAM

| Component | 1.5-9B | 1.0-9B (same config) |
|---|---|---|
| Total on card 2 (post-warmup, 262K KV Q8) | **20.7 GiB** | ~18.8 GiB |
| Δ | +1.9 GiB | — |

+1.9 GiB is likely from the marginally larger base file (5.63 GB vs 5.0 GB) plus KV overhead of the slightly larger vocab/config. Still fits comfortably in 24 GB with headroom for a co-loaded small model.

### Sampling A/B (preliminary)

Ornith 1.5-9B card recommends `presence_penalty=1.5` for general tasks. Preliminary test with `temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` on a 100-token generation:
- Decode tok/s dropped to **28.88** (vs 65.44 greedy)
- Quality subjectively strong (well-formed haiku + coherent 100-word explanation)

Note: temperature=1.0 sampling always hurts MTP acceptance vs greedy — the drafter's most-likely-token predictions are less likely to be selected. The 28.88 figure is temperature-driven, not `presence_penalty`-driven. A proper A/B needs:
- Same seed, same prompt, temp=1.0 with pp=1.5 vs temp=1.0 with pp=0.0
- ≥256-token generation each

Deferred to next session — the greedy-decode gate has passed decisively, sampling tuning can be optimized without blocking a swap decision.

### Stability

- Model load time: **12 s** cold from launcher → `/health` = 200
- No `xe engine reset` events in dmesg during bench window
- Health probe stable throughout, no wedges

## Success gate scorecard

| Gate | Threshold | Result | Verdict |
|---|---|---|---|
| Prefill within ±5% of 1.0 | ≥ 875 tok/s @ 500 | 1,133 | ✅ pass (+22% over threshold) |
| Decode ≥ 48 tok/s | 48 | 65.4 | ✅ pass (+36% over threshold) |
| MTP acceptance ≥ 55% | 55% | 82.7% | ✅ pass (+27pp over threshold) |
| No engine resets during 15-min bench | 0 | 0 | ✅ pass |

**All gates pass decisively.** Ornith 1.5-9B is a clean upgrade path from 1.0-9B on this stack.

## What's untested / open

- Long-form quality vs 1.0 under real chat + pi.dev workloads (not yet A/B'd in prod)
- Real-workload MTP acceptance rate under mixed temp settings (bench used greedy)
- Impact of `presence_penalty=1.5` on quality + throughput vs current `--repeat-penalty 1.05`
- Tool-call surface (`qwen3_xml` parser) — reported as reasoning model but currently launched with `--reasoning off`; may want to flip to `--reasoning qwen3` to surface `reasoning_content` field
- Whether c1 also benefits from swap, or if we should keep 1.0 available for regression baseline

## Config used

```bash
docker run -d --name llamacpp-sycl-c2 \
  --memory=14g --memory-swap=16g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -v /data/llm/Ornith-1.5-9B-GGUF:/models:ro \
  -p 0.0.0.0:8010:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  llama.cpp:sycl-f16 \
  -m /models/Ornith-1.5-9B-Q4_K_M.gguf \
  --model-draft /models/mtp-Ornith-1.5-9B-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --alias ornith-1.5-9b \
  -ngl 99 -ngld 99 \
  -c 262144 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 \
  --repeat-penalty 1.05 --repeat-last-n 256 --min-p 0.0
```

## Recommendation

Ready to accept 1.5-9B on card 2 in the LB pool. Three sensible next steps:

1. **Rejoin card 2 into Traefik LB as-is** (mixed pool: card 1 = 1.0, card 2 = 1.5). Traefik LB is round-robin; response `model` field will report either alias depending on which backend served the request. Fine for headroom-proxy which doesn't validate model name. Real-world A/B evidence accumulates automatically.
2. **Swap card 1 to 1.5 as well** (all-1.5), rejoin card 2. Cleanest — no version skew. Some risk if 1.5 has unknown regression under specific prompt shapes we haven't tested.
3. **Keep card 2 out of LB, do a soak day** on the isolated :8010 endpoint (direct calls only) to build confidence before rejoining.

Steve to decide.
