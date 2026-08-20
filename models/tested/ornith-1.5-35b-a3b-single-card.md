# Ornith 1.5 35B-A3B — Single-card B60 bench (IQ4_XS)

**Status:** Tested 2026-08-21, single-card SYCL on B60 #2, isolated (all card-2 services stopped + Traefik LB pulled). NOT deployed to prod. Reverted to 1.5-9B mirror after bench.
**HF quant:** [`bartowski/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) · IQ4_XS 19.28 GB, imatrix-calibrated, MTP layers baked in (Q4_0 for MTP weights)
**Base:** ornith-ai/Ornith-1.5-35B-A3B (arch `qwen35moe`, 35B total, ~3B active per token, 8+ experts, MTP built-in)
**Launcher:** `/data/llm/launch/start-llamacpp-sycl-ornith-35b-c2.sh` (temporary — bench only)
**Build:** `llama.cpp:sycl-f16` (b10433 — bartowski quantized on b10472 but our b10433 loaded it cleanly, no build bump needed)
**Card:** Intel Arc Pro B60 #2 (level_zero:1)
**Bench date:** 2026-08-21

## Headline numbers

| Metric | 1.5-35B-A3B (this) | 1.5-9B baseline (same b10433) | Δ |
|---|---|---|---|
| Prefill @ ~500 tok | **604 tok/s** | 1,133 | -47% |
| Prefill @ ~2K tok | **1,220 tok/s** | 1,930 | -37% |
| Prefill @ ~4K tok | **1,360 tok/s** | 2,040 | -33% |
| Decode + MTP (greedy, 512 tok) | **32.61 tok/s** | 65.44 | -50% |
| MTP acceptance rate | **32.5%** | 82.7% | -50pp |
| Mean accepted per draft | 0.97 | 2.48 | -60% |
| VRAM at 65K context | **22.5 GiB** (94% of card) | 20.7 GiB | +1.8 |
| Load time cold | 20 s | 12 s | +8 s |
| Engine resets during bench | 0 | 0 | ✅ |

**Verdict: not a slam-dunk single-card upgrade.** MoE decode drops ~50%, MTP is nearly useless with bartowski's Q4_0 MTP weights, VRAM has almost no headroom.

## What worked

- **b10433 loaded it cleanly** — no build bump required despite bartowski quantizing on b10472. `qwen35moe` arch supported.
- **Bartowski's baked-in MTP layers work** — `--spec-type draft-mtp` with just `-m model.gguf` (no `--model-draft`). Simpler wiring than the 1.5-9B two-file setup.
- **Stable under bench** — no engine resets, no hangs, container healthy throughout.
- **Chat template + reasoning parser** — template loaded fine, `--reasoning off` works, alias `ornith-1.5-35b-a3b` reported correctly.

## What didn't work well

### MTP acceptance collapse (32.5% vs 82.7% for 9B)

Bartowski documents that MTP layers are stored at **Q4_0** in imatrix quants because "imatrix calibration does not exercise them. Q4_0 is chosen for its speed which massively benefits MTP performance." In practice, this means the MTP head is much lower-quality than the main model — the draft tokens are less likely to match target-model greedy output, so acceptance craters.

For 1.5-9B we're using protoLabsAI's dedicated Q8_0 MTP head, which is trained to match the main model precisely. Very different acceptance profile (82.7% vs 32.5%).

Options to fix (untested):
- Use `mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF` standalone drafter (0.4B params, separate quant, probably higher acceptance)
- Use bartowski's Q8_0 quant (37.8 GB) which stores MTP layers at higher precision — but that doesn't fit on a single B60 anymore
- Run without MTP entirely (`--spec-type disabled`) — decode probably similar (32-40 tok/s), no wasted draft compute

### Prefill throughput regression (33-47% slower)

MoE prefill still has to touch a large fraction of the expert weights for gating decisions and prefill batching. The 3B "active" figure describes decode, not prefill. On a bandwidth-limited B60, reading 19 GB of weights vs 5.6 GB (9B) dominates prefill time. Expected result.

Note: the 8K and 12K bench sizes tokenized down to only ~4K each because the filler pattern hit the tokenizer's dictionary limit — real 8K/12K prefills would likely be closer to the 4K number (throughput plateaus) rather than significantly slower.

### Decode near halving (32.6 tok/s vs 65.4)

For a 35B MoE with 3B active params, single-B60 decode SHOULD be roughly comparable to a 3B dense model — memory bandwidth per token step is the bottleneck. 32.6 tok/s is consistent with reading ~3B params + shared attention state per step at B60's memory bandwidth (456 GB/s per tile). Not bad in absolute terms; just not obviously faster than 1.5-9B on the same card.

### VRAM headroom gone (22.5 / 24 GiB, 94%)

At 65K context we're at 94% VRAM. Any increase in context (or co-loaded services) would OOM. This confirms that **single-card 35B is only viable at reduced context** — the 262K context we run 1.5-9B at is out of reach single-card.

## The tradeoff analysis

Ornith 1.5-35B-A3B claims (from HF card, 5-run averages vs Ornith 1.0 baselines):

| Benchmark | 1.5-35B-A3B | 1.5-9B | Δ (35B advantage) |
|---|---|---|---|
| Terminal-Bench 2.1 (Terminus-2) | 67.8 | 46.2 | +21.6 |
| Terminal-Bench 2.1 (Claude Code) | 68.5 | 47.0 | +21.5 |
| SWE-bench Verified | 79.0 | 70.6 | +8.4 |
| SWE-bench Pro | 59.6 | 47.5 | +12.1 |
| NL2Repo | 46.2 | 32.4 | +13.8 |
| HLE (with tools) | 33.4 | 30.5 | +2.9 |
| GPQA Diamond | 89.2 | 86.4 | +2.8 |
| MCP-Atlas | 70.2 | 54.2 | +16.0 |
| BrowseComp | 67.6 | 56.4 | +11.2 |

**Cost paid for those benchmark wins on our stack (single card):**
- 50% slower decode
- 33-47% slower prefill
- 94% VRAM (no headroom for co-loaded services)
- MTP effectively broken (32.5% acceptance = barely helping)

For agentic coding workloads (which are throughput-tolerant but quality-sensitive), the tradeoff might be worthwhile — Terminal-Bench +21 points is a big real-world win, and 32 tok/s is still faster than a human reads. For pi.dev chat and interactive agents where wall-clock latency matters more, 1.5-9B remains the better choice.

## What to test next

1. **mudler's dedicated MTP drafter** (`Ornith-1.5-35B-A3B-APEX-MTP-GGUF`, 0.4B, standalone) — likely gets MTP acceptance back to 70%+ range. Decode could jump 50-80%.
2. **Ornith-ai official Q4_K_M (21.86 GB)** — official quant, no imatrix, MTP layers at full quality. Wouldn't fit on single card at any real context; needs tensor-split.
3. **Tensor-split across both B60s (task #142)** — the real intended deployment. Should give more VRAM for context, might restore MTP-friendly quant, but adds PCIe cross-card latency.
4. **vLLM XPU port** — Sergio Barrientos' work suggested vLLM XPU + MTP gets +5.2× prefill / +1.8× decode over llama.cpp SYCL on B60 for MoE specifically. If MoE is our future, vLLM XPU migration deserves serious consideration.

## Config used

```bash
docker run -d --name llamacpp-sycl-c2 \
  --memory=28g --memory-swap=30g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -v /data/llm/Ornith-1.5-35B-A3B-GGUF:/models:ro \
  -p 0.0.0.0:8010:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  llama.cpp:sycl-f16 \
  -m /models/Ornith-1.5-35B-A3B-IQ4_XS.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --alias ornith-1.5-35b-a3b \
  -ngl 99 -ngld 99 \
  -c 65536 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 \
  --top-k 20 --min-p 0.0
```

## Bench operational notes

- **Card 2 was fully evacuated** — Ornith 1.5-9B mirror stopped, embed-c2 stopped, tei-rerank-c2 stopped, llamacpp-categorise (E2B primary) stopped, e2b-wedge-watchdog paused
- **Card 1 alone served prod during ~30 min bench window** — Traefik LB stripped to single backend per service
- **Zero prod incidents observed** during bench — brain-eval's backoff behavior worked as designed, chat/pi.dev traffic continued serving from card 1 alone
- **Restoration was clean** — all 4 card-2 services back healthy within 90 s of tear-down, LB and watchdog restored, LB round-robin verified via `https://llm.levirge.com` returning "ornith-1.5-9b" model

## Recommendation

- **Do not deploy 35B-A3B single-card** on this stack. The 50% decode + broken MTP tradeoff isn't compensated by benchmark wins for our workload mix (chat + pi.dev + brain-eval categorise).
- **Retry with mudler's standalone MTP drafter** as a quick follow-up — if that fixes acceptance, decode could jump to 50+ tok/s and change the calculus.
- **The real test for 35B-A3B is tensor-split across 2× B60** (task #142), which needs the B580 migration first to free the cards. Once that's done, the split gives us:
  - Better quant options (Q4_K_M fits split, MTP layers at higher precision)
  - Real headroom for context (~13 GB KV per card at 262K)
  - Real-world data on B60 SYCL multi-GPU maturity
- **Keep 1.5-9B as prod chat** until at least one of those experiments shows a decisive win.

## Sources

- [bartowski/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) (IQ4_XS, imatrix, MTP baked in)
- [mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) (standalone drafter, future test)
- Baseline: `models/tested/ornith-1.5-9b-first-bench.md`
- Research brief: `docs/research/ornith-1.5-upgrade.md`
