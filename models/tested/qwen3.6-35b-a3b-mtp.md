# Qwen 3.6-35B-A3B (MTP variant) — Tested 2026-07-19, standout

**Status:** Benched. **The strongest reasoning-upgrade candidate found so far.** Nearly matches Ornith 9B's decode speed at 4× the parameter count. Not promoted — needs real quality bake-off (KB dual-eval + pi.dev win rate) before displacing Ornith.
**HF:** [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
**Arch:** `qwen35moe` with MTP layers embedded (llama.cpp b10068+ detects via `--spec-type draft-mtp`)

## Specs

| | |
|---|---|
| Parameters | 35.5B total / **3B active** (MoE) |
| Quant | UD-Q4_K_XL |
| File size | 21.28 GB |
| Context (trained) | 262,144 |
| Context (bench) | 32,768 |
| MTP | Embedded in base GGUF (no separate drafter file needed) |

## Benchmarks (b10068 SYCL, isolated)

| Metric | Value | vs Ornith prod | vs base Qwen 3.6-35B-A3B (pre-MTP) |
|---|---|---|---|
| **Cold 12K prefill** | 798 tok/s (16.3s) | 89% of Ornith's 896 | flat |
| **Decode (12K cold)** | **40.5 tok/s** | 78% of Ornith's 51.8 | **+30% vs 31.1 base** |
| **Decode (5K + 300 gen)** | **49.0 tok/s** | 95% of Ornith's 51.8 | **+58% vs 31.1 base** ⭐ |
| **5K prefill** | 974.5 tok/s | 80% of Ornith's 1,213 | +18% vs 823 base |
| MTP draft acceptance | 77.8% (mean 2.56 accepted) | vs Ornith 70-76% | — |
| VRAM (32K ctx, KV Q8) | **24.4 GiB (tight)** | 2.2× Ornith | +4 GB vs base (MTP layers) |
| Correctness (chat + tool call) | ✓ | — | — |

## Config

```bash
docker run -d --name bench-sycl \
  --memory=22g --memory-swap=22g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/Qwen3.6-35B-A3B-MTP-GGUF:/models:ro \
  -p 0.0.0.0:8019:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --predict 300 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning off
```

## Verdict

**Speed:** Essentially at Ornith parity for the workload that matters (5K prefill + 300 tok decode). MTP head does exactly what unsloth advertises — took Qwen 3.6-35B-A3B from 31 tok/s decode (base, no MTP) to 49 tok/s (with MTP). +58% throughput just by using the right GGUF.

**Capacity:** 35.5B total params at only 3B active — much more knowledge and reasoning capacity than Ornith 9B while executing at similar throughput. This is the MoE promise finally paying off at our scale.

**VRAM trade-off:** 24.4 GiB / 24 at 32K context is tight. Ornith at 128K only uses 10.9 GiB. Going to 128K on Qwen 3.6-35B-A3B-MTP would require CPU offload or quant drop. Real cost of the extra capacity.

### ⚠ Co-residence problem

The isolated bench (24.4 GiB at 32K) **does not fit** alongside the current production stack:

| Component | Steady VRAM |
|---|---|
| llamacpp-embed | 0.5 GiB |
| llamacpp-rerank (fallback) | 0.5 GiB (up to 3-4 during active batches) |
| tei-rerank (patched) | 1.4 GiB |
| **Non-chat total** | **~2.4 GiB** |
| **Available for chat model** | **21.6 GiB** |

`UD-Q4_K_XL` at 32K context (24.4 GiB) exceeds this by 2.8 GiB. **Cannot promote without compromise.**

Paths to make it fit:
1. **Drop quant to `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`** (16.96 GB file, saves ~4.3 GB) → isolated ~20 GiB → co-resident ~22.4 GiB → **fits with 1.6 GiB headroom**. Small quality drop.
2. **Drop context 32K → 16K** → saves ~1.5 GiB of KV → tight fit at 32K equivalent-user-experience.
3. **Drop context 32K → 8K** → saves ~3 GiB → comfortable ~21.5 GiB total. Bad for a coding agent.
4. **Swap `llama.cpp:sycl-f16` rerank off when Qwen3.6 is loaded** → saves 0.5-4 GiB depending on activity. TEI-only rerank path.

**Recommended path if we promote:** rebench with `UD-IQ4_XS` (option 1). Same MTP head, smaller weights, may sacrifice ~2-3% decode but gain the VRAM headroom needed for real production.

### IQ4_XS bench (2026-07-19 follow-up) — trade doesn't pencil out

Rebenched `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` from the same MTP repo, same config, isolated:

| Metric | UD-Q4_K_XL | **UD-IQ4_XS** | Δ |
|---|---|---|---|
| Cold 12K prefill | 798 tok/s (16.3s) | 776 tok/s (16.8s) | -3% |
| Decode (12K cold) | 40.5 tok/s | 34.5 tok/s | -15% |
| **Decode (5K + 300 gen)** | **49.0 tok/s** | **31.6 tok/s** | **-36%** ⚠ |
| 5K prefill | 974 tok/s | 918 tok/s | -6% |
| MTP draft acceptance | 77.8% (mean 2.56) | 60.8% (mean 2.20) | -22 pp |
| VRAM (32K ctx, KV Q8) | 24.4 GiB | **21.1 GiB** | -3.3 GiB |
| Co-residence (with 2.4 GiB other) | 26.8 GiB (fail) | 23.5 GiB (0.5 GiB headroom) | tight fit |

**Two compounded losses:**

1. **IQ-family kernels have more decode overhead than K-quants on B60** — same finding we hit comparing Ornith Q4_K vs IQ4 variants: on this card, K-quants beat i-matrix quants for decode throughput.
2. **MTP acceptance dropped 22 percentage points** (77.8 → 60.8) — the smaller quant compromised the drafter head's calibration against the target distribution.

Compound decode drop: **-36% at the 5K + 300gen workload that matters most** (49 → 31.6 tok/s). That takes it from "nearly Ornith parity" to "notably slower than Ornith" — the whole reason we tested it was to preserve MTP throughput while making room for co-residence.

**Verdict:** IQ4_XS is not the escape hatch. Better options if we still want to promote a 35B-A3B variant:
- Claude APEX-MTP Compact (19.4 GiB isolated, 21.8 co-res, 36.9 tok/s decode) — already benched, only 35B-A3B variant that fits comfortably.
- Try `UD-Q4_K_S.gguf` (19.92 GB file) — K-quant family, likely preserves decode better than IQ4_XS. **Benched below — doesn't save enough VRAM to fit prod.**
- Try `UD-Q4_K_M.gguf` (21.11 GB file) — even closer to K_XL. Not yet benched.
- Or **stay with Ornith**. It's not worth trading 36% decode for a 4× parameter bump if the parameters can't be shown to translate to real capability wins on our specific workloads.

### Q4_K_S bench (2026-07-19 follow-up #2) — still doesn't fit

Rebenched `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (19.92 GB file, one K-quant step down from Q4_K_XL) to see if a smaller K-quant preserves decode better than IQ4_XS while still saving VRAM:

| Metric | UD-Q4_K_XL | **UD-Q4_K_S** | UD-IQ4_XS |
|---|---|---|---|
| Cold 12K prefill | 798 tok/s | **820 tok/s** | 776 |
| Decode (12K cold) | 40.5 tok/s | **44.3 tok/s** | 34.5 |
| Decode (5K + 300 gen) | **49.0 tok/s** | 37.7 | 31.6 |
| 5K prefill | 974 tok/s | **985 tok/s** | 918 |
| MTP acceptance | 77.8% | 64.8–71.1% | 60.8% |
| VRAM (32K, KV Q8) | 24.4 GiB | **24.1 GiB** | 21.1 GiB |
| Co-residence (+2.4 GiB) | 26.8 (fail) | **26.5 (still fails)** | 23.5 (tight fit) |

**Two surprises:**

1. **Q4_K_S actually beats Q4_K_XL on prefill** (820 vs 798 cold; 985 vs 974 at 5K) — smaller K-quant means faster kernel dispatch. First time seeing "smaller quant, faster prefill" in the K-quant family on this workload.
2. **But decode regresses on 5K+300gen** (49.0 → 37.7 tok/s) because MTP acceptance dropped (77.8% → 71%). The drafter head's calibration is quant-sensitive; even within K-quant family, the smaller variant compromises MTP more than the raw weight loss suggests.

**VRAM savings are only 0.3 GiB.** Not remotely enough to escape co-residence. The Q4_K family variants all cluster in a narrow band (21-24 GB isolated) that fails co-residence at 32K context.

### Final conclusion after three quant benches

The Q4_K variants of Qwen 3.6-35B-A3B-MTP are **speed-competitive with Ornith** but **fundamentally too VRAM-heavy** to fit alongside embed + rerank + TEI at 32K context. There is no "smaller quant of the winner" that both preserves decode and fits.

**Real paths forward if this model gets promoted:**
1. **Claude APEX-MTP Compact** (16.14 GB, 19.4 GB VRAM isolated) — only 35B-A3B variant that co-resides. Uses smaller mudler APEX quant + tighter reasoning distillation. See [that file](qwen3.6-35b-a3b-claude-distilled.md).
2. **Cut context from 32K → 16K on Q4_K_XL** — saves ~1.5 GiB KV. Would fit if we accept the smaller window. Bad for pi.dev `/deep_review` workloads.
3. **Evict `llamacpp-rerank` fallback container** to save ~0.5 GiB — leaves TEI-only rerank path. Might be enough to squeeze Q4_K_XL in.
4. **Stay with Ornith 9B + MTP.** 51.8 tok/s decode + 128K context + 10.7 GiB headroom is a genuinely good production config. Any Qwen 3.6-35B-A3B replacement needs to win a real capability bake-off to justify the co-residence acrobatics.

## What's next

Before promotion, needs the same eval lanes Ornith won on:
1. **Brain KB dual-eval** — 45-question categorise-quality bake-off (Ornith scored .80)
2. **pi.dev win rate** (Ornith scored 66.7%)
3. **Tool call reliability** at long context (Gemma 4 26B-A4B had loop issues that killed it)
4. **Context ceiling check** — can we get from 32K → 64K → 128K on this VRAM budget?

If it wins those, this replaces Ornith. If not, sits in `tested/` as the "reasoning candidate we tried".

## Bench provenance

- Session: 2026-07-19 21:07 local
- Image: `llama.cpp:sycl-f16` (b10068)
- Isolated: yes (Ornith stopped, only bench + embed + rerank running)
- Note: initial 2026-07-19 base bench used `unsloth/Qwen3.6-35B-A3B-GGUF` (base, no MTP file). Retest here uses the correct `-MTP-GGUF` repo variant.
