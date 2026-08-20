# Ornith 1.5 35B-A3B — APEX-MTP-Compact bench (mudler quant)

**Status:** Tested 2026-08-21, single-card SYCL on B60 #2. Follow-up to `ornith-1.5-35b-a3b-single-card.md` — testing whether mudler's APEX quant with Q8_0 MTP head fixes bartowski's poor MTP acceptance.
**HF quant:** [`mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF`](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) · file `Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf` (17.44 GB)
**APEX pitch:** layer-wise precision gradient — routed experts (89.6% of weights) compressed aggressively (only 8/256 fire per token), shared expert kept high, first/last layers high, MTP head **pinned to Q8_0** (per Compact tier docs).
**Base:** ornith-ai/Ornith-1.5-35B-A3B (arch `qwen3_5_moe`, 40 layers, 256 experts + 1 shared, 8 active per token, MTP as blk.40)
**Build:** `llama.cpp:sycl-f16` (b10433 — loaded qwen3_5_moe cleanly)
**Card:** Intel Arc Pro B60 #2 (level_zero:1) at 65K context

## Results — worse than bartowski, not better

| Metric | APEX-MTP-Compact (this) | bartowski IQ4_XS | 1.5-9B baseline |
|---|---|---|---|
| Weights on disk | 17.44 GB | 19.28 GB | 5.63 GB |
| Prefill @ ~500 tok | **497 tok/s** | 604 | 1,133 |
| Prefill @ ~2K tok | **1,019 tok/s** | 1,220 | 1,930 |
| Prefill @ ~4K tok | **1,211 tok/s** | 1,360 | 2,040 |
| Decode + MTP (greedy, 512 tok) | **25.6 tok/s** | 32.6 | 65.4 |
| MTP acceptance rate | **26.2%** (224/856) | 32.5% | 82.7% |
| Mean accepted per draft | 0.78 | 0.97 | 2.48 |
| Load time cold | 24 s | 20 s | 12 s |
| Engine resets during bench | 0 | 0 | 0 |

**MTP acceptance actually got worse** — 26.2% (APEX) vs 32.5% (bartowski) vs 82.7% (9B). And prefill/decode are both slower than bartowski IQ4_XS.

## Why the Q8_0 MTP head didn't help

Mudler's card documents that MTP is pinned to Q8_0 on Compact for exactly the reason we were trying to fix — "a drafter that mispredicts the target wastes the speculation it was added for." The high-precision MTP head is real. So why did acceptance drop?

Hypothesis: **APEX's aggressive routed-expert quantization skews the target model's output distribution far enough that even a perfect MTP head can't predict it.** The target isn't the "true" Ornith 1.5-35B any more — it's a heavily-quantized version whose specific token probabilities diverge from what the MTP head was trained on. The MTP layers were trained against the full-precision model; feeding them a compressed target creates a distribution mismatch that no MTP quantization quality can compensate for.

By contrast, bartowski's imatrix IQ4_XS uses imatrix calibration to preserve the output distribution more faithfully at the cost of a bigger file — so the Q4_0 MTP (which SHOULD be worse) actually gets more predictions right because it's predicting a more faithful target.

**Net: APEX's compression strategy is optimized for standalone quality-per-byte, not for speculative decoding compatibility.** The two goals are in tension when MTP is involved.

## Also notable

- Prefill throughput is ~18% lower than IQ4_XS despite smaller weights. Suggests APEX's quantization mix causes more per-token compute (mixed-precision expert dispatch may have more branching / less optimized SYCL kernels vs uniform K-quants).
- Zero engine resets, stable throughout — SYCL is happy with both quant families.

## What we've now ruled out for single-card 35B on B60

Two independent community quant approaches both fail on MTP acceptance:
- **bartowski imatrix IQ4_XS + Q4_0 MTP** → 32.5% acceptance, 32.6 tok/s decode
- **mudler APEX + Q8_0 MTP** → 26.2% acceptance, 25.6 tok/s decode

The 1.5-9B baseline achieves 82.7% MTP acceptance and 65 tok/s decode because we use protoLabsAI's separately-trained MTP drafter designed for the specific full-precision base model. There's no equivalent standalone MTP drafter published for 35B-A3B that would let us test the same architecture — every 35B option so far ships MTP embedded in the same file as the quantized base.

## Conclusion — path forward is not single-card

Single-card 35B-A3B is dead as a viable production upgrade on this stack. The bandwidth math (~30 tok/s max for 3B-active decode) combined with MTP's target/drafter distribution mismatch under compression means single-card 35B will always be significantly slower than 1.5-9B with functional MTP.

**Real paths forward, in priority order:**
1. **Tensor-split across 2× B60** (#142, blocked by #144 B580 migration) — bigger quants become viable (Q4_K_M 21.86 GB splits ~10.9 GB per card, leaving room for KV), MTP compatible with full-precision-target quants, and adds the cross-card benchmark data we need
2. **vLLM XPU** — Sergio Barrientos work suggested +5.2× prefill / +1.8× decode over llama.cpp SYCL for MoE on B60. Larger project (new runtime, new observability) but the natural pivot if #142 shows llama.cpp SYCL TP is inadequate for MoE
3. **Wait for a standalone 35B MTP drafter to be published separately** — none exist today; would need someone (protoLabsAI, googling shows nothing else) to distill an Ornith 1.5-35B MTP head against the full-precision model as they did for 9B

**Stay on Ornith 1.5-9B as prod** on both B60s until one of the above lands.

## Config used

```bash
docker run -d --name llamacpp-sycl-c2 \
  --memory=28g --memory-swap=30g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/cache/neo:/root/.cache/neo_compiler_cache \
  -v /data/llm/Ornith-1.5-35B-APEX:/models:ro \
  -p 0.0.0.0:8010:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  llama.cpp:sycl-f16 \
  -m /models/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --alias ornith-1.5-35b-a3b-apex \
  -ngl 99 -ngld 99 \
  -c 65536 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 \
  --top-k 20 --min-p 0.0
```

## Bench ops notes

- Same evacuation dance as `ornith-1.5-35b-a3b-single-card.md`: pulled card 2 from all 4 Traefik LB pools, stopped 4 card-2 containers, paused e2b-wedge-watchdog. Card 1 handled all prod traffic solo for ~15 min.
- Container restart + LB rejoin + watchdog resume all clean. Zero prod incidents observed.
- Round-robin verified via `https://llm.levirge.com/v1/chat/completions` returning `ornith-1.5-9b`.

## Sources

- [mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) (Compact 17.4 GB, MTP at Q8_0)
- [APEX quantization method](https://github.com/localai-org/apex-quant)
- Companion bench: `models/tested/ornith-1.5-35b-a3b-single-card.md` (bartowski IQ4_XS)
- Baseline: `models/tested/ornith-1.5-9b-first-bench.md`
