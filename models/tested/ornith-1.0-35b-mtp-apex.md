# Ornith 1.0 35B (MTP APEX I-Compact) — Tested 2026-07-22

**Status:** Benched. **Fits co-residence comfortably (~3 GiB headroom)** but **decode is 32% slower than Ornith 9B prod** and MTP acceptance drops 12–20 pp. Not promoted — sits in `tested/` as the "IQ-quant MTP variant of Ornith 35B we tried." Awaits a K-quant MTP variant to be worth a real capability bake-off.
**HF (target):** [`deepreinforce-ai/Ornith-1.0-35B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B)
**GGUF (self-contained target + MTP):** [`SC117/Ornith-1.0-35B-MTP-APEX-GGUF`](https://huggingface.co/SC117/Ornith-1.0-35B-MTP-APEX-GGUF)
**Base:** Qwen 3.5 MoE post-trained by DeepReinforce AI for agentic coding

## Specs

| | |
|---|---|
| Parameters | 35B total / **3B active per token** |
| Experts | 256 routed, 8 active per token |
| Layers | 40 transformer + 1 MTP layer (embedded) |
| Arch | `qwen3_5_moe` (same class as prod Ornith 9B and Qwen 3.6-35B-A3B) |
| Context (trained) | 262,144 |
| Context (bench) | 32,768 |
| MTP | 1 embedded MTP layer (785 tensors) — no separate drafter file |
| Multimodal | ships with `mmproj-F16.gguf` (vision projector, 0.9 GB) |
| Quant variant | APEX I-Compact (17.02 GB, IQ-family) |

APEX bundles ship as target+MTP in a single GGUF, matching the Qwen 3.6-35B-A3B-MTP layout llama.cpp handles via `--spec-type draft-mtp`. Only IQ-family quants are published for MTP — no K-quant MTP variant exists yet.

## Benchmarks (b10068 SYCL, isolated, single-slot)

Bench method matches the [Qwen 3.6-35B-A3B family sweep](qwen3.6-35b-a3b-mtp.md) exactly: unique UUID prefix per prompt, `cache_prompt=false`, real 300-token generation task (transformer attention explanation, `ignore_eos=true`), 3 samples per size.

| Metric | Value | vs Ornith 9B prod | vs Qwen 3.6-35B-A3B UD-Q4_K_XL MTP |
|---|---|---|---|
| Cold 5K prefill | 816 tok/s | -33% vs 1,213 | -16% vs 974 |
| Cold 12K prefill | 802 tok/s | -10% vs 896 | flat |
| **Decode (5K + 300 gen)** | **35.4 tok/s** | **-32% vs 52** | **-28% vs 49** |
| Decode (12K + 300 gen) | 36.9 tok/s | -29% vs ~52 | -9% vs 40.5 |
| MTP acceptance | **56.3% (562/999)** | -12 to -20 pp vs 68–76 | -22 pp vs 77.8 |
| Estimated VRAM (32K ctx, KV Q8) | ~19 GiB isolated | 1.8× Ornith 9B | -5 GiB vs Q4_K_XL |
| **Co-residence** (+1.9 GiB) | **~20.9 GiB (fits, 1.2 GiB headroom)** | vs 12.7 co-res | vs 26.8 (fails) |
| Correctness (chat) | ✓ | — | — |

Prefill and decode-per-sample were tightly consistent across the three samples per size (spread <5 tok/s).

## Config

```bash
docker run -d --name bench-ornith35 \
  --memory=22g --memory-swap=22g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/Ornith-1.0-35B-MTP-APEX-GGUF:/models:ro \
  -p 0.0.0.0:8019:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/Ornith-1.0-35B-MTP-APEX-I-Compact.gguf \
  -ngl 99 -c 32768 --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --predict 300 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning off
```

## Verdict

**Not promoting.** Same story as our Qwen 3.6-35B-A3B UD-IQ4_XS finding:

1. **IQ-quant kernel decode penalty on B60.** APEX I-Compact is IQ-family; same finding we hit on the Qwen 3.6-35B-A3B sweep — IQ quants underperform K quants on Battlemage. Decode drops 28–32% vs the Q4_K_XL Qwen we already benched.
2. **MTP acceptance drops with smaller quant.** 56% here vs 77.8% on Qwen Q4_K_XL. Consistent with the "MTP acceptance is quant-sensitive" finding — drafter calibration degrades faster than raw quant math suggests.
3. **35B/3B active × IQ-Compact loses to 9B dense × Q4_K_M + MTP on this silicon.** Ornith 9B prod hits 52 tok/s decode at 10.9 GiB VRAM footprint. Ornith 35B APEX I-Compact hits 35 tok/s at ~19 GiB. The parameter count doesn't translate to throughput.

**What would change the answer:**
- **K-quant MTP variant lands.** If someone publishes `Ornith-1.0-35B-MTP` in UD-Q4_K_S or UD-Q4_K_M format (K-quant + embedded MTP), the Qwen 3.6-35B-A3B family sweep suggests decode would climb to ~45–49 tok/s and MTP acceptance to 70–78%. Watch [SC117](https://huggingface.co/SC117) and [unsloth](https://huggingface.co/unsloth/Ornith-1.0-35B-GGUF) for K-quant + MTP combos.
- **Capability wins the bake-off.** Even at 35 tok/s decode, if this beats Ornith 9B by a clear margin on KB dual-eval + pi.dev win rate (Ornith 9B baselines: .80 and 66.7%), the throughput trade might be worth it for reasoning-heavy tasks. The APEX author's BenchLocal scores (100 ToolCall / 92 BugFind / 89 HermesAgent-20 in no-thinking mode) suggest agentic capability is strong — but that's the author's bench on RTX 5070 Ti, not our KB corpus.
- **Multimodal need emerges.** This is the only VLM-capable option we've benched. If pi.dev / brain need image inputs, this becomes the default candidate for that reason alone.

## What's next

If a K-quant MTP variant appears:
1. Re-bench at UD-Q4_K_S and UD-Q4_K_M with same methodology
2. If decode ≥45 tok/s and MTP ≥70%, run KB dual-eval + pi.dev bake-off vs Ornith 9B
3. If Ornith 35B wins the bake-off, promote and add mmproj vision support for pi.dev

Otherwise this file stays as a data point: Ornith 35B on B60 needs K-quants + MTP combined, and neither release nor community has published that combination.

## Bench provenance

- Session: 2026-07-22 13:38 local
- Image: `llama.cpp:sycl-f16` (b10068, commit `571d0d540`)
- Coordination window: brain-eval handoff `46e4232929f`, ~10-minute :8002 downtime for VRAM
- Isolated: Ornith 9B prod stopped for bench; embed + TEI stayed up
- Full bench script preserved at `/tmp/ornith35_bench2.py` on llm.local
