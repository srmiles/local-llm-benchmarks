# Qwen 3.6-27B — Tested, not promoted

**Status:** Benched via LM Studio Vulkan (early) then llama.cpp SYCL. Dense-27B throughput uncompetitive vs MoE.
**HF:** `bartowski/Qwen3.6-27B-GGUF` (Q4_K_XL)

## Specs

| | |
|---|---|
| Parameters | 27B dense |
| Quant | Q4_K_XL |
| File size | ~16 GB |
| Context (deployed) | 128,000 |

## Benchmarks

### On b10068 SYCL (2026-07-19, isolated re-test)

| Metric | Value |
|---|---|
| Cold 12K prefill | 374 tok/s (27.5s) |
| Pure decode (warm) | ~18 tok/s |
| VRAM (32K, KV Q8) | 20.2 GiB |
| Co-residence with prod stack (2.4 GiB other) | 22.6 GiB / 24 — fits with 1.4 GiB headroom |

### Historical (initial bench, older builds)

| Metric | Value |
|---|---|
| Decode | ~22 tok/s |
| Prefill @ 12K | ~380 tok/s |
| VRAM (loaded) | ~17 GB |

## Verdict

Roughly half Gemma 4 26B-A4B's decode throughput at similar VRAM budget. Confirms the dense-vs-MoE pattern already established with Devstral 24B and Gemma 4 12B dense.

**Re-tested 2026-07-19 on b10068** to see whether the XMX+oneDNN FA path (which lifted Ornith's dense-9B GQA by +42%) would rescue dense-27B. It didn't — 374 tok/s cold prefill actually slightly WORSE than the earlier ~380 tok/s number. Decode ~18 tok/s vs ~22 tok/s previously. b10068's dense-model wins on Ornith 9B don't scale up to Qwen 3.6-27B dense; the model is still bandwidth-bound on this card at this size. **Finding 5 (MoE > dense on Battlemage) holds firmly** — confirming that the 35B-A3B MoE variants (49 tok/s decode with MTP) dominate this dense-27B (18 tok/s) at similar VRAM.
