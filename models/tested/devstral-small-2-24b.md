# Devstral Small 2 24B — Tested, not promoted

**Status:** Benched on b9777 SYCL. Dense penalty visible; not competitive against 26B-4B-active MoE.
**HF:** `mistralai/Devstral-Small-2506` · unsloth Q4_K_XL GGUF

## Specs

| | |
|---|---|
| Parameters | 24B dense |
| Quant | UD-Q4_K_XL |
| File size | ~14 GB |
| Context (deployed) | 128,000 |

## Benchmarks

| Metric | Value |
|---|---|
| Decode | ~18 tok/s |
| Prefill @ 12K | ~340 tok/s |
| VRAM (loaded) | ~15 GB |

## Verdict

Dense 24B decode is ~2.5× slower than Gemma 4 26B-A4B (44 tok/s) on the same hardware. Illustrates Finding 5 from the blog: **MoE beats parameter count on Battlemage.** No capability advantage on pi.dev workload to justify the throughput hit — not pursued.
