# Nemotron-3-Embed-1B (Q4_K_M) — Tested 2026-07-22

**Status:** Benched. Not promoted — 6–9× slower than incumbent EmbeddingGemma-300M on B60 SYCL, and current brain embed throughput isn't the bottleneck. Retained as capability upgrade if we need stronger multilingual retrieval or a 2048-dim embedding.
**HF (weights):** [`nvidia/Nemotron-3-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) · GGUF from [`zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF`](https://huggingface.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF)
**Base:** [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512), pruned via NVIDIA ModelOpt NAS and distilled from the 8B teacher
**Family:** Part of NVIDIA's Nemotron-3 Embed release (Jul 15, 2026); 8B sibling ranked #1 on RTEB.

## Specs

| | |
|---|---|
| Parameters | 1.14B (dense) |
| Architecture | `ministral3` (encoder used bidirectionally) |
| Hidden / embed dim | 2048 |
| Max sequence | 32,768 |
| File size (Q4_K_M) | 715 MB |
| Pooling | mean |
| Query/doc prefixes | `query: ` / `passage: ` |
| Multilingual | 34 languages |
| Published RTEB (BF16) | 72.4 NDCG@10 |
| Published MMTEB retrieval (BF16) | 71.0 |

## Benchmarks (b10068 SYCL, isolated, Q4_K_M)

Config: `--embedding --pooling mean -ngl 99 -c 32768 -b 2048 -ub 2048 --parallel 1 -fa off`.

`-fa on` produces NaN embeddings on this arch (bidirectional attention on the SYCL FA path). Must run with FA disabled until llama.cpp adds an encoder FA path or ministral3-specific handling.

| Input | Batch | n_tok | Median | tok/s | emb/s |
|---|---|---|---|---|---|
| short 65 tok  | 1  |    63 |  36 ms |  1,738 | 28 |
| med   665 tok | 1  |   663 | 167 ms |  3,976 |  6 |
| long 1800 tok | 1  | 1,803 | 537 ms |  3,361 |  2 |
| short 65 tok  | 8  |   504 | 282 ms |  1,789 | 28 |
| short 65 tok  | 32 | 2,016 | 1.12 s |  1,794 | 29 |
| short 65 tok  | 64 | 4,032 | 2.24 s |  1,796 | 29 |
| med   665 tok | 3  | 1,989 | 498 ms |  3,993 |  6 |

Peak throughput is ~4,000 tok/s at ~500-token inputs. Batched requests scale linearly (no parallelism win under `--parallel 1`, single-slot FIFO).

### Correctness

Model-card similarity table reproduced under Q4_K_M + SYCL to within 1–3 pp of the BF16 reference. Retrieval geometry is preserved by Q4_K_M quantization:

|      | d[0]   | d[1]   | d[2]   | d[3]   |
|------|--------|--------|--------|--------|
| q[0] | 0.8114 | 0.0436 | 0.0097 |-0.0131 |
| q[1] | 0.0549 | 0.6902 | 0.0047 | 0.0769 |
| q[2] |-0.0083 |-0.0261 | 0.5959 | 0.1450 |
| q[3] |-0.0123 | 0.0408 | 0.0996 | 0.7569 |

Diagonal matches card's expected 0.81/0.65/0.65/0.77.

## Head-to-head vs prod (EmbeddingGemma-300M Q8_0, `:8004`)

Same probe, same host, same SYCL runtime, both isolated:

| Metric | Nemotron-3-Embed-1B (Q4_K_M) | EmbeddingGemma-300M (Q8_0) | Ratio |
|---|---|---|---|
| Params | 1.14B | 300M | 3.8× |
| Embed dim | 2048 | 768 | 2.7× |
| Max seq | 32,768 | 8,192 | 4× |
| Multilingual | 34 langs | primarily EN | — |
| Published RTEB | 72.4 | ~57 (from HF blog leaderboard) | +15 pp |
| Weight file (bench quant) | 715 MB | 305 MB | 2.3× |
| **65-tok single** | 36 ms, 28 emb/s | **14 ms, 72 emb/s** | 2.6× faster (EG) |
| **665-tok single** | 167 ms, 6 emb/s | **28 ms, 36 emb/s** | 6× faster (EG) |
| **1800-tok single** | 537 ms, 2 emb/s | **78 ms, 13 emb/s** | 7× faster (EG) |
| **short batch 64** | 2.24 s, 29 emb/s | **252 ms, 253 emb/s** | 9× faster (EG) |
| Steady VRAM | ~1.0 GiB | 0.5 GiB | 2× |
| FA on SYCL | broken (NaN, arch-specific) | works | — |

EmbeddingGemma wins the throughput race decisively because (a) it's 4× smaller, (b) its BERT-style path uses SYCL FA (Nemotron can't), and (c) llama.cpp's encoder batching kernel handles it efficiently.

## Config used

```bash
docker run -d --name bench-embed \
  --memory=8g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/Nemotron-3-Embed-1B-GGUF:/models:ro \
  -p 0.0.0.0:8009:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/nemotron-3-embed-1b-q4_k_m.gguf \
  --embedding --pooling mean \
  -ngl 99 -c 32768 -b 2048 -ub 2048 \
  --parallel 1 \
  -fa off \
  --host 0.0.0.0 --port 8000 --metrics
```

Endpoint: `POST /v1/embeddings` (OpenAI-compatible). Prefix inputs with `query: ` for retrieval queries and `passage: ` for indexed documents.

## Verdict

**Not promoting.** EmbeddingGemma-300M is 6–9× faster on the same silicon for the actual brain workload (~500-1800 token document chunks), and prod embed throughput is not the current bottleneck (rerank + chat dominate).

**Would promote if:**
1. **Multilingual retrieval matters.** Nemotron covers 34 languages; EmbeddingGemma is EN-heavy. If brain ingests non-English source material, retrieval quality gap is real (+15 pp NDCG@10 headline).
2. **Higher-dim embeddings need to land.** Some downstream RAG tasks benefit from richer 2048-dim vectors vs 768-dim. Retrieval MRR@10 uplift on tough queries is often 5–10 pp when dim doubles.
3. **llama.cpp adds encoder FA on SYCL.** Right now `-fa on` produces NaN because the SYCL FA kernel assumes causal mask. If ministral3 encoder gets FA support upstream, Nemotron speeds up substantially and the gap closes.

**Watch:**
- **NVFP4 Blackwell quant** exists for this model (`nvidia/Nemotron-3-Embed-1B-NVFP4`) — irrelevant to B60 (Battlemage/Xe2 has no NVFP4 path), but illustrates NVIDIA is investing in embed-model quantization.
- **8B sibling.** `Abiray/Nemotron-3-Embed-8B-GGUF` exists; likely 3–4× slower than 1B here. Only worth benching if 1B quality doesn't cut it.
- **TEI XPU-IPEX path.** Not yet tried. TEI supports arbitrary encoder architectures on Intel XPU via IPEX and might unlock better throughput than the llama.cpp SYCL non-FA path. Would need model weights (BF16 safetensors), not GGUF.

## Bench provenance

- Session: 2026-07-22 13:15 local
- Image: `llama.cpp:sycl-f16` (b10068, commit `571d0d540`)
- Isolated: Ornith + embed prod + TEI all running; bench container was only additional process. B60 had ~11.4 GiB free before this bench.
- Quant: community Q4_K_M by [zenmagnets](https://huggingface.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF).
