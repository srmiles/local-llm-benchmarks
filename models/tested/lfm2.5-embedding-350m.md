# LFM2.5-Embedding-350M (Q8_0) — Tested 2026-07-22

**Status:** Benched. Roughly parity with EmbeddingGemma-300M on throughput at same VRAM class. Retained as a **multilingual retrieval upgrade candidate** — 11 languages vs EG's English-heavy training. Not promoted; brain workload is currently English and embed is not the bottleneck.
**HF:** [`LiquidAI/LFM2.5-Embedding-350M`](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) · [GGUF](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M-GGUF)
**Base:** LFM2.5-350M-Base with bidirectional attention patches (first bidirectional member of the LFM2 family)

## Specs

| | |
|---|---|
| Parameters | 350M dense |
| Architecture | LFM2 (hybrid conv+attn) with bidirectional patches |
| Hidden / embed dim | 1024 |
| Pooling | CLS (model default) |
| Max sequence | 8,192 (native), longer via extrapolation |
| Multilingual | 11 languages: en, es, de, fr, it, pt, ar, sv, no, ja, ko |
| File size (Q8_0) | 378 MB |
| First-party GGUF | yes (LiquidAI publishes it) |
| FA on SYCL | works (unlike Nemotron-3-Embed and ColBERT variant) |

## Correctness

Query: "Represent this query for retrieving relevant documents: What is the capital of France?"

| Doc | Cosine sim |
|---|---|
| "Paris is the capital and largest city of France." | **0.5429** |
| "London is the capital of the United Kingdom." | 0.2914 |
| "Berlin is the capital of Germany." | 0.2454 |

Ranking correct, gap between right and wrong is substantial.

## Throughput (b10068 SYCL, isolated, `--pooling` = model default = CLS)

| Input | Batch | n_tok | Median | tok/s | emb/s |
|---|---|---|---|---|---|
| short 65t | 1  |    52 |  13.7 ms |  3,795 | 73 |
| med 665t  | 1  |   552 |  32.9 ms | 16,765 | 30 |
| long 1800t| 1  | 1,502 |  69.2 ms | 21,717 | 15 |
| short 65t | 8  |   416 | 100.5 ms |  4,141 | 80 |
| short 65t | 32 | 1,664 | 399.1 ms |  4,169 | 80 |
| short 65t | 64 | 3,328 | 803.1 ms |  4,144 | 80 |
| med 665t  | 3  | 1,656 |  95.9 ms | 17,271 | 31 |

## Head-to-head vs EmbeddingGemma-300M Q8_0 (prod)

Same probe, same host, same isolated conditions:

| Metric | LFM2.5-Embedding-350M | EmbeddingGemma-300M | Winner |
|---|---|---|---|
| Params | 350M | 300M | comparable |
| Embed dim | **1024** | 768 | LFM2 (richer) |
| **65-tok single** | 13.7 ms, 73 emb/s | 14.0 ms, 72 emb/s | tie |
| **665-tok single** | 32.9 ms, 30 emb/s | 28.0 ms, 36 emb/s | EG (+18%) |
| **1800-tok single** | 69.2 ms, 15 emb/s | 77.7 ms, 13 emb/s | LFM2 (+13%) |
| **64× short batch** | 803 ms, 80 emb/s | **252 ms, 253 emb/s** | **EG (3.2×)** |
| Multilingual | 11 languages | primarily English | LFM2 |
| Steady VRAM | ~0.6 GiB | ~0.5 GiB | EG (marginal) |
| FA on SYCL | works | works | tie |

EG dominates on **batched short inputs** (3.2×), which is the shape of brain's real embed workload — indexing ingest is many small chunks per batch. LFM2 slightly wins on single long inputs but that's not a common workload.

## Config

```bash
docker run -d --name bench-lfm2-embed \
  --memory=3g \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/LFM2.5-Embedding-350M-GGUF:/models:ro \
  -p 0.0.0.0:8009:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/LFM2.5-Embedding-350M-Q8_0.gguf \
  --embedding \
  -ngl 99 -c 8192 -b 2048 -ub 2048 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 --metrics
```

No `--pooling` flag — takes model default (CLS). Explicitly passing `--pooling mean` gets logged as a mismatch warning but still runs. `-fa on` works (unlike Nemotron-3-Embed which needed FA off).

## Verdict

**Not promoting today.** EG dominates batched short (the actual brain workload shape), and multilingual isn't a real requirement yet.

**Would promote if:**
1. **Multilingual content ingestion becomes real.** Brain ingesting non-English documentation, papers, or user-provided content at scale → LFM2.5-Embedding likely wins retrieval quality outright vs EG (LiquidAI's claim: beats Qwen3-Embedding-0.6B on MKQA-11 cross-lingual and NanoBEIR multilingual, at 350M vs 600M).
2. **RAG quality wall on high-dim vectors.** 1024-dim vectors carry ~1.75× more information per doc than 768-dim; some retrieval failure modes on tough queries respond to that. Would need an evaluation harness (RTEB or a bespoke KB dual-eval variant) to know.
3. **LFM2-ColBERT-350M** promotes and we consolidate to a single first-party LiquidAI stack — LFM2.5-Embedding for dense first-stage, LFM2.5-ColBERT for rerank, mutual multilingual coverage.

## Bench provenance

- Session: 2026-07-22 20:40 local
- Image: `llama.cpp:sycl-f16` (b10068, commit `571d0d540`)
- Isolated: prod stack (Ornith 9B, EmbeddingGemma, TEI) all running; bench container on `:8009` added ~0.6 GiB VRAM, no other change
- Both LFM2 and EG bench used identical probe script and prompts
