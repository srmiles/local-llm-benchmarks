# LFM2.5-ColBERT-350M (Q8_0) — Tested 2026-07-22

**Status:** Benched. ~2.7× slower than production TEI bge-reranker-v2-m3 for the same 25-pair rerank workload, and doesn't unify embed+rerank without a 21× storage cost. **Not a rerank replacement**, but retained as a **combined multilingual retriever + reranker** if brain ever shifts to a ColBERT-native architecture.
**HF:** [`LiquidAI/LFM2.5-ColBERT-350M`](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) · [GGUF](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M-GGUF)
**Base:** LFM2.5-350M with bidirectional patches, trained as ColBERT late-interaction retriever

## Specs

| | |
|---|---|
| Parameters | 350M dense |
| Output | **Per-token vectors** (not pooled) |
| Per-token dim | 128 |
| Query prefix | `[Q] ` |
| Doc prefix | `[D] ` |
| Query length | 32 tokens (padded) |
| Doc length | 512 tokens (truncated) |
| Multilingual | 11 languages |
| File size (Q8_0) | 378 MB |
| Pooling flag | `--pooling none` (llama.cpp) |
| Endpoint | `/embedding` (llama.cpp native, not `/v1/embeddings`) |
| FA on SYCL | **broken** (NaN vectors, same bidirectional-attention issue as Nemotron-3-Embed) — **must run with `-fa off`** |

## Serving path

llama.cpp with `--embeddings --pooling none -fa off` serves the per-token vectors. Client applies MaxSim scoring:

```python
def maxsim(q_emb, d_emb):
    return (normalize(q_emb) @ normalize(d_emb).T).max(axis=1).sum()
```

LiquidAI's reference script adds tokenizer-driven padding and a skiplist (punctuation, stopword tokens dropped from doc vectors). For a latency bench the skiplist is throughput-neutral and can be omitted; for retrieval quality it matters.

## Correctness

Query: "How do transformers use attention to model dependencies?"

Top 5 by MaxSim:

| Score | Doc |
|---|---|
| 9.40 | The Transformer architecture, introduced in 2017, relies on self-attention. |
| 9.20 | Feed-forward networks apply position-wise transformations after attention. |
| 9.11 | Multi-head attention allows the model to jointly attend to different subspaces. |
| 9.08 | The encoder-decoder attention lets the decoder attend to the encoder outputs. |
| 9.07 | The GPT family uses decoder-only transformer architecture. |

Bottom 3 (correctly demoted):

| Score | Doc |
|---|---|
| 8.64 | Vector databases index high-dimensional embeddings for similarity search. |
| 8.62 | Cross-encoders score query-document pairs jointly for reranking. |
| 8.55 | Cosine similarity is a common distance metric for text embeddings. |

Ranking is correct — the transformer-attention docs dominate, off-topic RAG-adjacent docs are demoted.

## Bench (b10068 SYCL, isolated, -fa off)

25-pair rerank (1 query + 25 candidate docs, batched in single `/embedding` call, MaxSim in Python/numpy):

| Metric | Value |
|---|---|
| **25-pair rerank latency** | **293.6 ms** (median over 5 samples, spread <2 ms) |
| Single-query embed latency | 11.5 ms |
| Compute breakdown | ~275 ms GPU embed + ~18 ms client MaxSim |

## Head-to-head vs bge-reranker-v2-m3 (TEI XPU-IPEX, prod)

Same 25-pair rerank workload:

| Metric | LFM2.5-ColBERT-350M | bge-reranker-v2-m3 (TEI :8008) | Winner |
|---|---|---|---|
| Params | 350M | 568M | ColBERT (smaller) |
| **25-pair latency (isolated)** | **293.6 ms** | **109 ms** | **bge (2.7×)** |
| VRAM | ~0.6 GiB | 1.4 GiB | ColBERT |
| Multilingual | 11 languages | 100+ languages | bge (broader) |
| Server framework | llama.cpp SYCL | TEI XPU-IPEX | — |
| FA on SYCL | broken | works | bge |
| Scoring model | late interaction (MaxSim) | cross-encoder (single score head) | — |
| Storage cost per doc (if used as retriever) | ~64 KB (512 tok × 128 dim × 1 byte Q8) | N/A (score only) | — |

bge is faster because it's a fused cross-encoder — one forward pass produces one score per pair, and TEI's IPEX backend fuses attention efficiently. ColBERT needs to embed all tokens then transfer them for client MaxSim.

## Verdict

**Not a rerank replacement for bge-reranker-v2-m3.** 2.7× slower with no quality data yet to justify the trade.

**Would be interesting if:**
1. **Unified embed + rerank stack via ColBERT.** ColBERT can serve as both first-stage retriever (with per-token vector index) AND reranker (query online, no index needed for rerank pass). Would consolidate EmbeddingGemma + bge-reranker into one LFM2.5-ColBERT model. **But** per-token vector storage is ~64 KB/doc for a 512-token doc, vs 3 KB/doc for EmbeddingGemma-300M single vectors — a **~21× storage blowup**. For brain's document scale, this is meaningful.
2. **Multilingual retrieval quality wins the bake-off.** LiquidAI claims "best-in-class accuracy across 11 languages" for LFM2.5-ColBERT. If a real multilingual eval shows a clear margin, the rerank-only path could be worth 2.7× latency.
3. **`-fa on` gets fixed upstream.** Same SYCL bidirectional-attention FA bug we hit on Nemotron-3-Embed. If llama.cpp adds encoder FA, ColBERT would speed up ~30-40% and the gap to bge closes to maybe 2×.

## Bench provenance

- Session: 2026-07-22 20:50 local
- Image: `llama.cpp:sycl-f16` (b10068, commit `571d0d540`)
- Isolated: prod stack + bench container on `:8009`, `-fa off` required
- Full bench script preserved at `/tmp/colbert_rerank_bench2.py` on llm.local
- Skiplist masking omitted for latency bench (has no throughput effect; matters for retrieval quality)
