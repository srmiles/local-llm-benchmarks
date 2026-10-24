# Nemotron-3.5-Embed 1B / 8B and Nemotron-3.5-Rerank-8B (NVIDIA) — Tested 2026-10-24

**Status:** Embed benched on llama.cpp b11850 with exact reference parity; rerank not completed.

- **Embed-1B loses to EmbeddingGemma-300M (prod) on all three text datasets** and runs at ~⅓ the throughput. Q8_0 and F16 are indistinguishable on quality.
- **Embed-8B is the first embedder here to beat EmbeddingGemma on prose retrieval** — NDCG@10 **+0.046 on scifact, +0.037 on nfcorpus** — but loses on code, and embeds at **5.6–7.9 docs/s vs 96–127**: ~16× slower, with 5.3× larger vectors.
- **llama.cpp's brand-new support is numerically exact**: cosine ≥ 0.9997 per text against NVIDIA's own sentence-transformers output at Q8_0, and 1.0000000 at F16.
- **Rerank-8B has no result.** It cannot load in TEI 1.9.3 or llama.cpp, and the PyTorch-XPU run hit a GPU memory fault 47 minutes in.
- **Side result that matters more:** prod `bge-reranker-v2-m3` *lowers* NDCG@10 when reranking EmbeddingGemma's top 50, on both datasets, and prod TEI agrees on a subset (finding **#62**).

Harness, results and parity data: [`configs/benchmarks/20261024-xmx-bolt-embed/embed/`](../../configs/benchmarks/20261024-xmx-bolt-embed/embed/).

**HF:** [`nvidia/Nemotron-3.5-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) · [`nvidia/Nemotron-3.5-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) · [`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)
**Released:** 2026-10-07 (embed) / 2026-10-15 (rerank) · **License:** OpenMDW 1.1
**Predecessor:** [Nemotron-3-Embed-1B](nemotron-3-embed-1b.md) (July; benched on b10068)

## Specs

| | Embed-1B | Embed-8B | Rerank-8B |
|---|---|---|---|
| HF arch | `Mistral3Model` (text tower `Ministral3Model`, `is_causal: false`) | same | `Mistral3ForSequenceClassification` + `2_Dense` head |
| Dim | 2048 | 4096 | 1 logit |
| Pooling | mean, L2-normalised | mean | — |
| Prefixes | `query: ` / `passage: ` via the chat template's system message | same | `query` / `document` roles via chat template |
| Context | 32,768 | 32,768 | 32,768 |
| Needs | transformers 5.20 (card) | same | same |
| Published scores | ViDoRe V3 / MIRACL-VISION (visual-document retrieval) — **no text-only MTEB on the card** | — | — |

That last row frames everything below: these are visual-document retrievers being measured on plain-text BEIR.

## Runtime support — what loads where

| Runtime | Embed | Rerank | Evidence |
|---|---|---|---|
| **TEI 1.9.3** (`tei:xpu-ipex-nomemleak`) | ✗ | ✗ | Three independent blockers: the router exits on `max_position_embeddings` missing at the top of `config.json` (it lives under `text_config`); transformers 4.51.3 raises `KeyError 'ministral3'`; the `TokenizersBackend` tokenizer class does not exist in 4.x. |
| **llama.cpp b11850** | ✓ | ✗ | Upstream [#30877](https://github.com/ggml-org/llama.cpp/pull/30877) "support Nemotron-3(.5)-Embed" (2026-10-21) + [#30974](https://github.com/ggml-org/llama.cpp/pull/30974) tokenizer fix. Converted GGUF: arch `mistral3`, `attention.causal = false`, `pooling_type = 1` (mean), pre-tokenizer `nemotron-3.5-embed`. The converter rejects the reranker: `Model Ministral3ForSequenceClassification is not supported`. |
| **PyTorch-XPU 2.14.1 + transformers 5.20 + sentence-transformers** | (not used) | loads; faulted | native XPU, no IPEX; needs `torchvision` for the Pixtral processor |

Two operational notes from the conversion:

- `llama quantize` inside the server image **initialises SYCL even to quantise** and dies with `can not find preferred GPU platform` without `/dev/dri`. `convert_hf_to_gguf.py --outtype q8_0` writes Q8_0 directly on CPU instead.
- Both transformers 5.19 and 5.20 log `Unrecognized keys in rope_parameters for 'rope_type'='yarn': {'apply_yarn_scaling'}` on these configs. It is not a version skew, and the exact parity below says it does not matter for these input lengths.

## Method

- **Server:** `launch-embed.sh` (prod embed flags) on card 2 `:8020`, `-c 8192 --parallel 4 -b/-ub 2048 --embeddings`. EmbeddingGemma re-run on the same image as the anchor.
- **Prompts:** client-side reproduction of the model's `chat_template.jinja`: `"query: " + text` / `"passage: " + text`, with the template's ZWNJ escaping after `[` and `<`. **Documents are `title + "\n" + text`** — the card does not specify a title join, so this is a choice, and a different join could move the scores.
- **Parity:** 32 queries + 64 documents vs a sentence-transformers `encode_query`/`encode_document` reference on CPU (fp32 for both sizes). This tests the conversion, the quant and the client-side prompt rendering together.
- **Retrieval:** BEIR scifact (5,183 docs / 300 queries), nfcorpus (3,633 / 323), csn-python. Same `bench.py` as the 2026-10-11 EmbeddingGemma 2 run, plus a `--style nemotron` prompt mode and 1024-dim truncation.

## Parity — exact

| | cos mean | cos min | sim-matrix Pearson |
|---|---|---|---|
| Embed-1B **F16** | **1.0000000** | 0.9999996 | 0.99999996 |
| Embed-1B Q8_0 | 0.99989 | 0.99963 | 0.99993 |
| Embed-8B Q8_0 | 0.99990 | 0.99971 | 0.99996 |

The port is correct, so the scores below are the models', not llama.cpp's.

## Retrieval (NDCG@10, full dim; b11850-patched, card 2)

| Model | scifact | nfcorpus | csn-python | docs/s | p50 / p95 query latency |
|---|---|---|---|---|---|
| **EmbeddingGemma-300M QAT Q8_0** (prod, dim 768) | 0.791 | 0.386 | **0.959** | **96–127** | 11–17 / 16–38 ms |
| Embed-1B Q8_0 (2048) | 0.751 | 0.366 | 0.920 | 29–41 | 7–11 / 9–27 ms |
| Embed-1B F16 (2048) | 0.751 | 0.365 | 0.919 | 35–47 | 10–13 / 12–21 ms |
| **Embed-8B Q8_0** (4096) | **0.837** | **0.423** | 0.943 | **5.6–7.9** | 28–39 / 32–135 ms |

EmbeddingGemma reproduced its 2026-10-11 numbers on the new image (scifact 0.7911, nfcorpus 0.3855 vs 0.3858), so the anchor is good.

**Truncation** (re-normalised; NVIDIA says end-truncation to 1024/512 still works):

| Model @ dim | scifact | nfcorpus | csn-python |
|---|---|---|---|
| EmbeddingGemma @ 768 / 512 | 0.791 / 0.776 | 0.386 / 0.375 | 0.959 / 0.961 |
| Embed-1B Q8 @ 1024 / 512 | 0.743 / 0.717 | 0.356 / 0.339 | 0.916 / 0.910 |
| Embed-8B Q8 @ 1024 / 512 | **0.806** / 0.751 | 0.383 / 0.325 | 0.942 / 0.934 |

**Truncating the 8B to 1024 keeps the scifact lead but loses nfcorpus** (0.383 vs 0.386). At 512 it is behind EmbeddingGemma everywhere. Storage-for-quality is not a free trade on this model.

**F16 is faster than Q8_0 for the 1B** (35–47 vs 29–41 docs/s) at identical quality — a 1B fits trivially, so F16 is the right pick if it is ever used.

## Rerank

### bge-reranker-v2-m3 (prod model) — reranking makes EmbeddingGemma worse

`rerank.py`: sentence-transformers `CrossEncoder` on XPU (fp16, `max_length` 1024), rescoring EmbeddingGemma's top 50 per query.

| | scifact NDCG@10 | nfcorpus NDCG@10 | pairs/s | peak mem |
|---|---|---|---|---|
| EmbeddingGemma alone | 0.791 | 0.386 | — | — |
| + bge-reranker-v2-m3, top-50 | **0.753 (−0.038)** | **0.353 (−0.033)** | 144–151 | 1.42 GiB |

Because this contradicts what a reranker is for, it was cross-checked against **production TEI `:8008`** — a different code path for the same model — on the first 40 scifact queries × the same top-50 (`tei-check.py`): **0.880 → 0.871**. The direction reproduces on the production path. The magnitudes are not comparable (40-query subset vs 300; per-query harness scores were not saved). Finding **#62**.

### Nemotron-3.5-Rerank-8B — GPU fault, no result

bf16 on XPU, batch 4, 17.4 GiB on card 2. At **03:30:30**, about 47 minutes into scifact's 15,000 pairs:

```
xe 0000:0b:00.0: [drm] Tile0: GT0: Engine memory CAT error [1]: class=ccs, logical_mask: 0x1, guc_id=3
xe 0000:0b:00.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs, logical_mask: 0x1, guc_id=3, state=0x249
xe 0000:0b:00.0: [drm] Tile0: GT0: Timedout job: seqno=6, ... in python [543369]
```

After the reset the process spun at 100% of one host thread with the card at 50 W (41 W idle) — the `urEventWait` spin signature from finding #43. It was terminated; the engine reset was clean this time, and `:8011` came back on card 2 with no further faults. **Throughput before the fault was low enough that a full two-dataset run would have taken ~2 hours** (no progress counter was logged; the estimate is from 47 min for under one dataset).

**Verdict: Rerank-8B is not runnable on this stack today** — not in TEI, not in llama.cpp, and not stably in PyTorch-XPU (finding #63).
