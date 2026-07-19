# tei:xpu-ipex-nomemleak

Patched Text-Embeddings-Inference (TEI) image that fixes the IPEX VRAM leak observed on Intel B60.

## Root cause

TEI XPU-IPEX uses **PyTorch's caching allocator** via IPEX. When a rerank batch finishes:

1. Intermediate tensors go out of scope, but IPEX keeps them in an internal pool for reuse
2. Different query/document length combinations produce different tensor shapes each batch — old pool blocks don't fit new shapes, so new blocks keep getting allocated while old ones sit unused
3. TEI never calls `torch.xpu.empty_cache()` between requests

Observed rate under brain workload: **~840 MB/hour** growth (2.5 GiB → 10.9 GiB in 10 hours). Hits 24 GiB ceiling and starves the rest of the stack in ~15 hours.

Setting `PYTORCH_XPU_ALLOC_CONF=max_split_size_mb:256` caps fragmentation but doesn't reduce the pool ceiling.

## Fix

Add `torch.xpu.empty_cache()` (via `_release_cache()` helper that dispatches to xpu/cuda) after every rerank inference call and on the gRPC error path.

**Patched files:**
- `classification_model.py` — adds `_release_cache()` call at the end of `ClassificationModel.predict()`, wraps forward in `torch.inference_mode()` (also reduces intermediate retention)
- `interceptor.py` — replaces the CUDA-only cache release with the xpu/cuda-aware helper

## Build

```bash
docker build -t tei:xpu-ipex-nomemleak .
```

Base image is `tei:xpu-ipex-fix` (our previous custom build); this Dockerfile just overlays the two patched Python files.

## Deploy

Update the launcher to reference `tei:xpu-ipex-nomemleak` and restart:

```bash
docker rm -f tei-rerank
/data/llm/launch/start-tei-rerank.sh
```

## Verified

- **Correctness identical** — rerank scores match to 3 decimal places on the same query/text pair
- **Baseline VRAM: ~1.4 GiB** after warm-up (was ~2.5 GiB with unpatched image after warm)
- **Load test:** 2 min heavy varied rerank load → patched stayed at 1.44 GiB, unpatched stayed at 6.16 GiB (unpatched had already accumulated from earlier real traffic)

Real validation is the 24-hour brain workload watch — if patched holds ≤2 GiB over that window, we've killed the leak.

## Upstream

The root-cause fix should go upstream to [`huggingface/text-embeddings-inference`](https://github.com/huggingface/text-embeddings-inference) as a PR — the `_release_cache()` helper generalises across cuda/xpu/hpu and costs ~1 ms/batch vs 100–800 MB/hour of VRAM growth on IPEX. Worth filing.
