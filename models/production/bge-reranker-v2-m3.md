# bge-reranker-v2-m3 — Production rerank

**Status:** Production on **TEI XPU-IPEX (`:8008`)**. llama.cpp SYCL fallback on `:8007` **retired 2026-07-19** after TEI's empty_cache patch demonstrated 10+ hours of steady 1.4 GiB VRAM under real brain traffic. Launcher preserved at `/data/llm/launch/start-llamacpp-rerank.sh` for on-demand relaunch if TEI ever fails.
**HF:** [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
**Launchers:** [`start-tei-rerank.sh`](../../configs/launchers/start-tei-rerank.sh) (prod, :8008) · [`start-llamacpp-rerank.sh`](../../configs/launchers/start-llamacpp-rerank.sh) (fallback, :8007)

## Two serving paths

### 1. TEI XPU-IPEX on `:8008` (production, 7–9× faster)

| Metric | Value |
|---|---|
| Latency (25 pairs, isolated bench) | **109 ms** |
| Latency (observed, real brain workload) | 49–221 ms/batch (median ~130 ms) |
| Queue time (real workload) | 30–100 ms |
| VRAM (steady, patched image) | **1.43 GiB** (holding flat over 9+ hours real workload) |
| Format | HF safetensors (float16) |
| Backend | text-embeddings-inference on Intel XPU + IPEX (empty_cache patched) |

```bash
docker run -d --name tei-rerank \
  --restart unless-stopped \
  --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --shm-size=4g --ipc=host \
  -v /data/llm/bge-reranker-v2-m3-hf:/data:ro \
  -p 0.0.0.0:8008:80 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e PYTORCH_XPU_ALLOC_CONF=max_split_size_mb:256 \
  tei:xpu-ipex-nomemleak \
  --model-id /data --port 80 --dtype float16 --auto-truncate \
  --max-client-batch-size 64 \
  --max-batch-tokens 32768 \
  --max-concurrent-requests 128
```

**VRAM leak — fixed 2026-07-19.** Original `tei:xpu-ipex-fix` image leaked ~840 MB/hour under brain workload (2.5 → 10.9 GiB in 10 hours). Root cause: IPEX allocator holds intermediate tensors in a caching pool and TEI never called `torch.xpu.empty_cache()` between batches. Replaced with [`tei:xpu-ipex-nomemleak`](../../configs/images/tei-xpu-ipex-nomemleak/README.md) which adds the release call to `ClassificationModel.predict()` and the gRPC error path. **Confirmed working after 9 hours of real brain traffic**: VRAM 1.43 GiB flat vs unpatched projection of ~10 GiB. Fix validated empirically.

### 2. llama.cpp SYCL on `:8007` (retired 2026-07-19; on-demand fallback only)

| Metric | Value |
|---|---|
| Latency (25 pairs) | 800–1,000 ms (7–9× slower) |
| VRAM | ~4 GB |
| Format | GGUF Q8_0 |
| Context | 65,536 |
| Slots | 8 |

```bash
docker run -d --name llamacpp-rerank \
  --device=/dev/dri \
  -v /data/llm/bge-reranker-v2-m3:/models:ro \
  -p 8007:8000 \
  --restart unless-stopped \
  -e ZES_ENABLE_SYSMAN=1 \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 120s --health-retries 3 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/bge-reranker-v2-m3-Q8_0.gguf \
  --reranking \
  -ngl 99 \
  -c 65536 \
  --parallel 8 \
  --host 0.0.0.0 --port 8000 \
  -b 2048 -ub 2048 \
  --no-mmap \
  --metrics
```

## Notes

- TEI wins because IPEX's fused encoder-attention kernel beats llama.cpp's generic BERT path on this exact workload — reranker is encoder-only, no autoregressive decode, so llama.cpp's decode-focused optimizations don't help
- Title-prefix truncation bug (fixed in July) required verifying at `-c 16384` — could silently invert scores
- Custom TEI image (`tei:xpu-ipex-fix`) applies IPEX patches missing from upstream tei-xpu image
- Endpoints: Tailscale `http://100.70.193.48:8008/rerank`, LAN `http://192.168.1.253:8008`
- Both containers can coexist; brain hybrid-search rerank stage points at `:8008` by default
