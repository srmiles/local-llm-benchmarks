FROM tei:xpu-ipex-fix

# Overlay patched Python backend files that add torch.xpu.empty_cache() calls
# to fix the IPEX allocator VRAM leak (observed ~840 MB/hour under brain workload).
# See ./README.md for root-cause analysis.

COPY classification_model.py /usr/src/backends/python/server/text_embeddings_server/models/classification_model.py
COPY interceptor.py          /usr/src/backends/python/server/text_embeddings_server/utils/interceptor.py
