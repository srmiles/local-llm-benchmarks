#!/usr/bin/env python3
"""CPU fp32 reference via sentence-transformers encode_query/encode_document (NVIDIA's own template). usage: ref.py <model_dir> <out.npy>"""
import json, sys, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(6)
P = json.load(open("parity_raw.json")); t0 = time.time()
m = SentenceTransformer(sys.argv[1], device="cpu", model_kwargs={"dtype": getattr(torch, sys.argv[3]) if len(sys.argv) > 3 else torch.float32})
q = m.encode_query(P["queries"], batch_size=4, normalize_embeddings=True)
d = m.encode_document(P["docs"], batch_size=4, normalize_embeddings=True)
e = np.vstack([q, d]).astype(np.float32); np.save(sys.argv[2], e)
print("saved", e.shape, "nan", bool(np.isnan(e).any()), round(time.time() - t0, 1), "s")
