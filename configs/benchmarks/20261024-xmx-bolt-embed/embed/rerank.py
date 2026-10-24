#!/usr/bin/env python3
"""Rerank top-K candidates from a first-stage embedding run with a CrossEncoder on XPU; ndcg@10 before/after + pairs/s.
usage: rerank.py <model_dir> <tag> [--k 50] [--dtype bf16] [--first results/<stage1>.npz]
First stage: EmbeddingGemma prod embeddings saved by bench (or BM25-free: uses saved D/Q from stage1 npz)."""
import argparse, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load, metrics, raw_doc
from sentence_transformers import CrossEncoder
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag")
ap.add_argument("--k", type=int, default=50); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--datasets", default="scifact,nfcorpus"); ap.add_argument("--first", default="results/stage1")
a = ap.parse_args()
dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
m = CrossEncoder(a.model, device="xpu", model_kwargs={"dtype": dt}, max_length=1024)
out = {"tag": a.tag, "model": a.model, "k": a.k, "dtype": a.dtype, "datasets": {}}
for name in a.datasets.split(","):
    corpus, queries, qrels = load(name); f = np.load(f"{a.first}-{name}.npz", allow_pickle=True)
    qids, dids, S = list(f["qids"]), list(f["dids"]), f["S"]
    top = np.argsort(-S, axis=1)[:, :a.k]
    pairs = [(queries[q], raw_doc(corpus[dids[j]])[:4000]) for i, q in enumerate(qids) for j in top[i]]
    m.predict(pairs[:32], batch_size=a.bs)  # warmup
    torch.xpu.synchronize(); t0 = time.time()
    sc = np.asarray(m.predict(pairs, batch_size=a.bs), dtype=np.float32).reshape(len(qids), a.k)
    torch.xpu.synchronize(); el = time.time() - t0
    S2 = np.full_like(S, -1e9)
    for i in range(len(qids)): S2[i, top[i]] = sc[i]
    out["datasets"][name] = {"before": metrics(S, qids, dids, qrels), "after": metrics(S2, qids, dids, qrels),
                             "pairs": len(pairs), "pairs_per_s": round(len(pairs) / el, 1),
                             "peak_mem_gib": round(torch.xpu.max_memory_allocated() / 2**30, 2)}
    print(name, json.dumps(out["datasets"][name]), flush=True)
json.dump(out, open(f"results/rerank-{a.tag}.json", "w"), indent=1)
