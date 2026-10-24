#!/usr/bin/env python3
"""Cross-check rerank.py against prod TEI (:8008, same bge-reranker-v2-m3): same 40 scifact queries x top-50."""
import json, sys, urllib.request, numpy as np
sys.path.insert(0, ".")
from bench import load, metrics, raw_doc
corpus, queries, qrels = load("scifact"); f = np.load("results/stage1-scifact.npz", allow_pickle=True)
qids, dids, S = list(f["qids"])[:40], list(f["dids"]), f["S"][:40]
top = np.argsort(-S, axis=1)[:, :50]; S2 = np.full_like(S, -1e9)
for i, q in enumerate(qids):
    texts = [raw_doc(corpus[dids[j]])[:4000] for j in top[i]]
    req = urllib.request.Request("http://localhost:8008/rerank", data=json.dumps({"query": queries[q], "texts": texts, "truncate": True}).encode(), headers={"Content-Type": "application/json"})
    for r in json.load(urllib.request.urlopen(req, timeout=120)): S2[i, top[i][r["index"]]] = r["score"]
qr = {str(q): qrels[str(q)] for q in qids}
print("40q before", json.dumps(metrics(S, qids, dids, qr))); print("40q TEI   ", json.dumps(metrics(S2, qids, dids, qr)))
