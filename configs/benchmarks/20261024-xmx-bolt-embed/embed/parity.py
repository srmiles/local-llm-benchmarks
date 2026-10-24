#!/usr/bin/env python3
"""Compare a llama.cpp endpoint (client-rendered Nemotron prompts) to the ST CPU reference. usage: parity.py <url> <ref.npy> <tag>"""
import json, sys, numpy as np
sys.path.insert(0, ".")
from bench import post, norm, esc
url, ref, tag = sys.argv[1:4]
P = json.load(open("parity_raw.json")); R = np.load(ref)
texts = ["query: " + esc(x) for x in P["queries"]] + ["passage: " + esc(x) for x in P["docs"]]
E = np.vstack([post(url, texts[i:i + 8])[0] for i in range(0, len(texts), 8)])
nq = len(P["queries"]); cos = (norm(E) * norm(R)).sum(1)
Sg, Sr = norm(E[:nq]) @ norm(E[nq:]).T, norm(R[:nq]) @ norm(R[nq:]).T
r = dict(tag=tag, dim=int(E.shape[1]), cos_mean=float(cos.mean()), cos_min=float(cos.min()), cos_q_mean=float(cos[:nq].mean()),
         cos_doc_mean=float(cos[nq:].mean()), frac_ge_0_99=float((cos >= 0.99).mean()), simmatrix_pearson=float(np.corrcoef(Sg.ravel(), Sr.ravel())[0, 1]))
print(json.dumps(r)); json.dump(r, open(f"results/parity-{tag}.json", "w"), indent=1)
