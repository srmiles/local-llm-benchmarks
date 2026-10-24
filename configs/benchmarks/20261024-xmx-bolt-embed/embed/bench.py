#!/usr/bin/env python3
"""Retrieval-quality + throughput/latency bench against a llama.cpp /v1/embeddings endpoint.
usage: bench.py --url http://127.0.0.1:8020 --tag v2-q8 [--datasets scifact,nfcorpus,csn-python]
Prompts (identical for v1/v2): query='task: search result | query: ', doc='title: {t} | text: {x}' (t='none' if empty)
"""
import argparse, json, os, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QP = "task: search result | query: "
CODE_QP = "task: code retrieval | query: "

TRUNC = {"n": 0}

def _raw(url, texts, model, timeout):
    req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps({"model": model, "input": texts}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return np.array([x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])], dtype=np.float32), d.get("usage", {}).get("prompt_tokens", 0)

def post(url, texts, model="bench", timeout=180):
    """On 5xx (input exceeds slot/ubatch tokens): split batch; single oversize text -> halve until it fits."""
    try:
        return _raw(url, texts, model, timeout)
    except urllib.error.HTTPError as e:
        if e.code < 500:
            raise
        if len(texts) > 1:
            m = len(texts) // 2
            a, ta = post(url, texts[:m], model, timeout); b, tb = post(url, texts[m:], model, timeout)
            return np.vstack([a, b]), ta + tb
        t = texts[0]
        for _ in range(6):
            t = t[: len(t) // 2]; TRUNC["n"] += 1
            try:
                return _raw(url, [t], model, timeout)
            except urllib.error.HTTPError as e2:
                if e2.code < 500:
                    raise
        raise

def jl(p):
    return [json.loads(l) for l in open(p)]

def load(name):
    d = os.path.join(HERE, "data", name)
    corpus = {(c.get("_id") or c.get("id")): c for c in jl(f"{d}/corpus.jsonl")}
    queries = {(q.get("_id") or q.get("id")): q["text"] for q in jl(f"{d}/queries.jsonl")}
    qrels = {}
    for r in jl(f"{d}/qrels/test.jsonl"):
        if float(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = float(r["score"])
    queries = {k: v for k, v in queries.items() if str(k) in qrels}
    return corpus, queries, qrels

STYLE = {"s": "gemma"}

def esc(x):
    # Nemotron-3.5-Embed chat_template.jinja mark_user_text(): ZWNJ after '[' and '<'
    return x.replace("[", "[\u200c").replace("<", "<\u200c")

def raw_doc(c):
    t = (c.get("title") or "").strip()
    return f"{t}\n{c['text']}" if t else c["text"]

def doc_text(c, maxchars):
    if STYLE["s"] == "nemotron":  # template: system prefix rstrip + " " + user content
        return ("passage: " + esc(raw_doc(c)))[:maxchars]
    t = (c.get("title") or "").strip() or "none"
    return f"title: {t} | text: {c['text']}"[:maxchars]

def query_text(name, q):
    if STYLE["s"] == "nemotron":
        return "query: " + esc(q)
    return (CODE_QP if name.startswith("csn") else QP) + q

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def embed_all(url, texts, bs, workers):
    batches = [texts[i:i + bs] for i in range(0, len(texts), bs)]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda b: post(url, b), batches))
    dt = time.time() - t0
    return np.vstack([r[0] for r in res]), sum(r[1] for r in res), dt

def metrics(S, qids, dids, qrels, k=10):
    nd, mrr, r10, r100 = [], [], [], []
    top = np.argsort(-S, axis=1)[:, :100]
    for i, q in enumerate(qids):
        rel = qrels[str(q)]
        ranked = [str(dids[j]) for j in top[i]]
        gains = [rel.get(d, 0.0) for d in ranked[:k]]
        dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        nd.append(dcg / idcg if idcg else 0)
        rr = next((1 / (r + 1) for r, d in enumerate(ranked[:k]) if rel.get(d, 0) > 0), 0)
        mrr.append(rr)
        r10.append(len({d for d in ranked[:10] if rel.get(d, 0) > 0}) / len(rel))
        r100.append(len({d for d in ranked if rel.get(d, 0) > 0}) / len(rel))
    return dict(ndcg10=float(np.mean(nd)), mrr10=float(np.mean(mrr)), recall10=float(np.mean(r10)), recall100=float(np.mean(r100)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default="scifact,nfcorpus,csn-python")
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--maxchars", type=int, default=6000)
    ap.add_argument("--lat-n", type=int, default=100)
    ap.add_argument("--style", default="gemma", choices=["gemma", "nemotron"])
    a = ap.parse_args(); STYLE["s"] = a.style
    out = {"tag": a.tag, "url": a.url, "datasets": {}}
    for _ in range(5):  # warmup
        post(a.url, ["warmup"])
    for name in a.datasets.split(","):
        corpus, queries, qrels = load(name)
        dids = list(corpus); qids = list(queries)
        dt_txt = [doc_text(corpus[d], a.maxchars) for d in dids]
        q_txt = [query_text(name, queries[q]) for q in qids]
        D, dtok, ddt = embed_all(a.url, dt_txt, a.bs, a.workers)
        Q, qtok, qdt = embed_all(a.url, q_txt, a.bs, a.workers)
        full = D.shape[1]
        res = {"truncated_requests": TRUNC["n"], "n_docs": len(dids), "n_queries": len(qids), "dim": full, "doc_tokens": dtok,
               "doc_embed_s": round(ddt, 2), "docs_per_s": round(len(dids) / ddt, 1), "doc_tok_per_s": round(dtok / ddt, 0),
               "quality": {}}
        if os.environ.get("STAGE1_DUMP"):
            np.savez(os.environ["STAGE1_DUMP"] + f"-{name}.npz", qids=np.array(qids, dtype=object), dids=np.array(dids, dtype=object), S=norm(Q) @ norm(D).T)
        for dim in sorted({full, 1024, 512, 256, 128}, reverse=True):
            if dim > full:
                continue
            Dn, Qn = norm(D[:, :dim]), norm(Q[:, :dim])
            res["quality"][str(dim)] = metrics(Qn @ Dn.T, qids, dids, qrels)
        # single-request latency (sequential), queries only
        lats = []
        for t in q_txt[:a.lat_n]:
            t0 = time.time(); post(a.url, [t]); lats.append((time.time() - t0) * 1000)
        res["lat_ms_p50"] = round(statistics.median(lats), 1)
        res["lat_ms_p95"] = round(sorted(lats)[int(len(lats) * 0.95) - 1], 1)
        out["datasets"][name] = res
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        json.dump(out, open(os.path.join(HERE, "results", a.tag + ".json"), "w"), indent=1)
        print(name, json.dumps(res["quality"][str(full)]), "docs/s", res["docs_per_s"], "p50", res["lat_ms_p50"], flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", a.tag + ".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
