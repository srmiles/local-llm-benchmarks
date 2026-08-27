#!/usr/bin/env python3
"""Controlled decode probe: every run is exactly N tokens.

bench-candidate.py lets the server stop at EOS. On this model the two arms hit
EOS at very different rates on the filler prompt (v2.0 ran 19/20 to full length,
v3.0 only 10/20), and a run that stops at 8 tokens reports first-token latency
dressed up as tok/s. That drags a median without any decode difference existing.
ignore_eos forces equal work per run so the arms are actually comparable.
"""
import argparse, json, statistics, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8020)
ap.add_argument("--name", required=True)
ap.add_argument("--runs", type=int, default=20)
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--out", required=True)
a = ap.parse_args()
U = f"http://localhost:{a.port}"

def metrics():
    with urllib.request.urlopen(f"{U}/metrics", timeout=10) as r:
        t = r.read().decode()
    m = {}
    for ln in t.splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        try:
            k, v = ln.rsplit(" ", 1); m[k] = float(v)
        except Exception: pass
    return m

def run():
    body = {"prompt": f"[uuid:{uuid.uuid4()}] Write a detailed technical explanation.",
            "n_predict": a.tokens, "ignore_eos": True, "cache_prompt": False,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    req = urllib.request.Request(f"{U}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())

m0 = metrics()
vals, ns = [], []
errors = []
for i in range(a.runs):
    try:
        d = run(); t = d["timings"]
    except Exception as e:
        errors.append(f"run{i+1}: {e}")
        print(f"  run{i+1:2d}: ERROR {e}", flush=True)
        continue
    vals.append(t["predicted_per_second"]); ns.append(t["predicted_n"])
    print(f"  run{i+1:2d}: {t['predicted_per_second']:.2f} tok/s  (n={t['predicted_n']})", flush=True)
if not vals:
    json.dump({"name": a.name, "errors": errors, "ok_runs": 0}, open(a.out, "w"), indent=1)
    raise SystemExit(f"all runs failed: {errors[:2]}")
try:
    m1 = metrics()
except Exception:
    m1 = dict(m0)
def dl(k): return m1.get(k, 0) - m0.get(k, 0)
dr, dt, ac = dl("llamacpp:spec_decode_num_drafts_total"), dl("llamacpp:spec_decode_num_draft_tokens_total"), dl("llamacpp:spec_decode_num_accepted_tokens_total")
if dr == 0:
    for k in m1:
        if "draft" in k: print("   metric:", k, m1[k]-m0.get(k,0))
res = {"name": a.name, "runs": a.runs, "tokens": a.tokens,
       "all_full_length": len(set(ns)) == 1 and ns[0] == a.tokens,
       "decode": {"median": round(statistics.median(vals), 2),
                  "mean": round(statistics.mean(vals), 2),
                  "stdev": round(statistics.stdev(vals), 2),
                  "min": round(min(vals), 2), "max": round(max(vals), 2)},
       "spec": {"drafts": dr, "draft_tokens": dt, "accepted": ac,
                "acceptance_pct": round(100*ac/dt, 1) if dt else None,
                "accepted_per_draft": round(ac/dr, 2) if dr else None},
       "per_run": [round(v,2) for v in vals], "ok_runs": len(vals), "errors": errors}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("name","all_full_length","decode","spec")}, indent=1))
