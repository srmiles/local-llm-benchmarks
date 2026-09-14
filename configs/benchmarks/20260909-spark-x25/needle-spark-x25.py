#!/usr/bin/env python3
"""Needle-in-a-haystack across depth and length for Spark-X2.5-4B.

sliding_window=512 covers 27 of 36 layers; only 9 full-attention layers can carry
long-range retrieval. A single mid-context hit at 32K does not test that, so this
sweeps depth (start/quarter/middle/three-quarter/end) at increasing lengths.
"""
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8020)
ap.add_argument("--name", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--lengths", default="8000,32000,64000,120000")
args = ap.parse_args()
URL = f"http://localhost:{args.port}/v1/chat/completions"

NEEDLE = "The maintenance passphrase for the Battlemage rack is CORAL-ANVIL-77."
KEY = "CORAL-ANVIL-77"
FILLER = "Routine telemetry line: all subsystems nominal, no action required. "

def ask(text, timeout=1200):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 48,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or ""), round(time.time()-t0, 1), d.get("usage", {})

results = {"name": args.name, "needle": KEY, "grid": []}
for L in [int(x) for x in args.lengths.split(",")]:
    n_fill = max(2, L * 4 // len(FILLER))
    for label, frac in [("start", 0.0), ("quarter", 0.25), ("middle", 0.5),
                        ("three_quarter", 0.75), ("end", 0.98)]:
        body = [FILLER] * n_fill
        body.insert(int(len(body) * frac), NEEDLE + " ")
        prompt = ("".join(body) +
                  "\n\nQuestion: what is the maintenance passphrase for the Battlemage rack? "
                  "Answer with the passphrase only.")
        try:
            txt, wall, usage = ask(prompt)
            ok = KEY in txt.upper()
        except Exception as e:
            txt, wall, usage, ok = f"ERROR {e}", -1, {}, False
        ntok = usage.get("prompt_tokens")
        results["grid"].append({"target_tokens": L, "prompt_tokens": ntok,
                                "depth": label, "pass": ok, "wall_s": wall,
                                "answer": txt.strip()[:80]})
        print(f"{L:>7} tok (actual {ntok})  depth={label:<14} "
              f"{'PASS' if ok else 'FAIL'}  {wall}s  {txt.strip()[:60]!r}")

npass = sum(1 for g in results["grid"] if g["pass"])
results["summary"] = {"passed": npass, "total": len(results["grid"])}
print(f"\n{args.name}: {npass}/{len(results['grid'])} retrieved")
if args.out:
    json.dump(results, open(args.out, "w"), indent=2)
    print("wrote " + args.out)
