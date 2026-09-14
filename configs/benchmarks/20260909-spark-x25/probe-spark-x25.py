#!/usr/bin/env python3
"""Functional probe for Spark-X2.5-4B — the claims the tok/s bench cannot test.

The model card's headline is agentic: tau3-bench 30.4 vs Qwen3.5-9B's 9.3,
MCP-Atlas 54.6 vs 47.4, BrowseComp 40.9 vs 8.3. bench-candidate.py hits /completion
(raw), so it never exercises the chat template, thinking mode or tool calling.
This does, plus a needle test because sliding_window=512 across 27 of 36 layers is
an aggressive bet for a model advertising 1M context.

usage: probe-spark-x25.py --port 8020 --name spark-q4 [--out x.json]
"""
import argparse, json, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8020)
ap.add_argument("--name", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--needle-tokens", type=int, default=32000)
args = ap.parse_args()
URL = f"http://localhost:{args.port}/v1/chat/completions"

def chat(messages, tools=None, thinking=None, max_tokens=512, temp=0.6, timeout=900):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temp,
            "top_p": 0.95, "top_k": 20}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    d["_wall_s"] = round(time.time() - t0, 2)
    return d

def msg(d):
    return d["choices"][0]["message"]

results = {"name": args.name, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tests": {}}
def record(k, ok, **kw):
    results["tests"][k] = dict(pass_=ok, **kw)
    print(f"[{'PASS' if ok else 'FAIL'}] {k}" + (f"  {kw.get('note','')}" if kw.get('note') else ""))

# 1. coherence, thinking off -----------------------------------------------
try:
    d = chat([{"role": "user", "content": "In one sentence: why is a sliding-window "
               "attention layer cheaper than a full-attention layer?"}], thinking=False)
    txt = msg(d).get("content") or ""
    leaked = "<think>" in txt
    ok = len(txt.strip()) > 40 and not leaked
    record("coherence_thinking_off", ok, chars=len(txt), think_tag_leaked=leaked,
           wall_s=d["_wall_s"], sample=txt[:400])
except Exception as e:
    record("coherence_thinking_off", False, error=str(e))

# 2. thinking on ------------------------------------------------------------
try:
    d = chat([{"role": "user", "content": "A shelf holds 3 red books, 5 blue books and "
               "twice as many green books as red. How many books total? Show reasoning."}],
             thinking=True, max_tokens=1024)
    m = msg(d)
    txt = (m.get("content") or "")
    reasoning = m.get("reasoning_content") or ""
    # 3 red + 5 blue + 6 green = 14
    ok = "14" in (txt + reasoning)
    record("thinking_on_arithmetic", ok, answer_has_14=ok,
           reasoning_chars=len(reasoning), content_chars=len(txt),
           wall_s=d["_wall_s"], sample=(reasoning or txt)[:400])
except Exception as e:
    record("thinking_on_arithmetic", False, error=str(e))

# 3. tool calling — the headline claim --------------------------------------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_meter_reading",
        "description": "Return the latest interval reading for an electricity meter.",
        "parameters": {
            "type": "object",
            "properties": {
                "meter_no": {"type": "string", "description": "Meter serial number"},
                "date": {"type": "string", "description": "ISO date, e.g. 2026-09-08"},
            },
            "required": ["meter_no"],
        },
    },
}]
try:
    d = chat([{"role": "user", "content": "What did meter E4471290 read on 2026-09-08?"}],
             tools=TOOLS, thinking=False)
    m = msg(d)
    tc = m.get("tool_calls") or []
    ok, note, argsd = False, "no tool_calls emitted", None
    if tc:
        fn = tc[0]["function"]
        try:
            argsd = json.loads(fn["arguments"])
        except Exception:
            argsd = fn["arguments"]
        ok = fn["name"] == "get_meter_reading" and isinstance(argsd, dict) \
             and argsd.get("meter_no") == "E4471290"
        note = f'{fn["name"]}({argsd})'
    record("tool_call_single", ok, note=note, n_calls=len(tc), args=argsd,
           wall_s=d["_wall_s"])
except Exception as e:
    record("tool_call_single", False, error=str(e))

# 4. tool result round-trip --------------------------------------------------
try:
    convo = [
        {"role": "user", "content": "What did meter E4471290 read on 2026-09-08?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_meter_reading",
                         "arguments": '{"meter_no":"E4471290","date":"2026-09-08"}'}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": '{"meter_no":"E4471290","date":"2026-09-08","kwh":41.7,"unit":"kWh"}'},
    ]
    d = chat(convo, tools=TOOLS, thinking=False)
    txt = msg(d).get("content") or ""
    ok = "41.7" in txt
    record("tool_result_roundtrip", ok, note=("cited 41.7" if ok else "did not cite the value"),
           wall_s=d["_wall_s"], sample=txt[:400])
except Exception as e:
    record("tool_result_roundtrip", False, error=str(e))

# 5. needle in a haystack — SWA 512 stress ----------------------------------
try:
    NEEDLE = "The maintenance passphrase for the Battlemage rack is CORAL-ANVIL-77."
    filler = "Routine telemetry line: all subsystems nominal, no action required. "
    n_fill = max(1, args.needle_tokens * 4 // len(filler))
    body = [filler] * n_fill
    body.insert(len(body) // 2, NEEDLE + " ")
    haystack = "".join(body)
    d = chat([{"role": "user", "content": haystack +
               "\n\nQuestion: what is the maintenance passphrase for the Battlemage rack? "
               "Answer with the passphrase only."}], thinking=False, max_tokens=64, temp=0.0)
    txt = msg(d).get("content") or ""
    ok = "CORAL-ANVIL-77" in txt.upper()
    record("needle_mid_context", ok, approx_tokens=args.needle_tokens,
           wall_s=d["_wall_s"], sample=txt[:200])
except Exception as e:
    record("needle_mid_context", False, error=str(e))

# 6. structured output -------------------------------------------------------
try:
    d = chat([{"role": "user", "content": "Return ONLY a JSON object with keys "
               '"city" and "country" for the Eiffel Tower. No prose, no code fence.'}],
             thinking=False, max_tokens=128, temp=0.0)
    txt = (msg(d).get("content") or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    parsed, ok = None, False
    if m:
        try:
            parsed = json.loads(m.group(0))
            ok = str(parsed.get("city", "")).lower().startswith("paris")
        except Exception:
            pass
    record("structured_json", ok, parsed=parsed, wall_s=d["_wall_s"], sample=txt[:200])
except Exception as e:
    record("structured_json", False, error=str(e))

npass = sum(1 for t in results["tests"].values() if t["pass_"])
results["summary"] = {"passed": npass, "total": len(results["tests"])}
print(f"\n{args.name}: {npass}/{len(results['tests'])} passed")
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
sys.exit(0 if npass == len(results["tests"]) else 1)
