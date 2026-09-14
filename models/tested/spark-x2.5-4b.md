# Spark-X2.5-4B (XHToken) — Tested 2026-09-09

**Status:** Benched. **Strongest challenger the chat slot has had.** A dense **4B** decodes **87.74 tok/s unassisted** against Ornith 1.5-9B's 77.55 *with its MTP drafter running*, prefills **1.93× faster** (3,643 vs 1,892 @ 12K), and does it in **6.44 GiB against 11.33**. Retrieval holds at 20/20 across a depth × length needle sweep out to 106K tokens. **Not promoted** — the same rule that governed Qwen3.8-9B-Distill applies here and applies harder, because this is not an architectural sibling of the incumbent: it needs a `docs/track2`-style qualitative bake-off on the pi.dev corpus before anyone touches `:8002`.

**HF:** [`XHToken/Spark-X2.5-4B`](https://huggingface.co/XHToken/Spark-X2.5-4B) · [official GGUF](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF) (used here)
**License:** Apache 2.0
**Publisher:** XHToken — trained on Huawei Ascend clusters
**Released:** 2026-09-03 · official GGUF updated 2026-09-07
**Arch:** `Spark2_5ForCausalLM` → GGUF arch **`spark2_5`**
**Drafter:** **none, in any format.** `config.json` declares no MTP/nextn layers, so the finding #28 trick has nothing to convert. Every number below is unassisted.

## Why this was interesting

**It claims to beat a 9B on agent work while being a 4B.** The published table puts it above Qwen3.5-9B on τ³-bench (30.4 vs 9.3), MCP-Atlas (54.6 vs 47.4), BrowseComp (40.9 vs 8.3) and SWE-Bench Pro (44.4 vs 33.8) — the exact axis on which Ornith 1.5-9B holds the chat slot and carries the pi.dev agent. Vendor self-reports, so worth nothing on their own; worth a bench slot because if even half of it survives contact, the size class of the chat slot is the wrong size class.

**And it forced a build.** `spark2_5` is absent from `b10809` — upstream merged support in [PR #27868](https://github.com/ggml-org/llama.cpp/pull/27868) on **2026-09-06**, two days after our cutover. Same shape as Ling-3.0-tiny in the b10566 round. The model card still points at the vendor's `XHToken/llama.cpp` fork; **that guidance is stale — mainline has it**, and the fork is not needed.

## Specs

| | |
|---|---|
| Total parameters | 4.1B (**dense**) |
| Architecture | `spark2_5` — hybrid sliding-window / full attention |
| Layers | 36 — **27 `sliding_attention` + 9 `full_attention` (3:1)** |
| Sliding window | **512** |
| Hidden dimension | 2,560 |
| Attention heads | 16 Q / 4 KV, `head_dim` 256, headwise output gate |
| FFN | intermediate 10,240, `gelu` |
| RoPE | θ 5e6 full-attn (`rope.dimension_count` 64, partial 0.25) / θ 1e4 SWA (256, partial 1.0) |
| Vocabulary | 131,072, tied embeddings |
| Context | **1,048,576** native |
| MTP | **absent** |

The interesting bet is `sliding_window: 512` across 27 of 36 layers. Only nine full-attention layers carry long-range retrieval, which is what makes a 1M window affordable — and is exactly the thing that could quietly not work. It is tested below rather than assumed.

## Setup (llama.cpp SYCL on Intel Arc Pro B60)

```bash
MODEL_DIR=/data/llm/Spark-X2.5-4B-GGUF MODEL=Spark-X2.5-4B-Q4_K_M.gguf \
  CTX=131072 MEM=12g IMAGE=llama.cpp:sycl-f16-b10867-patched-b3ce098f7 \
  /data/llm/launch/bench-slot.sh
```

Stock `bench-slot.sh`, no model-specific flags, no `EXTRA_ARGS`. Loads in **10 s** against Ornith's 55 and E2B's 40.

## Benchmarks (b10867-patched, isolated card 2, 20 runs × 300 tok @ temp 0.6 / top-p 0.95 / top-k 20)

### Quant comparison

| | Decode mean | σ | Prefill @ 12K | Peak VRAM |
|---|---|---|---|---|
| **Q4_K_M** (2.60 GB) | **87.74** | 0.35 | **3,643** | **6.44 GiB** |
| Q8_0 (4.38 GB) | 56.25 | 0.10 | 3,105 | 8.09 GiB |
| delta | **−35.9%** | — | **−14.8%** | +1.65 GiB |

**Q4_K_M, and it is not close.** Q8_0 buys nothing measurable and costs a third of decode. Decode ratio 1.56 against a weight-size ratio of 1.68 — near-proportional, so **decode on this model is bandwidth-bound, not compute-bound**, and the quant is the throughput dial (finding #52).

### Full numbers, Q4_K_M

| Metric | Value |
|---|---|
| Decode | 87.74 mean · 87.72 median · **σ 0.35** (min 87.22, max 88.38) |
| Prefill | 1,948 @ 500 · 3,581 @ 2K · 3,622 @ 5K · **3,643 @ 12K** |
| Peak VRAM | 6.44 GiB |
| Load | 10 s |
| Drafter | none |

**Prefill rises monotonically with context and is still climbing at 12K** — the opposite of Ling-3.0-tiny, which was rejected for falling from 2,293 to 1,218 over the same range.

## Head-to-head

Ornith and E2B re-run on **b10867** in the same session; Gemma 4 26B-A4B carried from b10809 (see the anchor note below).

| | Spark-X2.5-4B Q4_K_M | Ornith 1.5-9B **+MTP** | Gemma 4 E2B **+MTP** | Gemma 4 26B-A4B **+MTP** |
|---|---|---|---|---|
| Decode mean | **87.74** | 77.55 | 188.06 | 62.66 |
| Decode σ | **0.35** | 13.79 | 25.58 | 20.24 |
| Prefill @ 12K | **3,643** | 1,892 | 3,328 | 1,669 |
| Peak VRAM | 6.44 GiB | 11.33 | **3.96** | 20.69 |
| Load | **10 s** | 55 s | 40 s | — |
| Drafter | **none** | Q8_0 MTP | Google MTP | Q8_0 MTP |

**+13.1% decode and +92.6% prefill over the production chat slot, at 57% of its VRAM, with no drafter at all.** It also out-prefills Gemma 4 E2B, a model less than half its size.

**σ 0.35 against Ornith's 13.79 is not a rounding detail.** Finding #31 established that the bimodality in every drafted row on this box is an acceptance artifact — σ collapsed from 10.57 to 0.05 when Qwen3.8-9B-Distill's head was removed. Spark has no head, so it has no bimodality: every one of 20 runs landed between 87.22 and 88.38. For an agent slot, a p99 that equals the median is worth real money, and it is the one axis on which a drafted model **cannot** compete.

## Long context — the SWA-512 question

20 × needle-in-a-haystack, five depths × four lengths, greedy, thinking off (`needle-spark-x25.py`):

| Prompt tokens | start | quarter | middle | ¾ | end |
|---|---|---|---|---|---|
| 7,111 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 28,291 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 56,521 | ✅ | ✅ | ✅ | ✅ | ✅ |
| **105,931** | ✅ | ✅ | ✅ | ✅ | ✅ |

**20/20, no depth bias, no degradation with length.** Nine full-attention layers are sufficient for exact retrieval at 106K. Time-to-answer at 106K ranged 36.6–56.6 s, consistent with the measured prefill rate.

**Scope:** this tests *retrieval*, which is the cheapest long-context capability. It does not test multi-hop reasoning over a long window, and it does not touch the 1M claim — 131K is the bench ceiling and 106K is what a 131K window actually holds after templating.

## Functional probe

`probe-spark-x25.py`, via `/v1/chat/completions` with `--jinja` — the speed bench hits `/completion` and never exercises the template. **6/6 on both quants.**

| Test | Result |
|---|---|
| Coherence, thinking off | ✅ no `<think>` leakage into `content` |
| Thinking on, arithmetic | ✅ correct, trace in `reasoning_content` (393 chars), answer in `content` |
| Tool call | ✅ `get_meter_reading({meter_no: "E4471290", date: "2026-09-08"})` — inferred the optional arg |
| Tool result round-trip | ✅ *"The meter E4471290 read **41.7 kWh** on 2026-09-08."* |
| Needle @ 32K | ✅ |
| Structured JSON | ✅ `{"city":"Paris","country":"France"}`, no fence, no prose |

Tool calling and the `enable_thinking` switch both work out of the box under `--jinja`, with `reasoning_content` separated the way the Nemotron slot already expects — so the agent-client conventions in `configs/AGENT-CLIENTS.md` carry over unchanged.

## Build note — and the anchor

`spark2_5` needs ≥ **b10867**. Built `llama.cpp:sycl-f16-b10867-patched-b3ce098f7`: upstream `f3f1a8f27` + the 8-commit local `ggml-sycl` series, **which rebased with zero conflicts**.

Because Spark cannot load on *any* earlier build, no A/B is possible for it — so the incumbents were re-run on the new image instead, per finding #51's rule about knowing which half of a delta you own:

| Anchor | b10809 | b10867 | Δ (mean) |
|---|---|---|---|
| Ornith 1.5-9B decode mean | 76.63 | 77.55 | +1.2% |
| Ornith prefill @ 12K | 1,901.6 | 1,892.4 | −0.5% |
| Ornith MTP acceptance | 76.8% | 77.2% | +0.4 pp |
| Gemma 4 E2B decode mean | 192.88 | 188.06 | −2.5% |
| Gemma 4 E2B prefill @ 12K | 3,336.2 | 3,327.6 | −0.3% |
| Gemma 4 E2B acceptance | 97.7% | 98.4% | +0.7 pp |

**Flat, and flat in opposite directions — the 58-commit bump moved nothing.** Ornith's *median* rose 9.2%, which is the finding #31 bimodality flipping clusters, not a gain; the mean is the honest comparator at σ 13.79. Spark's numbers therefore stand directly against the published b10809 table with no build caveat (finding #53).

## Verdict

**Earns the bake-off, and earns it more strongly than Qwen3.8-9B-Distill did.**

**1. The speed win is unambiguous, unlike the last challenger.** Qwen3.8-9B-Distill won +13.5% on median but only +3.3% on mean, and the gap was bimodality. Spark wins +13.1% on *mean* with σ 0.35, plus +92.6% prefill and −43% VRAM. There is no statistical asterisk to apply.

**2. Do not cut over on speed — and the integration risk is real this time.** Qwen3.8-9B-Distill was byte-for-byte Ornith's shape; launcher flags carried over and only capability was in question. Spark is a different architecture, a different tokenizer (131K vs 248K vocab), a different publisher, and a fresh upstream implementation **three days old at time of bench**. Ornith 1.5-9B is a DeepReinforce agentic-coding post-train carrying pi.dev. A general-purpose 4B is not a drop-in for that, whatever BFCL says.

**3. The freed VRAM may be the real prize.** At 6.44 GiB it leaves ~4.9 GiB more headroom on card 1 than Ornith does, against a card sitting at ~20.5 GiB with four co-resident services. That is a co-residence argument, not just a speed one.

**4. No drafter is a ceiling and an opportunity.** Every other chat-slot model here is drafted; Spark beats them without one. If XHToken or a third party ships an MTP head, the finding #28 conversion path applies and the headroom is large. Nothing to do until one exists.

**5. The 1M context claim is unverified.** 106K is tested and clean. 131K is the harness ceiling. Anything above that is the vendor's word.

## The drafter question — settled 2026-09-09

Spark has no MTP head and **none can be built**: the safetensors index holds exactly 290 tensors, layers 0–35 with none missing and none spare, `model.embedding.weight` + `model.norm.weight` as the only non-layer entries, and no `mtp`/`nextn`/`eagle`/`medusa` naming anywhere. Tied embeddings mean there is not even a spare `lm_head`. The finding #28 conversion path is closed.

But **`Spark-X2.5-1.7B` is vocabulary-identical** — 131,072 tokens, same `spark2_5` arch, same `head_dim` 256, same `sliding_window` 512 — so the *classic* draft-model path (`--spec-type draft-simple`) is open. Tested:

| | undrafted | + 1.7B, n-max 3 | + 1.7B, n-max 5 (p-min 0.6) |
|---|---|---|---|
| Decode mean | 87.74 | 93.67 | **93.79** |
| Decode σ | **0.35** | 2.53 | 0.76 |
| Acceptance | — | 98.7% | **99.7%** |
| Accepted/draft | — | 2.95 / 3 | 4.95 / 5 |
| Prefill @ 12K | **3,643** | 2,585 | 2,583 |
| Peak VRAM | **6.44 GiB** | 10.29 | 10.31 |

**It works, and it is still not worth taking.** +6.8% decode costs **−29% prefill and +3.85 GiB (+60%)** — which spends exactly the VRAM advantage that made this model interesting against Ornith — and forfeits the σ 0.35 property.

**The mechanism is the useful part.** Decode is *flat* between n-max 3 and 5 (+0.1%) while the drafter returns **4.95 of 5.00 tokens at 99.7% acceptance**. The drafter is giving back essentially everything asked of it and throughput does not move, so the ceiling is **the draft model's own forward cost**, not acceptance: each extra draft token costs a full 1.7B forward that consumes what the accepted token wins. A same-family sibling drafts with near-perfect alignment — size ratio does not hurt acceptance at all — but a 2.4× ratio caps the achievable speedup at ~+7% regardless. That is the structural difference from a one-layer MTP head, which drafts nearly free and therefore *does* scale with n-max (finding #55).

## Watch items

- **Qualitative bake-off vs Ornith 1.5-9B on the pi.dev corpus** — tool-calling reliability, edit-diff correctness, scaffold quality. The only thing that decides a cutover. Same instrument as `docs/track2-quality-bakeoff.md`.
- **Sampling divergence.** The GGUF ships `temp 1.0 / top_p 0.95 / top_k −1` and the card says all published evals used thinking mode at those settings. This bench used the house convention (0.6 / 0.95 / 20) for comparability. **A quality bake-off should use the vendor's numbers**, or it is testing the wrong model.
- **Thinking-mode cost.** Every published benchmark is thinking-on; the speed table above is thinking-off. Token cost of the trace on real agent traffic is unmeasured and could erase the decode win.
- **Long-context reasoning, not just retrieval** — 20/20 needles says nothing about multi-hop work at 100K.
- **Upstream implementation is three days old.** Watch for correctness fixes to `src/models/spark2-5.cpp` before trusting it in production.
- **`Spark-X2.5-1.7B`** — same family, sibling release, plausible categorise-slot candidate against Gemma 4 E2B's 3.96 GiB.

## Files on disk

```
/data/llm/Spark-X2.5-4B-GGUF/
├── Spark-X2.5-4B-Q4_K_M.gguf     2.60 GB — benched, recommended ⭐
└── Spark-X2.5-4B-Q8_0.gguf       4.38 GB — benched, rejected (−35.9% decode)
```

Raw results: `/data/llm/benchmarks/20260909-spark-x25/` — also committed under [`configs/benchmarks/20260909-spark-x25/`](../../configs/benchmarks/20260909-spark-x25/).

## References

- [Model card — XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) · [official GGUF](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF)
- [llama.cpp PR #27868 — Spark2_5ForCausalLM support](https://github.com/ggml-org/llama.cpp/pull/27868) (merged 2026-09-06)
- [Ornith 1.5-9B — production chat](../production/ornith-1.5-9b.md) · [Qwen3.8-9B-Distill — the previous challenger](qwen3.8-9b-distill.md)
- Findings **#52–#54**
