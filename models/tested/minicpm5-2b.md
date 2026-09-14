# MiniCPM5-2B (OpenBMB) — Tested 2026-09-09

**Status:** Benched. **Beats Gemma 4 E2B on decode and prefill simultaneously, and sets a new repo decode record.** With the official DSpark drafter at `--spec-draft-n-max 7 --spec-draft-p-min 0.6`: **302.65 tok/s decode at 100.0% acceptance** against E2B's 188.06, with prefill **3,614 vs 3,328**. Undrafted it prefills at **5,029 tok/s @ 12K — the fastest prefill of any model in this repo** and +51% over the incumbent. **The catch is VRAM:** 8.33 GiB drafted, 5.24 GiB solo, against E2B's 3.96 — which matters on the 12 GiB B580 node where the categorise pair actually lives.

**HF:** [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) · [official GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) · [DSpark drafter](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark) · [DSpark GGUF](https://huggingface.co/aj9o9/MiniCPM5-2B-DSpark-GGUF) (used here)
**License:** Apache 2.0
**Publisher:** OpenBMB
**Released:** 2026-09-08 (GGUF) · DSpark GGUF 2026-09-07
**Arch:** `LlamaForCausalLM` → GGUF arch **`llama`** — **loads on any build**, no rebuild needed
**Drafter:** **official DSpark**, DFlash family. Q8_0 (0.35 GB) and F16 (0.65 GB) both tested.

## Why this was interesting

**It is the successor to the fastest model this rig has ever run.** [MiniCPM5-1B](minicpm5-1b.md) hit 187 tok/s decode and 4.6K prefill in July and was stopped only to reclaim ~3 GiB. The 1B had **no drafter**; the 2B ships a first-party one.

**And DFlash-family drafters have never misbehaved here.** Muse Glimmer hit 100% acceptance on b10433; LFM2.5's DSpark was the reason that model got a slot at all. That record now extends to a third: **99.7–100.0%**.

## Specs

| | |
|---|---|
| Total parameters | ~2.5B (dense) |
| Architecture | `LlamaForCausalLM` — plain llama, no hybrid attention |
| Layers | 42 |
| Hidden dimension | 2,048 |
| Attention heads | 16 Q / 2 KV, `head_dim` 128 |
| FFN | intermediate 6,144 |
| RoPE | θ 5e6 |
| Vocabulary | 130,560, **not** tied |
| Context | 131,072 |
| Drafter | DSpark (DFlash family), published separately |

## Setup

```bash
MODEL_DIR=/data/llm/MiniCPM5-2B-GGUF MODEL=MiniCPM5-2B-Q4_K_M.gguf \
  DRAFT=MiniCPM5-2B-DSpark-Q8_0.gguf \
  SPEC_ARGS="--spec-type draft-dspark --spec-draft-n-max 7 --spec-draft-p-min 0.6" \
  CTX=131072 MEM=12g /data/llm/launch/bench-slot.sh
```

## Benchmarks (b10867-patched, isolated card 2, 20 runs × 300 tok @ temp 0.6 / top-p 0.95 / top-k 20)

### Drafter and head quant

| Config | Decode mean | σ | Acceptance | Prefill @ 12K | Peak VRAM |
|---|---|---|---|---|---|
| solo, no drafter | 137.14 | 1.27 | — | **5,029** | **5.24 GiB** |
| + DSpark **F16** (0.65 GB) | 218.83 | 2.29 | 99.9% | 3,624 | 8.59 |
| + DSpark **Q8_0** (0.35 GB) | **233.83** | 2.70 | 99.7% | 3,617 | 8.33 |

**Q8_0 beats F16 by 6.9% decode while being smaller** — the 0.2 pp of acceptance the bigger head buys is worthless. Head-to-target ratio is 22% at Q8_0 and **42% at F16**; this is the [Qwen3.8-9B-Distill](qwen3.8-9b-distill.md) rule measured directly rather than inferred (finding #56).

### n-max sweep, DSpark Q8_0

| n-max | p-min | Decode mean | σ | Acceptance | Accepted/draft | Prefill @ 12K | VRAM |
|---|---|---|---|---|---|---|---|
| 3 | 0.0 | 233.83 | 2.70 | 99.7% | 2.98 / 3 | 3,617 | 8.33 |
| 5 | 0.0 | 270.33 | 2.18 | 100.0% | 4.98 / 5 | 3,634 | 8.33 |
| 7 | 0.0 | 300.89 | 4.76 | 99.6% | **6.85** | 3,622 | 8.40 |
| **7** | **0.6** | **302.65** | **1.87** | **100.0%** | **6.86** | 3,614 | **8.33** |
| 9 | 0.0 | 300.19 | 6.71 | 99.3% | **6.82** | 3,633 | 8.33 |
| 9 | 0.6 | 301.40 | 3.50 | 100.0% | **6.86** | 3,645 | 8.33 |
| 11 | 0.6 | 300.78 | 2.96 | 99.8% | **6.85** | 3,636 | 8.38 |

**Ceiling is at n-max 7 and it is the drafter's, not the harness's.** The p-min-0.0 arms were run specifically to rule out the flag: `accepted_per_draft` saturates at **6.82–6.86 with p-min off**, identical to with it on, so ~6.85 is the DSpark head's effective depth. Everything above n-max 7 is inert — not harmful, just wasted.

**What p-min actually buys here is variance, not throughput** — σ 4.76 → 1.87 at n-max 7, 6.71 → 3.50 at n-max 9, at ~0 decode cost. Directly opposite to [finding #34](../../docs/findings.md), where p-min 0.6 was worth **+26% decode** on Nemotron. The discriminator is acceptance: Nemotron's real traffic accepted 46%, so p-min pruned bad drafts; here acceptance is ~100%, so there are none to prune (finding #57).

Prefill is flat (3,614–3,645) across the entire sweep, confirming the drafter's prefill cost is fixed rather than n-max-dependent — same shape Nemotron showed.

### Prefill curve

| | @500 | @2K | @5K | @12K |
|---|---|---|---|---|
| solo | 4,113 | **5,989** | 5,927 | **5,029** |
| + DSpark Q8_0 | 3,402 | 4,093 | 4,075 | 3,617 |

**5,989 @ 2K undrafted is the highest prefill number in this repo.** It declines 16% by 12K — the shape that got [Ling-3.0-tiny](ling-3.0-tiny.md) rejected, but far milder there (Ling fell 47%, and ended *below* where it started).

## Head-to-head vs the categorise incumbent

Both on b10867-patched, same session, same card.

| | MiniCPM5-2B solo | MiniCPM5-2B +DSpark n7 | Gemma 4 E2B +MTP |
|---|---|---|---|
| Decode mean | 137.14 | **302.65** | 188.06 |
| Decode σ | 1.27 | 1.87 | 25.58 |
| Prefill @ 12K | **5,029** | 3,614 | 3,328 |
| Prefill @ 2K | **5,989** | 4,093 | 3,421 |
| Acceptance | — | **100.0%** | 98.4% |
| Peak VRAM | 5.24 GiB | 8.33 GiB | **3.96 GiB** |
| Load | 25 s | 25 s | 40 s |

## Functional probe

6/6 on the drafted n-max 7 config (`probe-spark-x25.py`): coherence with thinking off, thinking-on arithmetic with `reasoning_content` separated, tool call with the optional arg inferred, tool-result round-trip, needle @ 32K, and clean structured JSON with no fence.

## Verdict

**Two viable configurations, and which one wins depends on a measurement nobody has taken.**

**1. Drafted (302.65 tok/s, 8.33 GiB) is the headline but probably the wrong config for categorise.** The categorise workload is prefill-heavy with short outputs. The drafter costs **28% of prefill** to buy decode that workload barely consumes — finding #31's trade running backwards. It would be the right config for a chat or agent slot.

**2. Solo (5,029 prefill @ 12K, 5.24 GiB) is the one to bake off against E2B.** +51% prefill over the incumbent at +1.28 GiB, no drafter to maintain, σ 1.27.

**3. VRAM is the real obstacle, and it is not on the B60.** The categorise pair runs on `llm2.local`, a **12 GiB B580** currently at ~5.8 GiB with three services. Solo at 5.24 GiB is a +1.28 GiB swap for E2B; **drafted at 8.33 GiB likely does not fit alongside embed and rerank.** The B60 side has room; the B580 side decides.

**4. Zero integration risk on the runtime side.** Plain `llama` arch — unlike Spark-X2.5-4B this needed no build, and it will load on the current production b10809 image unchanged.

## Watch items

- **Quality bake-off vs Gemma 4 E2B on the Tier A corpus** — the `docs/track2-quality-bakeoff.md` instrument. E2B won its slot on a quality *tie* plus 3.1× latency; this needs the same test before it displaces anything.
- **Measure the actual categorise prefill:decode ratio.** Both the E2B-vs-MiniCPM5 choice and the solo-vs-drafted choice hinge on it, and it has never been measured on real traffic — only assumed.
- **B580 co-residence.** Solo needs verifying at 5.24 GiB alongside embed + rerank on a 12 GiB card; drafted probably will not fit.
- **Prefill decline past 12K** — 16% from the 2K peak, untested beyond 12K.
- **DSpark head sourced from a third party.** `aj9o9/MiniCPM5-2B-DSpark-GGUF`, not OpenBMB. OpenBMB publishes the drafter only as safetensors; converting it in-house would remove the dependency.

## Files on disk

```
/data/llm/MiniCPM5-2B-GGUF/
├── MiniCPM5-2B-Q4_K_M.gguf          1.56 GB — benched, recommended ⭐
├── MiniCPM5-2B-Q8_0.gguf            2.68 GB — on disk, not benched
├── MiniCPM5-2B-DSpark-Q8_0.gguf     0.35 GB — drafter, recommended ⭐
└── MiniCPM5-2B-DSpark-F16.gguf      0.65 GB — drafter, rejected (−6.9% decode)
```

Raw results: `/data/llm/benchmarks/20260909-round2/`, committed under [`configs/benchmarks/20260909-round2/`](../../configs/benchmarks/20260909-round2/).

## References

- [Model card](https://huggingface.co/openbmb/MiniCPM5-2B) · [GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) · [DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark)
- [MiniCPM5-1B — the predecessor](minicpm5-1b.md) · [Gemma 4 E2B — categorise incumbent](../production/gemma-4-e2b-categorise.md)
- Findings **#55–#57**
