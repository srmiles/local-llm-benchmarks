# Research project — Ornith 1.5 upgrade evaluation

**Type:** Follow-up research + testing (model swap candidate)
**Priority:** Medium-high — direct drop-in for current Ornith 1.0-9B; also opens MoE variant option
**Trigger:** Steve posted 6 HF URLs 2026-08-19 for the new Ornith 1.5 family
**Status:** Research seeded from HF model cards. No testing yet.

## The Ornith 1.5 family

Three sizes released 2026-08-19 by Ornith Team ("From Self-Scaffolding to Self-Improvement"). All are reasoning models — assistant turn opens with `<think>...</think>` before the final answer. Same qwen3.5 lineage as our current Ornith 1.0-9B, so tokenizer / chat template / tool-call surface are all familiar.

| Model | Arch | Params (active) | BF16 size | Q4_K_M size | Fits card |
|---|---|---|---|---|---|
| Ornith-1.5-9B | qwen3_5 (dense) | 9B | ~19 GB | 5.63 GB | Single B60 (24 GB), replaces current 1.0-9B |
| Ornith-1.5-35B-A3B | qwen35moe (MoE) | 35B / 3B active | ~71 GB | 21.7 GB | Q4_K_M just barely on one B60 (21.7 GB in 24), tight; safer on both cards with tensor split |
| Ornith-1.5-397B | large MoE | 397B | — | — | Not viable on B60 stack |

Multimodal: **the 9B non-GGUF card now advertises image-text-to-text** (uses `AutoProcessor` + `AutoModelForMultimodalLM` with image+text messages). This is new vs Ornith 1.0. The GGUF card doesn't mention image support — llama.cpp probably doesn't wire it up, so for our SYCL stack we'll be text-only unless we move to vLLM XPU.

## Ornith-1.5-9B (dense, direct 1.0-9B replacement)

**Architecture:** qwen3_5 (same family we already run). Dense 9B. Reasoning model with `<think>` blocks.

**Context:** 262,144 tokens native. YaRN factor 4.0 → ~1M effective. Static YaRN, so only enable if the workload needs it.

**Serving flags (matches our current 1.0-9B launcher pattern):**
- `--reasoning-parser qwen3`
- `--tool-call-parser qwen3_xml` (vLLM) / `qwen3_coder` (SGLang)
- `--enable-auto-tool-choice`
- `--enable-prefix-caching`

**Sampling (from the model card):**
- General: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
- Coding: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`

Note: presence_penalty=1.5 is unusual — Ornith 1.0 didn't need it. Worth A/B on our real chat workload before flipping.

**MTP drafter:** **Available from protoLabsAI** (same third party we already use for 1.0). Ornith-ai itself doesn't publish drafters for either 1.0 or 1.5 — they come from community publishers:
- **`protoLabsAI/Ornith-1.5-9B-MTP-GGUF`** — 0.5B params, published 2026-08-21 (~1h before this check). Direct successor to our current `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` drafter (used via `--model-draft /models/mtp-Ornith-1.0-9B-head-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3` in our launcher).
- Same MTP wiring pattern should just work — swap the base + swap the drafter, keep the flags.
- Worth re-measuring acceptance rate on real workload; new head, different training.

**Benchmarks vs 1.0-9B** (from card — 5-run average):

| Benchmark | 1.5-9B | 1.0-9B | Δ |
|---|---|---|---|
| Terminal-Bench 2.1 (Terminus-2) | 46.2 | 43.1 | +3.1 |
| Terminal-Bench 2.1 (Claude Code) | 47.0 | 40.6 | +6.4 |
| SWE-bench Verified | 70.6 | 69.4 | +1.2 |
| SWE-bench Pro | 47.5 | 42.9 | +4.6 |
| SWE-bench Multilingual | 54.4 | 52.0 | +2.4 |
| NL2Repo | 32.4 | 27.2 | +5.2 |
| SWE Atlas QnA | 20.6 | 17.9 | +2.7 |
| HLE (no tools) | 20.2 | 16.8 | +3.4 |
| HLE (with tools) | 30.5 | 26.4 | +4.1 |
| GPQA Diamond | 86.4 | 82.5 | +3.9 |
| MCP-Atlas | 54.2 | 49.4 | +4.8 |
| Toolathlon-Verified | 41.2 | 33.4 | +7.8 |
| WideSearch | 59.5 | 55.8 | +3.7 |
| BrowseComp | 56.4 | 44.8 | +11.6 |
| ClawEval | 66.5 | 63.1 | +3.4 |

Uniform improvement across every category, biggest gains on agentic (BrowseComp +11.6, Toolathlon +7.8, Terminal-Bench Claude-Code +6.4). Also beats Qwen3.5-9B by wide margins (SWE-bench Verified +17.4, HLE +5.5 no-tools).

## Ornith-1.5-35B-A3B (MoE variant)

**Architecture:** qwen35moe. 35B total, ~3B activated per token.

**Sizes:** Q4_K_M 21.7 GB / Q5_K_M 25.3 GB / Q6_K 29.2 GB / Q8_0 37.8 GB / BF16 71.1 GB.

**Fit on our stack:** Q4_K_M at 21.7 GB fits on ONE B60 (24 GB) but leaves only ~2 GB for KV cache — untenable for our 262K-context Ornith use. Realistic options:
- **Tensor split across both B60s:** Q4_K_M 21.7 GB / 2 ≈ 10.8 GB per card, leaves ~13 GB each for KV cache. Comfortable. But we'd lose the mirror-load pattern (both cards serving Ornith independently) — becomes a single logical instance shanked across two GPUs.
- **Q5_K_M or Q6_K on split:** 12.6 or 14.6 GB per card, still fine.
- **Single-card BF16 or Q8_0:** doesn't fit either card.

**Sampling (35B card):** `temperature=0.6, top_p=0.95, top_k=20` general; `temperature=1.0` to reproduce benchmarks. Different from the 9B recipe — no `presence_penalty=1.5`.

**Benchmarks vs Ornith-1.0-35B-A3B and Qwen3.6-35B-A3B (which we've already tested):**

| Benchmark | 1.5-35B-A3B | 1.0-35B-A3B | Qwen3.6-35B-A3B | Δ vs Qwen |
|---|---|---|---|---|
| Terminal-Bench 2.1 (Terminus-2) | 67.8 | 64.2 | 52.5 | +15.3 |
| Terminal-Bench 2.1 (Claude Code) | 68.5 | 62.8 | 49.2 | +19.3 |
| SWE-bench Verified | 79 | 75.6 | 73.4 | +5.6 |
| SWE-bench Pro | 59.6 | 50.4 | 49.5 | +10.1 |
| SWE-bench Multilingual | 71.4 | 69.3 | 67.2 | +4.2 |
| DeepSWE | 22 | 0 | 0 | +22 |
| Frontier-Bench v0.1 | 5.1 | 1.4 | 1.4 | +3.7 |
| NL2Repo | 46.2 | 34.6 | 29.4 | +16.8 |
| SWE Atlas QnA | 39.8 | 37.1 | 15.5 | +24.3 |
| HLE (no tools) | 25.6 | 20.8 | 21.4 | +4.2 |
| HLE (with tools) | 33.4 | 30.1 | 28.9 | +4.5 |
| GPQA Diamond | 89.2 | 86.2 | 86 | +3.2 |
| MCP-Atlas | 70.2 | 64.4 | 62.8 | +7.4 |
| Toolathlon-Verified | 48.7 | 42.4 | 41.7 | +7.0 |
| WideSearch | 67.8 | 63.4 | 60.1 | +7.7 |
| BrowseComp | 67.6 | 63.5 | 62 | +5.6 |
| ClawEval | 72.5 | 69.8 | 68.7 | +3.8 |

vs the Qwen3.6-35B-A3B we've already benched: substantial wins on every coding + agentic benchmark. Terminal-Bench and DeepSWE deltas are especially big.

Also notable — the 35B-A3B beats **Qwen3.5-397B** (the frontier dense model) on coding across the board (SWE-bench Pro +8.0, DeepSWE +21, Frontier-Bench +3.7, NL2Repo +9.4, SWE Atlas QnA +19.4) at ~3B activated params. If real, this is the strongest ratio in the family.

## Runtime requirements

Both models: `Transformers ≥ 5.8.1`, `vLLM ≥ 0.19.1`, `SGLang ≥ 0.5.9`.

Our llama.cpp SYCL b10433 already runs Ornith 1.0-9B (qwen35 arch). qwen3_5 for 1.5-9B and qwen35moe for 35B-A3B should be within the same lineage but need explicit confirmation from llama.cpp release notes — the exact arch string in `config.json` is what matters for the loader. **Verify:** does llama.cpp b10433 recognize `qwen3_5` and `qwen35moe`?

## Deployment plan (proposed)

### Phase 1 — 1.5-9B drop-in test (low risk)
- [ ] Download `ornith-ai/Ornith-1.5-9B-GGUF` Q4_K_M to `/data/llm/Ornith-1.5-9B-GGUF/`
- [ ] Download `protoLabsAI/Ornith-1.5-9B-MTP-GGUF` drafter (Q8_0 head) to same dir
- [x] ~~Check for a matching MTP drafter~~ — confirmed 2026-08-21: protoLabsAI shipped `Ornith-1.5-9B-MTP-GGUF` 1h before this check (same publisher as our current 1.0 drafter). Direct swap.
- [ ] Confirm llama.cpp b10433 loads it cleanly (arch recognition, no unknown-tensor errors)
- [ ] Bench on card 1 alongside current 1.0-9B — same prompts, measure prefill / decode tps + real chat/pi.dev output quality
- [ ] Try `presence_penalty=1.5` and without; see if it helps or hurts our workload
- [ ] Measure MTP acceptance rate of protoLabsAI 1.5 head vs current 1.0 head (~40-50% baseline)
- [ ] Decision: swap Ornith launcher `--alias ornith-1.0-9b` → `--alias ornith-1.5-9b`, update `-m` and `--model-draft` paths, keep `--spec-type draft-mtp --spec-draft-n-max 3` flags. Update headroom-proxy and nightly-review.sh only if quality wins are clear.

### Phase 2 — 35B-A3B feasibility (medium risk, bigger reward)
- [ ] Only after 1.5-9B is either shipped or ruled out
- [ ] Download Q4_K_M (21.7 GB) — same disk footprint as our current models combined-ish
- [ ] Test tensor-split across both B60s: `-ts 1,1 -ngl 99 -sm layer` (or `row` if layer split can't balance activated experts)
- [ ] Bench MoE decode on B60 SYCL — Sergio Barrientos vLLM XPU work suggests MoE on B60 wants vLLM more than llama.cpp; we may hit the same wall
- [ ] If throughput acceptable and quality substantially better than 1.5-9B: shape decision between "one big Ornith 35B split across both cards" vs current mirror pattern
- [ ] If throughput inadequate: park until vLLM XPU migration

### Phase 3 — vLLM XPU stack (adjacent)
- MoE + Ornith 1.5-35B-A3B is the natural moment to move to vLLM XPU. Combine with LMCache (see `docs/research/lmcache-evaluation.md`) and this becomes the strongest possible categorise/chat stack.

## Non-goals

- Not touching E2B — Gemma 4 E2B stays for categorise; Ornith swap is chat + pi.dev only
- Not attempting 1.5-397B on B60 stack — not enough VRAM even split across both cards at Q4
- Not chasing multimodal on llama.cpp path (needs vLLM path if we want image input on 1.5-9B)

## Open questions

1. **Is there an Ornith-1.5-9B MTP drafter?** Not mentioned on either the 1.5-9B or 1.5-9B-GGUF card. If no drafter → decode will be slower than current 1.0-9B+MTP. Check the Ornith-1.5 collection for a drafter artifact before committing.
2. **Does llama.cpp b10433 support `qwen3_5` dense + `qwen35moe`?** Card lists Transformers/vLLM/SGLang minimums but silent on llama.cpp. Check llama.cpp release notes and open issues for the exact arch strings.
3. **Multimodal path** — the 1.5-9B advertises image support. Is this useful for any of our workloads? (Chat, pi.dev, brain-eval categorise — none of these currently take images, but pi.dev could conceivably benefit.)
4. **`presence_penalty=1.5`** — is this training-time regularization that carried into inference, or a genuine recommendation? Ornith 1.0-9B didn't use it and worked fine.

## Sources

- https://huggingface.co/ornith-ai/Ornith-1.5-9B (base, multimodal)
- https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (deployed variant, fetched 2026-08-19)
- https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF (MoE, fetched 2026-08-19)
- https://huggingface.co/ornith-ai/Ornith-1.5-397B, /Ornith-1.5-35B-A3B, /Ornith-1.5-397B-GGUF (not fetched — 397B out of stack budget)
- Collection: https://huggingface.co/collections/ornith-ai/ornith-15 (12 items, may include drafters)
- Ornith 1.5 blog (referenced by cards): https://ornith.ai/ornith_1_5.html

## Push to brain when reachable

- Vault: `llm-local`
- Type: `research` or `project`
- Link to: `ornith-1.0-9b`, `qwen-3.8-27b` (tested/parked comparator), `2nd-b60-arrival-playbook`, `lmcache-evaluation` (adjacent), `gemma-4-e2b-categorise` (unaffected but adjacent)
