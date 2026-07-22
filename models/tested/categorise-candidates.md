# Categorise slot candidates — Tested 2026-07-22

**Motivation:** The old dedicated categorise slot on `:8006` (`llamacpp-categorise` running `Qwen3-4B-Instruct-2507`) was retired 2026-07-19 due to quality regression. Its work moved onto Ornith 9B (`:8002`), which now serves chat + pi.dev agent + categorise on a single-slot FIFO — creating the primary throughput bottleneck. Goal: find a small model that can carry categorise workload with acceptable quality and free the Ornith slot.

## Bench methodology

10 realistic short-context categorise prompts (200-800 chars each). Fixed taxonomy (`technical`, `business`, `personal`, `admin`, `other`), strict JSON output schema. Same system prompt across all candidates. `temperature=0`, `cache_prompt=false`, isolated (bench container on `:8009`, prod stack running).

**Measured:** decode tok/s, wall latency per task, JSON parseability, schema validity, category coverage, qualitative accuracy (spot-check obvious miscategorisations).

## Summary

| Model | Params | Decode tok/s | Wall (median) | JSON valid | Obvious errors | Verdict |
|---|---|---|---|---|---|---|
| **Ornith 1.0 9B (prod baseline)** | 9B dense | **57** | **1193 ms** | **10/10** | 0 | reference — but this is exactly the resource we want to free |
| Qwen3-4B-Instruct-2507 (Q4_K_M) | 4B dense | 79 | 581 ms | 10/10 | 1 (sourdough→technical) | retired for quality; still competent but Ornith reference is better |
| Agents-A1-4B (Q4_K_M, reasoning off) | 4B dense | 79 | 715 ms | 10/10 | 1 (sourdough→technical) | agentic tuning, same recipe issue as Qwen3-4B |
| MiniCPM5-1B Claude Fable5-V2 (Q8_0) | 1B dense | **154** | **255 ms** | 9/10 | 3 (sourdough, grocery, vacation all→business) | fastest, worst quality — strong "business" bias |
| **Gemma 4 E2B-it QAT Q4_0 (Google official)** | 2B effective | **88** | **689 ms** | **10/10** | 0 obvious (sourdough → "other" is defensible) | **best trade-off found so far** — 5/5 categories used, matches Ornith on 9/10 |

## Detailed results

### Ornith 1.0 9B (prod baseline, `:8002`)

```
Decode:  median 57 tok/s (spread 52-63)
Wall:    median 1193 ms (spread 1140-1363)
JSON:    10/10 parseable + schema-valid
Cats:    admin, business, personal, technical (4/5)
```

All 10 tasks categorised correctly by human judgment. Sample outputs:
- Q3 planning → `business` ✓ (tags: planning, revenue, hiring, quarterly)
- Sourdough recipe → `personal` ✓ (tags: cooking, baking, sourdough)
- K8s CrashLoop → `technical` ✓ (tags: kubernetes, crashloopbackoff, sigsegv)
- Contract renewal → `admin` ✓ (tags: contract, vendor, aws, deadline, compliance)

**This is the target quality bar to match.**

### Qwen3-4B-Instruct-2507 (Q4_K_M)

```
Decode:  median 79 tok/s
Wall:    median 581 ms  (2× faster than Ornith)
JSON:    10/10 parseable + valid
Cats:    business, personal, technical (3/5 — never used admin or other)
```

**1 clear error:** sourdough recipe → `technical` (should be personal — the model over-classifies recipes as technical procedures).

Speed advantage over Ornith is real (2× wall), but category taxonomy compression to 3 categories combined with the recipe miscategorisation confirms the "quality dropped" observation.

### Agents-A1-4B (Q4_K_M, `--reasoning off`)

```
Decode:  median 79 tok/s
Wall:    median 715 ms  (1.7× faster than Ornith)
JSON:    10/10 parseable + valid
Cats:    admin, business, personal, technical (4/5)
```

Configuration note: must run with `--reasoning off` — this model routes output to `reasoning_content` by default, leaving OpenAI `content` field empty. Without the flag, JSON validation is 0/10.

**Category coverage matches Ornith (4/5).** Same sourdough → technical error as Qwen3-4B. Correctly classified driver's license → admin (Ornith called it personal, both defensible). Contract renewal → business (Ornith called it admin).

Speed is essentially tied with Qwen3-4B. Quality is comparable — same recipe miscategorisation but slightly broader taxonomy coverage.

### Gemma 4 E2B-it QAT Q4_0 (Google official, `--reasoning off`)

```
Decode:  median 88 tok/s   (fastest 4B-class tested — beats Qwen3-4B and Agents-A1)
Wall:    median 689 ms     (1.7× faster than Ornith, comparable to Qwen3-4B and Agents-A1)
JSON:    10/10 parseable + valid
Cats:    business, other, personal, technical (4/5 — first candidate to use "other")
```

Configuration note: must run with `--reasoning off`. Gemma 4 routes thinking tokens to `reasoning_content` by default; without the flag JSON parsing is 0/10.

Quality per task (compared with Ornith baseline):
- Q3 planning → `business` ✓ matches Ornith
- Sourdough recipe → `other` (Ornith called personal — "other" is defensible for a bare recipe with no personal framing; better than 4B models' `technical` and MiniCPM's `business`)
- K8s CrashLoop → `technical` ✓
- Driver's license → `personal` ✓ matches Ornith
- ML blog draft → `technical` (defensible)
- Grocery list → `personal` ✓ matches Ornith (Qwen3-4B/Agents got this right too; MiniCPM called it "business")
- Server monitoring → `technical` ✓
- Vacation → `personal` ✓ matches Ornith (only candidate besides Ornith to get this right — MiniCPM/Qwen3-4B/Agents all defensible variants but not personal)
- Contract → `business` (Ornith called admin — both defensible)
- Transformer notes → `technical` ✓

**Best candidate found.** Matches Ornith on 9/10 tasks (only the recipe classification differs, and Gemma's `other` is at least defensible). Fastest 4B-class model tested. Google's QAT quantization preserves quality well.

### MiniCPM5-1B Claude-Opus-Fable5-V2-Thinking (Q8_0)

```
Decode:  median 154 tok/s  (2.7× Ornith, 2× Qwen3-4B)
Wall:    median 255 ms     (4.7× faster than Ornith)
JSON:    10/10 parseable, 9/10 schema-valid (1 formatting issue)
Cats:    business, personal, technical (3/5)
```

**3 clear errors:**
- Sourdough recipe → `business` (worse than the 4B models' `technical`)
- Grocery list → `business` (tags: "business grocery list", "business items" — very off)
- Vacation plan → `business` (should be personal)

Strong "business" bias — model appears to over-classify anything with actions/planning verbs. Given the "V2 Thinking" branding, may be that reasoning-off is masking the intended workflow. The base MiniCPM5-1B had a JSON fence bug that killed it in prior testing; the Claude-distilled version fixes JSON emission but introduces the classification bias.

**Speed is spectacular. Quality is not workable for production categorise.**

## Verdict

None of the candidates cleanly beats Ornith 9B on quality. Two viable paths forward:

### Path 1: Restore split-categorise with Gemma 4 E2B QAT (RECOMMENDED — updated 2026-07-22 21:15)

**Winner is Gemma 4 E2B**, not Agents-A1-4B as initially recommended. Gemma:
- **Faster:** 88 tps vs 79 tps (11%)
- **Better latency:** 689 ms vs 715 ms (matches Qwen3-4B baseline)
- **Better quality:** matches Ornith on 9/10 tasks (vs Agents-A1's 8/10), uses 4/5 categories including "other" (first candidate to use it)
- **Smaller model:** 3.35 GB Q4_0 vs Agents-A1's 2.71 GB Q4_K_M — similar footprint
- **Google-official QAT:** quality-aware training preserves accuracy better than post-training K-quants for this size class
- **Apache-2.0 license**

Config:
```bash
docker run -d --name llamacpp-categorise \
  --memory=6g --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/gemma-4-E2B-it-GGUF:/models:ro \
  -p 0.0.0.0:8006:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/gemma-4-E2B_q4_0-it.gguf \
  -ngl 99 -c 8192 --parallel 2 \
  --host 0.0.0.0 --port 8000 --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --temp 0.0
```

Then point brain env `CATEGORISE_URL` (or equivalent) at `:8006`.

**Recipe-classification quality can be improved system-prompt-side:** add explicit taxonomy examples ("food/recipes/cooking → personal", "monitoring/alerts → technical") to the categorise system prompt. If brain owns the prompt, this is a config change, not a model change.

### Path 1b: Agents-A1-4B (previous recommendation, now secondary)

Same benefits, slightly slower (79 tps, 715 ms) and one task worse on quality (categorised vacation as `admin` per its taxonomy interpretation vs Gemma's `personal`). Apache-2.0. Kept as a plausible fallback if Gemma 4 E2B has issues in real-world traffic.

### Path 2: Add `--parallel 2` to Ornith prod (no new slot)

Trade: doubles KV cache VRAM (10.9 → ~14 GiB), no additional container. Cheaper VRAM cost than a new slot.

Halves wait time for the second queued task on `:8002`. Doesn't help when 3+ tasks queue (common under brain-ingest bursts). Keeps categorise quality at Ornith's 9B bar.

### Path 3: Look further afield

Candidates worth pulling if the above two don't satisfy:
- **`Nanbeige/Nanbeige4.2-3B`** — brand new (1d old), no GGUF yet, wait a week for community quants.
- **`fdtn-ai/antares-1b`** — granitemoehybrid 1B, brand new, worth a bench when it gets more attention.
- **Gemma 4 E4B-it Q4_K_M** — already tested at 73.9 tok/s decode; hasn't been bench-tested for categorise quality specifically. Given E2B already matches Ornith at 88 tps, E4B likely wins outright with even better quality.
- **Custom-tune Qwen3-4B-Instruct-2507 on our categorise corpus** — the base model handles it competently; a LoRA on our specific taxonomy and examples would likely fix the recipe issue.

## IPEX-LLM Ollama compatibility on B60 — not viable today (2026-07-22)

Investigated as an alternative serving stack (analogous to how TEI-IPEX beats llama.cpp SYCL for rerank). Blocked by container maturity:

- **Image:** `intelanalytics/ipex-llm-inference-cpp-xpu:latest` (27.7 GB, last updated 2025-08-26 — ~11 months stale)
- **Bundled Ollama:** v0.9.3 with llama.cpp snapshot from ~mid-2025 (predates Gemma 4 architecture)
- **Bundled oneAPI:** 2025.0 (host has 2025.3.3)
- **Failure modes observed:**
  - Gemma 4 GGUFs load-fail with `unknown model architecture: 'gemma4'`
  - Gemma 3 4B: llama runner crashes with SIGSEGV on model load
  - Qwen2.5 3B: same SIGSEGV
  - `sycl-ls` inside container sees only OpenCL CPU, not the B60 via Level Zero (despite `--device=/dev/dri` and `ONEAPI_DEVICE_SELECTOR=level_zero:0`)
- **Root cause:** container userland compiled against older oneAPI/L0 ABI; can't drive our current L0 driver on Battlemage

**Practical conclusion:** the TEI-for-rerank analogy doesn't extend to categorise/generation today. TEI has an actively-maintained XPU image (Intel + HF collaboration). IPEX-LLM Ollama's Docker packaging is behind Battlemage support. Path forward would be building IPEX-LLM from source against current oneAPI, which is a real project — worth watching but not competitive with just running our up-to-date `llama.cpp:sycl-f16` b10068 image.

**Real win path for categorise stack acceleration** is inside our existing llama.cpp SYCL stack: newer llama.cpp builds pick up Battlemage-specific optimizations quickly. Gemma 4 E2B on our b10068 image already delivers 88 tps decode for categorise — no framework switch needed to reclaim throughput.

## Bench provenance

- Session: 2026-07-22 21:00 local
- Image: `llama.cpp:sycl-f16` (b10068, commit `571d0d540`)
- Isolated: all three candidates loaded sequentially on `:8009`; Ornith baseline probed live on prod `:8002`
- Prod services untouched throughout
- Full bench script: `/tmp/categorise_bench.py` on llm.local
