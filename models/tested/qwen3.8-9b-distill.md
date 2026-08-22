# empero-ai Qwen3.8-9B-Distill + MTP — Tested 2026-08-21

**Status:** Benched. **Real challenger to the production chat slot** — 73.97 tok/s against Ornith 1.5-9B's 65.15 on the same build, same window, at effectively identical VRAM and prefill. **Not promoted on speed alone:** Ornith is an agentic-coding post-train carrying the pi.dev workload, and a 13.5% decode win means nothing if tool-calling or edit-diff quality regresses. Needs a `docs/track2`-style qualitative bake-off first.

**HF:** [`empero-ai/Qwen3.8-9B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill) · [their GGUF](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF) · [**our MTP heads**](https://huggingface.co/srmiles/Qwen3.8-9B-Distill-MTP-GGUF)
**License:** Apache 2.0
**Publisher:** empero-ai — distillation of Qwen 3.8 onto [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)
**Released:** 2026-08-15
**Arch:** `qwen3_5` / `Qwen3_5ForConditionalGeneration` — **byte-for-byte the same shape as Ornith 1.5-9B**
**Drafter:** none published upstream. **Built locally** and uploaded — see [`../hf-uploads/qwen3.8-9b-distill-mtp.md`](../hf-uploads/qwen3.8-9b-distill-mtp.md).

## Why this was interesting

**Zero integration risk.** Diffing its `config.json` against Ornith 1.5-9B's: same 32 layers, same 24 `linear_attention` + 8 `full_attention` pattern (3:1), same hidden 4,096, same `head_dim` 256, same `mtp_num_hidden_layers: 1`, same 262k context. The production launcher works unchanged apart from the model and head paths — so this is a pure capability A/B against the current chat slot rather than a plumbing exercise.

**And it had no drafter.** The model carries an MTP head in its weights but empero's GGUF repo publishes main-model quants only. Benched as-published it runs unassisted and lands ~20-25% low for reasons that have nothing to do with the model. That gap is what prompted building the head locally, which turned into finding #28.

## Specs

| | |
|---|---|
| Total parameters | 9.65B (dense) |
| Architecture | `qwen3_5` hybrid linear-attention |
| Layers | 32 — 24 `linear_attention` + 8 `full_attention` (3:1) |
| Hidden dimension | 4,096 |
| Attention heads | 16 Q / 4 KV, `head_dim` 256, `attn_output_gate: true` |
| Linear attention | 16 key heads / 32 value heads, head dim 128, conv kernel 4 |
| FFN | intermediate 12,288 |
| Vocabulary | 248,320 |
| Context | 262,144 |
| MTP | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` |
| Modalities | text + image + video → text (vision tower present, untested here) |

## The missing drafter

empero ships five main-model quants and no head. Built from the BF16 safetensors:

```bash
convert_hf_to_gguf.py Qwen3.8-9B-Distill-hf \
  --mtp --outtype q8_0 --outfile mtp-Qwen3.8-9B-Distill-head-Q8_0.gguf
```

18 tensors, 2.43 GB, architecture `qwen35`, loaded first try, **81.4% acceptance** — inside the band Ornith's third-party head reaches (84.7%). Both Q8_0 and BF16 are now published at [`srmiles/Qwen3.8-9B-Distill-MTP-GGUF`](https://huggingface.co/srmiles/Qwen3.8-9B-Distill-MTP-GGUF).

**Q8_0 is the recommended default for this one, deliberately breaking the BF16-only house rule.** ~2.03B of the head's 2.28B parameters are `token_embd` + `output` (248,320 × 4,096, twice) — a full copy of the vocab matrices, not the MTP block. At BF16 the head is 4.56 GB against a 5.38 GB target, ~85% of it; the Gemma assistant heads are ~5%. The discriminator is **head-to-target ratio**, not precision dogma. Full reasoning in the [upload doc](../hf-uploads/qwen3.8-9b-distill-mtp.md).

## Setup (llama.cpp SYCL on Intel Arc Pro B60)

```bash
docker run -d --name llamacpp-qwen38-9b \
  --memory=14g --memory-swap=14g --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/Qwen3.8-9B-Distill-GGUF:/models:ro \
  -p 0.0.0.0:8020:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  llama.cpp:sycl-f16-next-bb4caa754 \
  -m /models/Qwen3.8-9B-Q4_K_M.gguf \
  --model-draft /models/mtp-Qwen3.8-9B-Distill-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -ngld 99 \
  -c 262144 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 --top-k 20 --min-p 0.0 \
  --host 0.0.0.0 --port 8000 --metrics
```

Identical to `start-llamacpp-sycl-ornith-1.5-c2.sh` apart from the two paths.

## Benchmarks (b10566, isolated card 2, 20 runs × 300 tok @ temp 0.6 / top-p 0.95 / top-k 20)

### With and without the drafter

| | Decode med | σ | Prefill @ 12K | Peak VRAM |
|---|---|---|---|---|
| unassisted | 56.69 | **0.05** | **2,366** | 10.91 GiB |
| **+ Q8_0 MTP head** | **73.97** | 10.57 | 2,020 | 14.76 GiB |
| delta | **+30.5%** | — | **−14.6%** | **+3.85 GiB** |

**The head costs prefill to buy decode (finding #31).** Right trade for decode-bound chat and agent work; *not* automatically right for a prefill-heavy short-output workload like categorise.

### Full numbers with the head

| Metric | Value |
|---|---|
| Decode | 73.97 median · 65.56 mean · σ 10.57 (min 47.69, max 75.00) |
| Acceptance | 81.4% — 4,236 accepted of 5,203 draft tokens |
| Accepted per draft | 2.43 at `--spec-draft-n-max 3` |
| Prefill | 1,178 @ 500 · 1,914 @ 2K · 1,957 @ 5K · 2,020 @ 12K |
| Peak VRAM | 14.76 GiB |

## Head-to-head vs the production chat slot

Same build, same card, same window, same harness:

| | Qwen3.8-9B-Distill | Ornith 1.5-9B (prod) | Δ |
|---|---|---|---|
| Decode median | **73.97** | 65.15 | **+13.5%** |
| Decode mean | 65.56 | 63.48 | +3.3% |
| Prefill @ 12K | **2,020** | 1,987 | +1.7% |
| Prefill @ 5K | 1,957 | 1,946 | +0.6% |
| Acceptance | 81.4% | **84.7%** | −3.3 pp |
| Accepted/draft | 2.43 | **2.53** | −0.10 |
| Peak VRAM | 14.76 GiB | 14.62 GiB | +0.14 |

## Verdict

**Earns a quality bake-off, and nothing more than that.**

**1. The speed win is real but narrower than the headline.** +13.5% on median, only +3.3% on mean — because decode is bimodal, clustering near 74 and near 50 tok/s. Ornith shows the same shape (56–75). The unassisted run settles the cause: σ collapses from 10.57 to **0.05** without a drafter, so the bimodality is an acceptance artifact, not thermal or scheduler noise (finding #31).

**2. Do not cut over on speed.** Ornith 1.5-9B is a DeepReinforce agentic-coding post-train, RL-tuned for scaffold generation and solution rollouts, carrying the pi.dev agent. This is a general distillation of Qwen 3.8. The two are not interchangeable on the axis that matters, and the repo already has the right instrument for deciding — the `docs/track2-quality-bakeoff.md` pattern used for Ornith 9B vs Gemma 4 E2B.

**3. Its acceptance is slightly worse**, which also means it has less headroom on `--spec-draft-n-max` than Nemotron does: at p ≈ 0.898 the modelled optimum is ~4 with a ~+1% gain, below the p ≈ 0.914 break-even for pushing to 7. Not worth sweeping.

**4. The drafter work is reusable regardless.** Even if this model never ships, finding #28 came out of it: any model declaring `mtp_num_hidden_layers` can now be given a head locally, at a quant we choose, without waiting on a third-party upload.

## Watch items

- **Qualitative bake-off vs Ornith 1.5-9B on the pi.dev corpus** — tool-calling reliability, edit-diff correctness, scaffold quality. This is the only thing that decides a cutover.
- **Vision** — the model carries a full vision tower (`deepstack_visual_indexes`, video preprocessor) that was not exercised. Only Muse Glimmer-30B currently offers vision on this stack, at 25.3 tok/s; a 74 tok/s vision-capable 9B would be a materially different option if the tower works under llama.cpp.
- **Which content classes the head misses on** — the bimodality is understood in cause but not in content.

## Files on disk

```
/data/llm/Qwen3.8-9B-Distill-GGUF/
├── Qwen3.8-9B-Q4_K_M.gguf                       5.78 GB — main model (empero)
├── mtp-Qwen3.8-9B-Distill-head-Q8_0.gguf        2.43 GB — MTP head (built here) ⭐
└── mtp-Qwen3.8-9B-Distill-head-BF16.gguf        4.57 GB — MTP head, canonical precision
```

BF16 safetensors were deleted after conversion. Conversion venv preserved at `/data/llm/build/convert-venv`.
Raw bench results: `/data/llm/benchmarks/20260821/qwen3.8-9b-distill-mtp.json`, `/data/llm/benchmarks/20260821-nmax/qwen3.8-9b-nodraft.json`.

## References

- [Model card — empero-ai/Qwen3.8-9B-Distill](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill)
- [Our MTP head upload](https://huggingface.co/srmiles/Qwen3.8-9B-Distill-MTP-GGUF) · [upload doc](../hf-uploads/qwen3.8-9b-distill-mtp.md)
- [Ornith 1.5-9B — production chat](../production/ornith-1.5-9b.md)
- [Full bench write-up + findings #25-#32](2026-08-21-tier1-tier2-bench.md)
