# NVIDIA Nemotron 3.5 Lightning 30B-A3B — Tested 2026-08-21

**Status: DEPLOYED 2026-08-22** on `llm.local:8011`, B60 card 2, dedicated whole card, for agent testing. **Retuned for the agent workload the same day** — `--spec-draft-p-min 0.6` and `temp 0.2` lifted real code-generation decode from 35.02 to 50.71 tok/s (+44.8%); see [finding #34](../../docs/findings.md) and the retuning section below. Synthetic bench: 91.91 tok/s decode at `--spec-draft-n-max 7` with 99.5–100% MTP acceptance — beats the reasoning fallback (Gemma 4 26B-A4B, 62.84) on every axis. Deliberately **not** in any Traefik pool. Deployment detail and the rejected 262K config are at the bottom of this page.

**HF:** [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) · [bartowski GGUF](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) (used here) · [ggml-org GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) · [unsloth](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) · [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
**License:** nvidia-open-model-license (`license:other`)
**Publisher:** NVIDIA
**Released:** 2026-08-01 (BF16), 2026-08-04 (NVFP4)
**Arch:** `nemotron_h` / `nemotron_h_moe` — Mamba2 hybrid MoE. Present in **both** b10433 and b10566; this model was never gated on the rebuild.
**Drafter:** `mtp-…-Q8_0.gguf`, first-party MTP head shipped in the same bartowski repo. Requires `--spec-type draft-mtp`.

## Why this was interesting

Two reasons, and the second one turned out to matter more than the first.

**It is the 30B-A3B-class model that actually fits.** Ornith 1.5-35B-A3B failed twice on this stack — 32.5% and 26.2% MTP acceptance — but on *drafter alignment under compression* (finding #24), not on VRAM. Nemotron arrives at 31.6B/A3B with a first-party MTP head matched to the release, which is precisely the failure mode Ornith could not escape.

**It isolates a variable that had been confounded for six days.** Qwen 3.8-27B was parked at 23.0 tok/s with the note *"revisit when SYCL SSM gets XMX GEMM"* — blaming the hybrid SSM layers. But that model is also **dense** 27B, so SSM cost and dense-bandwidth cost were tangled in a single data point and nobody had a *sparse* hybrid to separate them. Nemotron is exactly that: same architectural class, ~3B active instead of 27B.

## Specs

| | |
|---|---|
| Total parameters | 31.6B |
| Active per token | ~3B (128 routed experts, **6 active**) |
| Architecture | `nemotron_h` — Mamba2 SSM / attention hybrid MoE |
| Layers | 52 |
| Hidden dimension | 2,688 |
| Attention heads | 32 Q / 2 KV (GQA 16:1) |
| Mamba2 | 64 heads × 64 head dim, SSM state 128, 8 groups |
| Expert FFN | intermediate 1,856 |
| Vocabulary | 131,072 |
| Context | 262,144 |
| MTP | `num_nextn_predict_layers: 1` — one draft layer, shipped in-weights |
| Modalities | text → text |

## Quantizations (bartowski ladder, GiB) — the interesting part

```
IQ2_XXS 17.54   Q2_K    17.61   Q4_0    17.75   Q3_K_M  18.46   Q3_K_XL 19.02
IQ4_XS  17.62   Q3_K_S  17.64   Q2_K_L  17.78   Q4_1    19.44   Q4_K_S  21.61
Q4_K_M  23.73   Q4_K_L  23.85   Q5_K_S  23.06   Q5_K_M  25.11   Q8_0    32.60
```

**The whole ladder from IQ2_XXS to Q4_K_M spans 17.5 → 23.7 GiB.** Only the 128 routed experts compress; the dense trunk and Mamba2 SSM state tensors do not. Two consequences (finding #26):

- **Q4_K_M cannot be served on a 24 GiB B60 at all** — 23.73 GiB of weights before a 2.03 GiB MTP head and any KV. Arithmetic, not tuning. A 23 GB download was abandoned at 99% on discovering this.
- **Dropping below Q4 buys nothing.** Q2_K saves 0.14 GiB over Q4_0 for real quality loss. The reflexive "step down one quant to fit" move is dead for this architecture family.

**Q4_0 @ 17.75 GiB is the only sensible choice** — largest non-IQ quant that fits (IQ excluded by finding #13), and SYCL's reordered Q4_0 path is already well-trodden via the Gemma 4 QAT models.

## Setup (llama.cpp SYCL on Intel Arc Pro B60)

Production launcher: [`configs/launchers/start-llamacpp-nemotron-agent.sh`](../../configs/launchers/start-llamacpp-nemotron-agent.sh). Equivalent to:

```bash
docker run -d --name llamacpp-nemotron \
  --memory=24g --memory-swap=24g --device /dev/dri \
  --group-add "$(getent group render|cut -d: -f3)" \
  --group-add "$(getent group video|cut -d: -f3)" \
  -v /data/llm/nemotron-3.5-lightning-30b-a3b-GGUF:/models:ro \
  -p 0.0.0.0:8011:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  -e NEO_CACHE_PERSISTENT=1 \
  llama.cpp:sycl-f16-next-bb4caa754 \
  -m /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
  --model-draft /models/mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 7 \
  -ngl 99 -ngld 99 \
  -c 131072 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on -ub 2048 -b 2048 \
  --jinja --reasoning off \
  --predict 2048 --top-k 20 --min-p 0.0 \
  --host 0.0.0.0 --port 8000 --metrics
```

**Critical flags:**
- `--spec-draft-n-max 7` — **not the default 3.** Worth +16.3% here, and 8 falls off a 34% cliff (finding #30). Never exceed 7 on this card.
- `-ngld 99` — offload the draft head too.
- Use the **Q8_0** MTP head, not the Q4_0 one also published in that repo — finding #24.
- `--host 0.0.0.0 --port 8000` is **mandatory**, not cosmetic: omit them and llama.cpp binds its default 8080 while the container publishes 8000, and health checks fail with a connection reset on a model that actually loaded fine.

## Benchmarks (b10566, isolated card 2, 20 runs × 300 tok @ temp 0.6 / top-p 0.95 / top-k 20)

### `--spec-draft-n-max` sweep

| n-max | Decode med | σ | Acceptance | Accepted/draft | VRAM | Prefill @ 12K | Verify batch |
|---|---|---|---|---|---|---|---|
| 3 | 79.05 | 1.06 | 99.9% | 2.98 / 3.00 | 21.99 | 1,766 | 4 |
| 5 | 86.94 | 0.62 | 99.9% | 4.97 / 5.00 | 22.08 | 1,762 | 6 |
| 6 | 90.10 | 4.41 | 99.8% | 5.94 / 6.00 | 22.13 | 1,762 | 7 |
| **7** ⭐ | **91.91** | 1.05 | 99.5% | 6.85 / 7.00 | 22.18 | 1,760 | **8** |
| 8 | 60.31 | 0.42 | 99.8% | 7.79 / 8.00 | 22.67 | 1,754 | 9 |
| 9 | 63.93 | 0.07 | **100.0%** | 8.97 / 9.00 | 22.71 | 1,764 | 10 |
| 10 | 67.24 | 0.19 | 99.9% | 9.68 / 10.00 | 22.76 | 1,752 | 11 |

**Acceptance never degrades — 99.5–100% at every setting.** At n-max 9 the drafter returned 8.97 of a possible 9.00 tokens per draft while decode sat at 63.93. The drafter is not the constraint anywhere on this curve; the cap is the target's verification-batch cost, which is flat-ish to batch 8 and then jumps +71%.

### Prefill (server-side `prompt_ms`, 3 samples/size, at n-max 7)

| Prompt | 500 | 2K | 5K | 12K |
|---|---|---|---|---|
| tok/s | 760 | 1,518 | 1,629 | **1,760** |

Prefill is flat across the entire n-max sweep (~1,760 @ 12K), confirming the cliff is specific to the decode/verify path.

## Comparison to peers (same build, same card, same harness)

| Model | Decode | Prefill @ 12K | Acceptance | VRAM |
|---|---|---|---|---|
| **Nemotron 30B-A3B + MTP @ n-max 7** | **91.91** | **1,760** | **99.5%** | 22.18 GiB |
| Ling-3.0-tiny | 91.86 | 1,218 | — | 5.72 GiB |
| Qwen3.8-9B-Distill + MTP | 73.97 | 2,020 | 81.4% | 14.76 GiB |
| Ornith 1.5-9B + MTP (prod chat) | 65.15 | 1,987 | 84.7% | 14.62 GiB |
| Gemma 4 26B-A4B QAT + MTP (reasoning fallback) | 62.84 | 1,592 @ 4K | 97.2% | 19.9 GiB |
| **Qwen 3.8-27B + MTP (dense hybrid, parked)** | **23.0** | **333** | 57.9% | 22.3 GiB |

## Verdict

**Fastest large model this stack has run, and it corrected a wrong diagnosis.**

**1. The "SYCL SSM penalty" does not exist (finding #25).** Nemotron is the same architectural class as Qwen 3.8-27B and decodes **4× faster** (91.91 vs 23.0) while prefilling **5.3× faster** (1,760 vs 333). The variable was never the Mamba2 layers — Qwen 3.8-27B activates all 27B of its parameters per token, Nemotron ~3B. The 23 tok/s is the dense bandwidth wall already recorded for Muse Glimmer-30B dense (25.3) and Laguna XS-2.1 (29.5). Sparse hybrid-linear-attention models are **not** blocked on upstream SYCL work and should be benched on arrival; only *dense* hybrids above ~12B are penalised, and no upstream kernel work will fix that.

**2. It beats the reasoning fallback on every axis** — +46% decode over Gemma 4 26B-A4B, +11% prefill, at 99.5% vs 97.2% acceptance.

**3. It needs the whole card — and got one.** 22.18 GiB peak against a 24 GiB card leaves ~2 GiB, so it displaces every co-resident service. That made it the natural payload for **task #144**, which completed 2026-08-22: embed/rerank/categorise moved to the B580 node `llm2.local`, card 2 was freed, and Nemotron took it. Better use of a freed B60 than the task #142 tensor-split of Ornith 1.5-35B-A3B, which has failed twice on MTP acceptance under compression.

**4. Decode stability is exceptional.** σ = 0.33 tok/s at n-max 3 against 16.41 for Ornith and 35.26 for LFM2.5. Near-perfect acceptance removes the accept/reject variance that makes every other MTP row on this box bimodal — a tight σ is itself evidence of high acceptance.

## Watch items

- **Confirm the batch-8 SYCL fast-path hypothesis in kernel source.** The +71% step-cost discontinuity crossing verify batch 9 is measured and replicated (see finding #30), but the explanation is inferred, not read.
- **Re-check the n-max boundary after each llama.cpp bump** — it is a kernel-path property and an upstream change could move it.
- **Q4_K_S at 21.61 GiB** is the only untested rung between Q4_0 and the unusable Q4_K_M. It would fit weights-only but leaves nothing for the head; probably not worth the download.
- **NVFP4 variant** — irrelevant on Intel today, relevant if the stack ever gains Blackwell.

## Deployment (2026-08-22)

Live on `llm.local:8011`, B60 card 2 (`level_zero:1`), alone on the card. Launcher: `configs/launchers/start-llamacpp-nemotron-agent.sh`. **No Traefik route** — agent-test traffic cannot reach a production pool and vice versa. Reachable on the LAN at `http://192.168.1.253:8011` and over Tailscale.

What the card carries now, versus before:

| | before 2026-08-22 | after |
|---|---|---|
| B60 card 1 | Ornith + embed + rerank + categorise, ~20.7 GiB | unchanged, ~20.5 GiB |
| B60 card 2 | mirror of all four, ~20.7 GiB | **Nemotron alone, ~22.8 GiB** |
| B580 `llm2.local` | — | embed + rerank + categorise, ~5.8 GiB of 12 |

### `-c 262144` was tried and rejected

The model's native context is 262,144 and it *does* load there — but the margin is not real:

| context setting | idle VRAM | peak under load | headroom |
|---|---|---|---|
| `-c 262144` | 23.38 GiB | **23.88 GiB at a 105K-token request** | **0.12 GiB** |
| **`-c 131072`** ⭐ | 21.86 GiB | **22.26 GiB at a 70K-token request** | **1.74 GiB** |

At 262K the card reached **24,450 of 24,576 MiB — 126 MiB free** — and decode collapsed to **19.96 tok/s** on that request, roughly a quarter of normal. Nothing crashed, but there is no margin for a transient and the throughput penalty makes the extra context worthless anyway. The launcher now defaults to `131072` with that reasoning inline; `CTX=262144 ./start-llamacpp-nemotron-agent.sh` still overrides it if someone wants to re-measure on a future build.

### Measured on the deployed config (`-c 131072`, n-max 7)

| Prompt | Prefill tok/s | Decode tok/s | Peak VRAM |
|---|---|---|---|
| 2K | 1,354 | 89.86 | 22.06 GiB |
| 12K | 1,750 | 88.03 | 22.18 GiB |
| 32K | 1,661 | 74.23 | 22.18 GiB |
| 64K | 1,439 | 75.11 | 22.21 GiB |
| 70K | 1,321 | 72.19 | 22.26 GiB |

Decode holds 72–90 tok/s from 2K to 70K of context. The 12K figures reproduce the bench rows above, which is the check that the deployed config matches what was measured.

### Retuned for the agent workload (2026-08-22) — finding #34

The deployed config above came straight from the bench: 91.91 tok/s at 99.5% MTP acceptance. **Real agent traffic did not reproduce it.** Forty minutes of live opencode/pi.dev use at 82K context gave **46.2% acceptance (8,066 accepted / 17,469 drafted), mean accepted chain 4.23, decode 23.6–58.3 tok/s** — and the split by turn type was stark:

| turn shape | acceptance | mean chain |
|---|---|---|
| short replies / tool calls (40–220 tok) | 0.60 – **0.87** | 5.2 – 7.1 |
| long code generations (700–6,300 tok) | **0.25** – 0.46 | 2.8 – 4.2 |

The cause was `--spec-draft-p-min`, which defaults to **0.00** and was never set: the MTP head ran the full 7 forward passes on every step regardless of its own confidence. Swept on a 20K-token code-generation probe at `temp 0.2`, `n-max 7` fixed:

| `--spec-draft-p-min` | decode tok/s | acceptance | mean chain | drafted |
|---|---|---|---|---|
| 0.00 (was deployed) | 41.16 | 43.5% | 4.04 | 1,381 |
| 0.30 | 41.71 | 46.1% | 3.86 | 1,280 |
| 0.50 | 46.88 | 58.7% | 3.93 | 988 |
| **0.60** ⭐ | **52.04** | 70.9% | 4.19 | 833 |
| 0.75 | 49.73 | 77.8% | 3.55 | 702 |
| 0.90 | 44.35 | 85.3% | 3.21 | 551 |

**+26.4% for one flag**, with accepted tokens essentially flat (601 → 591) while drafted tokens fall 40%. Acceptance climbs monotonically across the whole sweep while decode peaks at 0.60 — tuning on acceptance would have picked 0.90 and cost 15%, which is finding #32's rule stated as plainly as it gets.

Sampling was the smaller half. The slot had been serving NVIDIA's **chat** default of `temp 1.0` because the launcher never set `--temp`; `temp 0.2 / top-p 0.9` is worth +7–9%. A 2×2 probe, 800 output tokens per cell:

| workload | sampling | p-min 0.00 | p-min 0.60 |
|---|---|---|---|
| code | temp 1.0 | 35.02 | 46.36 |
| **code** | **temp 0.2** | 37.64 | **50.71** |
| prose | temp 1.0 | 23.74 | 36.62 |
| prose | temp 0.2 | 25.37 | 38.24 |

**Combined: 35.02 → 50.71 tok/s on the code workload, +44.8%.** Note also that this drafter handles **code better than prose** (50.71 vs 38.24) — the workload was never the problem, the drafting policy was.

`n-max` stays at **7**. With p-min at 0.60 the mean chain settles at 4.19, so p-min truncates long before n-max binds; finding #27's ceiling (never 8+ on this card) is unchanged. `NMAX` and `PMIN` are now env-overridable in the launcher for future sweeps.

Raw results: `/data/llm/benchmarks/20260822-codetune/`. Probe: `/data/llm/benchmarks/probe-codetune.py`, sweep driver `/data/llm/benchmarks/sweep-pmin.sh`.

### Context growth is a client-side cost, not a server one

Prefix reuse works: LCP similarity is 0.99+ on most turns, and `--cache-ram` (8 GiB default) and `-ctxcp` (32 checkpoints × 8,192 spacing = 262K of coverage) are both adequate untouched. What hurts is the size of the conversation itself. At 82,763 tokens prefill has decayed to **1,031 tok/s** from 1,772 at 12K, so each time the client rewrites history — two events in the observed window, `f_sim_best` dropping to 0.16 and 0.14, i.e. opencode compaction — the server re-prefills **70,500 tokens in 68 seconds**. Halving the agent's compaction threshold roughly halves both that stall and the decode penalty. No server flag fixes it.

### Operational: the slot can wedge, and `/health` will not tell you

2026-08-23, ~100 minutes after the retune: a generation stalled mid-stream at `n_gen = 1211`, the client's cancel never produced a `slot release`, and `/slots` kept reporting `is_processing: true` on the dead task with the GPU idle at 50 W. `--parallel 1` meant every later request queued behind it indefinitely. **`/health` returned `200` the whole time** — it checks the HTTP listener, not the inference slot — and `--restart unless-stopped` never fired because the process never exited. Only the preceding 40+ turns' clean completions and this single cancel exist in the log, so the cause is not established; a restart cleared it.

`:8011` now runs [`nemotron-wedge-watchdog.sh`](../../configs/watchdogs/nemotron-wedge-watchdog.sh) (systemd unit alongside it), which restarts the container after two consecutive 90-second windows of **zero server-log output with a request in flight**. It deliberately does *not* use the `e2b-wedge-watchdog.sh` detector: that one keys on completion counters, which a long generation freezes by design — measured, a healthy 4,000-token turn reached `frozen 3/6`. Full reasoning in finding #35.

### Gotcha found during deployment

The first launcher revision omitted `--host`/`--port`, so llama.cpp bound its **default 8080** while the container published 8000 — health checks failed with `Recv failure: Connection reset by peer` even though the model had loaded fine. The repo's other launchers all pass `--host 0.0.0.0 --port 8000` explicitly; that is not optional.

## Files on disk

```
/data/llm/nemotron-3.5-lightning-30b-a3b-GGUF/
├── NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf         19.06 GB — main model
└── mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf      2.18 GB — MTP draft head
```

Raw bench results: `/data/llm/benchmarks/20260821/nemotron-3.5-lightning-30b-a3b.json` and `/data/llm/benchmarks/20260821-nmax/nemotron-nmax{3,5,6,7,8,9,10}.json`.

## References

- [Model card — NVIDIA-Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- [Full bench write-up + findings #25-#32](2026-08-21-tier1-tier2-bench.md)
- [Candidate sweep that shortlisted it](2026-08-21-new-candidates-sweep.md)
- [`docs/findings.md`](../../docs/findings.md) — #25 (SSM), #26 (quant ladder), #27 + #30 (n-max)
