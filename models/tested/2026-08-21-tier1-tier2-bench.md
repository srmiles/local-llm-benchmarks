# Tier 1 + Tier 2 candidate bench — 2026-08-21

Benches for the four candidates shortlisted in [`2026-08-21-new-candidates-sweep.md`](2026-08-21-new-candidates-sweep.md), plus an Ornith 1.5-9B reference re-run on the same build.

**Build:** `llama.cpp:sycl-f16-next-bb4caa754` = **b10566**, commit `bb4caa754`, image `c45646db48cf`, built 2026-08-21 14:33 UTC. Rebuilt from b10433 because Ling-3.0-tiny needs `bailingmoe3`, which b10433 lacks (bartowski's GGUF landed 08-18, four days after the b10433 cutover). `nemotron_h_moe`, `lfm2moe` and `gemma4-assistant` are all present in both builds.

**Isolation:** card 2 (`level_zero:1`) taken fully out of service 14:44–15:04 UTC — chat mirror `:8010`, primary categorise `:8009`, `llamacpp-embed-c2`, `tei-rerank-c2` and their three watchdog services all stopped. Card 1 served every Traefik route solo for the window. Card 2 idled at 243 MiB before the first model loaded. Bench slot bound to `:8020`, never `:8010`, so the LB could not route live traffic into a bench model.

**Methodology:** `/data/llm/benchmarks/bench-candidate.py`. Prefill = server-side `prompt_ms`, 3 samples × {500, 2K, 5K, 12K}. Decode and acceptance = **20 runs × 300 tokens at production sampling** (`temp 0.6 / top-p 0.95 / top-k 20`), not greedy — small-N greedy is variance-dominated (brain `e2255203a12d5080`). Acceptance accumulated from `llamacpp:spec_decode_*` metric deltas across all 20 runs. VRAM sampled from `xpu-smi` every 3s, peak reported.

> **VRAM caveat:** peak observed during a bench that never exceeds 12K context, so these are directly comparable *to each other* but lower than prod steady-state at full context. The Ornith reference reads 14.62 GiB here against 20.7 GiB in production at 262K. Read the column as a ranking, not as a co-residence budget.


> **Re-run 2026-08-26 — read these numbers with a ±10% band.** This bench was replayed on a locally patched build and **four control models moved +12.9%, +9.5%, +0.7% and −3.3%** despite the change under test being unable to affect them. The cause is that this run had **no saved runner script**, so per-model contexts were not reproducible — peak VRAM differs inconsistently between the two runs (ornith 14.62 → 10.83 GiB, lfm2.5 8.63 → 10.50). The Ornith reference row here is also degraded: σ 16.41 with one sample at 0.0 tok/s. Treat this table as a ranking, not as a baseline for measuring single-digit or low-teens effects. [Re-run and finding #40.](2026-08-26-sycl-patches-default-bench.md)

---

## Results

| Model | Quant | Params | Decode med | Decode mean (σ) | Prefill 2K / 5K / 12K | Accept | Peak VRAM |
|---|---|---|---|---|---|---|---|
| **Nemotron 3.5 Lightning 30B-A3B** | Q4_0 + MTP Q8_0 | 31.6B / A3B | **78.95** | 78.89 (0.33) | 1,536 / 1,658 / **1,772** | **99.8%** · 2.98/draft | 21.99 GiB |
| **LFM2.5-8B-A1B + DSpark** | Q4_K_M + DSpark Q8_0 | 8B / A1B | **168.25** | 161.40 (35.26) | 2,667 / 3,156 / **3,665** | 79.8% · **5.51/draft** | 8.63 GiB |
| **Qwen3.8-9B-Distill + MTP** | Q4_K_M + self-converted Q8_0 head | 9.65B dense hybrid | **73.97** | 65.56 (10.57) | 1,914 / 1,957 / 2,020 | 81.4% · 2.43/draft | 14.76 GiB |
| Ling-3.0-tiny | Q4_K_M | 7.9B / A0.8B | 91.86 | 90.28 (3.75) | 2,293 / 1,811 / 1,218 | — (no drafter) | 5.72 GiB |
| *Ornith 1.5-9B — reference* | Q4_K_M + MTP Q8_0 | 9B dense hybrid | *65.15* | *63.48 (16.41)* | *1,910 / 1,946 / 1,987* | *84.7% · 2.53/draft* | *14.62 GiB* |

The reference row reproduces the repo's recorded 65.44 tps / 82.7% acceptance / 2,040 prefill to within noise, which is what validates the harness, the new build and the card isolation for the other four rows.

> All rows above ran at `--spec-draft-n-max 3`. **Nemotron's number is superseded — at n-max 7 it does 91.91 tok/s** (finding #30). The Qwen3.8-9B row now also has an unassisted baseline of 56.69 (finding #31).

---

## Finding #25 — the SYCL SSM penalty was never about SSM

Qwen 3.8-27B was parked at 23.0 tps with *"revisit when SYCL SSM gets XMX GEMM"*. That diagnosis was wrong, and this bench is what separates the variables.

Nemotron 3.5 Lightning is a `nemotron_h` **Mamba2 hybrid** — the same class of architecture, 52 layers, 128 experts / 6 active. It decodes at **78.95 tok/s**, faster than Ornith 1.5-9B (65.15) and faster than Gemma 4 26B-A4B (62.84), and prefills at **1,772 @ 12K** against Qwen 3.8-27B's 333.5.

The difference is not the SSM layers. It is that Qwen 3.8-27B activates all 27B of its parameters per token while Nemotron activates ~3B. The 23 tps was the dense-27B bandwidth wall — exactly the ~24 tps ceiling already recorded for Muse Glimmer-30B dense (25.3) and Laguna XS-2.1 (29.5). SSM cost and dense-bandwidth cost were confounded in a single data point.

**Consequence:** the hybrid-linear-attention class is *not* blocked on upstream SYCL work. Sparse hybrids are among the fastest things this stack has run. Only *dense* hybrids above ~12B are penalised, and that is a bandwidth property they share with every dense model on Battlemage.

## Finding #26 — `nemotron_h` barely quantises, and it decides the quant for you

The full bartowski ladder, in GiB:

```
IQ2_XXS 17.54   Q2_K    17.61   Q4_0    17.75   Q3_K_M  18.46   Q3_K_XL 19.02
IQ4_XS  17.62   Q3_K_S  17.64   Q2_K_L  17.78   Q4_1    19.44   Q4_K_S  21.61
Q4_K_M  23.73   Q4_K_L  23.85   Q5_K_M  25.11   Q6_K    31.95   Q8_0    32.60
```

From IQ2_XXS to Q4_K_M the entire range is 17.5 → 23.7 GiB. The floor sits at ~17.5 GiB because only the 128 experts compress — the dense trunk and the Mamba2 SSM state tensors do not.

Two consequences:

- **Q4_K_M (23.73 GiB) cannot be served on a 24 GiB B60 at all**, let alone beside a 2.03 GiB MTP head. This is not a tuning problem; it is arithmetic.
- **Dropping below Q4 buys nothing.** Q2_K saves 0.14 GiB over Q4_0 and costs real quality. The usual "step down one quant to fit" move is dead for this architecture family.

**Q4_0 at 17.75 GiB is the only sensible choice** — largest non-IQ quant that fits, and IQ is excluded by finding #13. Worth noting Q4_0 is also the quant the Gemma 4 QAT models use, which are the fastest things on this box; SYCL's reordered Q4_0 path is well-trodden.

## Finding #27 — 99.8% MTP acceptance, and `--spec-draft-n-max 3` is now the limiter

Nemotron's MTP head accepted **4,478 of 4,486 draft tokens — 99.8%, at 2.98 accepted per draft against a hard ceiling of 3.00**. Highest ever on this stack; the previous best was Gemma 4 26B-A4B at 97.2% / 2.90.

At 2.98/3.00 the drafter is essentially never wrong, which means `--spec-draft-n-max 3` is leaving throughput on the table — every draft is being truncated by the flag, not by a rejection.

**Swept later the same day. The answer is `--spec-draft-n-max 7` — 91.91 tok/s, +16.3%, free — and 8 falls off a cliff.** See finding #30 below.

Decode stability supports the same read: σ = 0.33 tok/s across 20 runs, against 16.41 for Ornith and 35.26 for LFM2.5. Near-perfect acceptance removes the accept/reject variance that makes every other MTP row on this box bimodal.

## Finding #28 — self-converted MTP heads work

The empero GGUF repo ships no MTP head, so one was built from the BF16 safetensors:

```
convert_hf_to_gguf.py /data/llm/Qwen3.8-9B-Distill-hf --mtp --outtype q8_0 \
  --outfile mtp-Qwen3.8-9B-Distill-head-Q8_0.gguf
```

18 tensors, 2.43 GB, loaded first try and hit 81.4% acceptance — right in the band the Ornith head reaches (84.7%). The venv is preserved at `/data/llm/build/convert-venv` (torch 2.13.0+cpu, transformers 5.15.1).

This removes a standing dependency. Until now every drafter came from a third-party upload — protoLabsAI for Ornith, Google's safetensors for the Gemma 4 assistants. Any model shipping `mtp_num_hidden_layers` in its config can now be given a head locally, at a quant we choose. The converter also exposes `--dspark`, so DSpark heads are buildable the same way.

## Finding #29 — DSpark drafts wide where MTP drafts deep

LFM2.5-8B-A1B's DSpark head accepted **5.51 tokens per draft** at `--spec-draft-n-max 7` (block size 9) versus MTP's ~2.4–3.0 at n-max 3. Lower acceptance *rate* (79.8% vs 84.7%) but more than double the tokens per accepted draft, because it is a block-diffusion drafter proposing a whole block rather than a short chain.

That is what puts it at **168.25 tok/s** — the fastest decode in the repo outside MiniCPM5-1B, on a model with 4× the active parameters of Gemma 4 E2B.

---

## Finding #30 — the n-max sweep: verification batch size is the cap, and batch 9 is a cliff

Second window, 20:33–21:04 UTC, same card, same build, same harness. n-max 3 re-run first as an in-window control — it reproduced at 79.05 / 99.9% / 2.98 against the morning's 78.95 / 99.8% / 2.98.

| n-max | decode med | σ | acceptance | accepted/draft | VRAM | prefill @ 12K | verify batch (n+1) |
|---|---|---|---|---|---|---|---|
| 3 | 79.05 | 1.06 | 99.9% | 2.98 / 3.00 | 21.99 | 1,766 | 4 |
| 5 | 86.94 | 0.62 | 99.9% | 4.97 / 5.00 | 22.08 | 1,762 | 6 |
| 6 | 90.10 | 4.41 | 99.8% | 5.94 / 6.00 | 22.13 | 1,762 | 7 |
| **7** | **91.91** | 1.05 | 99.5% | 6.85 / 7.00 | 22.18 | 1,760 | **8** |
| 8 | 60.31 | 0.42 | 99.8% | 7.79 / 8.00 | 22.67 | 1,754 | 9 |
| 9 | 63.93 | 0.07 | **100.0%** | 8.97 / 9.00 | 22.71 | 1,764 | 10 |
| 10 | 67.24 | 0.19 | 99.9% | 9.68 / 10.00 | 22.76 | 1,752 | 11 |

**Acceptance never degrades — it is 99.5–100% at every single setting.** At n-max 9 the drafter returned 8.97 of a possible 9.00 tokens per draft, essentially flawless, while decode sat at 63.93. The drafter is not the constraint anywhere on this curve; what changes is what it costs the target to verify the block.

And that cost is not a smooth curve. Throughput climbs monotonically to n-max 7, **collapses 34% at n-max 8**, then *recovers* through 9 and 10. Sorted by verification batch size (`n_max + 1`) the pattern is clean: every batch of 4–8 performs well, every batch of 9+ is penalised. The likely explanation is a SYCL fast path for verification batches up to 8 with a fallback above it — consistent with the power-of-two geometry sensitivity already documented for this backend, and supported by the ~0.5 GiB VRAM step at the same 7→8 boundary. **Stated as a hypothesis: it has not been confirmed in the kernel source.** Prefill is flat at ~1,760 across the whole sweep, so whatever it is, it is specific to the decode/verify path.

Two practical consequences beyond "use 7":

- **Do not tune this by stepping upward until throughput stops improving.** That procedure walks straight off the cliff at 8, loses a third of throughput, and invites the conclusion that 7 was noise.
- **Do not read a decode collapse as a drafter problem** without checking the acceptance counters. Here acceptance was perfect at exactly the points where throughput cratered. The two failure modes are indistinguishable from the throughput number alone and have opposite fixes.

Re-check after any llama.cpp bump — a kernel-path property can move.

## Finding #30b — the cliff replicates; the optimum does not

Third window, 21:21–21:34 UTC. Gemma 4 26B-A4B (QAT Q4_0 + Google's official MTP Q8_0) — different model family, different drafter provenance, different quant — swept over the same range.

| n-max | decode med | σ | acceptance | accepted/draft | VRAM | prefill @ 12K | verify batch |
|---|---|---|---|---|---|---|---|
| 3 | 59.46 | 3.45 | 88.7% | 2.65 | 17.85 | 1,634 | 4 |
| **5** | **65.34** | 11.63 | 81.6% | 4.07 | 17.85 | 1,632 | **6** |
| 7 | 61.88 | 14.66 | 75.7% | 5.25 | 17.85 | 1,631 | 8 |
| 8 | 44.10 | 6.88 | 83.8% | 6.63 | 18.95 | 1,636 | 9 |

**The cliff is in the same place and is the same size.** Step cost crossing 7 → 8 rises **+71.3%** here against Nemotron's **+70.6%** — two architectures agreeing to within 0.7 points — with the same ~0.5–1.1 GiB VRAM step at the same boundary. That is about as much confirmation as a black-box measurement can give that the batch-9 boundary is a kernel property, not a model one. **`n_max ≤ 7` is a hard rule for this card.**

**The optimum inside that range is not.** Gemma peaks at **n-max 5 (+9.9%)**, and 7 is worse (+4.1%). The reason is visible in the acceptance column: Gemma's acceptance *decays* along the chain, 88.7% → 81.6% → 75.7%, where Nemotron's is flat at 99.9% → 99.9% → 99.5%. Step-cost growth per drafted token is nearly the same for both (+0.161 vs +0.174 of a baseline step), so chain decay is the entire difference between an optimum of 5 and an optimum of 7.

**Prediction vs outcome, honestly.** The framework predicted best-n = 7 at +14% for this model. Direction and rough magnitude were right; the optimum was off by one step. The error traces to the input: the predicted `p` came from the repo's recorded 97.2% acceptance, measured at Gemma's *production* sampling (`temp 1.0`, `top-k 64`), while the bench harness holds sampling at `temp 0.6 / top-p 0.95 / top-k 20` for cross-model comparability. Same model, same drafter, same build — 97.2% vs 88.7%, purely from sampling. See finding #32.

**That caveat was tested and did not hold.** Re-swept at Gemma's own production sampling (`temp 1.0 / top-k 64`), fourth window 21:44–21:53 UTC:

| n-max | harness decode | harness acc | prod decode | prod acc |
|---|---|---|---|---|
| 3 | 59.46 | 88.7% | 58.77 | 77.8% |
| **5** | **65.34 (+9.9%)** | 81.6% | **64.75 (+10.2%)** | 83.7% |
| 7 | 61.88 (+4.1%) | 75.7% | 60.88 (+3.6%) | 65.0% |

Acceptance moves up to 10.9 points between the two configs; decode moves less than 1% at every setting, and **the optimum is n-max 5 under both**. The worry that a flag tuned at harness sampling would mislead a slot serving different sampling was real in principle and absent in practice.

**Recommendation, unqualified: run the Gemma 4 26B-A4B reasoning fallback at `--spec-draft-n-max 5`.** Worth ~+10%.

The wider lesson is in finding #32: sweep on decode throughput, and read acceptance only to understand the shape — flat acceptance means push to 7, decaying acceptance means stop at 5. Never carry an acceptance number from one bench into another as a model input, which is exactly the error that produced the wrong +14%-at-7 prediction.

## Finding #31 — what the drafter actually costs

empero Qwen3.8-9B-Distill, benched unassisted in the same window to close the gap left by the morning run:

| | decode med | σ | prefill @ 12K | peak VRAM |
|---|---|---|---|---|
| unassisted | 56.69 | **0.05** | **2,366** | 10.91 GiB |
| + Q8_0 MTP head | 73.97 | 10.57 | 2,020 | 14.76 GiB |
| delta | **+30.5%** | — | **−14.6%** | **+3.85 GiB** |

The +30.5% decode is the number that justifies the head on a chat workload. But the head also costs 14.6% of prefill and 3.85 GiB, and the categorise workload is prefill-heavy with short outputs — where that trade can go negative even as the decode figure improves. **Decide per workload shape, not per model.**

This also closes the bimodal-decode question from the morning run. Unassisted σ is **0.05**; with the drafter it is 10.57. The bimodality is entirely an acceptance artifact — not thermal, not scheduler noise. Unassisted decode on this card is about as stable as a measurement gets.

---

## Per-model verdicts

### Nemotron 3.5 Lightning 30B-A3B — **strongest result in this round**

79 tok/s, 99.8% acceptance, 1,772 prefill @ 12K. It beats the current reasoning fallback (Gemma 4 26B-A4B, 62.84 / 97.2% / 1,592) on every axis at a slightly larger parameter count.

**But 21.99 GiB peak leaves ~2 GiB of headroom, so it does not co-reside.** Card 1 currently carries chat + embed + rerank + categorise; Nemotron would need the card to itself. That makes it a natural fit for the **task #144 B580 migration** — once embed, rerank and E2B move off, a freed B60 running Nemotron as the reasoning slot is a better use of the card than a tensor-split 35B-A3B (task #142), given 35B-A3B already failed twice on MTP acceptance under compression (finding #24).

Run it at **`--spec-draft-n-max 7`** — 91.91 tok/s, +16.3% over the default 3 (finding #30).

### Qwen3.8-9B-Distill + MTP — **real challenger to the prod chat slot**

**+13.5% decode over Ornith 1.5-9B** (73.97 vs 65.15 median) at effectively identical VRAM (14.76 vs 14.62) and identical prefill (2,020 vs 1,987 @ 12K). Acceptance is slightly behind (81.4% vs 84.7%).

Arch-identical to Ornith 1.5-9B — same 32 layers, same 24 linear / 8 full pattern, same hidden 4096, same head_dim 256, same 262k context — so the prod launcher works unchanged apart from the model and head paths.

**Do not cut over on speed.** Ornith is an agentic-coding post-train carrying the pi.dev workload; this is a general distillation of Qwen 3.8. A 13.5% decode win means nothing if tool-calling or edit-diff quality regresses. This needs the same treatment Ornith 1.5 got: a `docs/track2`-style qualitative bake-off on the real pi.dev corpus. The speed result earns it that bake-off and nothing more.

Note the mean/median gap (65.56 vs 73.97) — decode is bimodal, clustering at ~74 or ~50. Ornith shows the same shape (56–75). Worth understanding before either is trusted at a single number.

### LFM2.5-8B-A1B + DSpark — **fastest categorise candidate, wrong size**

168.25 tok/s decode and 3,665 prefill @ 12K, against Gemma 4 E2B's 138.8 isolated and 3,681 @ **2K**. It holds full prefill throughput out to 12K where E2B was only ever measured at 2K, and it is an 8B-class model rather than a 2B.

**8.63 GiB against E2B's 3.4 GiB in the co-residence budget** is the problem — +5.2 GiB per card, which the current four-service layout has nowhere to put. Same shape of answer as Nemotron: excellent model, blocked on the card budget rather than on merit. Revisit with task #144.

Also the first *official* llama.cpp-format drafter Liquid has shipped (2026-08-19), and `--spec-type draft-dspark` worked out of the box.

### Ling-3.0-tiny — **weakest; not worth a follow-up**

91.86 tok/s at 5.72 GiB with no drafter available. The disqualifier is prefill: **2,293 @ 2K falling to 1,218 @ 12K.** Throughput that *decreases* with context length is backwards — every other model here rises or holds — and the categorise workload is prefill-heavy (finding on prefill-dominated short-output tasks). Whatever the `bailingmoe3` SYCL path is doing at longer contexts, it is not something to build on.

It cost a build bump to test. That bump was still worth it: b10566 is now on the box with `bailingmoe3`, `nemotron_h_moe` and `qwen3next` support, and the reference re-run says it costs nothing against b10433.

---

## Recommended next actions

1. ~~Re-bench Nemotron at `--spec-draft-n-max 5` and 7.~~ **Done — use 7, worth +16.3%. See finding #30.** Follow-on: confirm the batch-8 fast-path hypothesis in the SYCL kernel source, and re-check the boundary after the next llama.cpp bump.
2. **Qualitative bake-off: Qwen3.8-9B-Distill vs Ornith 1.5-9B** on the pi.dev corpus. Speed says challenger; only quality decides a cutover.
3. **Fold Nemotron and LFM2.5 into the task #144 planning** as the models that justify freeing a card, rather than treating the B580 migration as pure consolidation.
4. ~~Investigate the bimodal decode on both 9B hybrids.~~ **Answered by finding #31** — unassisted σ is 0.05 vs 10.57 with a drafter, so it is acceptance-driven. Open question is narrower now: *which* content classes the head misses on.
5. **Fix DNS on `llm.local` permanently.** `systemd-resolved` was flapping UDP↔TCP against the IPv6 link-local upstream on `enp7s0` (`fe80::f02f:4bff:fe41:cd66`) and timing out every query through 127.0.0.53 — it broke a pip install and one HF download mid-session. Worked around at runtime with `resolvectl dns enp7s0 192.168.1.254`, which **reverts on network restart**. The IPv4 CoreDNS upstream answered correctly throughout.

## Artifacts on `llm.local`

```
/data/llm/benchmarks/bench-candidate.py          harness (prefill/decode/acceptance/VRAM, JSON out)
/data/llm/benchmarks/20260821/*.json|.log        raw results, 5 models
/data/llm/benchmarks/20260821-nmax/*.json|.log   n-max sweep + unassisted 9B baseline
/data/llm/launch/bench-slot.sh                   parametrised bench slot, card 2, :8020
/data/llm/launch/bench-card2-free.sh             open a bench window
/data/llm/launch/bench-card2-restore.sh          close it, with health verification
/data/llm/build/convert-venv/                    convert_hf_to_gguf.py env (torch 2.13.0+cpu)
/data/llm/build/convert-empero-mtp.sh            MTP head conversion recipe
```

Stack verified restored at 15:05 UTC: `:8002 :8010 :8004 :8012 :8006 :8009 :8008 :8013` all OK, both cards ~20 GiB, all three card-2 watchdogs running, LB serving.
