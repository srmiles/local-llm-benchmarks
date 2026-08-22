# Key findings — B60 Pro LLM stack

Numbered findings accumulated over the stack's build-out. Referenced from the main README as `finding #N`.

1. **MoE beats dense on Battlemage.** 26B-4B-active decodes ~2× faster than 12B dense at similar quality.

2. **Post-training K-quant beats QAT Q4_0 at ≥26B — with build-specific caveat that has since reversed.** Reverses at 4B (finding was model-size-dependent). Also briefly reversed at 26B-A4B on b10215+MTP where QAT edged out Q4_K_M by ~10% decode. On b10256 the ordering restored to K-quant winning: Q4_K_M + MTP prefill 1,475 tps vs QAT + MTP 1,335 tps (+10% for K-quant), decode ~parity. QAT still saves 2.3 GiB VRAM at 26B-A4B. **Bottom line by build:** on b10068 K-quant wins (original finding); on b10215+MTP QAT briefly won; on b10256+MTP K-quant wins again but by a smaller margin. **Update 2026-08-21:** on b10433 re-bench, QAT decode recovered to 62.8 tps and now leads Q4_K_M (47.7 tps) by a large margin — the QAT decode "regression" flagged 2026-08-14 (50.5 tps ⚠) is gone. Root cause not fully diagnosed (NEO cache warmth, or upstream patch under same tag). Current recommendation: **QAT at any size on b10433** as of 2026-08-21.

3. **MTP drafters are worth +5–15%** when a purpose-built head exists (Gemma 4 official, Ornith community). **Amended 2026-08-21:** MTP acceptance depends on target/drafter distribution alignment — see finding #24 for the compression corollary that hurts 35B-A3B on our stack.

4. **`-ub 2048` is the SYCL sweet spot.** 4096 regresses on this card; monotonic climb from 16 → 2048 then plateau.

5. **`-fa on` is mandatory** — turns 36s "warm" re-prefills into 0.55s cache hits.

6. **TEI XPU-IPEX crushes llama.cpp for rerank** — 7–9× on 25-pair batches. Requires periodic restart (weekly) to reclaim VRAM growth. See #22 for the architecture-specific nuance.

7. **`--jinja` is mandatory for tool-calling reliability** — the built-in template handler doesn't emit Gemma 4's tool delimiters.

8. **XMX+oneDNN FA (llama.cpp b10068) is a modest win on dense-GQA models.** Initial cold-vs-warm comparison overstated it as "+42% throughput / -47% wall time" — methodology-matched isolated probe on Ornith 9B shows the real uplift is **~3% at 8K/12K, flat at short context**. Effect on Gemma 4 MoE is similar. **Lesson: never trust a first-look uplift claim that pairs a cold-start baseline against a warmed candidate.** Always re-probe under identical conditions.

9. **b10068 also carries a silent Q4_K get_rows correctness fix** — the older build had a subtle bug in Q4_K row gather that affected Ornith and MiniCPM5 decodes. No perceptible quality change post-swap, but it's closed regardless.

10. **MTP variants matter more than base model choice for A3B MoEs.** Qwen 3.6-35B-A3B base was 31 tok/s decode; same architecture with MTP head enabled (via the `-MTP-GGUF` sibling repo) jumps to 49 tok/s — **+58% purely from picking the right repo.** Always check for `-MTP-GGUF` variants of MoE candidates.

11. **b10068's XMX FA win doesn't scale to dense-27B.** Ornith dense-9B GQA got +42% cold prefill from b10068. Qwen 3.6-27B dense-27B got flat-to-slightly-negative. b10068's XMX FA optimises FA vec kernels — small models can afford the launch overhead, larger dense models are still bandwidth-bound.

12. **Co-residence budget dominates viability.** A 24 GiB isolated bench that beats Ornith is meaningless if it leaves 0 GiB for embed + rerank + TEI. Always subtract ~2.4 GiB of non-chat services before deciding if a candidate can actually ship.

13. **MTP acceptance is quant-sensitive** — same architecture, same base weights, same MTP head, but changing from Q4_K_XL → Q4_K_S drops MTP acceptance from 77.8% → 71%, and IQ4_XS drops it further to 60.8%. The drafter head's calibration against the target degrades faster than raw quant math would suggest. Meaningful lesson for anyone hoping "just quantise smaller" is a free move on MTP models.

14. **Smaller K-quant can be FASTER on prefill.** Counter-intuitive but measured: Q4_K_S beats Q4_K_XL on cold 12K prefill (820 vs 798 tok/s) and 5K prefill (985 vs 974). The smaller weights let more of the model stay in cache during prefill's memory-bound phase. Decode reverses this — larger K-quant wins because MTP acceptance recovers.

15. **Two SYCL processes on a single B60 contend severely.** Split-slot experiment 2026-07-22: running `llamacpp-sycl` (Ornith) and `llamacpp-categorise` (Gemma 4 E2B) concurrently on the same B60 drops both from isolated speed to ~30% (Ornith 52 → 16 tps, Gemma 88 → 12 tps). Root cause: Level Zero context-switch overhead + shared kernel dispatch queue. **Split-slot architecture on B60 requires a dedicated GPU per SYCL process.** Resolved 2026-08-15 with 2nd B60 (E2B on card 2, Ornith on card 1). Fully resolved 2026-08-22 with 64 GB RAM upgrade enabling dual-load mirror pattern.

16. **Google QAT Q4_0 beats post-hoc K-quants at 2-4B on B60.** Gemma 4 E2B QAT hit 88 tps decode vs Qwen3-4B and Agents-A1-4B post-training K-quants at 79 tps. QAT preserves output distribution better than post-training quants at small sizes, AND Q4_0 layout dispatches faster than Q4_K_M on Battlemage. Reverses at ≥26B where K-quants pull ahead again (though see finding #2 amendment for b10433).

17. **Reasoning-tuned models (Gemma 4, Agents-A1) route output to `reasoning_content` by default** — `content` is empty, breaking any structured-JSON workflow. Fix: pass `--reasoning off` to `llama-server`. Without it, 0/10 JSON parseability. With it, 10/10. Small (~5%) decode-speed cost.

18. **Prefill/decode asymmetry matters more than raw decode tps for model choice.** Same B60 (456 GB/s bandwidth), same workload shape (5K prompt + 200 gen), measured under real skill_server load:

    | Model | Prefill tps | Decode tps | Decode % of BW ceiling |
    |---|---|---|---|
    | Gemma 4 E2B QAT Q4_0 (3.35 GB) | **1,600** | 30 | 22% |
    | Ornith 9B + MTP Q4_K_M (5.0 GB) | 1,000 | **50** | 55% |

    Q4_0 wins prefill (large batched matmul), Q4_K_M wins decode. Total-time math flips depending on output length:

    | Workload | Prompt | Gen | Gemma total | Ornith total | Winner |
    |---|---|---|---|---|---|
    | Categorise (short JSON) | 5K | 200 | 3.1 + 6.7 = 9.8s | 5.0 + 4.0 = 9.0s | ~tie |
    | Summarise | 5K | 500 | 3.1 + 16.7 = 19.8s | 5.0 + 10 = 15.0s | Ornith |
    | Chat/agent | 3K | 800 | 1.9 + 26.7 = 28.6s | 3.0 + 16 = 19.0s | Ornith |

    **Lesson: never pick a small model purely on prefill/isolated-decode headline numbers.** Compute total-time for the actual workload shape. Track prefill/decode ratio in future benches, not just headline decode tps.

19. **b10215 delivers ~2× prefill via SYCL oneMKL GEMM XMX FA (#25025), invisible to short-prompt/decode-only benches.** Discovered 2026-08-01 during brain-eval Track 2 ingest arm — both Ornith 9B and Gemma 4 E2B independently hit ~3,000 tps prefill on real 2-5K token brain-ingest prompts vs historical ~1,600 tps on b10068. **Meta-lesson: post-upgrade validation needs to include production-shape prompts, not just decode microbenches.**

20. **Isolated `/completion` probe vs real-workload prefix-cached wall-clock measures different things. Never mix them in a delta.** 2026-08-04 lesson: comparing Ornith 9B on b10215 vs b10256 initially showed a -40% "regression" on b10256 that was pure methodology artifact. When re-run with matched methodology (both isolated `/completion` cold probes on the same 5K prompt), b10215 delivered 1,442 tps and b10256 delivered 1,789 tps — a **+24% real improvement**, not a regression. **Rule:** pick one methodology and hold it constant across every arm; document which one you used inline with the number.

21. *(reserved — no finding published)*

22. **TEI's advantage over llama.cpp SYCL is architecture-specific, not universal for encoder-only models.** BERT-family cross-encoders (bge-reranker-v2-m3, older embed models): TEI wins big (7-9× rerank, 19× on Qwen3-Embedding-0.6B). Gemma3-encoder (EmbeddingGemma-300M): llama.cpp SYCL wins single-embed by 1.85× (23,208 vs 12,500 tps), TEI wins batch throughput by 24%. **Rule:** always A/B the same model on both runtimes before assuming an "encoder = TEI" advantage generalizes. TEI's win regime is batch throughput; llama.cpp SYCL's win regime is single-request latency.

23. **B60 draws ~100W actual vs 220W TDP under sustained LLM load — compute engines pinned but half the die is idle.** Measured 2026-08-01 via `xpu-smi stats`:

    | Metric | Value | Interpretation |
    |---|---|---|
    | Power | **97-98W** of 220W cap | 44% of design TDP |
    | GPU frequency | **2,400 MHz** (RP0 boost, max) | Card at max clock, not throttled |
    | Core temp / Mem temp | 53°C / 54°C | Cold |
    | Compute engine util | **99.99%** | Pinned |
    | Copy engine util | 90% | Active |
    | Media / 3D / Display util | 0% each | Idle |
    | Memory BW util | 20% | 91 of 456 GB/s |

    The 55% TDP headroom is unused because media encoders, 3D pipeline, ray tracing, display controllers sit idle under LLM inference. The interesting number is 20% memory BW + 100% compute — kernels are compute-limited, not memory-limited. Future kernel improvements can still lift decode. **Practical:** don't undersize PSU based on TDP × N; size on measured LLM load × N + spike margin. Don't confuse "GPU util 21%" (xpu-smi averages all engines) with under-utilization — look at `ENGINE_GROUP_COMPUTE_ALL_UTILIZATION` (99.99%) instead.

24. **MoE MTP under compression fails when target-model distribution drifts from what the drafter was trained against.** Discovered 2026-08-21 while trying to bench Ornith 1.5-35B-A3B on a single B60. **Bartowski IQ4_XS** (imatrix, MTP head at Q4_0): 32.5% MTP acceptance, 32.6 tok/s decode. **Mudler APEX-MTP-Compact** (aggressive routed-expert compression, MTP head **pinned at Q8_0**): acceptance actually **dropped further to 26.2%**, decode 25.6 tok/s. Higher-quality MTP head made things worse because APEX's aggressive expert quantization shifted the target's output distribution far enough that even a perfect MTP drafter can't predict it — the MTP was trained against full-precision Ornith. Bartowski's imatrix calibration preserved the target distribution more faithfully, so its Q4_0 MTP (which SHOULD be worse) actually predicts more accurately. **Practical:** for MoE + MTP, target quantization method matters more than MTP head precision. The reference config is what Google does with Gemma 4 26B-A4B: publish a co-designed MTP drafter matched to the exact quantized base (their `gemma-4-26B-A4B-it-qat-assistant-MTP-Q8_0.gguf` gets us 97.2% acceptance because the drafter was trained against the QAT model, not the BF16 one). For Ornith 35B-A3B, no such matched-pair drafter exists today from ornith-ai; the community options all embed MTP layers alongside compression that breaks target/drafter alignment. Real path forward: tensor-split across 2× B60 (allows bigger quant that preserves distribution better) or wait for a properly-distilled 35B drafter to be published. See [`models/tested/ornith-1.5-35b-a3b-single-card.md`](../models/tested/ornith-1.5-35b-a3b-single-card.md) and [`models/tested/ornith-1.5-35b-a3b-apex-mtp.md`](../models/tested/ornith-1.5-35b-a3b-apex-mtp.md).

25. **The "SYCL SSM penalty" does not exist — it was dense-bandwidth cost wearing a disguise.** Qwen 3.8-27B was parked 2026-08-15 at 23.0 tok/s with the note *"revisit when SYCL SSM gets XMX GEMM"*. That diagnosis was wrong, and it blocked a whole architecture family for six days. Benched 2026-08-21: **Nemotron 3.5 Lightning 30B-A3B** — a `nemotron_h` Mamba2 hybrid, 52 layers, 128 experts / 6 active — decodes at **78.95 tok/s** and prefills at **1,772 @ 12K**, against Qwen 3.8-27B's 23.0 and 333.5. Same class of architecture, 3.4× the decode. The variable was never the SSM layers: Qwen 3.8-27B activates all 27B parameters per token, Nemotron activates ~3B. The 23 tps is the dense bandwidth wall already documented for Muse Glimmer-30B dense (25.3) and Laguna XS-2.1 (29.5) — SSM cost and dense-bandwidth cost were confounded in a single data point, and nobody had a sparse hybrid to separate them. **Practical:** sparse hybrid-linear-attention models are not blocked on upstream SYCL work and should be benched on arrival. Only *dense* hybrids above ~12B are penalised, and that is a property they share with every dense model on this card — no upstream kernel work will fix it. **Meta-lesson:** when a single model is slow and it differs from the baseline on two axes at once, do not name one of them as the cause. Park it as unexplained until a model that varies only one axis comes along.

26. **`nemotron_h` barely quantises, and that decides the quant for you.** The full bartowski ladder for Nemotron 3.5 Lightning 30B-A3B, in GiB: IQ2_XXS 17.54 · Q2_K 17.61 · IQ4_XS 17.62 · Q3_K_S 17.64 · **Q4_0 17.75** · Q2_K_L 17.78 · Q3_K_M 18.46 · Q3_K_XL 19.02 · Q4_1 19.44 · Q4_K_S 21.61 · **Q4_K_M 23.73** · Q5_K_M 25.11 · Q6_K 31.95 · Q8_0 32.60. From IQ2_XXS to Q4_K_M the entire range is 17.5 → 23.7 GiB, because only the 128 routed experts compress — the dense trunk and the Mamba2 SSM state tensors do not. Two consequences: **Q4_K_M cannot be served on a 24 GiB B60 at all** (23.73 GiB of weights, before a 2.03 GiB MTP head and any KV — arithmetic, not tuning), and **dropping below Q4 buys nothing** (Q2_K saves 0.14 GiB over Q4_0 for real quality loss). The reflex move of "step down one quant to make it fit" is dead for this architecture family. **Q4_0 at 17.75 GiB is the correct choice** — largest non-IQ quant that fits, IQ excluded by finding #13, and SYCL's reordered Q4_0 path is already well-trodden via the Gemma 4 QAT models. **Practical:** pull the whole size ladder from the HF API before downloading anything for a new architecture; the shape of the ladder is itself a finding.

27. **99.8% MTP acceptance is achievable, and at that point `--spec-draft-n-max 3` becomes the limiter.** Nemotron 3.5 Lightning's MTP head accepted **4,478 of 4,486 draft tokens across 20 sampled runs — 99.8%, at 2.98 accepted per draft against a hard ceiling of 3.00**. Previous best on this stack was Gemma 4 26B-A4B at 97.2% / 2.90. At 2.98/3.00 drafts are being truncated by the flag, not rejected by the target model, so throughput is being left on the table. **Resolved 2026-08-21 by the n-max sweep — see finding #30. `--spec-draft-n-max 7` is the setting: 91.91 tok/s, +16.3% over n-max 3, for free.** Corroborating signal: decode σ = 0.33 tok/s over 20 runs, against 16.41 for Ornith 1.5-9B and 35.26 for LFM2.5-8B-A1B. Near-perfect acceptance removes the accept/reject variance that makes every other MTP row on this box bimodal — **a tight decode σ is itself evidence of high acceptance**, and a bimodal decode distribution is evidence the drafter is missing on some content classes. Confirmed directly: empero Qwen3.8-9B-Distill run *without* a drafter has σ = 0.05, against σ = 10.57 with one. Bimodality is an acceptance artifact, not thermal or scheduler noise.

28. **MTP heads can be self-converted; the third-party drafter dependency is over.** Until 2026-08-21 every drafter on this stack came from someone else's upload — protoLabsAI for Ornith, Google's official safetensors for the Gemma 4 assistants. empero's Qwen3.8-9B-Distill GGUF ships main-model quants only, so the head was built locally: `convert_hf_to_gguf.py <hf-dir> --mtp --outtype q8_0 --outfile mtp-<name>-head-Q8_0.gguf` → 18 tensors, 2.43 GB, loaded first try, 81.4% acceptance (Ornith's third-party head reaches 84.7% on the same build). Venv preserved at `/data/llm/build/convert-venv` (torch 2.13.0+cpu, transformers 5.15.1). **Practical:** any model shipping `mtp_num_hidden_layers` in its config can now be given a head locally, at a quant we choose — which matters directly for finding #24, where head precision and target quantisation have to be controlled together. The converter also exposes `--dspark` (build DSpark heads) and `--no-mtp` (publish target and draft as two files). **A missing drafter in a GGUF repo is no longer a reason to skip or discount a candidate** — but note that benching such a model *without* first converting the head produces a number ~20-25% low for reasons that have nothing to do with the model.

29. **DSpark drafts wide where MTP drafts deep — compare drafters on tokens-per-draft, not acceptance rate.** LFM2.5-8B-A1B's official DSpark head (`--spec-type draft-dspark --spec-draft-n-max 7`, block size 9) accepted **5.51 tokens per draft** at a 79.8% acceptance *rate*. Ornith 1.5-9B's MTP head, on the same build in the same window, scored a *higher* rate — 84.7% — but only 2.53 tokens per draft. The DSpark head wins decisively on throughput (168.25 vs 65.15 tok/s) despite the worse-looking headline percentage, because it is a block-diffusion drafter proposing a whole block rather than a short chain. **Practical:** acceptance % is not comparable across drafter families and ranking candidates by it will pick the wrong model. The throughput-relevant quantity is accepted-tokens-per-draft, which is bounded by `--spec-draft-n-max` and the drafter's trained block size (llama.cpp clamps n-max to the block size, so raising the flag past it does nothing). Record both numbers in every bench.


30. **Speculative decode throughput on B60 is capped by verification batch size, not drafter quality — and it falls off a cliff above batch 8.** Swept `--spec-draft-n-max` on Nemotron 3.5 Lightning 30B-A3B (Q4_0 + MTP Q8_0), 20 sampled runs each, isolated card, b10566:

    | n-max | decode med | σ | acceptance | accepted/draft | VRAM | verify batch (n+1) |
    |---|---|---|---|---|---|---|
    | 3 | 79.05 | 1.06 | 99.9% | 2.98 / 3.00 | 21.99 | 4 |
    | 5 | 86.94 | 0.62 | 99.9% | 4.97 / 5.00 | 22.08 | 6 |
    | 6 | 90.10 | 4.41 | 99.8% | 5.94 / 6.00 | 22.13 | 7 |
    | **7** | **91.91** | 1.05 | 99.5% | 6.85 / 7.00 | 22.18 | **8** |
    | 8 | 60.31 | 0.42 | 99.8% | 7.79 / 8.00 | 22.67 | 9 |
    | 9 | 63.93 | 0.07 | **100.0%** | 8.97 / 9.00 | 22.71 | 10 |
    | 10 | 67.24 | 0.19 | 99.9% | 9.68 / 10.00 | 22.76 | 11 |

    **Acceptance never degrades.** It is 99.5–100% at every setting, and at n-max 9 the drafter returned 8.97 of a possible 9.00 tokens per draft — essentially flawless — while decode sat at 63.93. The drafter is not the constraint anywhere on this curve. What changes is the cost of the target verifying the block.

    And that cost is not a smooth curve. Decode climbs monotonically to **n-max 7 (91.91)**, collapses **34% at n-max 8 (60.31)**, then *recovers* through 9 and 10. Ranked by verification batch size (`n_max + 1`), every batch of 4–8 performs well and every batch of 9+ is penalised. **Hypothesis (not confirmed in kernel source): the SYCL backend has a fast path for verification batches up to 8 and falls back above it** — consistent with the power-of-two geometry sensitivity already documented for this backend (`parallel=4/c=8192`, `parallel=8/c=16384`, `parallel=16/c=32768`). Prefill is flat at ~1,760 @ 12K across the whole sweep, so this is specific to the decode/verify path. VRAM steps ~0.5 GiB at the same 7→8 boundary, which is a second hint that a different code path is taken.

    **Practical:**
    - **Use `--spec-draft-n-max 7` on this card** for any MTP drafter with high acceptance. Free +16.3% over the reflexive default of 3.
    - **Do not tune by stepping upward until throughput stops improving.** That procedure walks off the cliff at 8, loses 34%, and concludes the peak at 7 was noise. Sweep the whole range or go straight to 7.
    - **Do not read a decode drop as a drafter problem** without checking `llamacpp:spec_decode_num_accepted_tokens_total` / `..._draft_tokens_total`. Here acceptance was perfect at every point where throughput cratered. The two failure modes look identical from the throughput number alone and have opposite fixes.
    - Worth re-checking after any llama.cpp bump — this is a kernel-path property, so an upstream change could move the boundary.

    **Replicated 2026-08-21 on a second architecture.** Gemma 4 26B-A4B (QAT Q4_0 + Google's official MTP Q8_0) — a different model family, different drafter provenance, different quant — hits the same wall in the same place:

    | n-max | Nemotron step cost | Gemma step cost |
    |---|---|---|
    | 7 → 8 | **+70.6%** | **+71.3%** |
    | VRAM step | 22.18 → 22.67 GiB | 17.85 → 18.95 GiB |

    Two architectures, agreeing on the size of the discontinuity to within 0.7 points, with the same VRAM signature at the same boundary. **Treat `n_max ≤ 7` as a hard rule for this card**, independent of model.

    **But the optimum inside that range is model-specific, and it is set by how acceptance decays along the chain.** Nemotron's acceptance is flat (99.9% → 99.9% → 99.5% at n = 3 → 5 → 7), so every extra drafted token pays and the optimum is the last legal one, 7. Gemma's decays (88.7% → 81.6% → 75.7%), so marginal tokens stop covering their cost and it peaks at **n-max 5** (65.34 tok/s, +9.9%), falling back to +4.1% at 7. Step-cost growth per drafted token is nearly identical for the two (+0.161 vs +0.174 of a baseline step), so acceptance decay is the whole difference.

    **Recipe for any drafted model on this card:** bench n-max 3 and 7. If accepted-per-draft scales ~linearly (acceptance holds), take 7. If acceptance sags, add a point at 5 and take the winner. Never test 8+.

31. **A drafter costs prefill to buy decode.** empero Qwen3.8-9B-Distill benched with and without its Q8_0 MTP head, same build, same window, `Q4_K_M` target:

    | | decode med | σ | prefill @ 12K | peak VRAM |
    |---|---|---|---|---|
    | unassisted | 56.69 | **0.05** | **2,366** | 10.91 GiB |
    | + Q8_0 MTP head | 73.97 | 10.57 | 2,020 | 14.76 GiB |
    | delta | **+30.5%** | — | **−14.6%** | +3.85 GiB |

    The +30.5% decode is the headline, but the drafter also costs **14.6% of prefill** and **3.85 GiB** of VRAM. On a decode-bound chat workload that trade is obviously right. On the categorise workload it is not obvious at all — that workload is prefill-heavy with short outputs (see the workload-shape finding), so a drafter can be a net loss there even while the decode number improves. **Decide per workload shape, not per model.** And note the σ collapse from 10.57 to 0.05 without the drafter: unassisted decode on this hardware is extremely stable, so any bimodality in a drafted row is the drafter, not the card.


32. **Acceptance is sampling-dependent; the n-max optimum turned out not to be. Tune on throughput, not on acceptance.** Gemma 4 26B-A4B (QAT Q4_0 + Google MTP Q8_0) swept twice on the same build, same card, same prompt corpus — once at the cross-model harness sampling (`temp 0.6 / top-p 0.95 / top-k 20`) and once at its own production sampling (`temp 1.0 / top-k 64`):

    | n-max | harness decode | harness acc | prod decode | prod acc |
    |---|---|---|---|---|
    | 3 | 59.46 | 88.7% | 58.77 | 77.8% |
    | **5** | **65.34 (+9.9%)** | 81.6% | **64.75 (+10.2%)** | 83.7% |
    | 7 | 61.88 (+4.1%) | 75.7% | 60.88 (+3.6%) | 65.0% |

    **Acceptance moves a lot with sampling — up to 10.9 points at the same n-max — while decode barely moves at all** (within ~1% at every setting). And the optimum is n-max 5 under both, at essentially the same gain. A third figure exists for the same pairing: the repo's recorded **97.2%**, which reproduces under neither sampling here, so corpus matters too.

    Two consequences, and the second one corrected an earlier prediction:

    - **Acceptance percentages are only comparable within one harness, one corpus and one sampling config.** Quoting them across benches — as was done to predict this model's optimum from its recorded 97.2% — produces a wrong answer. That prediction said n-max 7 at +14%; the truth is 5 at +10%.
    - **But the tuning decision was robust to all of it.** The concern that a flag tuned at harness sampling would be wrong for a slot serving different sampling was tested directly and did not materialise. Decode throughput is the stable signal; acceptance is a diagnostic for *why* a setting wins, not the thing to optimise.

    **Practical:** sweep n-max on decode tok/s. Read acceptance only to understand the shape — flat acceptance means push to 7, decaying acceptance means stop at 5. Do not carry an acceptance number from one bench into another as an input.

33. **A/B benching hardware behind a live Traefik LB pool measures LB load-share, not silicon.** Discovered 2026-08-22 during the B580-vs-B60 head-to-head. First run — both backends live in the `llm-categorise` pool, Gemma 4 E2B + Google MTP: B580 128.19 tps decode / B60 92.54 tps → apparent **+40% B580 advantage**. Pulled both `:8009` backends from the Traefik LB and re-ran identical script identical corpus identical build: B580 128.19 tps / **B60 159.55 tps** → real result is **B580 20% *slower*** on decode. The +40% "advantage" was pure LB contention on B60 (other traffic hitting `https://llm.levirge.com/v1/categorise` was queuing behind the bench request). Prefill was unaffected — it saturates faster than the LB queue depth changes. **Rule:** for any hardware A/B, both ends must be out of the LB pool for the duration. Same class of error as finding #20 (mixing methodologies produces artifacts) — here the mixing is "bench request + LB traffic" instead of "isolated probe + prefix-cached wall-clock." See [`models/tested/2026-08-22-b580-vs-b60-e2b.md`](../models/tested/2026-08-22-b580-vs-b60-e2b.md).
