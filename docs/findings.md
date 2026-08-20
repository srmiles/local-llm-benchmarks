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
