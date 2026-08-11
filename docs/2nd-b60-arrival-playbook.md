# 2nd B60 Arrival Playbook

Second Intel Arc Pro B60 arrives week of 2026-08-11. This doc consolidates everything that changes when it lands: what to try day-1, the strategic card 1 / card 2 split, and the deferred candidates that unblock when the second card is live. Also captures the most important upstream finding of the week — Sergio Barrientos's vLLM XPU MTP campaign on B70 — with reproduction steps.

## The big upstream finding: vLLM XPU + MTP crushes llama.cpp SYCL on MoE

Sergio Barrientos ([sergiiob.dev](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/) + [GitHub cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)) ran a 19-run campaign on Intel Arc Pro B70 comparing vLLM XPU vs llama.cpp SYCL on Qwen 3.6-35B-A3B — **exactly the MoE model we've been benching**. Headline: **vLLM XPU + MTP hits 133 tps decode / 8,718 tps prefill single-stream — 1.8× decode / 5.2× prefill over llama.cpp SYCL on the same card**.

### Same-model comparison (his B70 numbers vs our B60)

| Runtime | Prefill @ 5K | Decode | VRAM | Notes |
|---|---|---|---|---|
| **Our llama.cpp SYCL b10256 on B60** | 766 tps | 43 tps | 23.1 GiB | Q4_K_XL GGUF, MTP fused, `-fa on`, `-ctk q8_0` |
| His llama.cpp SYCL b10255+ on B70 | 1,498-1,662 tps | 58-74 tps | ~23 GiB | Same GGUF, larger die (~2× XMX vs B60) |
| **His vLLM XPU + MTP on B70** ⭐ | **8,718 tps** | **133 tps** | ~22 GiB | GPTQ-Int4 + MTP-preserved, patched |

### B60 scaling projection

B60 has ~62% of B70's XMX engines (160 vs 256), same memory bandwidth class (456 vs 608 GB/s), 24 GB vs 32 GB VRAM. Compute-bound workloads (prefill) scale roughly with XMX count; memory-bound workloads (decode) scale with bandwidth.

| Metric | B70 (his) | B60 projected (~62% XMX) | vs current B60 llama.cpp | 
|---|---|---|---|
| Prefill (compute-bound) | 8,718 tps | **~5,400 tps** | **+605% (7×)** |
| Decode (bandwidth-bound + MTP) | 133 tps | **~82 tps** | **+91% (2×)** |

Even the conservative projection puts vLLM XPU + MTP on B60 at ~5.4K prefill / 82 decode — a massive jump over the 766/43 we measured on llama.cpp.

### The four unlock patches (Sergio's cookbook)

All applied in-container at boot against `intel/vllm:0.21.0-xpu-int4moe`; no rebuild needed. Published MIT at [`SergiioB/intel-arc-pro-b70-inference-cookbook/patches/`](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/tree/master/patches).

| # | Patch | Fixes | Root cause |
|---|---|---|---|
| 1 | `patch_xpu_int4_moe_v4.py` | Native int4 MoE crash | C++ `is_B_int4 = (B_dtype == at::kChar)` requires int8, but GPTQ packs uint8; kernel treated weights as BF16 → shape check crash. Fix: store int8 via `implement_zp`. |
| 2 | `patch_mtp_bf16_draft.py` (BF16 draft) | `KeyError: w2_weight` on MTP load | Draft inherits target's GPTQ quant_config; checkpoint's MTP experts are BF16 fused tensors. Fix: strip quant_config for any prefix containing `mtp`. |
| 3 | `patch_mtp_bf16_draft.py` (kwarg strip) | `XpuFusedMoe __init__ unexpected kwarg is_fp8` | vLLM's xpu_moe.py passes `is_fp8`/`is_mxfp4` but XPU kernels auto-detect dtype. Fix: drop those kwargs at call site. |
| 4 | `patch_mtp_bf16_draft.py` (GDN assert) | `AssertionError: spec_sequence_masks is None` | XPU GDN SYCL kernel already takes explicit spec tensors; the boolean `spec_sequence_masks` is metadata-only and never reaches the kernel. **The assert was a guardrail, not a real limit** — this refutes the community's prior "XPU GDN incompatible with speculative decoding" verdict. |

### Critical caveats

1. **Dense on vLLM is BLOCKED on XPU.** `KeyError: PlatformEnum.XPU` in `choose_scaled_mm_linear_kernel` — no FP8 linear kernel registered for XPU. **Ornith 9B is dense** — vLLM won't help there. llama.cpp SYCL remains the only dense path until an XPU FP8 kernel lands upstream. Watch: [SergiioB DENSE-FP8-GAP doc](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/DENSE-FP8-GAP.md).
2. **MTP + concurrency incompatible.** The XPU GDN `causal_conv1d` state machine can't mix speculative and non-speculative tokens in one batch. **Choose: MTP (single-user, ~82 tps on B60) OR concurrency (no MTP, ~C16 aggregate).** Cannot have both until XPU GDN supports mixed batches.
3. **Model dependency.** Sergio uses [`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4`](https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4) (22.4 GB). Generic GPTQ checkpoints have MTP layers stripped during quantization; this "MTP-preserved" variant explicitly keeps them. For Laguna XS.2 or other MoE candidates, need equivalent preservation.
4. **Power finding.** MoE self-limits to ~140W draw; raising cap to 230W actually made B70 -8% slower. Sweet spot: **MoE @150W, Dense @180W**. B60's default cap already sits in the MoE-friendly range.
5. **Multi-turn 128K works with `--enable-prefix-caching`.** Cold load ~40s at 122K, then follow-up TTFT 1.4s — genuinely usable long-context sessions. Requires the flag or every turn re-prefills.

## Current stack recap (single B60, pre-arrival)

All llama.cpp services on `llama.cpp:sycl-f16` → b10256 (commit `6c8dcaa7a`) since cutover 2026-08-04. Rollback tag `llama.cpp:sycl-f16-b10215-safe` retained.

| Port | Container | Model | Purpose | VRAM |
|---|---|---|---|---|
| 8002 | `llamacpp-sycl` | Ornith 1.0 9B + MTP (dense) | chat / categorise / pi.dev agent | 10.9 GiB |
| 8004 | `llamacpp-embed` | EmbeddingGemma-300M QAT Q8_0 | brain embeddings | 2.65 GiB |
| 8008 | `tei-rerank` | bge-reranker-v2-m3 fp16 | rerank (TEI XPU-IPEX) | 0.9 GiB |
| 8009 | `bench-eval` | Gemma 4 E2B + Google MTP | warm-standby (Mode B co-load) | 5.8 GiB |

Total steady VRAM: ~20 GiB of 24. Compute contention (finding #15) means only one chat process actively dispatches at a time.

## What changes with 2nd B60

1. **VRAM budget doubles: 24 → 48 GiB effective** (24 per card).
2. **Cross-process SYCL contention (finding #15) is eliminated** — each SYCL context can pin to its own device via `ONEAPI_DEVICE_SELECTOR=level_zero:0` vs `:1`. Two chat models can dispatch simultaneously without the 3× throughput loss.
3. **Deferred candidates become viable:**
   - Qwen 3.6-35B-A3B-MTP (24.4 GiB @ 32K) — was co-res-hostile, now fits dedicated card
   - Laguna XS.2 (22.1 GiB @ 32K) — same story
   - Gemma 4 26B-A4B reasoning-fallback co-exists with Ornith prod
4. **New serving path unlocked: vLLM XPU + MTP on card 2 for MoE workloads.** llama.cpp on card 1 for dense (Ornith) prod.
5. **Multi-card llama.cpp SYCL** (via [PR #26234](https://github.com/ggml-org/llama.cpp/pull/26234) `dev2dev_memcpy_forward` merged in b10215+) becomes usable if we ever want a single logical serving path across both cards — but per-card pinning is simpler and matches the vLLM-on-card-2 strategy better.

## Recommended card split

### Card 1 (`level_zero:0`) — **dense + embed prod, llama.cpp SYCL**

Unchanged from today. Stays on the proven llama.cpp SYCL b10256 stack.

| Service | Model | Notes |
|---|---|---|
| llamacpp-sycl (`:8002`) | Ornith 1.0 9B + MTP | Dense — no vLLM path available for this arch |
| llamacpp-embed (`:8004`) | EmbeddingGemma-300M | Wins single-embed by 1.85× vs TEI (finding #22) |
| tei-rerank (`:8008`) | bge-reranker-v2-m3 | Encoder-only, TEI's home turf (7-9× win) |

**Card 1 VRAM budget: 24 GiB.** Ornith + embed + TEI = ~14.5 GiB. Plenty of headroom for a peak Ornith session or bench-eval swap-in.

### Card 2 (`level_zero:1`) — **MoE experimentation + reasoning fallback**

New. Runs vLLM XPU with Sergio's patches for MoE, or falls back to llama.cpp SYCL for anything vLLM can't serve.

**Primary candidate:** Qwen 3.6-35B-A3B-MTP (GPTQ-Int4 MTP-preserved) via vLLM XPU. Expected ~5.4K tps prefill / ~82 tps decode based on B70→B60 scaling. Replaces Gemma 4 26B-A4B as reasoning fallback if quality holds.

**Secondary experiments (in order):**
1. **Laguna XS.2 Q4_K_M on llama.cpp SYCL** (no DFlash available upstream — see [candidate sweep](../models/tested/2026-08-06-new-candidates-sweep.md#laguna-xs2-poolside-33b-a3b-moe)). Coding-focused, worth pi.dev agent quality bake-off vs Ornith.
2. **gpt-oss-20b on llama.cpp SYCL** — agentic-quality candidate. 13.4 GiB fits with room to spare.
3. **Gemma 4 26B-A4B QAT + MTP** — dedicated reasoning fallback co-resident with card 1's chat. 17.5 GiB, headroom for concurrent dispatch.
4. **Qwen3.8-27B** when it lands ([scheduled watcher](qwen38-hf-watch) firing daily) — likely MoE per pre-release rumors.

## First-week test sequence

### Day 1 (arrival + basic install)

1. Physical install, driver check: `sudo xpu-smi discovery` must show 2 devices. If BIOS PCIe lane split needs changes, do that now.
2. Verify both cards visible in Docker: `docker run --rm --device /dev/dri llama.cpp:sycl-f16 sycl-ls` shows both level_zero devices.
3. Baseline no-change: existing Ornith prod pinned to `level_zero:0` (already default). Confirm brain workload unchanged.
4. Smoke test card 2 in isolation: relaunch bench-eval on `:8009` with `-e ONEAPI_DEVICE_SELECTOR=level_zero:1`. Send a few probe requests. Confirm no contention with Ornith on card 1 (measure Ornith throughput while bench-eval is under load — should be flat vs solo).

### Day 2-3 (vLLM XPU MTP validation)

Follow Sergio's cookbook quickstart on card 2 (`level_zero:1`), Qwen 3.6-35B-A3B-MTP-Preserved-GPTQ-Int4:

```bash
# 1. Pull the MTP-preserved GPTQ checkpoint (~22 GB)
hf download llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4 \
  --local-dir /data/llm/qwen3.6-35b-a3b-mtp-gptq

# 2. Grab Sergio's patches
git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook \
  /data/llm/build/sergio-b70-cookbook

# 3. Speculative config
echo '{"method":"mtp","num_speculative_tokens":1}' | sudo tee /data/llm/vllm-config/spec.json

# 4. Launch on card 2 (level_zero:1) — bench port :8011 to avoid conflict
sudo docker run -d --name vllm-mtp-bench --restart no \
  -p 8011:8000 \
  --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/render* | head -n1)" \
  -v /data/llm/qwen3.6-35b-a3b-mtp-gptq:/model:ro \
  -v /data/llm/build/sergio-b70-cookbook/patches/patch_xpu_int4_moe_v4.py:/patch_v4.py:ro \
  -v /data/llm/build/sergio-b70-cookbook/patches/patch_mtp_bf16_draft.py:/patch_mtp.py:ro \
  -v /data/llm/vllm-config/spec.json:/spec.json:ro \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=1 \
  --entrypoint bash intel/vllm:0.21.0-xpu-int4moe \
  -lc 'python /patch_v4.py && python /patch_mtp.py && SPEC=$(cat /spec.json) && \
       exec vllm serve /model --quantization gptq --dtype float16 \
         --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 1 \
         --language-model-only --speculative-config "$SPEC" \
         --cudagraph-capture-sizes 1 2 4 8 16 32'
```

**Look for `[B70] GDN XPU: spec decode active`** in the logs — confirms MTP is running through the patched path.

**Bench methodology:** same 3-prompt suite as prior benches (52 tok / 1.3K / 5K real workload). Compare against our llama.cpp SYCL Qwen 3.6-35B-A3B numbers (766 / 43). Expected result on B60 (scaled from Sergio's B70): 4-6K prefill, 70-90 decode. Confirm within that range; investigate if far off.

**Correctness check:** greedy `temp=0` replay + factual probes (17×23=391, capital of Australia=Canberra) — patched vLLM should produce byte-identical output to the unpatched eager path.

**KL/acceptance audit** vs eager path is the remaining correctness gate before promoting to prod. Sergio flagged this as an outstanding item.

### Day 4-7 (quality bake-offs + config decisions)

Once vLLM XPU + MTP is confirmed working, bake off against Gemma 4 26B-A4B (current reasoning fallback baseline):

1. **Speed:** vLLM Qwen 3.6-35B vs llama.cpp Gemma 4 26B-A4B QAT+MTP on the same 5K real-workload prompts. Both on card 2 (separately).
2. **Quality:** send both 20-30 representative reasoning prompts from brain workload. Compare outputs qualitatively + LLM-as-judge if you want a rubric score. Gemma 4 26B is the incumbent — Qwen 3.6-35B needs to at least match on your specific workload shape to justify the ops complexity.
3. **Multi-turn 128K:** run Sergio's `benchmarks/b70-multiturn-128k-test.py` (or equivalent). Confirm the 1.4s warm-TTFT pattern holds on B60. This is genuinely usable for long-context sessions in a way llama.cpp SYCL isn't at this context length.

**Decision matrix at end of week 1:**
- vLLM Qwen 3.6-35B beats Gemma 4 26B on speed AND quality → promote vLLM to reasoning-fallback slot on card 2
- Speed wins but quality loses → keep Gemma 4 26B, park the vLLM setup for potential coding-focused deployment
- Neither wins convincingly → defer, wait for Qwen3.8-27B (scheduled watcher) or upstream DFlash for Laguna

## Deferred candidates (revisit checklist)

Everything currently blocked on single-B60 constraints. Priority order for retest:

| Candidate | Current blocker | 2nd B60 resolves? | Priority |
|---|---|---|---|
| **Qwen 3.6-35B-A3B-MTP (vLLM XPU)** | Contention + serving path unproven | ✅ dedicated card + Sergio's cookbook | **P0** — highest expected uplift, best-attested path |
| Gemma 4 26B-A4B QAT + MTP reasoning fallback | Co-res with Ornith blocked | ✅ dedicated card | P1 — quick win, already benched, launcher exists |
| Laguna XS.2 Q4_K_M (llama.cpp) | 22.1 GiB co-res-hostile, no DFlash | ✅ VRAM ✓, DFlash still missing | P2 — coding bake-off if agent quality matters |
| gpt-oss-20b Q4_K_M | Slow decode (25 tps, no MTP) | Partial — dedicated slot fine but decode still slow | P3 — agentic tool-calling quality gamble |
| Qwen3.8-27B (rumored MoE + MTP) | Not released yet | ✅ if MoE — vLLM path candidate | P0 when it lands (scheduled watcher active) |
| Muse Glimmer-30B + DFlash (Meta, multimodal) | Bandwidth-bound at ~25 tps decode + fresh SYCL arch (kernels un-optimized) | Partial — dedicated card + kernel maturation | P4 today, **P2 if a vision workload emerges** — only vision-capable option in the lineup, retain on disk |

## Upstream items to watch

| Item | Repo | Why it matters | How to check |
|---|---|---|---|
| **XPU FP8 linear kernel** | vllm-project/vllm | Unblocks dense on vLLM — Ornith could migrate | Grep `choose_scaled_mm_linear_kernel` for `XPU` platform registration |
| **Upstream DFlash decoder contract for Laguna** | ggml-org/llama.cpp | Laguna XS.2 gets its native drafter without needing Poolside's fork | Watch [ggml-org/llama.cpp PRs for `laguna` or `dflash`](https://github.com/ggml-org/llama.cpp/pulls?q=laguna+OR+dflash) |
| **Qwen3.8-27B open weights** | Qwen HF org | Rumored 17B-active MoE — if true, direct upgrade over Qwen 3.6-35B-A3B | Scheduled task `qwen38-hf-watch` fires daily 07:00 AEST |
| **Vulkan mega-PR sub-PRs #24406, #24407** | ggml-org/llama.cpp | Both still open. Vulkan gets Xe FA + GEMM optimizations. Currently we're on SYCL because MTP works there; Vulkan may catch up | GitHub PR watch |
| **Intel oneAPI FP8 native support on Battlemage** | Intel oneAPI releases | If oneAPI adds FP8 XMX path, vLLM XPU FP8 kernel becomes trivially achievable | Intel Arc release notes |

## Power management for dual-card

Both cards draw ~100W each under LLM load (finding #21 — B60 self-limits regardless of TDP cap). With 2 cards active simultaneously: ~200W GPU draw. Your existing 750W Gold PSU is fine — leaves ~250W for CPU/board/spinup transients. No PSU upgrade needed.

Thermal: B60 runs cool (~53°C under load) with no forced airflow issues in your case. 2 cards in the same chassis should stay under 65°C provided case airflow is unchanged; monitor for the first few days via `xpu-smi stats -d 0 -m 0` and `-d 1 -m 0` in parallel.

Set MoE-friendly power cap on card 2 explicitly (matches Sergio's finding):

```bash
# Card 1 stays default (embed + dense chat = mixed workload, benefits from default)
# Card 2 = MoE-heavy (150W cap, per Sergio's data)
echo 150000000 | sudo tee /sys/class/hwmon/hwmon5/power1_cap  # verify hwmonN maps to card 2
```

## Rollback plan

If anything goes wrong on card 2, card 1 is untouched — brain workload continues on Ornith / embed / TEI as today. No prod risk. Rollback for card 2 is `docker rm -f` on the vLLM container and either leave card 2 idle or fall back to bench-eval on it.

## Success criteria for the first week

- [ ] Both B60s enumerate cleanly, no PCIe topology issues
- [ ] Sergio's 4 patches applied cleanly to `intel/vllm:0.21.0-xpu-int4moe` — no rebuild
- [ ] Qwen 3.6-35B-A3B-MTP GPTQ served via vLLM on card 2, MTP active per logs
- [ ] Bench matches expected B60 scaling (4-6K tps prefill, 70-90 tps decode)
- [ ] Correctness check passes (greedy replay + factual probes)
- [ ] Card 1 Ornith prod unchanged throughout — no regression under card 2 load
- [ ] Repo `README.md` production stack table updated to reflect dual-card layout

## References

- **Full campaign write-up:** [Sergio B. — Intel Arc Pro B70: vLLM vs llama.cpp — The Full MoE + Dense Showdown](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
- **Reproducible patches + harnesses:** [`SergiioB/intel-arc-pro-b70-inference-cookbook`](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) (MIT)
- **MTP-preserved checkpoint:** [`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4`](https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4)
- **Our deferred candidates:** [`models/tested/2026-08-06-new-candidates-sweep.md`](../models/tested/2026-08-06-new-candidates-sweep.md)
- **Current stack finding #15 (SYCL contention):** [`README.md`](../README.md) — this constraint disappears with 2nd B60
- **Finding #22 (TEI arch-specific advantage):** relevant for card 1 decisions on embed/rerank
