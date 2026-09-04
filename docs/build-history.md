# llama.cpp SYCL build history — B60 Pro

Journey through llama.cpp SYCL builds and the impact of each release on this stack. Referenced from the main README.

**Deployed builds (2026-09-04, verified from the running containers — this line was stale before today):**

| slot | image | base |
|---|---|---|
| `:8002` Ornith chat, `:8004` embed | `llama.cpp:sycl-f16` | b10688 + patches (`880b848f4`) |
| `:8006` categorise | `sycl-f16-concurrency-a6553043f` | b10688-era + concurrency patch |
| `:8011` Nemotron | `sycl-f16-b10742-patched-a82c13531` | b10742 + 9 patches incl. the #28159 revert |

**Candidate, built + benched 2026-09-04, NOT deployed:** `llama.cpp:sycl-f16-b10809-patched-f937ca544` — master `85d5703a3` + the 8-commit local series, revert dropped. [Rebase notes, A/B and the Nemotron GPU-reset block.](../models/tested/2026-09-04-b10809-rebase-ab.md)
**Rollback tags on disk:** every image in the table above, plus `sycl-f16-b10433-safe`, `sycl-f16-b10256-safe`, `sycl-f16-b10215-safe`.

## b10809 rebase (2026-09-04) — 73 upstream commits, local series carried

Base moved `be789c344` (b10742) → `85d5703a3` (master, b10809 series). Eight local `ggml-sycl` commits rebased; the local revert of #28159 dropped because upstream's follow-up (`d11b3cc7e`, #28173) landed in the window.

| model | decode median | prefill 12K | MTP acceptance |
|---|---|---|---|
| Ornith 1.5-9B + MTP | 61.39 → **76.99 tok/s (+25.4%)** | flat | 66.8% → **76.8%** |
| Gemma 4 E2B + MTP | 199.41 → 204.45 (parity) | flat | 99.2% → 97.7% |
| Gemma 4 26B-A4B + MTP | 66.35 → 64.95 (parity) | +1.1% | 68.2% → 60.3% |
| Nemotron 30B-A3B + MTP | **no data — GPU engine reset on both arms** | | |

Only Ornith clears the ~15% single-pass noise floor (finding #47). The gain tracks drafter acceptance (2.00 → 2.30 accepted/draft) with prefill flat as a control, and the leading — unconfirmed — explanation is the dropped revert restoring correct `n_layer_nextn` handling for the MTP head. See findings #49 and #50.

**Rebase trap worth carrying forward:** upstream `4aa6ffba2` generalised the Q4_K MMVQ path with a `reorder_vec_dot_shared_activations<T>` trait and a `static_assert` on `rows_per_sg`; our GLU-fusion patch's hardcoded `rows_per_sg = 2` at `ncols_dst == 2` becomes a **compile error** for every non-Q4_K type it adds. Fixed with an `if constexpr` gate. Our series sits on the code upstream keeps generalising — re-check semantics on every bump, not just conflict markers.

[Full write-up.](../models/tested/2026-09-04-b10809-rebase-ab.md)

## Local `ggml-sycl` patches on b10566 (2026-08-26)

Not upstream. Branch `all-fixes` off `bb4caa754` in `/data/llm/build/llama.cpp`; patches exported to `/data/llm/build/upstream-patches/`.

| image | commits | deployed |
|---|---|---|
| `llama.cpp:sycl-f16-moereorder` | `68e55f242` MoE reorder Q4_0/Q8_0 | **Nemotron `:8011`, 2026-08-26** |
| `llama.cpp:sycl-f16-q3kmoe` | + `9d1f04528` MoE reorder Q3_K, `ad76cd30a` TQ1_0/TQ2_0 `supports_op` guard | no |
| `llama.cpp:sycl-f16-allfixes` | + `419f85698` MMVQ chunking above batch 8 | no |

| change | measured impact |
|---|---|
| MoE reorder Q4_0/Q8_0 | per-op **q4_0 5.12× / q8_0 6.44× at n=1**, flat at n≥4. End-to-end **Nemotron Q4_0 +32%**, **Gemma 4 26B-A4B QAT +28.7%** (interleaved A/B vs parent commit) |
| MoE reorder Q3_K | **Ornith-1.5-35B APEX +21.4%** (90 of 123 expert tensors are Q3_K) |
| TQ1_0/TQ2_0 guard | `test-backend-ops` completes on SYCL for the first time: **1022/1022 MUL_MAT**, **869/869 MUL_MAT_ID** (previously aborted at 214 lines) |
| MMVQ chunking | removes the finding-#30 batch-8 cliff: Qwen3-4B batch 9 **+104%**; Nemotron at n-max 9 **97.41 vs 78.95 tok/s (+23.4%)**. Default build unchanged to ±0.4% — the cap is env-gated at 8 |

**Read the reorder numbers with finding #36 in hand:** the gain exists only at batch 1, so a high-acceptance synthetic bench shows +3.2% where real agent traffic shows +32%.

[Full write-up, per-op tables and the default-bench re-run.](../models/tested/2026-08-26-sycl-patches-default-bench.md)

## b10433 impact (measured 2026-08-14, isolated `/completion` probes on 5K real-workload prompts, warmup preflight)

| Model | b10256 → b10433 | Notes |
|---|---|---|
| Ornith 9B + MTP (K-quant) | prefill **+7%** (1,789 → 1,911), decode **+6%** (64.0 → 67.9), MTP 100% held | Biggest winner; SYCL Mamba/gated-delta-net optimizations (PRs #26612, #26643) benefit adjacent standard-attention kernels too |
| Gemma 4 E2B + Google MTP | prefill **+8%** (3,169 → 3,425), decode parity | Small consistent win |
| Gemma 4 26B QAT Q4_0 + MTP | prefill +3%, **decode -6%** (53.5 → 50.5), MTP acc -5pp | Q4_0-specific regression on 2026-08-14. **2026-08-21 re-bench cleared it** — QAT decode back to 62.8 tps, MTP acc 97.2%. Not fully diagnosed (NEO cache warmth or upstream patch under same tag). |
| Gemma 4 26B **Q4_K_M** + MTP | parity across all axes (1,473 tps prefill / 47.7 tps decode / 96.1% MTP acc) | K-quant path unaffected |

## b10256 impact (measured 2026-08-04, isolated `/completion` probes on 5K real-workload prompts, `cache_prompt: false`, warmup preflight)

| Model | b10215 prefill | b10256 prefill | Δ | Decode Δ | MTP acc Δ |
|---|---|---|---|---|---|
| Ornith 9B + MTP | 1,442 tps | **1,789 tps** | **+24%** | parity (66→64) | maxed (100%) |
| Gemma 4 E2B + Google MTP | 3,681 tps | 3,169 tps | ~parity | **+18%** (138→164) | **+25pp** (67→92%) |
| Gemma 4 26B QAT + MTP | 1,164 tps | **1,335 tps** | **+15%** | parity | maxed |
| Gemma 4 26B Q4_K_M + MTP | 1,180 tps | **1,475 tps** | **+25%** | parity | ~same |

Universal win across all four configs. Prefill 15-25% up on 3/4, ~parity on E2B (already near XMX GEMM ceiling). Likely from commits between b10215 and b10256 including PR#25852 (SYCL concat kernel parallelization) plus multiple MTP-verification path improvements.

## b10215 impact (retained for historical context)

Decode + MTP acceptance unchanged from b10068 within noise (verified on decode-only synthetic bench pre-cutover), but ~2× prefill uplift discovered 2026-08-01 during brain-eval Track 2 ingest bake-off — real workload prompts hit ~3,000 tps prefill on Ornith 9B via prefix-cache reuse. Root cause: SYCL oneMKL GEMM flash attention for XMX (#25025) merged post-b10068. See [`charts/b10215_prefill_uplift.png`](../charts/b10215_prefill_uplift.png).

## Methodology note (2026-08-04)

The b10215 "~3,000 tps prefill" figure above was measured via **real-workload wall-clock during brain-ingest** (multiple sequential prompts benefiting from prefix cache reuse across the pipeline). The b10256 comparison table uses **isolated `/completion` probes with `cache_prompt: false` + warmup preflight**, which measures single-request cold prefill only. Both numbers are legitimate but they measure different things — never mix them in a delta. Rule for future comparisons: pick one methodology and hold it constant across A/B. See finding #20.

## b10068 investigation resolved 2026-07-20

brain-eval flagged an Ornith 9B MTP acceptance drop on b10068 (46% vs 64% baseline, small-N greedy bench). Methodology-matched A/B disproved it: (a) both b9948 and b10068 produce different hashes across repeated identical greedy runs — SYCL FP non-determinism is pre-existing, not b10068-introduced; (b) isolated prefill probe shows b10068 delivers a real but modest **~3% uplift at long context, flat at short** (not the "+42%" originally claimed). Prod acceptance range under real workload (68-76%) is the ground truth.

## Journey summary — decode / prefill / warm-follow-up over time

| Stage | Decode | Cold 12K prefill | Warm follow-up |
|---|---|---|---|
| LM Studio Vulkan (start) | 33.6 tok/s | 127s @ 97 tok/s | 36s (cache broken) |
| SYCL out-of-box | 38.4 tok/s | 37s @ 388 tok/s | 36s |
| + FA on, `-ub 2048` (Config D) | 38.6 tok/s | 30s @ 477 tok/s | 0.66s |
| + `GGML_SYCL_F16=ON` rebuild | 40.1 tok/s | 24s @ 602 tok/s | 0.61s |
| + Q4_K_M post-training | 44.1 tok/s | 22.8s @ 632 tok/s | 0.55s |
| + Config C + MTP (bare-metal, b9948) | 50.0 tok/s | ~13.7s @ ~938 tok/s | ~0.55s |
| + b10068 rebuild (XMX+oneDNN FA, Ornith prod) | 51.8 tok/s | ~13.3s @ 969 tok/s (+3%) | ~0.55s |
| + b10215 rebuild (oneMKL GEMM XMX FA #25025, real workload) | 56 tok/s | ~4.3s @ ~3,000 tok/s (real w/ prefix cache) — or ~8.3s @ 1,442 tok/s isolated | ~0.55s |
| + b10256 rebuild (SYCL concat parallelization #25852 + MTP verification improvements, cutover 2026-08-04) | **64 tok/s** (+15%) | **~6.7s @ 1,789 tok/s isolated (+24% vs b10215 same methodology)** | ~0.55s |
| + b10433 rebuild (Mamba/gated-delta-net optimizations #26612 #26643, cutover 2026-08-14) | **67.9 tok/s** (+6%) | **~5.2s @ 1,911 tok/s isolated (+7%)** | ~0.55s |
| + Ornith 1.5-9B swap (2026-08-21, same b10433) | **65.4 tok/s** (~parity, MTP acc up to 82.7% from 76%) | prefill **2,040 tps @ 4K** (+54% vs 1.0 baseline) | ~0.55s |

**Overall vs LM Studio start (isolated `/completion` methodology):** LM Studio Vulkan 97 tps prefill → b10433 SYCL 1,911 tps prefill = **~20× cold prefill**. Warm-path cache hits: 36s → 0.55s = **~65×**. Decode: 33.6 → 65.4 tok/s = **+95%**.

(b9948 12K prefill row revised 2026-07-20 from isolated probe — earlier "22.8s @ 632" figure was cold-start against a bandwidth-contended stack; methodology-matched probe gives 14.2s @ 938 tok/s. This means most of the prefill journey landed with SYCL + FA + `-ub 2048` + F16 rebuild + Q4_K, then b10215's XMX GEMM FA more than doubled it again under real load.)
