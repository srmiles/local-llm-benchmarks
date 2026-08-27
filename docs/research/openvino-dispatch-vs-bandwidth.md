# Is llama.cpp SYCL dispatch-bound on the B60? — roofline verification

**Date:** 2026-08-25 · **Status:** desk verification only, no GPU time spent · **Trigger:** community A770 report (marfrit) claiming OpenVINO GenAI is ~3× llama.cpp SYCL on a hybrid-MoE model, diagnosed as dispatch-bound (~2,500 kernel launches/token traced with `SYCL_UR_TRACE`).

## TL;DR

**The claim is real, and it does not apply to Qwen 3.8-27B.** The roofline forbids the win on that model: 18.97 GB of dense weights on a 456 GB/s card caps decode at **24.0 tok/s** no matter what runtime executes the graph. We measure 23.0. Finding #25 ("dense-27B bandwidth wall, nothing upstream will fix it") is now confirmed by arithmetic rather than inference — **do not spend a bench window re-testing it.**

The headroom the report describes is in the **MoE** models, where we sit at 15–25% of roofline. But see the two caveats below before treating that as 3× waiting to be collected.

## The roofline test

Weight bytes that must cross the bus per decoded token ÷ 456 GB/s. For MoE, only *active* params (plus the `lm_head` matmul) are read.

| Model | Bytes/token (est.) | Roofline | Measured (drafted) | Est. unassisted | % of roofline |
|---|---|---|---|---|---|
| **Qwen 3.8-27B** Q4_K_M, dense | 18.97 GB (all of it) | **24.0** | 23.0 | ~19–20 | **80–96%** |
| **Gemma 4 26B-A4B** QAT Q4_0, 4B active | ~2.7 GB | ~169 | 62.84 | ~40–43 | **24–25%** |
| **Nemotron 3.5 Lightning 30B-A3B** Q4_0, 3B active | ~2.0 GB | ~228 | 91.91 @ n-max 7 | ~35–40 | **15–18%** |

Unassisted figures are extrapolated from finding #31 (the MTP head is worth +30.5% decode at 81.4% acceptance on the 9B); scaled up for Gemma/Nemotron's much higher acceptance, down for Qwen 3.8-27B's 57.9%. They are estimates. The Qwen row's conclusion does not depend on them — even the *drafted* 23.0 is 96% of the hard ceiling.

**Read the same way, the source report is internally coherent.** A770 is 560 GB/s; the model is 3B-active at ~4.85 bpw (~2.5 GB/token):

- llama.cpp 14.4 tok/s → ~36 GB/s → **6% of the card's bandwidth**
- OpenVINO 43 tok/s → ~95 GB/s → 17%
- CPU-only 15.5 tok/s on a 5700X (DDR4-3200 dual channel, ~40 GB/s achievable) → ~39 GB/s → **~90% of the CPU's ceiling**

The CPU did not beat the GPU. The CPU ran at its roofline while the GPU ran at 6% of its. That is what dispatch-bound looks like, and the ~2,500-launch trace is the mechanism.

## Two caveats that shrink the prize on this stack

**1. MTP is already a dispatch-overhead mitigation, and we run it everywhere.** A verify batch of `n_max+1` produces up to 8 tokens from one graph execution, so kernel launches *per token* fall by roughly the accepted-tokens-per-pass factor — 2.90 on Gemma, 3.98 on Nemotron. The A770 comparison was llama.cpp **with no drafter** vs OpenVINO. Our drafted numbers have already collected most of what a fused graph would buy. The honest question here is "OpenVINO fused graph vs llama.cpp + MTP", and OpenVINO GenAI ships EAGLE-3, not MTP-head support for these architectures — so an OpenVINO run would have to beat 62.84 / 91.91, not the unassisted baselines.

**2. OpenVINO's MoE advantage is not automatic — it is one fragile fusion pass.** [openvino#36270](https://github.com/openvinotoolkit/openvino/issues/36270) has Qwen3.6-35B-A3B at ~10 tok/s under OpenVINO vs 17–19 under llama.cpp Vulkan on the same iGPU, root-caused to *78 ops/layer × 40 layers = 3,120 unfused operations* — the identical failure mode, on the other side. Open, unresolved. The source report independently confirms the pass is brittle: mixed-precision IRs (`--ratio 0.8`) are rejected by it, and `ov::cache_dir` loses the expert weights on reload. When the pass fires you get the fused graph; when it doesn't you get 3,120 ops and a result worse than where you started.

## What to do instead

1. **Skip Qwen 3.8-27B.** Physics has already answered. Leave finding #25 as written; it is stronger now, not weaker.
2. **Free measurement first, no export, no rented box:** watch `xpu-smi` GPU utilisation during a steady decode on Gemma 4 26B-A4B, unassisted and at n-max 5. Low utilisation with the bus nowhere near 456 GB/s = dispatch-bound and the prize is real; high utilisation = we are closer to roofline than the estimates above suggest and the whole line of enquiry closes for one afternoon's work.
3. **If it survives that, use the llama.cpp OpenVINO backend** (`docs/backend/OPENVINO.md` upstream, and OpenVINO 2026.1 integrates llama.cpp) on the *same GGUF* before going anywhere near optimum-intel/NNCF. The report's export path cost a 494 GB rented Graviton box for AWQ+SE calibration; ours would too. Only pay that if the backend path shows the gap.
4. **Bench target is Gemma 4 26B-A4B**, not Nemotron — biggest roofline gap of the two, it is the reasoning-fallback slot so a win is deployable, and Google/Intel publish QAT weights, which is the likeliest source of a ready-made ratio-1.0 IR.
5. **Unrelated but still outstanding:** `--spec-draft-n-max 5` is *still* missing from `start-llamacpp-sycl-gemma4-mtp.sh`. +10%, measured at two sampling configs, free, and it is a prerequisite for any fair OpenVINO comparison on that model.

## Card facts used

Arc Pro B60: 24 GB GDDR6, 192-bit @ 19 Gbps = **456 GB/s**. A770 16GB: 256-bit @ 17.5 Gbps = 560 GB/s.
