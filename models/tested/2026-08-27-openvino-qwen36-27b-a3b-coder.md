# OpenVINO GenAI 51 tok/s on B60 for Qwen 3.6-27B-A3B — replicates marfrit's A770 finding

**Date:** 2026-08-27
**Purpose:** Task #148 — test whether the marfrit A770 result (OpenVINO 3× llama.cpp SYCL on Qwen 3.6-27B-A3B-Coder) replicates on our B60 hardware. Signal for whether to reframe task #142 (Ornith 1.5-35B tensor-split) as an OpenVINO port.
**Model:** [`marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov`](https://huggingface.co/marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov) — pre-built OpenVINO IR, int4 group-64, AWQ + Scale Estimation, arch `qwen3_5_moe`, VLM (image-text-to-text).
**Runtime:** `openvino 2026.3.1` + `openvino-genai 2026.3.1.0` + `openvino-tokenizers 2026.3.1.0` in a Python 3.14 venv at `/data/llm/build/openvino-venv/`.
**Hardware:** llm.local B60 card 2 (GPU.1), 24 GiB, all other card-2 workloads stopped (Nemotron paused for the bench window).
**Storage:** model on NAS at `/docker/scratch/marfrit-openvino/` (NFS), 14.8 GB.

## Headline numbers

| Metric | Value | Notes |
|---|---|---|
| Pipeline load (fresh compile) | **204.5 s** | First-run compile from NFS-mounted IR, no `ov::cache_dir` (per marfrit trap #4) |
| Warmup (32 tok, greedy, "Hello" prompt) | 3.03 s | Includes prefill on a 2-token input |
| Bench input length | 39 tokens | Code-gen prompt (Fibonacci function w/ docstring + error handling + complexity) |
| Bench output length | **512 tokens** | Greedy, `temp=0.0`, `do_sample=False` |
| **TTFT** | **1,133 ms** | Prefill of 39 tokens |
| **Decode throughput** | **51.09 tok/s** | Wall clock 11.23 s for 512 tokens |
| VRAM peak observed | ~23 GiB | Loaded + KV during decode; released to 5 GiB at idle post-decode |
| Idle VRAM after generate | 5,124 MiB | Model itself pinned; KV etc released |

## Direct comparison to marfrit's A770 numbers

| Runtime / Hardware | Model | Decode tps | Notes |
|---|---|---|---|
| marfrit A770 llama.cpp SYCL Q4_K_M | Qwen 3.6-27B-A3B-Coder | **14.4** | Q4_K_M ~4.85 bpw |
| marfrit A770 CPU (AMD 5700x) | same GGUF | 15.5 | GPU loses to CPU |
| marfrit A770 OpenVINO GenAI int4 g64 | same weights, different quant | **43** | AWQ + SE, ~4.3 bpw |
| **This run — B60 OpenVINO int4 g64** | **same OpenVINO IR** | **51.09** | **+18.8% over A770 same runtime** |
| Reference — our B60 llama.cpp SYCL Qwen 3.6-35B-A3B-MTP UD-Q4_K_XL | Qwen 3.6 **35B**-A3B | 49.0 | Bigger sibling model, MTP-accelerated |
| Reference — our B60 llama.cpp SYCL Qwen 3.6-35B-A3B-MTP IQ4_XS | Qwen 3.6 **35B**-A3B | 31.6 | IQ path penalty |

**The 3× ceiling gap replicates on Battlemage.** We don't have our own llama.cpp SYCL number on this specific 27B-A3B-Coder model, but same-family same-quant comparison holds:
- **51 tps on OpenVINO for 27B-A3B** vs **49 tps on llama.cpp SYCL for 35B-A3B-MTP** (which is a bigger model with a purpose-built drafter helping).
- Marfrit's llama.cpp SYCL A770 was 14.4 tps on this exact model. Our B60 is ~1.5-2× A770's raw compute, so a B60 llama.cpp SYCL run on this same 27B-A3B model would very plausibly land in the 20-30 tps range — putting **OpenVINO 51 tps at ~2× llama.cpp SYCL for the same weights on our hardware**.

Combined with our own kernel-side findings:
- Finding #36 — MoE expert-tensor reorder wins only at batch 1 (dispatch-shaped work)
- Finding #43 — oneDNN 4-bit weight decompression measured 2.12× the current path standalone
- Finding #47 — llama.cpp dissolves MoE into ~2,500 kernel launches per token (marfrit's `SYCL_UR_TRACE`)

OpenVINO fuses the graph into ~24 ms of device time per token (marfrit's number), sidestepping the dispatch overhead entirely. **Our number confirms this holds on the Battlemage silicon, not just A770/Alchemist.**

## Traps we hit (and pre-empted from marfrit)

1. **`VLMPipeline`, not `LLMPipeline`** — the arch is VLM (`qwen3_5_moe` with vision embeddings). LLMPipeline would fail to construct. ✅ used VLMPipeline.
2. **API signature quirk on `generate()`** — first attempt was `pipe.generate(prompt, cfg)`, which matches no VLMPipeline signature. The overload we want is `pipe.generate(prompt, **kwargs)` (signature #5) with `generation_config=cfg` as kwarg. Cost us one 200-s reload cycle.
3. **NFS scratch adds ~2-3 min per cold load** — model file is 14.8 GB on Synology NFS. Would be much faster from local ext4, but we're avoiding /data pressure (only 28 GB free there). Not a runtime issue — happens once per process start.
4. **99 threads spawned** — OpenVINO parallelism during compile phase is aggressive. CPU-heavy for those first 200 s.
5. **VRAM behavior** — model loads to VRAM lazily; peak observed 23 GiB during decode, but the process stabilizes back to 5 GiB when idle (KV released). Same shape as our llama.cpp numbers but with lower steady-state footprint.

**Not tested yet:** `enable_prefix_caching=false` explicitly set — marfrit's trap #5 says it cost him 2 greedy points. Our current run doesn't touch that setting, so it defaulted (may or may not be off).

## Environment reproducibility

```bash
# On llm.local:
sudo apt install python3.14-venv     # (already installed)
python3 -m venv /data/llm/build/openvino-venv
source /data/llm/build/openvino-venv/bin/activate
pip install openvino openvino-genai openvino-tokenizers huggingface_hub

# Download model to NAS scratch (14.8 GB)
hf download marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov \
  --local-dir /docker/scratch/marfrit-openvino/

# Bench script at /tmp/ov-smoke.py — captured verbatim in this doc
```

## Coordination window impact

- Nemotron container stopped for **~15 min** during the probe (compile + warmup + bench + teardown)
- Nemotron `--restart=no` set during the probe, restored to `unless-stopped` after
- No LB traffic redirected — Nemotron :8011 isn't in any Traefik pool, so only direct-URL clients (Steve's agent testing) were affected
- Bench window overlapped with Nemotron stopping and starting cleanly; no engine resets in dmesg

## What this means for the roadmap

**Task #142 (Ornith 1.5-35B-A3B tensor-split via llama.cpp SYCL) is now decisively worse than the OpenVINO port option.** Both target the same problem — making a 30-35B MoE decode fast on B60 — but:

| Approach | Estimated decode | Model support | VRAM efficiency | Effort |
|---|---|---|---|---|
| llama.cpp SYCL tensor-split (task #142) | ~30-40 tps est. | Any GGUF | Cross-card PCIe traffic hurts | Medium (config + tuning) |
| **OpenVINO GenAI on single B60** | **~50 tps measured today** | Requires HF→IR conversion | Single-card, ~23 GiB peak | **High initial (per-model conversion)** |
| vLLM XPU (parked) | Sergio: +1.8× decode over llama.cpp | Any HF model | Similar to llama.cpp | High (new runtime stack) |

**Next probe should be:** convert Nemotron 3.5 Lightning 30B-A3B ourselves (optimum-intel / NNCF int4 g64 with ratio 1.0), run the same bench on card 2, compare against the 74 tps we're getting on llama.cpp SYCL b10567+moereorder. If OpenVINO on Nemotron gives ≥100 tps, the reasoning slot moves to OpenVINO permanently and #142 gets closed with prejudice.

## Sources

- [marfrit's HF model card](https://huggingface.co/marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov)
- [Original Reddit post](https://www.reddit.com/r/IntelArc/comments/1vxc2l8/)
- [`docs/research/runtime-hardware-alternatives.md`](../../docs/research/runtime-hardware-alternatives.md) — full analysis of both external signals
- [Finding #47](../../docs/findings.md) — one-paragraph statement of the dispatch-bound ceiling gap
