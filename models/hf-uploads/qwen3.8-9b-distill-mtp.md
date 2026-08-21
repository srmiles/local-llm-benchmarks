# Qwen3.8-9B-Distill MTP head — HF upload

**Repo:** [`srmiles/Qwen3.8-9B-Distill-MTP-GGUF`](https://huggingface.co/srmiles/Qwen3.8-9B-Distill-MTP-GGUF) · uploaded 2026-08-21 · Apache 2.0

Standalone MTP draft heads for [`empero-ai/Qwen3.8-9B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill), converted from the official BF16 safetensors with `convert_hf_to_gguf.py --mtp` at commit `bb4caa754` (b10566).

| File | Size | sha256 |
|---|---|---|
| `mtp-Qwen3.8-9B-Distill-head-BF16.gguf` | 4.56 GB (4.25 GiB) | `5a3ac58e36407a0661a2c53cd629d644ff844ddbb7820c1e168be39351ec19ee` |
| `mtp-Qwen3.8-9B-Distill-head-Q8_0.gguf` | 2.43 GB (2.26 GiB) | `cdc47bb91e8e149c43b3ff86bddd522491f6ef990aeaf9ffecbf9df15db22f80` |

Both: architecture `qwen35`, 18 tensors, 2.28B params — the `blk.32.nextn.*` MTP block plus `token_embd`, `output`, `output_norm`.

## Why this exists

empero's model carries an MTP head in its weights (`mtp_num_hidden_layers: 1` in `config.json`) but [their GGUF repo](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF) publishes main-model quants only — BF16 / Q4_K_M / Q5_K_M / Q6_K / Q8_0, no head. With no head file there is nothing to hand `--model-draft`, so anyone running their GGUFs runs unassisted and loses the speculative path the model was built with.

Different failure mode from the [Gemma 4 assistant drafters](gemma-4-assistant-drafters.md), which existed but were unloadable (broken `gemma4_assistant` arch string). Here nothing was published at all. Same fix either way: convert from the official safetensors with llama.cpp's own converter.

This is also the first drafter this stack has built rather than downloaded — see finding #28. Every prior head came from a third party (protoLabsAI for Ornith, Google's safetensors for the Gemma assistants).

## Why Q8_0 and not BF16 only

[`gemma-4-assistant-drafters.md`](gemma-4-assistant-drafters.md) sets the house rule: publish drafters unquantized, because "170-862 MB unquantized costs negligible VRAM vs the target model, and quantizing a drafter risks MTP acceptance regression for no meaningful footprint saving."

That reasoning is correct for the Gemma heads and wrong for this one. The size ratio inverts:

| | Gemma 4 E2B assistant | Qwen3.8-9B-Distill head |
|---|---|---|
| Head (BF16) | 170 MB | **4,560 MB** |
| Target (Q4) | ~3.2 GB | 5.38 GB |
| Head as % of target | ~5% | **~85%** |

The reason is where the parameters sit: **~2.03B of the head's 2.28B is `token_embd` + `output`** (248,320 × 4,096, twice), not the MTP block. It carries a full copy of a 248k-token vocab's embedding and output projection. Those are exactly the tensors that quantize well, and Q8_0 measured **81.4% acceptance** — inside the band Ornith 1.5-9B's third-party Q8_0 head reaches (84.7%) on the same build in the same session.

So both are published: **Q8_0 as the recommended default**, BF16 as the canonical source for requantization. The house rule stands for small heads; the discriminator is head-to-target ratio, not precision dogma.

> Note the interaction with finding #24: head precision and target quantization have to be controlled *together*. Ornith 1.5-35B-A3B failed at 26.2% acceptance with a Q8_0 head because the target's compression had moved its output distribution. Q8_0 on the head is safe here because the target is a straightforward Q4_K_M, not an aggressively compressed MoE.

## Verification before upload

- GGUF metadata read back for both files: arch `qwen35`, 18 tensors, 2.28B params, expected tensor types (BF16: F32+BF16; Q8_0: F32+Q8_0).
- **BF16 head load-tested end to end** — served under `--spec-type draft-mtp` with `-ngl 0` (weights in RAM, GPU used only for SYCL init, no production impact), generated coherent output, and `llamacpp:spec_decode_*` confirmed it actually drafted (15 drafts / 43 draft tokens / 8 accepted). The Q8_0 head was validated by the full bench.
- Gotcha hit on the way: a SYCL build **aborts at backend registry load with `can not find preferred GPU platform` if no GPU is exposed at all**, so a CPU-only container needs `--device /dev/dri` even at `-ngl 0`. Same root cause as the known `llama-server --version` abort.

## Bench (Intel Arc Pro B60, b10566, Q4_K_M target + Q8_0 head)

| Metric | Value |
|---|---|
| Decode | 73.97 tok/s median · 65.56 mean · σ 10.57 |
| Acceptance | 81.4% (4,236 of 5,203 draft tokens) |
| Accepted per draft | 2.43 at `--spec-draft-n-max 3` |
| Prefill | 1,914 @ 2K · 1,957 @ 5K · 2,020 @ 12K |
| Peak VRAM | 14.76 GiB |

Full methodology and the Ornith 1.5-9B reference row: [`../tested/2026-08-21-tier1-tier2-bench.md`](../tested/2026-08-21-tier1-tier2-bench.md).

**Closed 2026-08-21** — unassisted baseline measured in the n-max window:

| | decode med | σ | prefill @ 12K | peak VRAM |
|---|---|---|---|---|
| unassisted | 56.69 | **0.05** | **2,366** | 10.91 GiB |
| + Q8_0 head | 73.97 | 10.57 | 2,020 | 14.76 GiB |
| delta | **+30.5%** | — | **−14.6%** | **+3.85 GiB** |

The head buys 30.5% decode and costs 14.6% of prefill plus 3.85 GiB. Right trade for chat, not automatically right for a prefill-heavy short-output workload like categorise. Model card updated with these numbers. See finding #31.

## Reproducing

```bash
# venv preserved on llm.local at /data/llm/build/convert-venv
hf download empero-ai/Qwen3.8-9B-Distill --local-dir Qwen3.8-9B-Distill-hf

python convert_hf_to_gguf.py Qwen3.8-9B-Distill-hf \
  --mtp --outtype bf16 --outfile mtp-Qwen3.8-9B-Distill-head-BF16.gguf
python convert_hf_to_gguf.py Qwen3.8-9B-Distill-hf \
  --mtp --outtype q8_0 --outfile mtp-Qwen3.8-9B-Distill-head-Q8_0.gguf
```

`--mtp` exports only the MTP tensors as a standalone draft GGUF; `--no-mtp` writes the target without them. The converter also has `--dspark` for DSpark-style heads — relevant if LFM2.5-8B-A1B ever needs a custom one (finding #29).

Scripts on `llm.local`: `/data/llm/build/convert-empero-mtp.sh` (Q8_0), `/data/llm/build/make-bf16-head.sh` (BF16 + cleanup), `/data/llm/hf-upload/push.sh` (repo create + upload).
