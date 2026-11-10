# HF candidates on b12017 — Mellum 2.1 MTP, Groot-19B-A3B, Flint, LFM3-1.7B, Spark-X2.6-4B — Tested 2026-10-31

**Status:** benched on b12017 (`sycl-f16-5047748f1-patched-c9bf2e9a6`), isolated card 2. Only **round 1** finished, because the window ended on the second of two CCS engine resets (finding #66). **None displaces an incumbent on speed.** LFM3-1.7B has the highest prefill in this repo (finding #67), so it is worth a quality bake-off against the categorise incumbent.

Raw results, runners, server logs and the kernel log: [`configs/benchmarks/20261031-b12017-hf/hf/`](../../configs/benchmarks/20261031-b12017-hf/hf/).

## The sweep

These are the models published by `ggml-org`, `bartowski` or `unsloth`, or trending, between 2026-10-15 (the previous sweep) and 2026-10-31. Two filters applied: llama.cpp support at b12017, and fitting one 24 GiB B60.

| Candidate | Arch | Shape | Target slot | Why |
|---|---|---|---|---|
| [`bartowski/Mellum2.1-12B-A2.5B-Thinking-MTP-GGUF`](https://huggingface.co/bartowski/Mellum2.1-12B-A2.5B-Thinking-MTP-GGUF) (10-30) | MoE | 12B / 2.5B active, embedded MTP layer | `:8011` | 127 tok/s undrafted on 2026-10-15 (Q4_0). MTP was new in this upload. |
| [`GrootTeam/Groot-19B-A3B-261023`](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023) (10-23) | `qwen3_5_moe` | 19.4B, 128 experts / 8 active, 3:1 linear/full attention, `mtp_num_hidden_layers: 1`, Apache-2.0 | `:8011` | Same arch family as Ornith. Q4_K_M is 11.6 GB. |
| [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) (10-21, [`ggml-org/Flint-GGUF`](https://huggingface.co/ggml-org/Flint-GGUF)) | `flint` | 39.8B, 256 experts / 6 active + shared, hidden 2048, 49 layers (14 full, 35 sliding-512), 262K ctx | `:8011` | Upstream support in [`a37cc4610`](https://github.com/ggml-org/llama.cpp/commit/a37cc4610) (#30928). |
| [`LiquidAI/LFM3-1.7B-GGUF`](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) (10-24) | `lfm3` | 1.7B | categorise | Upstream support in [`19cce6919`](https://github.com/ggml-org/llama.cpp/commit/19cce6919) (#31252). |
| [`bartowski/Spark-X2.6-4B-GGUF`](https://huggingface.co/bartowski/Spark-X2.6-4B-GGUF) (10-20) | as Spark-X2.5 | 4B dense | categorise / chat | Successor to [Spark-X2.5-4B](spark-x2.5-4b.md). |

Skipped:
- **Diffuspace/Clover-35B-A3B:** no `clover_moe` arch in llama.cpp.
- **Nanbeige4.3-33B:** dense 33B, and the dense Qwen3.8-27B ran 23 tok/s here ([2026-08-17](qwen-3.8-27b.md)).
- **Too big for one 24 GiB card:** Ling-3.1-flash (572B), Mistral-Large-4 (1T), MiniMax-M3.1-Flash, DeepSeek-V4.1-Flash, Ornith-1.5-122B.
- **Already benched:** NVIDIA Bolt E4B and 74B ([2026-10-24](nemotron-3.5-bolt.md)), LFM2.5-24B and Mellum Q4_0 undrafted (2026-10-15, not written up here).

## Method

Same rig as the [b12017 A/B](2026-10-31-b12017-ab.md):
- card 2 `:8020`, q8_0 KV, `-fa on`, `--reasoning off`
- `bench-candidate.py`: 20 × 300-token decode, prefill 500 / 2K / 5K / 12K
- `:8011` stopped for the window, with `card2-empty.sh` before every launch (finding #59)

All weights are **Q4_K_M**, so the XMX MMVQ kernel engages (finding #58).

Embedded MTP heads: Groot and Mellum both carry `nextn_predict_layers = 1` inside the main GGUF. llama.cpp loads them with `--spec-type draft-mtp` and no `--model-draft` (`load_mtp`, `common/common.cpp`). [`bench-slot.sh`](../../configs/launchers/bench-slot.sh) only passed `SPEC_ARGS` when `DRAFT` was set, so it gained a branch for this case. Runs that use a separate draft file behave as before.

## Results (round 1)

| Model | Config | Decode median | σ | Acceptance | Prefill 2K | Prefill 12K | Peak VRAM |
|---|---|---|---|---|---|---|---|
| Mellum 2.1-12B-A2.5B | Q4_K_M, solo | 81.10 | 0.25 | – | 2,350 | 2,498 | 10.33 GiB |
| Mellum 2.1-12B-A2.5B | Q4_K_M + embedded MTP, n-max 3 | **67.68** | 9.19 | 58.5% (1.75 / draft) | 2,118 | 2,216 | 11.17 GiB |
| Groot-19B-A3B | Q4_K_M, solo | **no result**: CCS reset at 02:37:05, about 67 tok/s at the time | | | | | ~13.9 GiB |
| Groot-19B-A3B | + embedded MTP | not run | | | | | |
| Flint | Q4_K_M, `-c 32768` | 67.80 | 0.17 | – | 1,206 | 942 | **22.79 GiB** |
| LFM3-1.7B | Q4_K_M, solo | 186.2 | 2.82 | – | **8,676** | **9,410** | 3.07 GiB |
| Spark-X2.6-4B | Q4_K_M, solo | 92.33 | 0.36 | – | 3,848 | 3,952 | 6.65 GiB |

Incumbents on the same image, from the [A/B](2026-10-31-b12017-ab.md):
- Nemotron 3.5 Lightning + MTP: 99.7 tok/s decode, 1,661 prefill at 2K, 22.08 GiB.
- Gemma 4 E2B + MTP: 274.4 tok/s, 4,633, 4.08 GiB.

### Reading it

- **Mellum: the embedded MTP head makes it 17% slower.** It accepts 58.5% at 1.75 tokens per draft, which does not pay for the verify step. Undrafted, it loses to Nemotron by 19% on decode, but has +42% prefill and less than half the VRAM.
  - Today's Q4_K_M solo (81.1) is well below the 2026-10-15 Q4_0 solo (127.3, on `08246a28f`). That is a different quant, a different build, and a cross-date comparison (finding #40), so this run can't separate the causes.
  - **Run Q4_0 next on this image before ruling Mellum out.**
- **Flint fits one B60, just: 22.8 GiB at 32K.** It decodes at 68 tok/s with very low σ and prefills at 942 at 12K. It is slower than Nemotron on both, at the same VRAM. There is no drafter. Its prefill falls between 2K and 12K, unlike every other model here. That is an unexpected result for a mostly-sliding-window model, and I have not found the cause.
- **LFM3-1.7B: 9,410 tok/s prefill at 12K and 8,676 at 2K.** At 2K that is about 1.9× E2B on the same image (4,633). At 12K it is about 1.9× MiniCPM5-2B's previous repo record (5,029 at 12K, [2026-09-09](minicpm5-2b.md)). It uses 3.1 GiB, which fits the B580 with room left. Its decode is 32% below E2B+MTP, and the model is undrafted. For a categorise workload of short outputs on long inputs, prefill may matter more (the open question left on the [MiniCPM5 page](minicpm5-2b.md)). **Quality is untested.**
- **Spark-X2.6-4B: 92.3 tok/s at σ 0.36,** about 5% above Spark-X2.5 (87.7). That one was on b10867, so the comparison is cross-build and cross-date (±10%) and does not count as a real change.
- **Groot: no data.** The CCS reset came during its first decode runs (finding #66).

## Next

1. LFM3-1.7B: quality bake-off against Gemma 4 E2B on the categorise corpus, on the B580 (llm2 `:8009`).
2. Mellum: run Q4_0, solo and with MTP at n-max 1–2, on b12017.
3. Groot: one rerun, only after the reset question in finding #66 is answered.
