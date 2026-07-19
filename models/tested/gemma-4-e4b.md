# Gemma 4 E4B — Tested (QAT vs K-quant reversal)

**Status:** Tested for categorise workload; loses to Qwen3-4B-Instruct-2507. Interesting for the **QAT-vs-K-quant reversal** it demonstrates at small scale.
**HF (QAT):** [`google/gemma-4-E4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)
**HF (K-quant):** post-training Q4_K_M variants (lmstudio-community, etc.)

## Specs

| | |
|---|---|
| Parameters | ~4B effective (MoE-adjacent) |
| Quant tested | QAT Q4_0 and post-training Q4_K_M |
| File size | ~2.5 GB |
| Context | 32,768 |

## Benchmarks (3-request categorise workload, --parallel 4, -ub 1024)

| Model | Decode tok/s | Prefill tok/s | JSON valid |
|---|---|---|---|
| Gemma 4 E4B **QAT Q4_0** | **73.9** | 376 | 3/3 |
| Gemma 4 E4B **Q4_K_M** | 68.3 | **466** | 3/3 |
| Qwen3-4B-Instruct-2507 Q4_K_M | ~94 | — | 3/3 |

## Key finding: QAT-vs-K-quant reversal at 4B

Finding 8 established that on **Gemma 4 26B-A4B**, post-training Q4_K_M beats QAT Q4_0 by ~+10% decode.
On **Gemma 4 E4B**, that split **inverts**:

- **QAT Q4_0 wins decode (+8%)**
- **QAT Q4_0 loses prefill (−19%)**

So the QAT-vs-K-quant winner is **model-size-dependent** on Battlemage, not a universal rule. Don't generalise Finding 8 down the scale.

## Verdict

Neither Gemma 4 E4B variant catches Qwen3-4B-Instruct-2507 on the categorise workload. Both lose to Qwen3-4B's ~94 tok/s decode. Not adopted, but useful evidence for the QAT reversal.

## Download quirks

- `google/gemma-4-E4B-it-qat-q4_0-gguf` shows a Gemma license gating banner in the HF web UI, but the direct file URL is anonymously downloadable
- Filename inside the QAT repo is `gemma-4-E4B_q4_0-it.gguf` (underscore-then-dash), not the conventional `gemma-4-E4B-it-q4_0.gguf`
- Community mirrors under `unsloth/`, `lmstudio-community/`, `bartowski/` all 401 for the QAT variant — go direct to Google's repo
