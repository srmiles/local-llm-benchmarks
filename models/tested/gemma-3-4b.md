# Gemma 3 4B — Tested for categorise

**Status:** Tested for categorise workload; lost to Qwen3-4B on JSON structure adherence and required system-role workaround.
**HF:** `google/gemma-3-4b-it` (community Q4_K_M GGUFs)

## Specs

| | |
|---|---|
| Parameters | 4B dense |
| Quant | Q4_K_M |
| File size | ~2.5 GB |
| Context | 32,768 |

## Benchmarks (categorise bake-off)

| Metric | Value |
|---|---|
| Decode | ~78 tok/s |
| JSON valid | 3/3 |

## Issue: no `system` role in chat template

Gemma 3's chat template rejects the `system` role — brain's categorise system prompt has to be merged into the `user` role for the template to accept it. Workable but adds a translation layer that Qwen3-4B doesn't need.

## Verdict

Fast enough on paper, but the template-fix requirement plus lower decode than Qwen3-4B-Instruct-2507 (~94 tok/s) made it a clear loss for the categorise slot. Not adopted.
