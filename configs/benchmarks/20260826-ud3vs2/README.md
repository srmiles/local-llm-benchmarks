# `20260826-ud3vs2` — Unsloth Dynamic 3.0 vs 2.0 runner

Committed per finding #40 ("commit the runner with the results"). Results and analysis: [`models/tested/2026-08-27-unsloth-dynamic-3-vs-2.md`](../../../models/tested/2026-08-27-unsloth-dynamic-3-vs-2.md).

Deployed at `/data/llm/benchmarks/20260826-ud3vs2/` on `llm.local`. Paths inside the scripts are absolute to that directory.

| File | Role |
|---|---|
| `chain.sh` | top-level driver — serialises fetch → KLD base → delete referee → fetch v2.0 → bench |
| `fetch-q8.sh` | Q8_0 referee (29.05 GB) + the MTP head |
| `fetch-quants.sh` | `v3` = `main`, `v2` = revision `f1bfb127c64f`; filenames collide so v2.0 lands in its own directory |
| `ppl-run.sh` | one `llama perplexity` run in a throwaway container on card 2 |
| `run-ud3vs2.sh` | phase 1 quality (PPL + KLD), phase 2 throughput, all four arms |
| `probe-decode.py` | decode probe with `ignore_eos` — the fix for the EOS contamination that invalidated the first throughput pass |
| `run-probe-pass.sh` | runs `probe-decode.py` across all four arms |
| `final-pass.sh` | drafter vs no-drafter isolation for the v3.0 Q3_K_XL wedge |
| `diag-q3v3.sh` | minimal wedge reproduction with the server log retained |
| `gguf-types.py` | per-tensor quant-type histogram parsed straight from a GGUF header — no deps |
| `restore-nemotron.sh` | tears down the bench slot and brings `:8011` back; called from every EXIT trap |
| `quality.json` | parsed PPL/KLD summary for all four arms |
| `bench-*.json` | first (EOS-contaminated) pass — retained for prefill and VRAM only |
| `probe-*.json` | the trustworthy decode numbers |

**Two gotchas worth keeping.** `probe-decode.py` originally read `llamacpp:n_drafts` / `n_draft_tokens` / `n_draft_accepted`, which silently return zero — the real names are `llamacpp:spec_decode_num_drafts_total`, `..._num_draft_tokens_total`, `..._num_accepted_tokens_total`. And the KLD base file costs **~127 MB per chunk**, so chunk count is a disk decision before it is a statistical one.
