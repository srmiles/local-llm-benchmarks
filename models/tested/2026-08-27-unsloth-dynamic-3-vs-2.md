# Unsloth Dynamic 3.0 vs 2.0 — Qwen 3.8-27B, matched quant names — 2026-08-27

**Question:** unsloth re-quantized `unsloth/Qwen3.8-27B-GGUF` and now advertises [Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) as ">10% top-1% better accuracy at the same size compared to every other provider". Does 3.0 actually beat the 2.0 files it replaced, on our hardware?

**Answer, short:** **yes on accuracy, no on throughput, and one file is unusable.** v3.0 is measurably closer to the Q8_0 referee at both sizes *while being smaller* — on mean, median and tail KLD, and on RMS Δp. (It is **not** uniformly better: it loses on 99.0% Δp at both sizes, and on top-1 agreement at 4-bit. See the table.) It also decodes **roughly 10–14% slower** at Q4_K_XL (see the warning box in the throughput section — the sign is solid, the magnitude is a cross-pass figure), alongside a much more heterogeneous block-type mix. And **v3.0's `UD-Q3_K_XL` wedges llama-server**: a token id of `-1` reaches the batch, initialisation fails, and with a drafter attached every subsequent request 500s until restart.

## The A/B is unusually clean

unsloth re-quantized **in place** on 2026-08-19 rather than publishing a new repo, so both versions are still reachable from one repo id:

| | revision | date |
|---|---|---|
| Dynamic **3.0** | `main` | re-quant 2026-08-19 |
| Dynamic **2.0** | `f1bfb127c64f` | 2026-08-15, last commit before the re-quant |

Same uploader, same base checkpoint, same filenames. Fetch the old arm with `hf download unsloth/Qwen3.8-27B-GGUF <file> --revision f1bfb127c64f`.

**The referee is `Qwen3.8-27B-Q8_0.gguf`** (29,047,086,048 B). The 08-19 re-quant deleted and rewrote most of the ladder but **never touched Q8_0** — verifiable rather than inferred: its LFS SHA256 is `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348` **in both revisions**, while the two quants under test differ (`bee238bb…` → `3f227079…`, `00cf92e6…` → `8c2a45ff…`). Same blob, so neither version was tuned against it. Recorded in `evidence-manifest.txt`. Its logits are the KL-divergence base for every arm below.

## v3.0 is smaller at the same name — which is the trap

| Quant | v2.0 | v3.0 | Δ size |
|---|---:|---:|---:|
| UD-Q4_K_XL | 17.92 GB (17,923,394,624 B) | 17.56 GB (17,559,178,144 B) | **−2.03%** |
| UD-Q3_K_XL | 13.44 GB (13,441,059,904 B) | 13.15 GB (13,146,393,504 B) | **−2.19%** |
| UD-Q2_K_XL | 10.68 GB | 9.83 GB | −7.9% |
| UD-IQ2_XXS | 9.01 GB | 7.27 GB | −19.4% |

The bottom two rows are HF blob sizes; those files were never downloaded. All four figures, and every byte count above, are in `evidence-manifest.txt`.

On a model already proven **dense-bandwidth-bound** (finding #25), fewer bytes mechanically means more tok/s. A throughput-only bench would therefore hand 3.0 a free "win" that says nothing about the accuracy claim, so quality is measured separately and first.

## Quality — KLD against the Q8_0 referee

120 chunks at `n_ctx 512` = **61,440 tokens** of wikitext-2 test, `--seed 1234`, `-ngl 99`, build `sycl-f16-allfixes`, card 2. Referee `Mean PPL(base) = 6.790810`.

| | **Q4_K_XL v2.0** | **Q4_K_XL v3.0** | | **Q3_K_XL v2.0** | **Q3_K_XL v3.0** |
|---|---:|---:|---|---:|---:|
| File size | 17.92 GB | **17.56 GB** | | 13.44 GB | **13.15 GB** |
| Mean PPL(Q) | 6.809793 ±0.0955 | **6.791899** ±0.0952 | | 6.949201 ±0.0972 | **6.896400** ±0.0969 |
| Mean ln(PPL(Q)/PPL(base)) | 0.002791 ±0.000670 | **0.000160** ±0.000627 | | 0.023057 ±0.001477 | **0.015429** ±0.001452 |
| Mean KLD | 0.004947 ±0.000090 | **0.004529** ±0.000070 | | 0.026075 ±0.000366 | **0.022724** ±0.000324 |
| Median KLD | 0.001988 | **0.001934** | | 0.011858 | **0.009868** |
| 99.0% KLD | 0.052930 | **0.044834** | | 0.271880 | **0.232513** |
| 99.9% KLD | 0.201935 | **0.161286** | | 0.821890 | **0.754063** |
| Same top-1 | 97.000% ±0.098 | 96.954% ±0.098 | | 92.817% ±0.148 | **93.529%** ±0.141 |
| RMS Δp | 2.005% ±0.053 | **1.909%** ±0.039 | | 4.688% ±0.080 | **4.226%** ±0.078 |
| 99.0% Δp *(v3.0 loses)* | **5.326%** | 5.541% | | **10.119%** | 10.675% |

**v3.0 wins on mean, median and tail KLD and on RMS Δp at both sizes, while being ~2% smaller.** Two exceptions keep this from being a clean sweep, and both matter:

- **99.0% Δp is worse for v3.0 at both sizes** (5.326 → 5.541 and 10.119 → 10.675). So the *typical* token is closer to the referee under v3.0 while the *unlucky 1%* moves further away — the opposite of the tail story KLD tells. The two metrics disagree because KLD is computed over the whole distribution and Δp only over the reference's top token.
- At 4-bit, **top-1 agreement marginally favours v2.0** (97.000 vs 96.954).

Two things the headline number hides:

- **At 4-bit the KLD gain is entirely in the tail.** Top-1 agreement moves −0.046 pp, which is **0.33σ** — noise. What moves is the far tail: 99.9% KLD drops **20.1%**. The cleanest way to see it is `ln(PPL(Q)/PPL(base))`, where **v2.0 sits 4.2σ above the referee and v3.0 is statistically indistinguishable from it** (0.000160 ±0.000627).
- **At 3-bit, top-1 agreement moves for real:** +0.712 pp at ~3.5σ. Read as error reduction that is 7.183% → 6.471% disagreement, **−9.9% relative** — essentially unsloth's ">10% top-1%" claim. **But it only materialises at the low-bit end.** Anyone quoting that number for a 4-bit quant is quoting a result that does not exist there.

## Why v3.0 is slower: it mixes many more block types

Per-tensor quant-type histogram straight from the GGUF headers (`gguf-types.py` in the bench dir), non-F32 tensors:

**UD-Q4_K_XL** — 4 quant types → 8

| v2.0 | tensors | G-elems | | v3.0 | tensors | G-elems |
|---|---:|---:|---|---|---:|---:|
| Q5_K | 325 | 18.82 | | Q5_K | 191 | 11.53 |
| IQ4_XS | 65 | 5.79 | | IQ4_XS | 70 | 5.88 |
| Q4_K | 97 | 1.29 | | Q4_K | 69 | 5.44 |
| Q6_K | 19 | 1.41 | | Q6_K | 56 | 3.49 |
| | | | | IQ4_NL | 6 | 0.50 |
| | | | | **Q3_K** | 3 | 0.27 |
| | | | | Q8_0 | 110 | 0.12 |
| | | | | **IQ3_S** | 1 | 0.09 |

**UD-Q3_K_XL** — 4 quant types → **13**, including four 2-bit ones

| v2.0 | tensors | G-elems | | v3.0 | tensors | G-elems |
|---|---:|---:|---|---|---:|---:|
| IQ4_XS | 357 | 13.10 | | IQ4_XS | 156 | 9.36 |
| IQ3_S | 130 | 11.59 | | IQ3_S | 111 | 8.21 |
| Q5_K | 18 | 1.36 | | IQ3_XXS | 34 | 2.35 |
| Q3_K | 1 | 1.27 | | Q3_K | 12 | 2.02 |
| | | | | Q5_K | 26 | 1.71 |
| | | | | **IQ2_S** | 15 | 1.28 |
| | | | | Q4_K | 36 | 1.14 |
| | | | | Q6_K | 7 | 0.42 |
| | | | | **IQ2_XS** | 4 | 0.36 |
| | | | | **Q2_K** | 3 | 0.19 |
| | | | | **IQ2_XXS** | 2 | 0.18 |
| | | | | IQ4_NL | 2 | 0.07 |
| | | | | Q8_0 | 98 | 0.03 |

(Tensor counts sum to 506 non-F32 in every column, so nothing is truncated. F32 is 360 tensors in all four files and is excluded throughout.)

That is the Dynamic 3.0 method made visible — per-tensor bit allocation, dropping unimportant tensors as low as 2-bit and lifting sensitive ones to Q6_K/Q8_0.

**What the histograms do and do not explain.** The clean observation is **heterogeneity**: v3.0 doubles the number of distinct block types at Q4_K_XL and more than triples it at Q3_K_XL, so a decode step dispatches across many more per-type kernels. The tempting follow-on — "v3.0 moves weight into the IQ family, which finding #13 showed underperforms K-quants here" — **is only true at Q4_K_XL, and barely** (IQ share 21.2% → 23.7% of non-F32 elements). At Q3_K_XL the IQ share actually **falls**, 90.4% → 79.8%; what rises there is sub-4-bit content generally, 47.1% → 53.4%. So:

> **The throughput loss is measured; the mechanism is not proven.** Block-type heterogeneity is the one thing that moves consistently in the same direction as the slowdown at both sizes, and it is a plausible dispatch cost, but this bench did not isolate it. A per-op `test-backend-ops perf` sweep across the specific types each file uses would settle it, and has not been run.

## Throughput — and why the first pass had to be thrown away

`bench-candidate.py` lets the server stop at EOS, and the arms hit EOS at very different rates on the standard filler prompt (`bench-per-run.json`):

| Arm | runs reaching 300 tok | median as reported | full-length runs only |
|---|---:|---:|---:|
| Q4_K_XL v2.0 | 19 / 20 | 27.95 | 26.23 – 28.12 |
| Q4_K_XL v3.0 | **10 / 20** | 20.04 | 21.77 – 25.56 |
| Q3_K_XL v2.0 | 13 / 20 | 11.37 | 9.99 – 11.50 |
| Q3_K_XL v3.0 ‡ | 1 / 20 | 10.88 | 10.88 (single run) |

‡ **This row is not EOS contamination at all** — its 19 missing runs are HTTP 500s from the wedge described below, not early stops. It is listed only for completeness; the EOS problem is the first three rows.

A run that stops after 8 tokens reports first-token latency dressed up as tok/s, so v3.0's 20.04 sits below its own full-length band. **Be precise about what this does and does not invalidate:** at Q4_K_XL the v2.0 > v3.0 ordering survives on the full-length subsets, and the two bands do not even overlap (26.23–28.12 vs 21.77–25.56), so the contamination was never what produced that gap — it only made its size untrustworthy. At Q3_K_XL nothing can be concluded from this pass: v3.0 has a single full-length run (10.88) sitting inside v2.0's 9.99–11.50 band. Same family of error as findings #33, #40 and #41: the artifact was in the harness, not the hardware.

Re-measured with `ignore_eos` so every run does identical work — `probe-decode.py`, 10 runs × 300 tokens, a real generative prompt, `--spec-draft-n-max 3`, no p-min (against finding #41's rule — these arms are therefore **not** production-representative, they are matched to each other), ctx 32768, card 2, same MTP head hardlinked into both dirs.

**The two passes are not on the same scale, and EOS is not why.** Every arm is much slower under the probe (v2.0 Q4_K_XL 27.95 → 19.68). That is the **prompt**: acceptance collapses from 96.9% on the filler to 55.2% on a real generative prompt. The filler prompt is the caveat already recorded in the README for the default bench; it inflates every speculative-decoding number on this rig.

| Arm | Decode tok/s (median) | σ | MTP acc | acc/draft | VRAM | Prefill @12K |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_XL **v2.0** | **19.68** | 2.04 | 55.2% | 1.65 | 20.36 GiB | **711.5** |
| Q4_K_XL **v3.0** | 16.95 | 0.90 | 53.5% | 1.60 | 20.02 GiB | 684.7 |
| Q3_K_XL **v2.0** | 7.89 | 0.50 | 55.8% | 1.66 | 16.39 GiB | 588.2 |
| Q3_K_XL **v3.0** | *unusable — see below* | | | | 16.13 GiB | 587.0 |
| Q3_K_XL v3.0, **no drafter** | 8.79 † | 0.04 | n/a | n/a | *not measured* | *not measured* |

† 9 runs, not 10 — one request 500'd and the arm recovered; see the wedge section. VRAM and prefill are blank for this row on purpose: `final-pass.sh` launches it with `DRAFT=""` and runs only `probe-decode.py`, which records neither, so the 16.13 GiB / 587.0 above belong to the **with-drafter** slot and would overstate a no-drafter one by roughly the 1.37 GB head.

Prefill, median tok/s by prompt size (filler-prompt bench, retained because prefill is unaffected by the EOS problem):

| Arm | ~500 | ~2K | ~5K | ~12K |
|---|---:|---:|---:|---:|
| Q4_K_XL v2.0 | 405.6 | 680.7 | 692.2 | **711.5** |
| Q4_K_XL v3.0 | 375.2 | 652.9 | 660.8 | 684.7 |
| Q3_K_XL v2.0 | 257.5 | 534.7 | 549.7 | 588.2 |
| Q3_K_XL v3.0 | 253.0 | 535.9 | 547.3 | 587.0 |

**v3.0 loses decode at Q4_K_XL while being 2% smaller.** Acceptance barely moves (55.2% → 53.5%, 1.65 → 1.60 per draft), so drafter alignment does not explain it.

> ⚠ **The size of that loss is softer than a single number suggests, and the reason is a methodology slip worth recording.** The v2.0 Q4_K_XL arm was measured **twice**, in two different passes: `run-probe-pass.sh` gave **18.82 (σ 1.13)** but with the acceptance counters reading zero (wrong metric names), and `final-pass.sh` re-ran that arm alone ~17 minutes later to recover them, giving **19.68 (σ 2.04)**. The v3.0 arm it is differenced against (16.95) comes from the **first** pass only. So the headline is a **cross-pass** comparison, which is exactly what findings #20, #33 and #40 warn against, and the two v2.0 measurements disagree by 4.6%.
>
> Depending on which v2.0 number is used the gap is **−9.9% to −13.9%**. The sign is not in doubt — every v2.0 measurement in both passes and both methodologies exceeds every v3.0 measurement — but **treat the magnitude as "roughly 10–14%", not as 13.9%.** Settling it needs the two arms re-run interleaved in one session. Compounding factor: the `final-pass.sh` re-run happened while unrelated card-0 work was live on the same host, and finding #33 is precisely a case of cross-card contention faking a double-digit delta.

**Q3_K_XL is not a speed play on this hardware at all.** It is 25% smaller than Q4_K_XL and decodes at **40% of its speed** (7.89 vs 19.68). On a bandwidth-bound dense model that is backwards, and it means the quant ladder below Q4_K buys nothing here: you give up accuracy *and* throughput. The 08-26 SoA reorder patches do not help — they extended MoE **expert** tensor reorder to Q3_K, and this model is dense.

## The v3.0 Q3_K_XL wedge

Reproduced in three of four attempts. With the MTP drafter attached, the slot usually dies and never recovers:

```
E init: invalid token[0] = -1
E decode: failed to initialize batch
E llama_decode: failed to decode, ret = -1
E spec        draft: llama_decode[1] returned -1
E init: invalid token[1] = -1
E srv        decode: Invalid input batch. off = 0, n_batch = 2048, ret = -1
E srv    send_error: task id = 149, error: Invalid input batch.
E srv  update_slots: decode() failed: Invalid input batch.
```

Run counts:

| Configuration | Successful runs | Behaviour |
|---|---|---|
| v3.0 Q3_K_XL + MTP (default bench) | 1 / 20 | run 1 fine, runs 2-20 all HTTP 500 |
| v3.0 Q3_K_XL + MTP (probe) | 7 / 10 | failed at run 8; *unknown* whether it would have recovered — the pre-fix probe had no `try/except` and aborted, issuing no further requests |
| v3.0 Q3_K_XL + MTP (repro) | 1 / 10 | wedged at run 2, never recovered |
| v3.0 Q3_K_XL, **no drafter** | 9 / 10 | one isolated 500, **recovered by itself** |
| v2.0 Q3_K_XL + MTP | 10 / 10 | clean |
| **v3.0 Q3_K_XL + MTP (`diag-q3v3.sh`)** | **4 / 4** | **did not reproduce** — all HTTP 200, `stop_type: limit`, 300 tokens, 7.59–8.22 tok/s |

**That fourth row is the honest caveat on this whole section.** `diag-q3v3.sh` ran the same file with the same drafter and did not fail once. The one thing it does differently is skip the prefill probes — it fires four completions at a freshly loaded slot, where the other three attempts had first pushed 500/2K/5K/12K-token prompts through. So the trigger is plausibly a *slot state* reached after long prefills rather than the weights alone, and "**every** request 500s" is only true once the slot has already been poisoned. Not enough runs to call the rate.

**The fault appears to originate in the model file, not the drafter** — the same file still threw a 500 with no drafter attached (`probe-q3_k_xl-v3-nodraft.json`, `run6`). **Caveat worth stating plainly:** that arm's committed server log is a `tail -60` covering only the last runs, so it does **not** contain the run-6 error, and there is no server-side proof the no-drafter 500 was the same `-1` fault rather than an unrelated one. This is the load-bearing claim of the section and it rests on one HTTP status code. **The drafter is what makes it fatal:** without speculation the server recovered on the very next request (9/10 clean), and with it the invalid `-1` enters the draft batch and the slot never comes back. The log line `spec draft: llama_decode[1] returned -1` names the drafter because that is where the poisoned batch is consumed, not where it originates — an easy thing to misread, and worth keeping straight.

The suspect is v3.0 Q3_K_XL's four 2-bit tensor types (IQ2_S, IQ2_XS, IQ2_XXS, Q2_K), which appear in none of the other three files, producing degenerate logits that make the sampler return `-1`. **Untested** — confirming it means requantizing the same tensors without the 2-bit types and re-running, which was not done.

Note this also means **`/health` returns 200 for a permanently wedged slot** — same lesson as the 2026-08-23 Nemotron wedge (finding #35). Any watchdog for a Q3_K-class slot needs the intra-request log detector, not `/health`.

## Trap: the MTP head is not Q4_0

`MTP/mtp-Qwen3.8-27B-Q4_0.gguf` was **renamed** from `mtp-Qwen3.8-27B-Q4_K.gguf` on 2026-08-19 and its contents match the old name, not the new one:

```
Q3_K  2 tensors  2.54 G-elems
Q4_K  4 tensors  0.32 G-elems
Q6_K  4 tensors  0.10 G-elems
F32   8 tensors
```

By element count it is overwhelmingly **Q3_K** (2.54 of 2.96 G-elems, 86%), so neither filename describes it well. It shares a filename with the ggml-org head our parked launcher uses (`mtp-Qwen3.8-27B-Q4_0.gguf`, 1.68 GB) but is a different file (1.37 GB) with different contents. **Do not assume the two are interchangeable because the paths match.** All four arms above used this unsloth head, held constant, so it does not confound the comparison.

## What this changes

- **Prefer Dynamic 3.0 when you care about output fidelity**, especially at 3-bit and below, where the top-1 gain is real. Prefer **2.0 when you care about tok/s on Battlemage** — it is roughly 10–14% faster at Q4_K_XL. If a workload is sensitive to worst-case tokens rather than average ones, 99.0% Δp goes the other way and 2.0 wins there too.
- **Avoid v3.0 `UD-Q3_K_XL` entirely on this stack** until the `-1` token is understood. If it must be used, run it without a drafter.
- **Qwen 3.8-27B stays parked**, but the parked row's numbers are misleading and have been corrected — see [`qwen-3.8-27b.md`](qwen-3.8-27b.md).

## Reproducing

Everything is in `/data/llm/benchmarks/20260826-ud3vs2/` on `llm.local`:

| Script | Does |
|---|---|
| `fetch-q8.sh` | Q8_0 referee + MTP head |
| `fetch-quants.sh v3 \| v2` | matched pairs, v2 pinned to `f1bfb127c64f` |
| `chain.sh` | serialises fetch → KLD base → delete referee → fetch v2 → bench (disk is the binding constraint: 91 GB of models against 102 GB free, so Q8_0 is deleted the moment its logits exist) |
| `ppl-run.sh` | one perplexity/KLD run in a throwaway container |
| `run-ud3vs2.sh` | quality then throughput, all four arms |
| `probe-decode.py` / `run-probe-pass.sh` | the `ignore_eos` decode probe |
| `final-pass.sh` | drafter-vs-no-drafter isolation for the wedge |
| `gguf-types.py` | per-tensor quant-type histogram from a GGUF header |
| `bench-per-run.json` | per-run tok/s and `predicted_n` for the first pass — the evidence for the EOS contamination |
| `probe-*.serverlog` / `probe-*.log` | server-side logs, including the `invalid token[0] = -1` traces |
| `diag-resp-*.json` | the four completions that did **not** reproduce the wedge |
| `evidence-manifest.txt` | file sizes, HF blob sizes for both revisions, LFS SHA256s, and the raw `gguf-types.py` output behind every histogram here |

The KLD base logits (`kld-base.dat`, 15.2 GB) cost ~127 MB per chunk, which is what caps chunk count — 120 chunks was chosen to fit alongside four model files, not for statistical reasons. Nemotron `:8011` is torn down for the whole run and restored by an EXIT trap; a `systemd-run --on-active=3h` safety restore was armed alongside it in case the driving session died.
