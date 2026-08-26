# SYCL backend patches + default-bench re-run — 2026-08-26

Four changes to `ggml-sycl`, written and measured on the B60 pair, then the 2026-08-21 five-model default bench replayed against them.

**Branch:** `all-fixes` off `bb4caa754` (b10566), in `/data/llm/build/llama.cpp`. Patches exported to `/data/llm/build/upstream-patches/`.

| commit | change |
|---|---|
| `68e55f242` | MoE expert-tensor reorder for **Q4_0 / Q8_0** |
| `9d1f04528` | MoE expert-tensor reorder for **Q3_K** |
| `ad76cd30a` | report MUL_MAT/MUL_MAT_ID **unsupported for TQ1_0/TQ2_0** |
| `419f85698` | **chunk MMVQ** above `MMVQ_MAX_BATCH_SIZE` instead of falling back to dequant+GEMM |
| *(added later)* | **`test-backend-ops`: cover Q3_K** in the `MUL_MAT_ID` perf cases |

> **Commit hashes above are from the first pass and have since been rebased.** Two review items were fixed after reading `CONTRIBUTING.md`/`AGENTS.md`: `opt_for_reorder_id` now reuses the existing `ggml_sycl_supports_reorder_mmvq()` predicate instead of a duplicate switch, and the over-long comments were cut to house style. Current series and its compliance blockers: [`configs/patches/`](../../configs/patches/README.md).

Images: `llama.cpp:sycl-f16-moereorder` (first commit only — **deployed to Nemotron `:8011`**), `sycl-f16-q3kmoe`, `sycl-f16-allfixes`.

---

## 1. The gap: MoE expert tensors never got the reorder that dense tensors have

`reorder_qw()`'s dense switch covers seven quant types. Its MoE branch (`ne[2] > 1`) covered three. Expert weights dominate per-token weight traffic in a sparse MoE, and they were the ones left out.

| type | reordered GEMV kernel exists | dense reorder | MoE reorder |
|---|:--:|:--:|:--:|
| Q4_0, Q8_0 | yes | yes | **added** |
| Q3_K | yes | yes | **added** |
| Q4_K, Q5_K, Q6_K | yes | yes | upstream |
| Q2_K | **no** | yes | still missing |

**The MoE work is only cheap when `reorder_vec_dot_q_sycl<T>` already exists** — that is the hard part. Q2_K is the sole remaining gap and needs a new GEMV kernel, so it is a different size of job.

Two traps, both of which produce silent corruption rather than a crash:

- The dense `reorder_qw_q4_0/q8_0/q3_k` take an `offset` parameter that looks like a per-expert hook. It is not — they `memcpy` from the start of the tensor regardless of offset and derive the scales pointer from an already-offset base. Dedicated slice-aware kernels are required.
- **The MoE reorder dispatch has no `GGML_SYCL_DEBUG` print.** An absent log line proves nothing. Verification has to be behavioural.

## 2. Per-op measurement, and how engagement was proven

The images are built `GGML_BACKEND_DL=ON`, so `test-backend-ops` was built **once** and pointed at each *shipped* `libggml-sycl.so` through `GGML_BACKEND_PATH`. What is measured is the production binary, not a parallel build.

`perf -o MUL_MAT_ID -b SYCL0`, GFLOPS, two shapes per type:

| type_a | n | baseline | patched | ratio |
|---|---:|---:|---:|---:|
| **q4_0** | 1 | 343.5 / 346.5 | 1760 / 1870 | **5.12× / 5.40×** |
| **q8_0** | 1 | 181.7 / 158.5 | 1170 / 920.6 | **6.44× / 5.81×** |
| q4_K *(control)* | 1 | 1150 / 1180 | 1160 / 1210 | 1.01× / 1.03× |
| q6_K *(control)* | 1 | 1040 / 1050 | 1040 / 1050 | 1.00× |
| q4_0 | 4 | 293.9 | 277.7 | 0.95× |
| q4_0 | 8 | 312.2 | 280.5 | 0.90× |
| q4_0 | ≥32 | — | — | 0.98–1.01× |

Three things fall out of this table at once:

1. **The reorder engages.** A 5–6× jump confined to exactly the two types the patch adds is not what a no-op produces. This matters because the code has no debug output — a passing correctness run could not distinguish "reordered" from "fell back".
2. **The already-reordered K-quants are a built-in control** and stay at 1.00–1.03×.
3. **The gain exists only at `n = 1`.** Everything at n ≥ 4 is flat. Keep this — section 5 turns on it.

Scattered ~10% dips at n=4/8 are harness noise, not regressions: untouched types move as much (`iq2_xs` 0.84×, `mxfp4` 0.90×).

**Q3_K's per-op number, obtained later.** `perf -o MUL_MAT_ID` originally had no `q3_K` shape at all, so Q3_K had only end-to-end evidence. Adding one (patch 0003) fixed that — and because the test binary is independent of the backend, the *new* cases can be pointed at the *old* `libggml-sycl.so`: **q3_K n=1 goes 252.6 → 401.1 and 253.8 → 404.6 GFLOPS, a consistent 1.59×.** Much smaller than q4_0's 5.16×, but applied across 90 of Ornith APEX's 123 expert tensors, which is what the +21.4% end-to-end reflects.

**A larger version of the same trap, worth recording.** A later verification pair, run on card 0 *while production was live*, showed ratios collapsing to 0.15–0.28 at large n. Not a regression — the effect is positional and hits every type:

| shape group | n=1 | n=8 | n=256 |
|---|---:|---:|---:|
| m=768 *(runs first)* | 1.03 | 1.04 | 1.11 |
| m=1792 *(runs second)* | 1.04 | 0.91 | **0.21** |

At m=1792/n=256 the untouched types collapse identically — **f32 0.34, f16 0.76, q4_K 0.20, q6_K 0.21, iq2_xs 0.19** — and these patches cannot touch f32 `MUL_MAT_ID` at all. The tail of the second run was starved by prod load on the shared card. **Run perf comparisons on an idle card, and always keep an untouched type in the table.**

## 3. End-to-end: three models, interleaved A/B against the immediate parent commit

Interleaved rather than blocked, so thermal or clock drift cannot masquerade as the effect.

**Nemotron 3.5 Lightning 30B-A3B Q4_0** — code prompt, `n-max 7 / p-min 0.6`:

| build | decode t/s | median |
|---|---|---:|
| baseline | 57.98 / 58.46 / 58.75 | 58.46 |
| patched | 73.06 … 83.58 (8 samples) | **77.26** |

**+32%.** Worst patched sample beats best baseline by 24%.

**Gemma 4 26B-A4B QAT Q4_0** (60 Q4_0 expert tensors) — `n-max 5 / p-min 0.6`, so verify batch 6 stays *below* the batch-8 gate and the MMVQ patch cannot contribute:

| round | build | decode med | σ | acceptance | peak VRAM |
|---|---|---:|---:|---:|---:|
| 1 | baseline | 65.84 | 4.50 | 92.8% | 17.69 GiB |
| 1 | **patched** | **82.14** | 7.79 | 85.0% | 17.69 GiB |
| 2 | baseline | 64.35 | 6.08 | 86.9% | 17.69 GiB |
| 2 | **patched** | **85.40** | 8.07 | 90.0% | 17.69 GiB |

**+28.7%** on mean-of-medians; **+32.7%** on a code prompt (16.735 → 12.61 ms/token). Round 1 is the strong evidence: the patched run had *lower* acceptance than its baseline and was still 25% faster, so the gain cannot be a drafting artefact.

**Ornith-1.5-35B APEX** (`qwen35moe`; 90 of 123 expert tensors are Q3_K), control = the Q4_0/Q8_0 commit so Q3_K is the only variable:

| round | control | Q3_K |
|---|---:|---:|
| 1 | 39.58 | **47.82** |
| 2 | 39.28 | **47.92** |

**+21.4%**, acceptance and VRAM flat.

**Internal consistency check worth noting:** this same model moved **+1.6%** when only Q8_0 was covered (3 expert tensors) and **+21.4%** with Q3_K (90 tensors). The gain tracks the share of expert weight traffic reordered — which is what the mechanism predicts and is much harder to fake than a single number.

Peak VRAM is identical to the byte in every pairing — the reorder is in-place.

## 4. The TQ guard, and the first complete op suite on this backend

`ggml_backend_sycl_device_supports_op()` returns `true` for MUL_MAT/MUL_MAT_ID on *any* `src0` type, but the backend cannot execute `TQ1_0`/`TQ2_0` by any route:

```
ggml/src/ggml-sycl/mmvq.cpp:2488:   fatal error: unsupport data type=tq2_0   (MMVQ)
ggml/src/ggml-sycl/convert.cpp:724: fatal error: unsupport data type=tq2_0   (dequant+GEMM fallback)
```

Gating `can_use_mul_mat_vec_q()` is **not** sufficient — that only moves the abort from `mmvq.cpp` to `convert.cpp`. The check belongs in `supports_op`, so ggml falls back to CPU.

| suite | before | after |
|---|---|---|
| `test -o MUL_MAT -b SYCL0` | rc=139, aborts at 214 lines | **rc=0, 1022/1022 passed** |
| `test -o MUL_MAT_ID -b SYCL0` | rc=139, aborts at 93 lines | **rc=0, 869/869 passed** |

This appears to be the first complete MUL_MAT/MUL_MAT_ID result for the SYCL backend. It is also a data point against upstream **#25455** (MUL_MAT_ID returning wrong results on a B70, 28/792 failing): on a healthy B60 all 869 pass.

## 5. Default bench replayed — and why Nemotron shows only +3.2% here

The 2026-08-21 five-model bench, replayed on `sycl-f16-allfixes`. Arm **replay** uses the 21 Aug settings exactly (n-max derived from the stored counters: `draft_tokens/drafts` = 2.99 → n-max 3, 6.90 → n-max 7; no p-min; `MMVQ_MAX_COLS` at its default 8). Arm **tuned** uses n-max 9 with `MMVQ_MAX_COLS=32`.

| model | 21 Aug | replay | Δ | tuned | Δ |
|---|---:|---:|---:|---:|---:|
| ornith-1.5-9b *(dense — control)* | 65.15 | 73.57 | +12.9% | 42.92 | −34.1% |
| qwen3.8-9b *(dense — control)* | 73.97 | 74.46 | +0.7% | 71.83 | −2.9% |
| lfm2.5-8b-a1b *(Q4_K/Q6_K experts — control)* | 168.25 | 184.24 | +9.5% | 184.59 | +9.7% |
| ling-3.0-tiny | 91.86 | 88.84 | −3.3% | — | — |
| **nemotron 30B-A3B** *(Q4_0 experts)* | 78.95 | 81.47 | **+3.2%** | **97.41** | **+23.4%** |

Prefill at 12K is unchanged everywhere (−0.1% to −3.8%) — nothing here touches prefill, so that is a clean fourth control.

### The reconciliation: the two fixes serve disjoint regimes

Nemotron gained **+32%** in the controlled A/B but only **+3.2%** here. Not a contradiction — section 2 predicts it exactly. The reorder gives 5.12× at **n = 1** and is flat at n ≥ 4.

- `bench-candidate.py` uses a repetitive filler prompt → **99.9% acceptance**. At n-max 3 nearly every step is a 4-wide verify: the no-gain region.
- The controlled A/B used a code prompt with `p-min 0.6` → 67–70% acceptance → drafts rejected constantly → many genuine **batch-1** decode steps.

So:

- **MoE reorder** pays where decode actually runs at batch 1 — real prompts, moderate acceptance, p-min truncation. **Production Nemotron lives here** (p-min 0.6, 63–71% on code), which is why the deployed slot measures 70.5–78.2 tok/s against ~58.5 before.
- **MMVQ chunking** pays where verify batch exceeds 8. That is the whole of the +23.4% tuned result; the reorder contributes nothing at batch 10.

**A high-acceptance synthetic prompt hides the MoE reorder completely.** Evaluated on the default bench alone, that patch looks worth 3%.

### The tuned arm is not a free win

n-max 9 **hurt** two models: ornith −34.1% (acceptance collapsed 84.7% → 45.7%) and qwen −2.9% (81.4% → 70.1%). Removing the batch-8 cliff removes a discontinuity; it does not make aggressive drafting profitable when acceptance falls away. The n-max optimum stays model-specific (findings #30, #32).

### Baseline confidence: ±10%, no better

Three controls that today's MoE work provably cannot touch moved **+12.9%, +9.5% and +0.7%**; Ling moved −3.3%. The cause is configuration, not silicon drift: **the 21 Aug run had no saved runner script**, and peak VRAM differs inconsistently across models (ornith 14.62 → 10.83 GiB, but lfm2.5 8.63 → 10.50). Ornith's 21 Aug row is also visibly degraded — σ 16.41 with one sample at 0.0 tok/s.

A runner (`default-bench-allfixes.sh`) and comparator (`compare-default.py`) are now saved on the box so future default benches are reproducible.

## 6. Prefill: the limiter, and why "flat prefill" was an artifact

The earlier observation that prefill sits flat at 2,797–2,868 tok/s across batch 1–16 was **a property of the harness, not of the backend**. That sweep varied `-npl` with `-ub` pinned at 512, and `-npl` does not change the micro-batch — so it cannot change per-token prefill cost. Flatness was the expected result.

Sweeping `-ub` instead (Qwen3-4B Q4_K_M, `-npp 4096 -npl 1 -fa on`):

| `-ub` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| prefill tok/s | 306.9 | 434.7 | 610.1 | 836.2 | 1096.2 | 1317.9 |

**A 4.3× swing from micro-batch alone**, monotonic with diminishing returns — the signature of a fixed per-forward-pass cost being amortised, not an arithmetic roof.

Order-of-magnitude check: the implied fixed cost is ~0.3–0.5 s/pass, while dequantising 4B params to fp16 and reading it back is ~16 GB at 456 GB/s ≈ 35 ms. An order of magnitude apart, so the dominant per-pass cost is probably overhead — pool alloc/free, oneDNN primitive setup, synchronisation — rather than dequant bandwidth. **Not isolated; do not treat the cause as known.** The model is dense, so the MoE prefill host-sync is not the explanation.

Two consequences: the launchers already run `-ub 2048 -b 2048`, right at the knee, so prefill tuning on this box is essentially done — and **XMX / `joint_matrix` investment stays deferred**, because prefill is nowhere near the arithmetic roof.

## Environment

```
llama.cpp   bb4caa754 (b10566) + the four commits above, .devops/intel.Dockerfile, GGML_SYCL_F16=ON
GPU         Intel Arc Pro B60 (Battlemage G21, 8086:e211), 24 GB — bench card, ZE_AFFINITY_MASK=1
OS          Ubuntu 26.04 LTS, kernel 7.0.0-29-generic, xe driver
oneAPI      2025.3, Level Zero 1.14.37020+3
harness     /data/llm/benchmarks/bench-candidate.py, 20 decode runs × 300 tokens, temp 0.6 / top-p 0.95 / top-k 20
results     /data/llm/benchmarks/20260826-{moereorder-gemma4,moereorder-ornith,q3k-ornith,q3k-ops,prefill-ub,default-allfixes}/
```

Zero GT resets across every bench window (last was 04:55, before any of them).

## Still open

- `Q2_K` expert tensors are still reordered by nothing, and unlike Q3_K that needs a new GEMV kernel rather than plumbing.
- The four patches are **not filed upstream** — `gh` is not installed on `llm.local`. Exports are committed here at [`configs/patches/`](../../configs/patches/README.md); source of truth is branch `all-fixes` in `/data/llm/build/llama.cpp`.
