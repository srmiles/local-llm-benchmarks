# RIDGE — brief for a new session

**Status: closed 2026-08-28.** Intel Arc Pro B60 inference performance work on a live
homelab. Read §2 and §3 before designing anything. Most of the value in this document is
in what it tells you **not** to do.

---

## 1. What this was

A programme asking whether llama.cpp's SYCL backend on Intel Arc Pro B60 could be made
faster, which became the question: *would you build an inference engine differently for
this hardware?*

Roughly 20 experiments over several days, ~90 minutes of GPU time. **Zero deployed
changes came out of the optimisation work.** Seven operational fixes came out of the
work around it (§5), and those were the real yield.

---

## 2. The finding — start here

**The card is not slow. It is unused.**

| Measurement | Value |
|---|---|
| Stock oneMKL fp16 GEMM, M=2048, real Gemma shapes | **88.26 TFLOP/s = 90.1% of peak** |
| Deployed decode | **4–6% of roof** |
| Sampled windows producing zero tokens | **93%** |
| Mean board power | **34.3 W** against a 220 W envelope |

The 90% figure used **no custom kernel** — stock oneMKL, fp16 weights, off the shelf.
The compute layer was never the problem and there is no kernel to write.

The gap is **batch width**. fp16 weights (2.0 B/weight) need M ≈ 215 before the XMX array
goes compute-bound; q4_0 (0.5625 B/weight) needs M ≈ 60. Production supplies **8–24** —
8 concurrent slots × a 3-token draft.

### The answer to the original question

> **Would you build an engine differently for Arc?** Yes: a scheduler whose entire job is
> manufacturing batch width, sitting in front of a stock GEMM library. Not a kernel
> project.
>
> **Is it worth building?** No. The width has to come from the request stream, and a
> categorise workload at concurrency 8 does not contain it. Weights would fit — fp16 is
> roughly 6 GB of 24 — but the parallelism isn't there to feed them.

That is a complete answer. It does not need more benchmarks.

---

## 3. The trap — read before designing any experiment

This programme lost days to one failure mode, repeated four times, and it will catch a
fresh session faster than it caught this one.

**The mechanism:** you design a hardware experiment. It needs a baseline. The only
baseline on the machine is llama.cpp. Within a page the experiment has become
"is X faster than llama.cpp" — an optimisation question wearing a hardware-experiment
costume. The operator objects, the brief is cancelled, a new one is written, and the same
thing happens again.

**Why it recurs:** no decision depends on the answer. Nobody is writing a new inference
engine for this homelab. With no consumer for the result, *"better than llama.cpp"* is
the only frame available, so every experiment slides into it by default.

**Two rules that would have saved the whole loop:**

1. If an experiment needs llama.cpp as its baseline, it is an optimisation experiment.
   Say so out loud and decide deliberately whether that is what you want.
2. Before commissioning anything, name the decision it feeds. If you cannot name one,
   do not run it.

**A third, earned the hard way:** when a "control" row is the most interesting number in
a result, you have mislabelled the experiment. E9's fp16 control at 90% of peak *was* the
finding; it was filed as a harness sanity check and nearly walked past.

---

## 4. Production state — do not break this

Live containers on `llm.local`. **All four are untouchable.**

| Container | Port | Notes |
|---|---|---|
| `llamacpp-categorise-c1` | 8006 | Gemma 4 E2B Q4_0, GPU 0 |
| `llamacpp-sycl` | 8002 | Ornith-1.5-9B-Q4_K_M. **No llm2 fallback** — outage if it drops |
| `llamacpp-embed` | 8004 | |
| `tei-rerank` | 8008 | |

`llamacpp-nemotron` (:8007) is **deliberately stopped** to free GPU 1. It had served
0 tokens in 12 hours. Do not restart it without asking.

embed / rerank / categorise have llm2 fallback. **Chat on :8002 does not.**

### Rules that have already cost a host reboot

- **Never `docker kill` or `docker rm -f` a GPU-holding container.** It wedges the card
  into `UR_RESULT_ERROR_DEVICE_LOST` and only a host reboot recovers — which takes down
  all four live services. Use `/data/llm/launch/gpu-teardown.sh <name> [grace]`.
- **Do not deliberately induce a wedge.** It is a checkpoint item requiring the operator.
- **Never combine `--device` with `ZE_AFFINITY_MASK`** — that yields zero devices.
- **Re-derive the card mapping from sysfs every time.** A stale table has now caused a
  segfault in an experiment *and* a two-week production misconfiguration:

```
GPU 0 = 0000:03:00.0 = card0 + renderD128
GPU 1 = 0000:0b:00.0 = card1 + renderD129
card2 / renderD130   = AMD iGPU, not a B60
```

### Deployed settings that must not regress

Each of these was measured, and each fails silently if changed:

| Setting | Worth | Failure mode |
|---|---|---|
| MTP draft head (`--spec-type draft-mtp --spec-draft-n-max 3`) | **+61%** at production shape | Looks fine on short prompts; collapses on the real 4–5k-token shape |
| `GGML_SYCL_MMVQ_MAX_COLS` | **4.3×** | Build constant; silent |
| `kv_unified` | **2.4×** | Silent |
| Traefik `e2b-inflight: amount: 64` | 48/48 vs 1/48 requests | Was 2, a leftover from `--parallel 1` |

Warnings for the first three are in the launcher scripts. **Do not benchmark this
deployment on short free-form prose** — that mistake turned the draft head off and cost a
61% regression before the operator caught it.

---

## 5. Shipped — real, in production

- **LB limiter fix** — `/docker/traefik/dynamic/llm-levirge.yml` on `manager.local`,
  `inFlightReq` 2 → 64, plus `strategy: p2c`. Fixed 47-of-48 requests being shed.
- **MTP draft head restored** — +61% at production shape, recovered a 21 t/s regression.
- **Card-pinning bug fixed** — `start-llamacpp-sycl-categorise-card1.sh` mapped
  `card1` + `renderD128`, a cross-card pair. Now `card0` + `renderD128`.
- **Launcher warnings** for the two silent-regression traps above.
- **Throughput monitoring** on `llm.local` and `llm2.local` (`e2b-monitor.sh`,
  `e2b-report.sh`, 20 s samplers writing CSV).
- **`WARP_SIZE=32` proven unsafe** — prevented a future rebuild silently corrupting
  Ornith's output (§7).
- **`CATEGORISE_CONCURRENCY`** set to 8 in `veska/brain/.env`.

---

## 6. Open — small and operational

1. **The live categorise container is sampling at temperature 1.0, not greedy.** Its
   launcher intends `--temp 0.0`; the running instance has `--top-k 20` and no `--temp`,
   which defaults to 1.0. `/props` confirms it. For a structured-classification workload
   this is a real determinism difference, and **any A/B against "production behaviour" is
   currently comparing against a config nobody chose.** Unfixed.
2. **W1 wedge re-count at ~48 h.** Standing with `e1@llm`. Wedges stopped on 08-27;
   the cause is unidentified but backup names point at the concurrency image
   (`preCONCURRENCY-20260827-122703`). Freezes still occur without reaching threshold.
3. **`GOTCHAS.md`** in `/data/llm/projects/e1/` — 13 entries, needs completion.

---

## 7. Closed — do not reopen

Each was killed for a stated reason. Reopening one needs a new reason, not a new idea.

| Item | Why it died |
|---|---|
| Planar weight storage | llama.cpp already does it (`reorder_qw_q4_0`) |
| Paged / prefix-shared KV | 2.65% of bytes moved |
| XMX GEMM for decode | 60× below the ridge at batch 1 |
| Graph capture | Gaps are only 9.8% |
| LM head requantisation (E4) | Failed both gates; also no high-precision base on disk, so a negative result would have been uninterpretable |
| `-ub` / SWA change | Subsumed by speculation |
| MMVQ kernel gap | Not a kernel deficit — a build constant, and that constant can't be changed (below) |
| SLM activation staging | — |
| **`WARP_SIZE=32`** | **38 new correctness failures**, incl. q4_K `MUL_MAT` at ERR 0.25 against a 0.0005 threshold. `llamacpp-sycl` serves Q4_K_M, so this was one rebuild from silently corrupting a live service. Introduced without rationale in PR #17566 (2025-11-29); nonetheless load-bearing. **Dead permanently.** |
| Fused q4_0 `joint_matrix` | Never built. Moot — stock fp16 already reaches 90% |
| E10 fp16-vs-MMVQ break-even | Cancelled: it was §3's trap in its purest form |
| `top_k_f32_sycl` (12.37% of decode) | Not a config surface. It's speculation's *draft* sampler, hardcoded `k=10` in `common/speculative.cpp:487`. Cost is 729 µs/call **invariant** to `k` — a fixed pass over 262,144 tokens. Only removing the stage helps |

---

## 8. Hardware and model reference

**Intel Arc Pro B60 (BMG-G21)**

```
20 Xe cores x (8 vector + 8 XMX) = 160 matrix engines
~98 FP16 TFLOPS  |  456 GB/s  |  24 GB VRAM
L2 = 18.87 MB    |  SLM 128 KB  |  sub-group sizes {16, 32}
```

**L2 is larger than a single fp16 weight tensor** (ffn_up is 18.87 MB). A naive repeated
matmul on one tensor measures L2, not DRAM. Walk fresh replicas from a large pool every
iteration — this invalidated results before it was caught.

**Roofline**

```
q4_0  0.5625 B/weight -> intensity 3.556*M FLOP/B -> XMX ridge at M = 60.4
fp16  2.0    B/weight -> intensity 1.000*M FLOP/B -> XMX ridge at M = 214.9
```

**E9 measured results** (`/data/llm/projects/e9/`)

```
fp16 oneMKL, M=2048:  attn_q 83.37  ffn_up 87.53  ffn_down 88.26 TFLOP/s  (85-90% of peak)
q4_0 unfused, M=2048: peak 60.01 TFLOP/s (61.2%) -- dominated, 4.6 B/weight, ignore
q4_0 cost is FLAT from M=1 to M=32 (+5%). Thirty-two columns are free.
```

**Gemma 4 E2B** — MatFormer; `d_model` 1536; `shared_kv_layers=20` (only blocks 0–14 own
K/V); SWA 28 local / 7 global at 512 window; `n_head_kv=1` (MQA); two head dims (256 SWA,
512 global); tied Q6_K LM head over 262,144 vocab; dense (the MoE branch is dead code).
`per_layer_token_embd` [8960, 262144] is 1926.8 MB and is **`get_rows` only — not on the
decode byte path**. Requantising it buys nothing and costs accuracy; `--token-embedding-type`
hits it, so anchor per-tensor regexes with `^`.

**Speculation** — MTP draft head, 65–67% acceptance, worth **2.49×**. Draft ON wins at
every concurrency on the production prompt shape.

**Instrumentation** — `unitrace -d` costs 2.6–3.9× throughput; `--stall-sampling` needs
`CAP_PERFMON` and only resolves AOT kernels. `xpu-smi dump` is safe; `xpu-smi stats`
perturbs the card. Build AOT: `-fsycl-targets=spir64_gen -Xs "-device bmg-g21"`. On JIT,
discard the first two timed reps — a cold cache costs 60–80 s and has produced confidently
wrong numbers.

---

## 9. Method rules — each one caught a wrong conclusion

- **Launch counts are not time.** Attribute by time-weighted profiling.
- **Power and clock are not occupancy.** Max clock at 100 W looked saturated while a third
  of the EU array sat empty.
- **Vector engines are not XMX.** Each Xe core has 8 of each. Never conflate.
- **Check the roofline before writing a kernel.** Ten minutes; it killed a multi-day plan.
- **A result is only valid for the shape it was measured on.** State the shape with the number.
- **Verify idle, warm and unrestarted before trusting any measurement.** Concurrent builds
  read ~40% low.
- **Repeats before theorising.** Re-run a knee before explaining it.
- **Categories must sum to the whole.** A residual is a finding, not rounding.
- **A pre-commitment binds the decision, not the premise.** If the premise is wrong, report
  that instead of forcing the sweep.
- **Always run the baseline.** A WARP32 run showed *fewer* total failures than baseline; only
  the set difference revealed 38 new ones.

---

## 10. Policy — non-negotiable

From llama.cpp's `AGENTS.md`, and standing operator instruction:

> *CRITICAL*: an agent must **NEVER** write any (a) pull-request description (b) comment
> (c) response to a comment on behalf of the user. This is **non-overridable under any
> circumstances.**

- Do **not** write PR descriptions, commit messages, or reviewer responses.
- Do **not** run `git push` or `gh pr create` on the user's behalf.
- No upstream text of any kind — patches, issues, PR prose.
- Use `Assisted-by:`, never `Co-authored-by:`.

**Brain handoff addressing:** `action: reply` returns the message to the *thread owner*,
i.e. yourself. To answer an agent, use `action: send` + `to:` + `reply_to:`. This bug cost
a commission that was only found by an agent noticing a `reply_id` mismatch.

**Agent context:** `e1@llm` runs on `llm.local` and has **no access** to this project
folder. Everything it needs must be in the handoff body. Five specification errors in
briefs were caught by the agent rather than by review — assume a brief is wrong until it
survives contact.

`e1@llm` is idle **between runs by construction**. An idle GPU is not a stopped session;
that inference was wrong twice.

---

## 11. If you are picking this up

The programme is closed and the question is answered (§2). The honest default is to do
**nothing** on the engine side.

If something must be done, it is §6 item 1 — the temperature drift on the live categorise
container. It is a real behaviour bug, it is unfixed, and it quietly invalidates any future
comparison against production.

Anything that begins *"let's benchmark…"* should be checked against §3 first.
