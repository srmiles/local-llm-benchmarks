# E1 — Weight-layout bandwidth microbenchmark (Intel Arc B60)

**Owner:** `e1@llm` · **Requested by:** `perf@llm` · **Est:** ~2 GPU-hours · **No llama.cpp changes**

You have none of the originating conversation's context. Everything needed is below.

---

## 1. Why this exists

On an Intel Arc Pro B60 running Gemma 4 E2B Q4_0 under llama.cpp's SYCL backend, token
decode achieves roughly **one third of the card's 456 GB/s memory roof**, while execution
units sit at **26.6% active against 79.2% thread occupancy**. Threads are resident and
stalled on memory. Occupancy is not the problem.

There are two candidate explanations and they lead to completely different engineering:

| Hypothesis | Mechanism | Fix if true |
|---|---|---|
| **H1 — Layout** | GGUF quant blocks interleave scales with quantised weights, forcing strided/masked reads that never coalesce | Store weights planar (`[qs][scales][dm]`), which is a storage-format change |
| **H2 — Latency** | Reads coalesce fine, but too few loads are in flight per thread to hide DRAM latency | Deepen software pipelining / SLM prefetch; layout is irrelevant |

**This experiment discriminates between them, in isolation, without a model or a server.**
It is the gate on a much larger piece of engine work, so it must be cheap and unambiguous.

Relevant prior finding: llama.cpp already has an opt-in SoA reorder that deinterleaves quant
blocks, and enabling it correctly for wide batches was worth 2.76–3.24× on this hardware. That
is suggestive of H1 but does not isolate it — the reorder changes kernel selection as well as
layout. Hence a standalone harness.

---

## 2. Deliverable

A single table, plus a one-line verdict. Nothing else is required.

```
layout                  access      GB/s     % of 456    vs interleaved
bf16 (control)          stream      ---.-      --.-%          --
q4_0 interleaved        stream      ---.-      --.-%        1.00x
q4_0 planar             stream      ---.-      --.-%        -.--x
q4_k interleaved        stream      ---.-      --.-%        1.00x
q4_k planar             stream      ---.-      --.-%        -.--x
q4_0 interleaved        gemv        ---.-      --.-%        1.00x
q4_0 planar             gemv        ---.-      --.-%        -.--x
q4_k interleaved        gemv        ---.-      --.-%        1.00x
q4_k planar             gemv        ---.-      --.-%        -.--x
```

### Decision rule — apply it explicitly and state the outcome

- **planar ≥ 1.15× interleaved on the `gemv` rows** → H1 confirmed. Layout is the lever.
  Report the ratio; it is the upper bound on what a planar storage format can buy.
- **planar < 1.15× interleaved** → **H1 is dead.** Say so plainly. The stall is latency, not
  coalescing. Recommend pivoting to a prefetch-depth study (vary loads-in-flight per thread at
  fixed layout) and do not start layout work.

The `gemv` rows decide it. The `stream` rows are there to show whether any difference survives
without the dequantise arithmetic, which tells you whether the cost is in the access pattern or
in the unpacking.

---

## 3. Environment

**Host:** `llm.local` (SSH alias already configured). Work in `/data/llm/build/e1-bandwidth/`.
103 GB free on `/`.

**Container — you must build and run inside it.** SYCL will not work on the bare host, and a
previous investigation silently produced zero results by forgetting this.

```
intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04   (already pulled locally)
```

Run pattern:

```bash
docker run --rm -it \
  --device /dev/dri/card1 --device /dev/dri/renderD129 \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  -v /data/llm/build/e1-bandwidth:/work -w /work \
  -e ZE_AFFINITY_MASK=0 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04 \
  bash -lc 'source /opt/intel/oneapi/setvars.sh >/dev/null && <your command>'
```

Compile ahead-of-time for the target so you are not measuring JIT:

```bash
icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g21" \
     -o e1 e1.cpp
```

If AOT fails, fall back to plain `-fsycl` **but discard the first two timed repetitions** —
a cold SYCL kernel cache costs 60–80 s and has produced confidently wrong numbers here before.

---

## 4. Card selection and the operational rules

Both cards carry production workloads. **This is a live homelab, not a test rig.**

| Card | `ZE_AFFINITY_MASK` | Device nodes | Resident workload | Free VRAM |
|---|---|---|---|---|
| 0 | `0` | `card1` + `renderD128` | chat :8002, embed :8004, categorise :8006 | ~4.2 GB |
| 1 | `1` | `card2` + `renderD129` | Nemotron :8007 | ~2.3 GB |

Note the off-by-one: **`/dev/dri/card1` is GPU 0**. Confirm with `xpu-smi discovery` before
trusting either.

**Prefer card 1** — a single tenant, easier to catch idle. Size the buffer to
`min(1.5 GB, 60% of free VRAM)` and assert it allocated before timing.

### Hard rules — violating these has already cost a host reboot

1. **Never `docker kill` or `docker rm -f` a container holding a GPU.** It wedges the card into
   `UR_RESULT_ERROR_DEVICE_LOST` and only a host reboot recovers it. Use
   `/data/llm/launch/gpu-teardown.sh <name> [grace]`.
2. **Do not stop, restart or reconfigure any production container.** This experiment needs none
   of them touched.
3. **Verify idle before every timed run** and record it alongside the result:
   ```bash
   timeout 8 xpu-smi dump --device 1 --metrics m,p --number 1 | tail -1
   curl -s -m 5 http://127.0.0.1:8007/metrics | grep -E 'requests_processing|requests_deferred'
   uptime
   ```
   Require board power < 40 W, `requests_processing 0`, load average < 2. If the card is busy,
   wait — do not measure through contention. A benchmark taken during a concurrent build
   produced numbers ~40% low earlier in this project.
4. If a run wedges the card (`DEVICE_LOST`, or `dmesg` showing `Engine reset: engine_class=ccs`),
   **stop and report**. Do not attempt recovery beyond letting the existing watchdog act.

---

## 5. What to build

One C++/SYCL file, `e1.cpp`. Roughly 300 lines. No dependencies beyond the oneAPI toolchain.

### 5.1 Data layouts

Generate synthetic buffers matching GGML's on-disk formats — do not read a real GGUF, the point
is the access pattern, not the values.

**`q4_0` interleaved** — GGML native, 18 bytes per 32 weights:
```c
struct block_q4_0 { uint16_t d; uint8_t qs[16]; };   // fp16 scale, then 32 x 4-bit
```

**`q4_0` planar** — same bytes, two separate allocations:
```
qs_plane[nblocks * 16]      // all quantised nibbles, contiguous
d_plane [nblocks]           // all scales, contiguous
```

**`q4_k` interleaved** — 144 bytes per 256 weights:
```c
struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
```

**`q4_k` planar** — three planes, matching the reorder llama.cpp performs:
```
qs_plane[nblocks * 128]  |  scales_plane[nblocks * 12]  |  dm_plane[nblocks * 2]
```

**`bf16` control** — a flat `uint16_t` array of the same *element* count. This is the practical
ceiling for the harness; if it does not reach 380–420 GB/s, the harness itself is the bottleneck
and every other row is meaningless. **Fix the control before reporting anything.**

### 5.2 Access patterns

**`stream`** — read every byte, accumulate into a sink the compiler cannot elide (a
`sycl::reduction` into a device scalar, or an atomic add of a cheap running XOR). Isolates raw
bandwidth from unpacking cost.

**`gemv`** — the real decode pattern. One row of weights dotted against an activation vector held
in shared local memory, one sub-group per row, dequantising as it goes. This is the row that
decides the experiment.

### 5.3 Parameters to sweep

Report the best configuration per layout so a bad launch geometry cannot be mistaken for a bad
layout.

- sub-group size: `16`, `32` (`[[intel::reqd_sub_group_size(N)]]`)
- rows per work-group: `1`, `2`, `4`
- work-group size: `128`, `256`, `512`
- optionally: `sycl::ext::oneapi::experimental::` sub-group block loads vs plain indexed loads —
  this is a direct probe of the coalescing question and worth including if time allows

### 5.4 Timing discipline

- Buffer must exceed last-level cache by a wide margin — at ≥ 1 GB this is satisfied comfortably.
- Warm: 3 untimed iterations. Measure: 10 timed. **Report the median**, and also the min and max
  so variance is visible.
- Time with SYCL events (`command_start` → `command_end`), not wall clock.
- `GB/s = bytes_read / seconds`, where `bytes_read` counts only the weight bytes actually
  touched — be explicit in the output about what is counted, so the planar and interleaved rows
  are comparing like with like. This is the easiest place to produce a fake result: planar and
  interleaved must move the **same number of weight bytes**.

---

## 6. Steps

1. Confirm the environment: `xpu-smi discovery`, map GPU index to `/dev/dri/*`, record driver and
   Level Zero versions.
2. Write `e1.cpp`. Get the **bf16 control** correct first and confirm it lands in the 380–420 GB/s
   band before building anything else. If it does not, the rest is noise.
3. Add `q4_0` interleaved + planar, `stream` pattern. Sanity-check that both report identical
   byte counts.
4. Add the `gemv` pattern for both.
5. Add `q4_k` interleaved + planar for both patterns.
6. Sweep the launch parameters; keep the best per layout.
7. Verify idle, run the full matrix, produce the table.
8. Apply the decision rule in §2 and state the verdict in one line.

---

## 7. Report back

Reply to this handoff (`action="reply"`, `from="e1@llm"`) with:

- the results table
- the verdict against the 1.15× threshold, stated plainly either way
- the bf16 control figure, so the harness can be trusted
- the idle-state readings captured alongside the timed runs
- the path to `e1.cpp` and the exact build command used
- anything that surprised you — particularly if `stream` and `gemv` disagree about which layout
  wins, which would be genuinely informative and is not the expected outcome

**A clean negative result is a full success here.** It saves days of engine surgery. Do not
massage the numbers toward H1 because the framing leans that way — the entire point of running
this in isolation is that the answer is trustworthy either way.

---

## 8. Method rules for this project

Earned expensively on 2026-08-27; each of these produced a wrong conclusion before it was caught.

- **Launch counts are not time.** Attribute by time-weighted profiling, never by kernel counts.
- **Power and clock are not occupancy.** Max clock at 100 W looked saturated while a third of the
  EU array was empty.
- **Vector engines are not XMX.** 20 Xe cores, each with 8 vector engines *and* 8 XMX engines.
  Do not conflate them.
- **Check the roofline before writing a kernel.** Ten minutes; it has already killed a multi-day
  plan on this project.
- **A result is only valid for the shape it was measured on.** State the shape with the number.
- **Verify idle, warm, and unrestarted before trusting any measurement.** Cold caches, concurrent
  builds and mid-run container restarts each produced a confidently wrong number here in one day.
