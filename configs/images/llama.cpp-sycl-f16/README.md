# llama.cpp:sycl-f16

Custom llama.cpp SYCL image tuned for **Intel Arc Pro B60 (Battlemage / Xe2)**. Serves all four llama.cpp containers in the stack (chat, embed, rerank, plus any ad-hoc bench containers).

**Current tag → build:** `llama.cpp:sycl-f16` → **b10433** (2026-08-14, commit `9b05354ec` — master HEAD at time of build; includes recent SYCL Mamba/gated-delta-net optimizations)
**Rollback tags:** `llama.cpp:sycl-f16-b10256-safe`, `-b10215-safe`, `-b10068-safe`
**Explicit tags on disk:** `-b10433` (current), `-qwen38` (identical to `-b10433`), `-b10308`, `-b10256`, `-muse` (b10256+PR#26841 for Muse Glimmer), `-b10215-safe`, `-b10068-safe`

## Why we build our own

The upstream `ghcr.io/ggml-org/llama.cpp:server-intel` image ships with:
- `GGML_SYCL_F16=OFF` — costs ~26% prefill on B60 vs `ON`
- No `-DCMAKE_BUILD_TYPE=Release` — missing `-DNDEBUG` and `-O3`
- Whatever upstream tag is current — moving target, no rollback

Our build tags what we ship, keeps the previous tag around, and is 30–35 min end-to-end via `systemd-run`.

## Build flags

| Flag | Where set | Why |
|---|---|---|
| `GGML_SYCL_F16=ON` | `docker build --build-arg` | Enables FP16 SYCL math path. **+26% prefill / +4% decode** vs OFF. |
| `-DCMAKE_BUILD_TYPE=Release` | Dockerfile default | Adds `-DNDEBUG` + `-O3`. Missing this is worth ~10–15% on decode. |
| `--target server` | `docker build --target` | We only need the server binary, skip the CLI + tests + tools stages. Halves the image size. |
| oneAPI 2025.3.3 base | `intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04` | Matches the driver stack on `llm.local` (compute-runtime 26.18.38308.1, IGC v2.34.4). |

Runtime env vars set by the launchers (not in the image):
- `ONEAPI_DEVICE_SELECTOR=level_zero:0` — pins to the B60. Every launcher passes this.
- Nothing else. **No `GGML_SYCL_*` runtime env vars are set anywhere** — so the PR #25042 rename (disable → enable semantics flip) was a non-event for us.

## Build

Runs against a local checkout at `/data/llm/build/llama.cpp` on `llm.local`.

```bash
cd /data/llm/build/llama.cpp

# Fetch new tags + checkout target
git fetch origin --tags --prune
git checkout b10068                # or newer

# Verify hot commits are in the tree (SYCL wins)
for sha in 32b741c c1063ac efb3036 d3fba0c 956973c; do
  git log --oneline HEAD | grep -q "^$sha" && echo "✓ $sha" || echo "✗ $sha MISSING"
done

# Build in background (survives SSH disconnect, logs to /tmp/llama-build.log)
sudo systemctl reset-failed llama-build 2>/dev/null
sudo systemd-run --unit=llama-build \
  --property=StandardOutput=file:/tmp/llama-build.log \
  --property=StandardError=append:/tmp/llama-build.log \
  --property=WorkingDirectory=/data/llm/build/llama.cpp \
  /usr/bin/docker build \
    --build-arg GGML_SYCL_F16=ON \
    --target server \
    -f .devops/intel.Dockerfile \
    -t llama.cpp:sycl-f16-b10068 \
    .

# Poll progress
tail -f /tmp/llama-build.log | grep -E '^#[0-9]+ (\[|DONE)'
```

Build takes **~35 min** on the Ryzen 5 5500GT box (6c/12t, 32 GB RAM). SYCL template instances (fattn variants especially) dominate the compile time.

## Promote to production

```bash
# Save the current prod tag as fallback
docker tag llama.cpp:sycl-f16 llama.cpp:sycl-f16-b9948-prev

# Promote the new build
docker tag llama.cpp:sycl-f16-b10068 llama.cpp:sycl-f16

# Restart all containers that use the tag
sudo /data/llm/launch/start-llamacpp-sycl-ornith.sh
sudo /data/llm/launch/start-llamacpp-embed.sh
sudo /data/llm/launch/start-llamacpp-rerank.sh
# minicpm5 if running:
# sudo /data/llm/launch/start-llamacpp-minicpm5.sh
```

Rollback is one retag + one restart per container.

## What's in b10433 (upgrade from b10256, 2026-08-14)

**125 commits ahead of b10256.** Motivating factor: Qwen 3.8-27B release (Mamba-hybrid arch), plus two recent SYCL optimizations for Mamba-family ops:

| SHA / PR | Title | Why it matters |
|---|---|---|
| `#26612` | sycl: coalesce the ssm_conv window loads | Mamba conv op optimized |
| `#26643` | sycl: fuse the gated-delta-net state writeback cpy | GatedDeltaNet op fused (relevant for Qwen 3.8, Nemotron 3.5) |
| `#23174` | SYCL: gated_delta_net K>1 (base) | Mature at this build |
| `#22149` | sycl: FILL/CUMSUM/DIAG/SOLVE_TRI/SSM_SCAN/GATED_DELTA_NET | All SSM+Mamba ops on SYCL |
| `9b05354ec` | HEAD (sync : ggml) | Latest master at time of build |

### Bench comparison (b10256 → b10433, 5K real workload, warm)

| Model + config | Metric | b10256 | b10433 | Δ |
|---|---|---:|---:|---:|
| Ornith 1.0 9B + MTP (Q4_K_M) | Prefill @ 5K | 1,789 | **1,911** | **+7%** |
| Ornith 1.0 9B + MTP | Decode | 64.0 | **67.9** | **+6%** |
| Ornith 1.0 9B + MTP | MTP acc | 100% | 100% | — |
| Gemma 4 E2B + Google MTP | Prefill @ 5K | 3,169 | **3,425** | **+8%** |
| Gemma 4 E2B + Google MTP | Decode | 136.7 | 136.1 | parity |
| Gemma 4 26B-A4B Q4_K_M + MTP | Prefill @ 5K | 1,475 | 1,473 | parity |
| Gemma 4 26B-A4B Q4_K_M + MTP | Decode | 47.9 | 47.7 | parity |
| Gemma 4 26B-A4B **QAT Q4_0** + MTP | Prefill | 1,335 | 1,377 | +3% |
| Gemma 4 26B-A4B **QAT Q4_0** + MTP | Decode | 53.5 | **50.5** | **-6% ⚠** |
| Gemma 4 26B-A4B **QAT Q4_0** + MTP | MTP acc | 100% | 94.8% | **-5pp ⚠** |

**Q4_0-specific regression flagged.** K-quant (Q4_K_M, Q8_0) untouched. Nothing in current prod uses Q4_0 so cutover is safe. If Q4_0 becomes needed later (rare — K-quant is preferred at ≥26B per finding #2), consider staying on `b10256-safe` for that specific service.

**Cutover 2026-08-14:** promoted b10433 → `llama.cpp:sycl-f16` alias. All 3 llama.cpp services restarted, ~60s downtime total.

## What's in b10256 (upgrade from b10215, 2026-08-04)

**41 commits ahead of b10215.** Notable SYCL / MTP-relevant merges:

| SHA | Title | Why it matters |
|---|---|---|
| `6c8dcaa7a` | sycl: parallelize the non-contiguous concat kernel (#25852) | HEAD. Concat is on the hot path for MoE routing and KV concatenation — parallelizing helps prefill on any workload that hits non-contiguous concats. |
| (commits between b10215..b10256) | multiple MTP verification path improvements | **MTP acceptance jumped 67% → 92-100%** across Ornith and Gemma 4 E2B in our post-cutover bench. This is the biggest surprise gain — the drafter verification kernel is producing much higher acceptance rates. |

**Cutover 2026-08-04:** promoted from b10215 to b10256 via `docker tag`. All three llama.cpp services (Ornith `:8002`, embed `:8004`, bench-eval `:8009`) restarted with new image, ~60s total downtime. Rollback preserved at `llama.cpp:sycl-f16-b10215-safe`.

## Isolated bench comparison (b10215 → b10256, real 5K workload, `cache_prompt: false`, warmup)

| Model + config | Metric | b10215 | b10256 | Δ |
|---|---|---:|---:|---:|
| Ornith 1.0 9B + MTP | Prefill @ 5K (isolated /completion) | 1,442 tps | **1,789 tps** | **+24%** |
| Ornith 1.0 9B + MTP | Decode | 65.9 tps | 64.0 tps | ~parity (-3%) |
| Ornith 1.0 9B + MTP | MTP acceptance | 100% (74/74) | 100% (74/74) | maxed |
| Gemma 4 E2B + Google MTP | Prefill @ 5K | 3,681 tps | 3,169 tps | ~parity |
| Gemma 4 E2B + Google MTP | Decode | 138 tps | **164 tps** (medium prompt) | **+18%** |
| Gemma 4 E2B + Google MTP | MTP acceptance | 67.8% | **92-99%** | **+25pp** |
| Gemma 4 26B-A4B QAT + MTP | Prefill @ 5K | 1,164 tps | **1,335 tps** | **+15%** |
| Gemma 4 26B-A4B Q4_K_M + MTP | Prefill @ 5K | 1,180 tps | **1,475 tps** | **+25%** |

Note: initial cross-build read showed a spurious Ornith prefill "regression" that turned out to be a methodology mismatch — the b10215 "baseline" of 3,000 tps was a Track 2 real-workload prefix-cached measurement, not an isolated /completion probe. Retest with matched methodology (both isolated cold /completion, warmup preflight) gave the +24% real number above. **See finding #20 in main README** for the full explanation.

## What's in b10215 (upgrade from b10068, 2026-07-31)

**147 commits ahead of b10068.** Notable SYCL/Battlemage/MTP-relevant merges:

| SHA | Title | Why it matters |
|---|---|---|
| `9d9a6d29f` | SYCL: add oneMKL GEMM flash attention for XMX-accelerated prompt processing (#25025) | Additional XMX FA path via oneMKL. Complements the earlier `32b741c` oneDNN FA — may boost prefill on prompt-heavy workloads. |
| `82dbc4f01` | llama : load MTP tensors only if they are really used (#26296) | MTP tensor loading optimization. Reduces model init memory pressure for `--spec-type draft-mtp` configs. |
| `000547513` | server: correct accepted tokens when need draft token replay (#26320) | **MTP acceptance counter fix.** Previous counts may have been slightly over- or under-reported when the draft token replay path was hit; new build's acceptance metrics are more accurate. |
| `d5d3e05bf` | [SYCL] support the missed types in cpy (#26005) | Completeness fix for SYCL cpy operations. |
| `1c5b89ff6` | sycl : support dev2dev memcpy by DEV2DEV_MEMCPY_FORWARD (#26234) | **Multi-GPU SYCL support.** Directly relevant to planned 2nd B60 deployment — enables efficient cross-GPU tensor transfers. |
| `a2be61dc8` | [SYCL] Support q2 mul_mat (#26231) | Q2 quantization now supported on SYCL. |
| `155372596` | sycl: fuse RMS_NORM + MUL (#26015) | Kernel fusion perf win. |
| PRs #22738, #23211, #24282 | gemma4-assistant MTP drafter architecture support | **Enables Google's official Gemma 4 E2B/E4B MTP drafters** (once we convert HF safetensors to GGUF with the correct `gemma4-assistant` arch string — community GGUFs use the wrong `gemma4_assistant` underscore convention and don't load). |

**Cutover 2026-07-31:** promoted from b10068 to b10215. Ornith prod on `:8002` restarted with new image. Real-workload decode + MTP acceptance identical to b10068 baseline within measurement noise (~38-40 tps decode, ~40% MTP on the synthetic bench; historical 50-58 tps came from live brain traffic with different prompt shapes). No regressions observed, no crashes, no garbled output. Rollback available via `docker tag llama.cpp:sycl-f16-b10068-safe llama.cpp:sycl-f16 && sudo /data/llm/launch/start-llamacpp-sycl-ornith.sh` (<1 min).

**Prefill uplift discovered 2026-08-01 during brain-eval Track 2 ingest bake-off** (missed in earlier decode-only synthetic bench): b10215 delivers **~2× prefill throughput** on real brain workload (2-5K token prompts) via new SYCL oneMKL GEMM flash attention for XMX (#25025). Both Ornith 9B and Gemma 4 E2B hit **~3,000 tps prefill** on ingest prompts vs historical ~1,600 tps on b10068. Not a Gemma-specific win — the XMX oneMKL FA path benefits every architecture that uses standard attention. Only surfaces on prompts >1K tokens (small prompts stay launch-overhead-bound and don't reach the XMX GEMM path). **Lesson: never trust "no meaningful change" claims from decode-only or short-prompt synthetic benches. Real workload prefill benefits weren't measurable until brain-eval's Tier A prompts (2-5K tok) hit the new build.**

**Known integration gap:** Community Gemma 4 E2B MTP drafter GGUFs (e.g. `AtomicChat/gemma-4-E2B-it-assistant-GGUF`) use `gemma4_assistant` (underscore) arch string, but upstream registered it as `gemma4-assistant` (hyphen). Loading fails with `unknown model architecture: 'gemma4_assistant'`, and even patching that string exposes `gemma4-assistant.context_length` metadata-key mismatch (all keys need converting). To use MTP with Gemma 4: convert `google/gemma-4-E2B-it-assistant` BF16 safetensors to GGUF ourselves via this build's `convert_hf_to_gguf.py`. Not blocking prod — categorise runs on Ornith anyway; only relevant when 2nd B60 arrives for a dedicated Gemma slot.

## What was in b10068 that we care about (SYCL / Battlemage)

Between b9948 and b10068 (291 commits), the hot ones for our stack:

| SHA | Title | Why it matters here |
|---|---|---|
| `32b741c` | [SYCL] Flash Attention with XMX engine via oneDNN (#25222) | Routes FA through Battlemage's XMX matrix engines. **Biggest cold-prefill lever** — took Ornith from 22.8s to 12.1s on our 12K cold test. |
| `c1063ac` | sycl: set fattn_vec_nthreads to 256 for Battlemage (#25205) | Tunes FA vec kernel launch geometry for BMG-G21 exactly. Pairs with the XMX path. |
| `efb3036` | sycl: add fused top-k MoE (#25217) | Fuses expert routing on Gemma 4 26B-A4B's MoE path. Modest win on our workload (~3–17% depending on prompt). |
| `d3fba0c` | sycl : fix get_rows Q2_K, Q4_K, Q5_K (#25656) | Correctness fix on Q4_K row gather. Our Ornith and MiniCPM5 GGUFs are Q4_K — older builds had a silent bug in decode for these weights. |
| `956973c` | Fix crash with draft-simple (#25720) | MTP spec-decode crash fix. Our Ornith + drafter path uses this. |
| `f5525f7` | server : fix draft model fit vs load inconsistency (#25056) | Draft/MTP loader fix. |
| `e7e3f35` | sycl : clamp softmax input to avoid underflow (#24941) | Numerical stability on SYCL. |
| `3d4cbdf` | sycl : use sycl func to fix AOT double type issue (#25081) | AOT build fix; relevant to us since we build AOT-eligible. |

Breaking change we had to audit (turned out to be safe):

| SHA | Title | Impact |
|---|---|---|
| `26145b3` | sycl : rename the env vars from "disable" to "enable" (#25042) | We set zero `GGML_SYCL_*` env vars in any launcher — only `ONEAPI_DEVICE_SELECTOR`. And `GGML_SYCL_F16=ON` is a build arg, not a runtime env. Safe. |

## Isolated bench comparison (b9948 → b10068)

| Model + config | Metric | b9948 | b10068 | Δ |
|---|---|---:|---:|---:|
| Ornith 1.0 9B + MTP | Cold 12K prefill | 22.8 s @ 632 tok/s | **12.1 s @ 896 tok/s** | **-47% / +42%** |
| Ornith 1.0 9B + MTP | Decode (12K cold) | ~50 tok/s | 51.8 tok/s | +4% |
| Ornith 1.0 9B + MTP | 5K prefill | ~830 tok/s | 1,213 tok/s | +46% |
| Gemma 4 26B-A4B + MTP | Cold 12K prefill | 21.5 s @ 655 tok/s | 20.0 s @ 650 tok/s | ~flat |
| Gemma 4 26B-A4B + MTP | 5K prefill | ~830 tok/s | 971 tok/s | +17% |
| Gemma 4 26B-A4B + MTP | Decode (peak, MTP-accepted) | ~50 tok/s | 53.0 tok/s | +6% |

The Ornith cold-prefill win is the killer number. The XMX FA path is disproportionately effective on Ornith's dense-9B GQA attention vs Gemma's MoE — MoE inference isn't attention-bound the same way.

## Build artifacts + tags

```
llama.cpp:sycl-f16                     → current prod (b10256 / 6c8dcaa7a)
llama.cpp:sycl-f16-b10256              → explicit b10256 tag
llama.cpp:sycl-f16-b10215-safe         → previous prod (b10215 / eb41d503b), kept for instant rollback
llama.cpp:sycl-f16-b10068-safe         → older prod (b10068), further-back rollback
llama.cpp:sycl-f16-next-eb41d503b      → explicit b10215 tag (was "next" pre-b10215-cutover)
```

Prune old tags with `docker image prune -a --filter "until=90d"` when the `/data/llm/docker` mount gets tight. Currently ~13 GB of llama.cpp images cached.

## Notes

- Build cache lives on `/data/llm/docker` (moved off root during the disk-full fix in June). Rebuilds after minor commits are ~5 min not 35 because ggml + SYCL template instances get reused.
- The Dockerfile is upstream's own `.devops/intel.Dockerfile` — we don't fork it; we just pass build args and pick the `server` target.
- If upstream ever changes `intel.Dockerfile` in a way that removes `GGML_SYCL_F16` as a build arg, we'll need to patch the file. Watch for that when tags jump major versions.
