# llama.cpp:sycl-f16

Custom llama.cpp SYCL image tuned for **Intel Arc Pro B60 (Battlemage / Xe2)**. Serves all four llama.cpp containers in the stack (chat, embed, rerank, plus any ad-hoc bench containers).

**Current tag → build:** `llama.cpp:sycl-f16` → **b10068** (2026-07-19)
**Previous prod tag (rollback):** `llama.cpp:sycl-f16-b9948-prev`

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

## What's in b10068 that we care about (SYCL / Battlemage)

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
llama.cpp:sycl-f16              → current prod (b10068)
llama.cpp:sycl-f16-b10068       → explicit b10068 tag
llama.cpp:sycl-f16-b9948-prev   → previous prod, kept for rollback
llama.cpp:sycl-f16-b9948        → explicit b9948 tag
llama.cpp:sycl-f16-b9777-backup → older baseline
llama.cpp:sycl-f16-full         → full-target variant (has CLI + tools; ~4.25 GB)
```

Prune old tags with `docker image prune -a --filter "until=90d"` when the `/data/llm/docker` mount gets tight. Currently ~13 GB of llama.cpp images cached.

## Notes

- Build cache lives on `/data/llm/docker` (moved off root during the disk-full fix in June). Rebuilds after minor commits are ~5 min not 35 because ggml + SYCL template instances get reused.
- The Dockerfile is upstream's own `.devops/intel.Dockerfile` — we don't fork it; we just pass build args and pick the `server` target.
- If upstream ever changes `intel.Dockerfile` in a way that removes `GGML_SYCL_F16` as a build arg, we'll need to patch the file. Watch for that when tags jump major versions.
