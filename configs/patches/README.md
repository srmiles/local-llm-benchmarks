# Local `ggml-sycl` patches

Applied to `llama.cpp` **`bb4caa754`** (b10566). Written and measured on the B60 pair 2026-08-26. **Not upstream** — `gh` is not installed on `llm.local`, so these have not been filed as PRs yet.

Source of truth is branch `all-fixes` in `/data/llm/build/llama.cpp`; these files are `git format-patch` exports kept here so the repo is self-contained.

| file | what it does |
|---|---|
| `0001-...-Q4_0-and-Q8.patch` | MoE expert-tensor reorder for Q4_0 / Q8_0 |
| `0002-...-Q3_K.patch` | MoE expert-tensor reorder for Q3_K |
| `0003-test-backend-ops-...perf-c.patch` | add Q3_K to the `MUL_MAT_ID` perf cases (they covered q4_0/q8_0/q4_K/q6_K/iq2_xs but not q3_K) |
| `0004-...-TQ1_0.patch` | report MUL_MAT/MUL_MAT_ID unsupported for TQ1_0/TQ2_0 |
| `0005-...-MMVQ_MAX_BATCH_SIZE.patch` | chunk MMVQ above batch 8 instead of falling back to dequant+GEMM |

Total: 191 insertions, 5 deletions across `ggml-sycl.cpp`, `mmvq.cpp`, `test-backend-ops.cpp`.

Apply in order:

```bash
git checkout -b all-fixes bb4caa754
git am configs/patches/000*.patch
docker build --target server -f .devops/intel.Dockerfile \
  --build-arg GGML_SYCL_F16=ON -t llama.cpp:sycl-f16-allfixes .
```

Patch 0005 is behaviourally inert unless `GGML_SYCL_MMVQ_MAX_COLS` is set above its default of 8.

**Measured impact, and the caveat that matters →** [`models/tested/2026-08-26-sycl-patches-default-bench.md`](../../models/tested/2026-08-26-sycl-patches-default-bench.md) · findings **#36–#38**.

The reorder patches only help at **batch 1**. A high-acceptance synthetic benchmark will show ~3% where real agent traffic shows +32%. Do not evaluate them with `bench-candidate.py`'s filler prompt alone.

## Upstream compliance status - NOT ready to submit

Checked against `CONTRIBUTING.md` and `AGENTS.md` (read 2026-08-26). Blockers:

- **Issue #27517 (the Q4_0/Q8_0 gap) is assigned to @newjordan.** `CONTRIBUTING.md` lists "already mentioned in an existing issue and assigned to someone" as grounds for closing a PR. Comment on the issue first and coordinate; do not open a competing PR.
- **PR descriptions and commit messages must be written by a human.** `AGENTS.md` prohibits AI-written PR text, commit messages and reviewer responses; `CONTRIBUTING.md` calls it strictly prohibited. **Every commit message in this series is a placeholder and must be rewritten.**
- **AI usage must be disclosed** in the PR template. Undisclosed use can mean a permanent ban.
- **One PR per change**, and new contributors are limited to **1 open PR** - so this is a queue, not a batch.
- **Bug-fix PRs need a reproducing issue plus a regression test** that fails before and passes after. Patch 0004 (TQ guard) has neither yet.
- **Full local CI** (`ci/README.md`) has not been run.

Read before proposing any of this: `sycl : add Q2_K reordered MMVQ and ESIMD kernels` (#26336) was merged and then **reverted** in #27486 for CI failures. Same family of change. The stated reason lives in a comment on #26336.
