# E2B wedge — 2026-08-15 late-session resolution log

Follow-on from `handoff-2026-08-15-e2b-wedge.md`. Session ran ~20:35–21:40 UTC with brain MCP connected.

## Outcome

**Stack is fully restored dual-card and stable post-upgrade.** Ornith card 1 `:8002`, E2B categorise card 2 `:8009` (`-fa on` + q8_0 KV, device-node isolated), embed `:8004`, TEI `:8008`, watchdog v2.1 + llm-monitor active. Dual-card retest after `apt upgrade` + reboot: 12+ min clean under full brain-eval backlog, **zero engine resets or GPU faults the entire boot** — pre-upgrade the same load wedged E2B in ≤3 min every time.

## Root-cause narrative (what 6 experiments established)

| Hypothesis | Test | Verdict |
|---|---|---|
| Compute-runtime / firmware drift | dpkg + apt history archaeology | **Dead** — nothing GPU-side changed between stable weeks and wedge era |
| Cross-card L0 contexts (whole `/dev/dri` passed → fds/VM-binds on both cards) | Device-node isolation (card2+renderD129 only) | **Dead** — wedged in ~2 min anyway |
| SYCL flash attention (buggy on Xe2) | `-fa off` + f16 KV | **Dead** — wedged identically |
| Load pattern alone | Card-2 unbind, E2B solo on card 1, same backlog | **Partial** — the GPU fault still occurred once, but server *recovered*; 14+ min clean |
| **Dual-card xe fault recovery** | All of the above combined | **Confirmed (pre-upgrade)** — fault is workload-triggered on any config; only dual-card fails to recover |
| Post-upgrade stack | Reboot + dual-card retest | **Stable** — 12+ min clean, zero faults |

## Mechanism (from the captured Xe devcoredump)

`/var/log/e2b-wedge-forensics/20260815T205511Z/xe-devcoredump-card2.bin` (520 KB, GuC log + engine state):

- Workload hits a GPU page fault at an **unmapped VA** — dmesg: `Fault response: Unsuccessful -ENOENT`
- Engine hangs mid-execution: ROW_INSTDONE `0x0` on all 6 slices (EU rows busy, compute walker never retires), ACTHD ~32 MB deep in batch
- GuC times out the job (`Timedout job seqno=482`) → engine reset (ccs `state=0x3`, sometimes bcs `state=0x289`, once with a memory CAT error)
- Level Zero never signals completion → llama-server main thread spins at 100% CPU (thread dump: 1 spinning thread, rest in `futex_wait`), `/health` still 200 → the classic wedge
- Bonus finding: GuC firmware 70.58.0 loaded, kernel *wanted* 70.54.0

## What changed in the 21:10 UTC `apt upgrade` (why the retest may now pass)

libze-intel-gpu1 26.18.38308.4 → **26.27.39122.12**, libigc2/libigdfcl2 2.34.4 → **2.38.3**, oneAPI runtime 2026.0.0 → 2026.1.1, libze1 1.28.6, linux-firmware-intel-graphics bump (GuC version string unchanged), docker-ce 29.7.2. Caveat: containers bundle their own SYCL userland, so honest attribution between "new stack" and "fresh boot state" is incomplete — the overnight soak decides.

## New facts worth keeping

- **On-card GSC firmware mismatch:** card 1 = `BMG__21.1177`, card 2 = `BMG__21.1182`. LVFS/fwupd has no B60 updates; harmonizing needs `igsc` + Intel image (as done manually on card 1 originally). Remaining hardening step.
- **Watchdog v1 failure:** logged "unreachable" every 20 s for 5.5 h (14:47→20:14 UTC) without restarting — categorise was down overnight. v2 replaced it; v2.1 (this session) captures forensics + Xe devcoredump *before* each restart.
- `intel_gpu_top` cannot read the xe driver (i915 PMU only) — per-engine data comes from DRM fdinfo `drm-cycles-*` deltas.
- llm-monitor's `xpu-smi` stream holds DRM fds on **all** cards and blocks `xe/unbind`; stop it first.
- docker-ce upgrades restart all containers (no live-restore) — bounced the test mid-run at 21:11.
- Ornith registers 2.4 GiB of host buffers against the *other* card's DRM instance when whole `/dev/dri` is passed — harmless per the isolation test, but device-node scoping is kept on the categorise launcher as hygiene.

## Config state after session

- `configs/launchers/start-llamacpp-sycl-categorise-card2.sh` (live copy at `/data/llm/launch/`): device-node isolated (`card2`+`renderD129`, selector `level_zero:0` since only one device visible), `-fa on` + q8_0 KV restored. Backups: `.bak-alldri-20260815`.
- `/data/llm/launch/e2b-wedge-forensics.sh` — new; called by watchdog pre-restart, also runs standalone.
- `/data/llm/launch/e2b-wedge-watchdog.sh` — v2.1 (forensics hook). Backup `.bak-v2-20260815`.
- Temp unbind-test launcher removed; card 2 rebound via reboot.

## Open items

1. Overnight soak verdict (scheduled check ~01:00 UTC / 9am AWST).
2. Harmonize GSC firmware across cards via igsc (needs Intel image).
3. If wedges return: file upstream xe kernel bug with the saved coredump; llama.cpp issue for the unmapped-VA access under Gemma-4-E2B+MTP sustained load; single-card fallback is proven stable.
4. RAM upgrade Sat 2026-08-22 (#104) unchanged; host swap pressure noted but not urgent.
