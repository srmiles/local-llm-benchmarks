# Handoff — 2026-08-15 E2B wedge investigation

**Purpose:** carry state to a fresh Claude session with brain MCP connected. This session ran into an unresolved E2B wedge pattern; documenting the full evidence so next session doesn't retrace the same paths.

**Stack state at end of session (safe, running):**

| Service | Card | Port | Image | Notes |
|---|---|---|---|---|
| `llamacpp-sycl` (Ornith 9B) | 1 | :8002 | `llama.cpp:sycl-f16` (b10433) | Chat + pi.dev. `-c 262144` (was `-c 131072 --cache-ram 8192`, swapped to bigger VRAM context for less host RAM pressure) |
| `llamacpp-embed` | 1 | :8004 | `llama.cpp:sycl-f16` (b10433) | Brain embeddings, EmbeddingGemma-300M |
| `tei-rerank` | 1 | :8008 | `tei:xpu-ipex-nomemleak` | `--memory=6g` (was 2g, OOM'd today post-b10433 restore) |
| `llamacpp-categorise` (Gemma 4 E2B) | 2 | :8009 | **`llama.cpp:sycl-f16-b10256-safe`** | Sole categorise backend. NEO cache mounted at `/data/llm/cache/neo` |
| **STOPPED** `llamacpp-categorise-c1` | 1 | :8006 | — | Second E2B instance, kept wedging when co-located with Ornith |
| **STOPPED** `bench-eval` on :8009 (card 1) | — | — | — | Replaced by llamacpp-categorise-c1, itself replaced by the current card-2 instance |
| **STOPPED** `llamacpp-sycl-qwen38` | — | — | — | Qwen 3.8-27B, benched then parked (see `models/tested/qwen-3.8-27b.md`) |

**Traefik (manager.local):** `categorise.srmiles.com` → single backend `http://192.168.1.253:8009` + `inFlightReq: amount: 1`. HTTP + HTTPS both direct-serve (redirect removed). CoreDNS split-DNS on manager.local: `categorise.srmiles.com` → `192.168.1.254` (mgr Traefik).

**Watchdog:** `e2b-wedge-watchdog.service` systemd unit polls `:8009/metrics` every 15s. Restarts `llamacpp-categorise` if metrics unreachable for 45s OR counters frozen with `in_flight > 0` for 60s. 60s cooldown post-restart. Log at `/var/log/e2b-watchdog.log`. Script at `/data/llm/launch/e2b-wedge-watchdog.sh`.

## The E2B wedge — what we know

**Symptom:** llama.cpp E2B (Gemma 4 E2B QAT Q4_0 + MTP drafter) enters a state where container hits 100% CPU, `/metrics` and `/v1/chat/completions` both hang, but Docker's `/health` probe still returns 200 (HTTP server responds, inference loop deadlocked). Only recovery is `docker restart`. NEO cache persistence (`--cache-ram`, `/data/llm/cache/neo` bind-mount) makes restart ~10s.

**Timeline that matters:**
- Weeks up to 2026-08-14: single B60, Ornith + E2B (as `bench-eval`) on one card — STABLE with real brain-eval load on Ornith, only synthetic probes on E2B
- 2026-08-15 ~00:59 UTC: xpu-smi 1.3.7 → 2.1.0 upgrade + auto-installed deps (`ocl-icd-libopencl1`, `libhwloc15`, `libxnvctrl0`, `libhwloc-plugins`)
- 2026-08-15 ~08:00 UTC: llama.cpp b10256 → b10433 cutover (all services)
- 2026-08-15 ~08:00-14:00 UTC: bench-eval + Ornith on b10433 healthy for 6+ hours — this window is important
- 2026-08-15 ~14:00 UTC: 2nd B60 physically installed, both cards enumerated as expected
- 2026-08-15 ~15:00 UTC onwards: E2B wedge cycle starts; every attempt to serve brain-eval categorise traffic wedges within 30-90 min

**Hypotheses tested and DISPROVEN today:**

| Hypothesis | Test | Result |
|---|---|---|
| `--parallel 2` triggers it | Set --parallel 2 on card 2 | Wedged in seconds + engine reset. But --parallel 1 also wedges. So parallel > 1 is worse but not the cause. |
| Traefik `inFlightReq > 2` triggers it | Bumped to 4, then 1 | inFlightReq=1 still wedges eventually. So concurrency at LB isn't the trigger. |
| SYCL cross-process contention on same card (Ornith + E2B) | Moved Ornith between cards | Wedge follows E2B, not Ornith. But E2B on its own card without Ornith also wedges. |
| `ocl-icd-libopencl1` package addition | Removed + reboot + retry | Wedge continued (also broke libOpenCL.so.1, had to reinstall) |
| b10433 SYCL regression | Swapped E2B to b10256-safe image | **b10256 wedges too**. Same pattern. Not a build regression. |
| Container ephemeral NEO cache miss on restart | Bind-mounted persistent /data/llm/cache/neo | Solved slow cold-start (30-90s → ~200ms) but does NOT prevent the wedges under sustained load. Separate issue. |
| Cross-process SYCL contention (Ornith on other card via PCIe P2P) | Moved Ornith card 1 → card 2, back | Wedges track with E2B under load regardless of Ornith placement |
| Host RAM pressure | Checked PSI, swap, container mem vs cap | 11 GiB available, PSI 0%, minimal swap. Not the trigger. |
| PCIe Gen 1 x1 (misread earlier this session) | Re-examined lspci chain | **False alarm** — the Gen 1 x1 was the B60's internal switch chip reporting downstream. Real host-to-card link is Gen 3 x8 = ~8 GB/s. Plenty. |
| Client-abort during cold-start warmup | Sent requests with 60s timeout | Real cause of "cold start looks wedged" was NEO cache JIT (fixed by mount). But sustained-load wedges are a separate real bug. |

**Hypotheses NOT tested yet (candidates for next session):**

1. **2nd B60 presence causes Xe driver state changes** — even when we route E2B to only one card, the OTHER card being present affects the Xe driver. Test by unbinding card 2 via `echo 0000:0b:00.0 > /sys/bus/pci/drivers/xe/unbind` and see if wedges stop on card-1-only.
2. **Intel compute-runtime version drift** — check what was on host during the "stable weeks" era vs now. There may have been an implicit upgrade via apt/unattended-upgrades.
3. **Xe driver GuC/HuC firmware version** — currently `bmg_guc_70.bin v70.58.0` + `bmg_huc.bin v8.2.10`. Check if these got updated recently.
4. **Specific brain-eval prompt patterns** — capture a few of the exact requests brain-eval sends and replay them offline. Any specific token sequence or prompt length that reliably triggers?
5. **`intel_gpu_top` streaming during a wedge** — see per-engine (RCS, BCS, VCS, VECS, CCS) utilization at the moment of wedge. Could reveal which engine is stuck.
6. **Try Vulkan backend instead of SYCL** — llama.cpp has a Vulkan build. If Vulkan on B60 is stable, SYCL kernel path is confirmed as the trigger.
7. **Bug report upstream** — file a llama.cpp issue with the exact repro. Attach engine reset dmesg lines, container logs showing the `cancel task` events, and note that Ornith on the same build doesn't hit it while E2B does.

**Definitive evidence collected today:**

- `dmesg`: repeated `xe 0000:XX:00.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs, guc_id=N, state=0x3, Timedout job in llama-server[PID]` events. 28 total resets across the day, spread across both cards depending on where E2B was running.
- Container logs at wedge time: rapid `stop: cancel task, id_task = N` events followed by no forward progress. `slot launch_slot_` events but no matching `stop processing` or `release`.
- Metrics: `llamacpp:prompt_tokens_total` and `llamacpp:tokens_predicted_total` counters freeze while `llamacpp:requests_processing` stays > 0.
- HTTP `/metrics` and `/v1/chat/completions` become unreachable (server socket accepts but response never comes). `/health` continues to return 200 because it's a cheap GET that doesn't exercise inference.

**What's known GOOD:**

- Ornith 9B + MTP on b10433 has been serving chat + pi.dev via `:8002` all day WITHOUT wedging. Same binary, same host, same driver — just a different model.
- E2B on `bench-eval` container ran healthy for weeks with minimal traffic. Only wedged under sustained brain-eval categorise load starting today.
- Watchdog auto-restart keeps prod service functional; observed ~10s downtime per wedge cycle (NEO cache = fast restart).

## What next session should do first

1. **Reauth brain MCP** (this session was blocked from it) — via `/mcp` in interactive Claude Code, or via claude.ai connector settings.
2. **Query brain vault:llm-local** for any prior notes on E2B/SYCL/Xe engine reset patterns — Steve has been logging findings to brain all year, and there may be relevant context I couldn't access this session.
3. **Read this handoff doc + `docs/2nd-b60-arrival-playbook.md`** for the pre-cutover expectations vs today's reality.
4. **Read `docs/monitoring-dashboard-scope.md`** — the dashboard exists and Steve showed a screenshot of it earlier. That's the observability substrate for the E2B wedge investigation.
5. **Check watchdog log at `/var/log/e2b-watchdog.log`** — running restart cadence will show whether the wedge is stable-frequency (structural) or getting worse.
6. **Run the "unbind card 2, retest single-GPU" experiment** if wedges continue — this is the highest-signal untested hypothesis. Command: `sudo bash -c 'echo 0000:0b:00.0 > /sys/bus/pci/drivers/xe/unbind'` then run E2B on card 1 with real load for 30 min. Rebind with `echo 0000:0b:00.0 > /sys/bus/pci/drivers/xe/bind`.

## Config files touched this session (in the repo)

- `README.md` — updated multiple times reflecting today's stack churn; may need reconciliation
- `configs/launchers/start-llamacpp-sycl-categorise-card2.sh` — created (E2B card 2 launcher)
- `configs/launchers/start-llamacpp-sycl-categorise-card1.sh` — created (E2B card 1 launcher, currently stopped)
- `configs/launchers/start-llamacpp-sycl-qwen38-27b.sh` — created (Qwen 3.8-27B, container stopped; Steve has been editing this file locally to add `--alias Qwen3.8-27B` and other tweaks — respect his edits)
- `configs/launchers/start-tei-rerank.sh` — bumped `--memory=2g → 6g`
- `configs/launchers/start-llamacpp-categorise-e2b.sh` — deleted (stale :8006 experiment launcher from July)
- `models/production/gemma-4-e2b-categorise.md` — moved from `parked/` to `production/`, added cutover data
- `models/tested/qwen-3.8-27b.md` — moved from `production/` to `tested/`, reframed as benched-not-deployed
- `docs/monitoring-dashboard-scope.md` — created (Claude Code project scope)
- `docs/handoff-2026-08-15-e2b-wedge.md` — this file

## Commits this session

- `cfe6e1b` — "2nd-B60 cutover: E2B categorise on card 2, Qwen 3.8 parked, TEI cap fix"
- `2a79e90` — "Port convention: E2B categorise :8010 -> :8009, reserve :8010 for card-2 Ornith"

Both local only — push to remote needs to happen from Steve's Mac (proxy 403 from this sandbox).

## Open pending tasks (not blocking, just carry-forward)

- **#104** Post-RAM-upgrade dual-load plan for Sat 2026-08-22 (30 → 64 GiB) — revisit dual-load Ornith + E2B on both cards + LB
- **#137** Full dmesg IRQ + PCIe topology investigation (partially done this session; PCIe finding was misread; IRQ + Xe driver deep-dive still open)
- **LMCache evaluation** — research + testing project seeded at `docs/research/lmcache-evaluation.md`. Steve raised the repo (`github.com/lmcache/lmcache`) as potentially interesting for categorise workload. Full brief in that doc. Push to brain vault:llm-local when reachable.

## Contact context

- Steve is on Perth time (AWST, UTC+8)
- Ornith serves chat + pi.dev via `http://llm.local:8002` (unchanged, never in the categorise route churn)
- brain-eval and prod-brain both hit `https://categorise.srmiles.com/v1/chat/completions` (Traefik → E2B :8009)
- Brain-eval applied client-side backoff on retries earlier today — should be a well-behaved client now
