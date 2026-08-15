# Monitoring Dashboard — Scope for Claude Code project

**Status:** Scope brief. Implementation lives in a separate repo (TBD name, e.g. `llm-local-monitor`). This document defines what to build and why; the Claude Code project should pick this up and produce a working v1.

**Owner:** Steve
**Target host:** `llm.local` (dual Intel Arc Pro B60, Ubuntu 26.04, 30 → 64 GiB RAM upgrade pending Sat 2026-08-22)
**Author:** Claude (working session 2026-08-15)

---

## 1. Goal

A single browser page I can leave open on a second monitor that answers, at a glance:

1. Is every service up and serving?
2. How hard is each GPU actually working right now?
3. Am I about to run out of VRAM, host RAM, or swap?
4. Is speculative decoding still paying off (MTP acceptance rate holding)?
5. Am I regressing on prefill/decode throughput vs. yesterday?

Secondary: enough historical retention (7–30 days) to spot slow drift, catch container OOMs after the fact, and compare pre/post upgrade windows (build cutover, driver, RAM).

## 2. Why now

- Stack just went dual-B60 — I no longer have a mental model of "which card is doing what" without SSHing in.
- llama.cpp build cadence is ~weekly with real perf swings (b10256 → b10433 had a Q4_0 regression I only caught by hand-benching).
- XPU-SMI 2.1 finally landed with dual-card syntax (`stats -d 0,1`), JSON output, and a streaming `dump` subcommand — the data source problem is now solved.
- Second B60 makes VRAM co-residence math non-trivial: I want a visual answer to "can I fit another 20 GiB model on card 1 right now?"

## 3. In scope for v1

### 3.1 What to display

**Per-GPU tiles (card 0, card 1):**
- Name / PCI BDF / driver / firmware
- VRAM used / total, plus stacked bar showing which container owns each chunk (derived by cross-referencing docker container → level_zero device selector)
- Power draw (W), temperature (°C), fan RPM if reported
- Compute engine utilisation (%), memory bandwidth utilisation if exposed
- Recent P2P activity (if xpu-smi exposes counters — verify)

**Per-service tiles (one per llama.cpp / TEI instance):**
- Endpoint URL + name + card assignment
- Health: `/health` 200 → green, timeout/5xx → red, unknown → grey
- Prefill tokens/sec (rolling 1 min from `llamacpp:prompt_tokens_total` / `llamacpp:prompt_seconds_total`)
- Decode tokens/sec (rolling 1 min from `llamacpp:tokens_predicted_total` / `llamacpp:tokens_predicted_seconds_total`)
- Requests in flight, requests/min
- KV cache usage % (from `llamacpp:kv_cache_usage_ratio`)
- **MTP acceptance ratio** (derived: `n_decode_total` vs. `tokens_predicted_total`; instances without a drafter show N/A)
- Container mem usage vs. cgroup cap; CPU %

**Host tile:**
- Host RAM used / total, swap used / total, load avg
- Disk free on `/data/llm` (model storage), on `/` (root)
- Uptime
- Current kernel + Intel compute-runtime version

**Alert strip (top of page):**
- Any service unhealthy for > 30 s
- Any GPU >95% VRAM for > 1 min
- Host swap > 512 MiB used
- MTP acceptance dropped >10 pp vs. 24 h baseline for a given service

### 3.2 Historical view (secondary tab or drawer)

- 24 h / 7 d / 30 d selector
- Per-service: prefill tps, decode tps, MTP acceptance, requests/min, p50/p95 request latency
- Per-GPU: VRAM used, power, temp, util
- Annotation overlay showing container restarts and llama.cpp image tag changes (parse from docker events / a small log the launcher scripts write)

### 3.3 Nice-to-have (v1 if trivial, v2 otherwise)

- One-click "restart this service" button (calls `docker restart <name>` via a tiny local shim — auth-gated to loopback only)
- Copy-to-clipboard of the current launcher command for a service (read from `/data/llm/launch/*.sh`)
- Live tail of last 20 log lines per service (from journald or `docker logs --tail`)

## 4. Data sources (confirmed available today)

| Source | Command / endpoint | Format | Notes |
|---|---|---|---|
| Per-GPU stats | `xpu-smi dump -d -1 -m ALL --loop-ms 1000 -j` | JSON stream | New in 2.1; use as primary. Legacy `xpu-smi stats -d 0,1 -j` for snapshot fallback. |
| GPU discovery | `xpu-smi discovery -j` | JSON | Enumerate cards on startup. |
| GPU topology | `xpu-smi topology --p2p r -j` | JSON matrix | Static, snapshot at boot. |
| llama.cpp per-instance | `http://<host>:<port>/metrics` | Prometheus text | Enabled on every launcher (`--metrics` flag). |
| llama.cpp health | `http://<host>:<port>/health` | JSON | Returns `{"status":"ok"}` when serving. |
| llama.cpp props | `http://<host>:<port>/props` | JSON | Model name, ctx, drafter path — populate service tiles at startup. |
| TEI rerank | `http://<host>:8008/metrics` | Prometheus text | Same shape as llama.cpp for our purposes. |
| Container stats | `docker stats --no-stream --format json` | JSON | Poll every 5 s. |
| Container inventory | `docker ps --format json` | JSON | Poll every 30 s; label-based service metadata (see §7). |
| Host RAM/swap/load | `/proc/meminfo`, `/proc/loadavg`, `/proc/uptime` | Text | Cheap, no deps. |
| Disk free | `statvfs` or `df --output=avail,size -B1` | Text | Poll every 60 s. |
| Kernel / compute-runtime | `uname -r`, `dpkg -s intel-opencl-icd \| grep Version` | Text | Snapshot only, on startup. |

Explicit ports today (subject to change — dashboard should discover, not hardcode):
- Card 0: `:8002` Ornith, `:8004` embed, `:8008` TEI rerank, `:8009` bench-eval
- Card 1: `:8010` Qwen 3.8-27B
- `:8003` headroom-proxy (Anthropic proxy, external service — dashboard should treat it as a black-box health check only, not scrape /metrics from it)

## 5. Delivery target

**Recommended: self-hosted small stack**, in this order of preference for v1:

1. **Preferred — bespoke single-page app + tiny collector.** A Go or Rust binary that:
   - Polls the sources above on a fixed schedule
   - Writes to an embedded time-series store (SQLite with Timescale-style tables, or DuckDB, or bolt+parquet — implementer's call)
   - Serves a static HTML+chart.js SPA on `:9090`
   - Runs as a systemd unit; single binary, no external deps beyond `xpu-smi` and `docker`
   - Rationale: matches the "self-hosted, no fluff" ethos of the rest of the stack; avoids Grafana's config sprawl; the data volume is trivial (a few hundred series at 1 Hz).

2. **Acceptable fallback — Prometheus + Grafana.** Node-exporter for host, a wrapper exporter that shells out to `xpu-smi dump` and re-emits Prometheus metrics, direct scrape of llama.cpp/TEI `/metrics`, cAdvisor for containers, Grafana with a hand-built dashboard JSON. Faster to ship v1, higher operational surface area long-term.

**Decision to make in the Claude Code project's kickoff:** pick option 1 or 2. Default to 1 unless there's a strong reason otherwise (e.g., I already have a Grafana box I want to reuse — I don't).

## 6. Non-goals

- Multi-host support (this is one machine).
- Alerting via email/Slack/PagerDuty. In-page alert strip only.
- Authentication beyond binding to a private LAN interface. This is a home lab.
- Log aggregation. Dozzle already handles that; the dashboard can link to Dozzle for a given container but shouldn't replace it.
- Anything about the headroom-proxy internals — that's Steve's external service.
- GPU model performance comparisons across builds. That's what the `local-llm-benchmarks` repo is for; the dashboard shows *live* state, not historical benchmarks.

## 7. Container labelling convention (implementer should adopt in launchers)

To make service auto-discovery robust, launcher scripts should add labels the dashboard can key off:

```bash
docker run -d --name llamacpp-sycl-qwen38 \
  --label monitor.service=llm \
  --label monitor.model="Qwen3.8-27B-Q4_K_M" \
  --label monitor.card=1 \
  --label monitor.port=8010 \
  --label monitor.endpoint=http://localhost:8010 \
  ...
```

Dashboard reads `docker ps --format json` filtered on `label=monitor.service=llm`, gets endpoint + card assignment for free. Fallback for unlabelled containers: try `docker inspect` and match on env vars / port bindings.

A follow-up chore in the benchmarks repo will retrofit these labels to the existing launcher scripts.

## 8. Success criteria for v1

- I can open one page and, within 5 seconds, answer all five questions from §1.
- Restart of any single component (a container, `xpu-smi`, the collector itself) doesn't take the dashboard down for more than 30 s.
- Idle resource cost < 200 MiB RAM, < 3% CPU on the collector; page render < 500 ms cold.
- 7 days of 1 Hz retention fits in < 500 MiB on disk.
- Zero configuration to add a new llama.cpp instance beyond starting the container with the labels above.
- Survives host reboot cleanly.

## 9. Open questions for the implementer

- Does `xpu-smi dump` return valid data for both cards concurrently in a single stream, or do we need two persistent streams? Verify before committing to the polling design.
- MTP acceptance ratio derivation from Prometheus counters — is `llamacpp:n_decode_total` incremented once per accepted target token, or per model call? Confirm against llama.cpp source before wiring the alert.
- On host reboot, container start order isn't guaranteed. Should the dashboard wait N seconds before flagging "service down" alerts on cold start? Probably yes, with a 90 s grace on first boot after uptime < 5 min.
- Should the collector write to `/data/llm` (survives OS reinstalls) or `/var/lib/<collector>` (standard)? Recommend `/data/llm/monitor/` given the rest of the stack lives there.

## 10. Handoff notes to Claude Code

- The `local-llm-benchmarks` repo already has the current state of every launcher script under `configs/launchers/`. Read those first — they define the service surface area.
- `docs/2nd-b60-arrival-playbook.md` in the same repo has the current per-card allocation and headroom math.
- xpu-smi 2.1.0 install steps are captured in this session's chat log; if the tool isn't present on `llm.local`, install the `.deb` from `github.com/intel/xpumanager/releases/v2.1.0` before starting.
- Steve prefers a working v1 over a comprehensive v2. Ship the tiles from §3.1 first; §3.2 historical view can be a follow-up PR.
