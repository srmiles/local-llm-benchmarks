# 2026-10-31 — b12017 A/B and HF candidates

Copied from `llm.local:/data/llm/benchmarks/` and `/data/llm/build/` after the runs. Write-ups: [b12017 A/B](../../../models/tested/2026-10-31-b12017-ab.md) · [HF candidates](../../../models/tested/2026-10-31-hf-candidates.md) · findings #64–#67.

| Path | What |
|---|---|
| `ab/build-5047748f1.sh` | builds `sycl-f16-5047748f1-patched-c9bf2e9a6` and `sycl-f16-5047748f1-clean` |
| `ab/chunkify.py` | resolves the `5320c87` conflict by porting a whole-tensor MoE reorder kernel to the 32 MiB chunked loop (finding #65) |
| `ab/run-5047748f1-all.sh`, `run-5047748f1-ab.sh`, `.log` | 3-arm (old / new / clean) interleaved A/B, arm order changed each round |
| `ab/summ-5047748f1.py` | per-round summary table |
| `ab/r{1,2,3}/*.json` | `bench-candidate.py` output per model × arm × round (r3 = Ornith + Nemotron only) |
| `patches/0001`–`0007` | the 7-commit local series as built, `git format-patch 5047748f1..c9bf2e9a6` |
| `hf/dl-hf-20261031.sh` | the five GGUF downloads |
| `hf/run-hf-20261031.sh`, `.log` | candidates part 1 (Mellum, Groot, …). Stopped by hand after the 02:37 Groot reset |
| `hf/run-hf2-20261031.sh`, `.log` | part 2 (Flint, LFM3, Spark). Stopped by hand after the 03:28 Flint reset |
| `hf/*.json`, `hf/*.serverlog` | per-run results and full server logs. The `groot-q4km-solo-r1` and `flint-q4km-r2` logs show the stall |
| `kernel-faults.txt` | `journalctl -k` lines for card 2 (`0000:0b:00.0`) from 00:40 UTC on |

Not committed: the devcoredump (512 KB, kept on the host at `/data/llm/benchmarks/20261031-hf/groot-ccs-reset-0237.devcoredump`; it is the only real dump of the day, see finding #66), and the GGUFs.
