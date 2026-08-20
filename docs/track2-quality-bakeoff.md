# Track 2 quality bake-off — Ornith 9B vs Gemma 4 E2B (2026-08-01)

Head-to-head brain-eval on real Tier A corpus, grammar-constrained (JSON schema server-side), sampler pinned, 131K context matched across arms, 3-repeat ingests to control variance. Judge: DeepSeek-V3.2 with retry-with-backoff.

| Metric | Ornith 9B + MTP | Gemma 4 E2B + Google MTP | Delta |
|---|---|---|---|
| Pass rate (3-repeat mean) | 0.556 | 0.422 | quality noise-band overlap |
| Mean score (LLM-judge) | 0.784 | 0.787 | ~tie |
| Classify median latency | 19,403 ms | 6,286 ms | **3.1× faster** |
| VRAM steady | 10.9 GiB | 5.8 GiB | **1.9× smaller** |
| Full-ingest wall-clock | 2,370s | 1,145s | **2.1× faster** |

**Interpretation: quality is indistinguishable inside ingest-run variance** (individual ingests swung 0.400–0.600 across all four runs regardless of model — ingest variance dominates model gap). E2B wins latency, VRAM, and wall-clock.

**Decision:** E2B approved to move into the categorise slot when a 2nd B60 arrives. Cross-process SYCL contention on a single card (finding #15) forbade concurrent dispatch on one card. Executed 2026-08-15 (E2B on card 2 `:8009`, Ornith on card 1) after the 2nd B60 was installed.

Charts and full methodology:
- [`charts/track2_quality_vs_speed.png`](../charts/track2_quality_vs_speed.png)
- [`charts/gemma_google_mtp.png`](../charts/gemma_google_mtp.png)
- [`models/production/gemma-4-e2b-categorise.md`](../models/production/gemma-4-e2b-categorise.md) — full methodology and caveats (Tier A only; Tier C confirmation pending; Track 1 chat/agent eval not run)
