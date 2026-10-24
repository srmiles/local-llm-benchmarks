# 2026-10-24 — b11850 XMX A/B, Nemotron 3.5 Bolt, Nemotron-3.5-Embed/Rerank

Copied from `llm.local:/data/llm/benchmarks/` after the runs. Write-ups: [XMX A/B](../../../models/tested/2026-10-24-b11850-xmx-ab.md) · [Bolt](../../../models/tested/nemotron-3.5-bolt.md) · [Embed/rerank](../../../models/tested/nemotron-3.5-embed-rerank.md) · findings #58–#63.

| Path | What |
|---|---|
| `ab/build-9ae73e68f.sh` | image build (`sycl-f16-9ae73e68f-patched-3bd0de68e`) |
| `ab/run-9ae73e68f-all.sh`, `run-9ae73e68f-ab.sh`, `.log` | 3-arm × 3-round interleaved A/B (old / new / noxmx). **Its log-line handoff to the next script is the race in finding #59 — do not reuse that pattern.** |
| `ab/r{1,2,3}/*.json` | `bench-candidate.py` output per model × arm × round |
| `bolt/run-bolt-embed-20261024.sh`, `.log` | the post-reboot single-owner runner: Bolt, then embed, then rerank, `card2-empty.sh` before every launch |
| `bolt/*.json` | Bolt E4B / 74B, solo and DFlash, two rounds (74B DFlash has none — it crashed) |
| `bolt/bolt-74b-dflash6-r1.serverlog` | the `outprod.cpp:12` assert + backtrace (finding #60) |
| `embed/bench.py` | BEIR retrieval bench; `--style nemotron` renders the Nemotron chat template client-side; `STAGE1_DUMP` saves the similarity matrix for reranking |
| `embed/ref.py`, `parity.py`, `parity_raw.json` | sentence-transformers CPU reference + parity check against the llama.cpp endpoint |
| `embed/rerank.py`, `tei-check.py` | CrossEncoder-on-XPU reranker of the stage-1 top-K; cross-check against prod TEI `:8008` |
| `embed/results/*.json` | retrieval, parity and bge rerank results (no Nemotron rerank result — GPU fault, finding #63) |
| `kernel-faults.txt` | `journalctl -k` excerpts for both card-2 incidents |

Not committed: the BEIR datasets (`data/` is a symlink to the 2026-10-11 run's copy on the host), the `.npy` references and `.npz` stage-1 matrices, the void pre-reboot Bolt results.
