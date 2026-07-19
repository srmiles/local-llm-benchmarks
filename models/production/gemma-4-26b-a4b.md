# Gemma 4 26B-A4B (it) — Reasoning fallback / historical prod

**Status:** Reserved for reasoning-heavy queries; launcher `start-llamacpp-sycl-gemma4-mtp.sh` on disk.
**HF (base):** [`lmstudio-community/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF)
**HF (drafter):** Google's official 26B-A4B assistant, community-packaged by Janvitos as MTP Q8_0
**Chat template:** Google's updated official Jinja (strip_thinking macro, OpenAI tool response handling)
**Launcher:** [`configs/launchers/start-llamacpp-sycl-gemma4-mtp.sh`](../../configs/launchers/start-llamacpp-sycl-gemma4-mtp.sh)

## Specs

| | |
|---|---|
| Parameters | 26B total / **4B active** (MoE) |
| Quant | Q4_K_M (post-training — beats QAT Q4_0 on Battlemage) |
| File size | ~15 GB |
| Context (trained) | 256K |
| Context (deployed) | 131,072 |
| MTP drafter | Google official assistant, Q8_0, 441 MiB |

## Benchmarks

| Metric | Q4_K_M base | + MTP (Config C) |
|---|---|---|
| Decode (steady-state) | 44.1 tok/s | **50.0 tok/s** (+15.6%) |
| Cold 12K prefill | 22.8s @ 632 tok/s | ~21.5s @ 655 tok/s |
| Warm follow-up | 0.55s | 0.55s |
| Concurrent 2×4K | 13.9s | ~13.9s |
| VRAM (loaded) | 20.9 GB | 22.8 GB |
| MTP acceptance | — | 78%, 3.30 mean accepted tok/draft |
| KB dual-eval (OpenRouter fp8) | .76 | — |
| KB dual-eval (local Q4_K_M) | .73–.76 | — |
| pi.dev win rate | 63.3% | 63.3% |

## Config (`start-llamacpp-sycl-gemma4-mtp.sh`)

```bash
docker run -d --name llamacpp-sycl \
  --restart unless-stopped \
  --memory=20g --memory-swap=20g \
  --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --health-cmd 'curl -fsS http://localhost:8000/health >/dev/null 2>&1 || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 180s \
  -v /data/llm/lmstudio-community/gemma-4-26B-A4B-it-GGUF:/models:ro \
  -v /data/llm/Gemma-4-Assistant:/draft:ro \
  -v /data/llm/templates:/templates:ro \
  -p 0.0.0.0:8002:8000 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  llama.cpp:sycl-f16 \
  -m /models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --model-draft /draft/gemma-4-26B-A4B-it-qat-assistant-MTP-Q8_0.gguf \
  --spec-type draft-mtp \
  -ngl 99 -ngld 99 \
  -c 131072 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 3072 \
  --parallel 1 \
  --host 0.0.0.0 --port 8000 \
  --metrics \
  -fa on -ub 2048 -b 2048 \
  --predict 2048 \
  --min-p 0.0 \
  --temp 1.0 --top-k 64 \
  --chat-template-file /templates/gemma-4-official-current.jinja \
  --jinja \
  --reasoning off
```

## Notes

- Was locked production for months; superseded by Ornith 9B after July 2026 dual-eval bake-off
- Post-training Q4_K_M beat QAT Q4_0 by +10% decode (Finding 8) — surprising for a QAT-shipped model
- `--jinja` **mandatory** for tool calls; without it the built-in template drops Gemma 4's `<|tool>`/`<tool|>` delimiters and the agent loops
- Config C = vLLM-fixed → then Google-official template. Fixes tool-loop drift Config B had at long context
- `--reasoning off` currently used because PEG parser 500s; Google official template makes `--reasoning on` viable but hasn't been re-locked
- MTP drafter absolutely worth it (+15.6% decode, 78% acceptance)
- 256K context loads but leaves zero margin; 128K–192K is the safe range
