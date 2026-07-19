# pi.dev provider configs

Merge into **`~/.pi/agent/models.json`** under the top-level `providers` key.

⚠ **Not `~/.pi/agent/mcp.json`** — that file is for MCP tool servers (stdio commands), not LLM endpoints.

pi.dev reloads `models.json` each time you open `/model` in the TUI; no restart needed.

## MiniCPM5-1B on `:8009` — [`minicpm5-models.json`](minicpm5-models.json)

```json
{
  "providers": {
    "minicpm5-local": {
      "baseUrl": "http://192.168.1.253:8009/v1",
      "api": "openai-completions",
      "apiKey": "dummy-key",
      "compat": {
        "supportsDeveloperRole": true,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "/models/MiniCPM5-1B-Q4_K_M.gguf",
          "contextWindow": 8192,
          "maxTokens": 2048,
          "reasoning": false,
          "input": ["text"]
        }
      ]
    }
  }
}
```

### Notes

- **`baseUrl`** — LAN IP of `llm.local`. If accessing over Tailscale, swap for the tailnet IP.
- **`contextWindow: 8192`** — deployed per-slot context is 8K (`-c 16384` split across `--parallel 2` slots). Model's trained max is 128K; to expose that, restart `llamacpp-minicpm5` with `-c 131072 --parallel 1` (VRAM cost ~2.5 GiB more for KV).
- **`maxTokens: 2048`** — conservative response cap. Model can go higher, but pi.dev tool loops benefit from a shorter ceiling.
- **`supportsDeveloperRole: true`** — MiniCPM5 uses a standard Llama chat template that accepts the `developer` role. Different from Gemma 4 which needed `false`.
- **`supportsReasoningEffort: false`** — model has Think/No-Think modes but we launch with `--reasoning off`; sending `reasoning_effort` would be ignored anyway.
- **`reasoning: false`** — matches server-side `--reasoning off`.

### Known behaviour

- **Tool calls: ✓** — clean OpenAI-format tool_calls with structured JSON arguments.
- **JSON response format: ⚠** — wraps output in ```json ... ``` fences. If pi.dev asks the model for pure JSON (categorise-style workload), strip fences client-side or add a grammar constraint server-side.

### Bench numbers (isolated, no contention)

- Decode: **~187 tok/s**
- Prefill @ 2K: **4,642 tok/s**
- Under co-residence with Ornith on `:8002`: expect 130–187 tok/s depending on VRAM headroom.
