# Agent clients on Nemotron `:8011`

Config for **pi.dev** and **opencode** against the Nemotron 3.5 Lightning 30B-A3B agent-testing slot.

| | |
|---|---|
| Endpoint | **`http://100.70.193.48:8011/v1`** (Tailscale — use this from any client not on the box) · `http://192.168.1.253:8011/v1` (LAN) |
| Model id | `nemotron-3.5-lightning-30b-a3b` — exactly as `GET /v1/models` reports it |
| Auth | none. Any non-empty `apiKey` string satisfies clients that insist on one. |
| Context | 131,072 |
| Max output | uncapped server-side (`n_predict = -1`) |
| Traefik | **none** — this port is deliberately outside every LB pool |
| Throughput | 72–90 tok/s decode, 1,300–1,750 prefill, measured 2K→70K of context |

Deployed by [`configs/launchers/start-llamacpp-nemotron-agent.sh`](launchers/start-llamacpp-nemotron-agent.sh). Model detail: [`models/tested/nemotron-3.5-lightning-30b-a3b.md`](../models/tested/nemotron-3.5-lightning-30b-a3b.md).

## Which address to use

**Use the Tailscale address `100.70.193.48:8011` from anything that isn't on `llm.local` itself.** The LAN address works, but only from a host actually sitting on `192.168.1.0/24` with a working route to `.253`.

A macOS client hitting the LAN IP produced:

```
Cannot connect to API: connect EHOSTUNREACH 192.168.1.253:8011 - Local (192.168....
```

`EHOSTUNREACH` is a routing-layer failure — no route to host. The packet never left for the server, so it is not a firewall rule, not a refused connection, and nothing to fix on `llm.local`. Verified at the time: `:8011` binds `0.0.0.0`, `ufw` is inactive, and another LAN host (`manager.local`) got `200` from `http://192.168.1.253:8011/v1/models`.

**The Mac is on `192.168.1.x` and reaches `.253` fine over SSH**, so this is not a subnet mismatch — it is one *process* on that machine being unable to use a path the machine itself has. In rough order of likelihood:

1. **macOS Local Network permission.** macOS 15+ gates LAN access per-application, and a denied app gets exactly `EHOSTUNREACH` on a `192.168.x` target while everything else on the box works normally. The grant follows the *binary that opens the socket* — the terminal app, or Node — so SSH from one app and opencode from another can differ. Check System Settings → Privacy & Security → Local Network.
2. **opencode running inside a container or VM** (OrbStack, Docker Desktop). A bridged container has no route to the host's LAN unless explicitly given one, and LAN targets fail this way while tailnet and public addresses still work.
3. **A Tailscale subnet route for `192.168.1.0/24`.** If a tailnet node advertises it and this client has `--accept-routes` on, traffic to `.253` is pulled into the tunnel; a stale or unreachable subnet router then yields `EHOSTUNREACH` even though the LAN is physically right there.

Triage on the Mac, from the same shell opencode runs in:

```bash
curl -sS -m 5 http://192.168.1.253:8011/v1/models   # curl OK but opencode fails -> per-app permission (cause 1)
route -n get 192.168.1.253 | grep -E 'interface|gateway'   # interface not your Wi-Fi/Ethernet -> cause 3
ifconfig | grep 'inet '                              # a second 192.168.x address -> interface selection
tailscale debug prefs | grep -i routeall             # true -> accept-routes is on, cause 3
```

Tailscale sidesteps all of it: same endpoint, stable path, works from any network. Confirmed `200` over the tailnet.

```bash
# triage from the client machine, in this order
curl -sS http://100.70.193.48:8011/v1/models   # tailnet — should be 200
curl -sS http://192.168.1.253:8011/v1/models   # LAN — EHOSTUNREACH means routing, not the server
```

MagicDNS name `llm.tail67d0e5.ts.net` also resolves if you prefer a name to an IP.

## Reasoning: off by default, on per request

The server runs `--reasoning auto --reasoning-format deepseek` with a server-wide default of `--chat-template-kwargs '{"enable_thinking":false}'`. So:

- A **plain request** behaves exactly like the old `--reasoning off` — no thought trace, clean `content`.
- A request carrying **`"chat_template_kwargs": {"enable_thinking": true}`** turns thinking on for that call alone.
- The trace comes back in **`message.reasoning_content`**, never in `content`. A client that ignores that field sees no difference beyond latency.

Verified end to end: same prompt, same client, **19 output tokens** with thinking off versus **1,796** with it on.

Two things that do *not* work, both tested:

- **`reasoning_effort` is inert.** `low` and `high` both produced zero reasoning. Tell clients the model has no effort levels — it is a boolean.
- **`reasoning_budget` is not honoured per request.** It is server-side only. With thinking on and a 700-token output cap the entire budget went to the trace and `content` came back **empty**. Budget generously — 8K+ output when thinking is on.

## pi.dev

Merge [`nemotron-models.json`](pi.dev/nemotron-models.json) into **`~/.pi/agent/models.json`** under the top-level `providers` key. Not `mcp.json` — that file is for stdio tool servers and its parser rejects provider blocks with `command is required for stdio transport`. pi.dev reloads `models.json` every time you open `/model`; no restart.

```json
{
  "providers": {
    "nemotron-local": {
      "baseUrl": "http://100.70.193.48:8011/v1",
      "api": "openai-completions",
      "apiKey": "dummy-key",
      "models": [
        {
          "id": "nemotron-3.5-lightning-30b-a3b",
          "name": "Nemotron 3.5 Lightning 30B-A3B (local B60)",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 131072,
          "maxTokens": 32768,
          "thinkingLevelMap": {
            "minimal": null, "low": null, "medium": null,
            "high": "on", "xhigh": null, "max": null
          },
          "compat": {
            "supportsDeveloperRole": true,
            "supportsReasoningEffort": false,
            "maxTokensField": "max_tokens",
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
              "enable_thinking": { "$var": "thinking.enabled" }
            }
          }
        }
      ]
    }
  }
}
```

- **`thinkingFormat: "chat-template"` + `chatTemplateKwargs`** is the piece that makes the toggle work — it maps pi's thinking state onto `chat_template_kwargs.enable_thinking`, which is what this server reads. Toggle it in the TUI the normal way; pi sends `true` or `false` accordingly.
- Deliberately **not** `thinkingFormat: "qwen-chat-template"`. That variant also sends `preserve_thinking`, which this template does not define — it uses `truncate_history_thinking`. The explicit `chat-template` form sends only the key that exists.
- **`thinkingLevelMap`** collapses pi's effort ladder to a single on/off, because the model has no effort levels. Only `high` is mapped; everything else is `null` so it does not appear as a choice.
- **`supportsDeveloperRole: true`** — verified, the template accepts the `developer` role. (Gemma 4 needed `false`.)
- **`supportsReasoningEffort: false`** — verified inert, see above.

> Untested here — pi.dev is not installed on `llm.local`, so this config is built from the documented schema plus the endpoint behaviour probed directly. The opencode config below *was* run end to end.

## opencode

Merge [`nemotron-opencode.json`](opencode/nemotron-opencode.json) into `~/.config/opencode/opencode.json` under `provider`. It defines **two providers against the same endpoint** — that is the toggle, because opencode model options are static per entry and the map key must be the served model id.

```json
{
  "provider": {
    "nemotron": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nemotron 30B-A3B (local B60)",
      "options": { "baseURL": "http://100.70.193.48:8011/v1", "apiKey": "local" },
      "models": {
        "nemotron-3.5-lightning-30b-a3b": {
          "name": "Nemotron 3.5 Lightning 30B-A3B",
          "limit": { "context": 131072, "output": 32768 }
        }
      }
    },
    "nemotron-think": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nemotron 30B-A3B — thinking (local B60)",
      "options": { "baseURL": "http://100.70.193.48:8011/v1", "apiKey": "local" },
      "models": {
        "nemotron-3.5-lightning-30b-a3b": {
          "name": "Nemotron 3.5 Lightning 30B-A3B (thinking)",
          "limit": { "context": 131072, "output": 32768 },
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "options": { "chat_template_kwargs": { "enable_thinking": true } }
        }
      }
    }
  }
}
```

Pick between them in `/models`, or on the CLI:

```bash
opencode run --model nemotron/nemotron-3.5-lightning-30b-a3b       "..."   # fast
opencode run --model nemotron-think/nemotron-3.5-lightning-30b-a3b "..."   # thinking
```

- Model-level **`options`** are merged into the request body — that is what carries `chat_template_kwargs`. Confirmed against the server: 19 output tokens on `nemotron`, 1,796 on `nemotron-think` for the same prompt.
- **`interleaved: {"field": "reasoning_content"}`** tells opencode where the trace lives so it renders as thinking rather than being dropped.
- `limit.context` / `limit.output` are what opencode uses to show remaining context; they are not sent to the server.

This merges alongside the existing `omniroute` provider — it does not replace it. `"model": "omniroute/auto/best-coding"` stays the default unless you change it.

> Note: the `opencode.service` unit on `llm.local` is currently **inactive**. Start it with `sudo systemctl start opencode` if you want the headless server on `:4096` to pick this up.

## Sampling

Server defaults come from the model's own `generation_config.json`: `temp 1.0`, `top_p 0.95`, `top_k 20`, `min_p 0.0`. NVIDIA ships `temperature: 1.0` deliberately for this model. Neither client overrides it unless you ask them to — lower it per request if agent output is too loose for edit-diff work.

## Verified against this endpoint

| Behaviour | Result |
|---|---|
| `GET /v1/models` id | `nemotron-3.5-lightning-30b-a3b` |
| `developer` role | accepted |
| Tool calling | clean OpenAI-format `tool_calls` with structured JSON arguments |
| Tool calling **with thinking on** | works — `tool_calls` and `reasoning_content` together |
| `reasoning_effort` | no effect at any level |
| Per-request `reasoning_budget` | not honoured |
| Per-request `reasoning_format` | not honoured; server-side flag only |
