# Research project — LMCache evaluation for categorise workload

**Type:** Follow-up research + testing
**Priority:** Medium — no production impact if we don't pursue, but potentially large speedup for categorise
**Trigger:** Steve raised `https://github.com/lmcache/lmcache` 2026-08-15 during E2B wedge debugging
**Status:** Not started — seed document. Push to brain vault:llm-local when brain MCP reauthed.

## What LMCache is

External KV-cache store that sits between the inference engine and the model. Stores computed KV blocks in a tiered hierarchy (GPU → CPU → local disk → optional remote) and looks up by prompt-prefix hash on every incoming request. Cache hits skip prefill entirely: request jumps straight to decode with restored KV state.

Originally UChicago research (paper: "CacheGen" and related), now productionized under `lmcache/lmcache` on GitHub. Distinct from llama.cpp's built-in `--cache-ram` (which is per-process LCP cache) — LMCache is a shared multi-tier store designed for many-request, many-node deployments.

## Why it matters for our categorise workload

Brain-eval categorise requests almost certainly have this shape:
- Shared system prompt (categorise instructions, taxonomy definition, few-shot examples): ~3-4 K tokens
- Unique per-request payload (email body, doc excerpt, message content): ~500-2000 tokens
- Short generation (JSON classification): ~30-80 tokens

Currently, every request re-prefills the shared 3-4 K prefix. That's the dominant cost. On E2B at 2,800 tps prefill, 4 K tokens = ~1.4 s of prefill work per request. If LMCache caches that prefix, hits reduce to ~200-400 ms wall (just decode of new tokens + prefill of the delta).

Expected impact if we get 90%+ hit rate on the boilerplate:
- Categorise wall-clock: 1.5-2 s → **0.2-0.4 s per request** (~4-5× speedup)
- Total load on E2B backend drops proportionally — fewer prefill kernel dispatches = less pressure on whatever's triggering the SYCL wedge cycle
- Would let a single-backend E2B comfortably serve brain-eval's real dispatch rate without wedging

## What to verify (research questions)

1. **Backend support.** LMCache is confirmed vLLM-integrated. What about:
   - llama.cpp? (unknown — need to check repo README + issues)
   - Any Intel Arc / SYCL / IPEX-LLM integration? Unlikely but worth checking.
   - If llama.cpp isn't supported: is there a middleware / proxy pattern that can bolt LMCache in front of llama.cpp?

2. **Prompt-prefix hit rate on real brain-eval traffic.** Sample ~1000 brain-eval categorise requests, hash their common prefixes, measure what percentage share a common N-token boilerplate.

3. **Deployment shape on our stack.** LMCache options:
   - Sidecar container per llama.cpp instance
   - Shared LMCache node (one container, all llama.cpp instances hit it)
   - Cross-node: LMCache on manager.local shared between llm.local and future GPU boxes
   - Cache tier config: how much GPU VRAM vs host CPU vs disk

4. **Combines with vLLM XPU move?** Sergio Barrientos found vLLM XPU + MTP delivers +5.2× prefill / +1.8× decode over llama.cpp on B60 for MoE. If we move categorise from llama.cpp to vLLM XPU, LMCache stacks on top natively. Combined win could be dramatic.

## Testing plan (when we get to it)

Phase 1 — feasibility
- [ ] Fetch LMCache repo README + supported-engines list; confirm current llama.cpp status
- [ ] Sample brain-eval categorise prompts, measure common-prefix distribution
- [ ] If shared prefix > 60% of tokens on typical request → strong candidate; if < 30% → not worth the integration cost

Phase 2 — proof of concept (if feasibility passes)
- [ ] Stand up LMCache alongside an isolated llama.cpp E2B instance (dev host, not prod)
- [ ] Replay captured brain-eval traffic
- [ ] Measure prefill token reduction + wall-clock delta

Phase 3 — production integration
- [ ] Add LMCache to the llm.local stack (or wherever ends up owning categorise inference)
- [ ] Cutover categorise traffic behind the LMCache-fronted backend
- [ ] Monitor cache hit rate + wedge frequency (bonus: cache hits skip most of the SYCL kernel dispatch that seems to be triggering wedges)

## Non-goals

- Not a replacement for the E2B wedge investigation — LMCache reduces load but doesn't fix the underlying SYCL bug
- Not a chat/pi.dev optimization (those don't have repeated boilerplate prompts)
- Not a rerank/embed optimization (TEI + embed already fast enough)

## Adjacencies

- vLLM XPU migration (parked follow-up from 2nd B60 arrival playbook)
- llama.cpp `--cache-ram` (already used on Ornith; LMCache is the more sophisticated version)
- brain-eval request patterns (need to capture ~1000 samples to estimate hit rate)

## Sources

- `https://github.com/lmcache/lmcache` (raised by Steve 2026-08-15, not yet fetched)
- Underlying paper: search for "LMCache" or "CacheGen" on arXiv
- vLLM integration doc referenced in vLLM 0.6+ docs

## Push to brain when reachable

When next session has brain MCP reauthed:
- Vault: `llm-local` (or a new `research` vault if preferred)
- Type: `project` or `research`
- Link to: `2nd-b60-arrival-playbook`, `gemma-4-e2b-categorise`, and the vLLM XPU exploration handoff (if that exists in brain history)
