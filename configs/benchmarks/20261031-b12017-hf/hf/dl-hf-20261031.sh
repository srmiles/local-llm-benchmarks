#!/usr/bin/env bash
set -uo pipefail
B=/data/llm/benchmarks/bench-models
dl() { echo "== $(date -u +%T) $1 $2"; hf download "$1" "$2" --local-dir "$B/$3" >/dev/null && echo "OK $2" || echo "FAIL $2"; }
dl LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q4_K_M.gguf lfm3-1.7b
dl bartowski/Spark-X2.6-4B-GGUF Spark-X2.6-4B-Q4_K_M.gguf spark-x2.6-4b
dl bartowski/Mellum2.1-12B-A2.5B-Thinking-MTP-GGUF Mellum2.1-12B-A2.5B-Thinking-MTP-Q4_K_M.gguf mellum2.1-mtp
dl bartowski/GrootTeam_Groot-19B-A3B-261023-GGUF GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf groot-19b-a3b
dl ggml-org/Flint-GGUF Flint-Q4_K_M.gguf flint
echo DLDONE $(date -u +%T)
