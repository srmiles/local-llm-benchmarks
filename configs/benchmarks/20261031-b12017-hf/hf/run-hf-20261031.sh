#!/usr/bin/env bash
# 2026-10-31: HF candidates on card 2 :8020, image b12017 (5047748f1-patched-c9bf2e9a6), 2 rounds.
#   :8011 slot: Mellum2.1-12B-A2.5B MTP (solo + embedded MTP), Groot-19B-A3B (embedded MTP if present), Flint Q4_K_M
#   categorise: LFM3-1.7B, Spark-X2.6-4B
# References on the same image: 20261031-5047748f1/r*/ (nemotron-new, gemma-4-E2B-new).
# Owns card 2 for the whole window: stops :8011 once, restores once (trap). card2-empty.sh before every launch.
set -uo pipefail
until grep -q '^DLDONE' /data/llm/build/dl-hf-20261031.log; do
  pgrep -f dl-hf-20261031.sh >/dev/null || { echo "download script died"; exit 1; }; sleep 30; done
grep FAIL /data/llm/build/dl-hf-20261031.log && echo "!! some downloads failed - those models will be skipped"
IMG=llama.cpp:sycl-f16-5047748f1-patched-c9bf2e9a6
B=/data/llm/benchmarks/bench-models
D=/data/llm/benchmarks/20261031-hf; mkdir -p $D
T0=$(date '+%F %T')
restore() { /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true; echo "== $(date -u +%FT%TZ) restoring llamacpp-nemotron =="; docker start llamacpp-nemotron; }
trap restore EXIT
nextn() { PYTHONPATH=/data/llm/build/llama.cpp/gguf-py python3 -c "
import sys,gguf; r=gguf.GGUFReader(sys.argv[1])
v=[int(f.parts[-1][0]) for k,f in r.fields.items() if k.endswith('nextn_predict_layers')]
print(v[0] if v else 0)" "$1" 2>/dev/null || echo 0; }
echo "== $(date -u +%FT%TZ) stopping llamacpp-nemotron =="
docker stop -t 120 llamacpp-nemotron; sleep 5
one() { local name=$1; shift
  echo "=== $name ==="
  local mf; mf=$(env "$@" bash -c 'echo $MODEL_DIR/$MODEL')
  [[ -f $mf ]] || { echo "!! $name model missing ($mf), skipped"; return 1; }
  /data/llm/launch/card2-empty.sh 120 || { echo "!! card2 not empty - ABORT window"; exit 1; }
  if ! env IMAGE=$IMG NAME=llamacpp-bench ALIAS=$name "$@" /data/llm/launch/bench-slot.sh; then
    echo "!! $name launch failed"; docker logs --tail 40 llamacpp-bench > $D/$name.serverlog 2>&1; /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true; return 1; fi
  python3 /data/llm/benchmarks/bench-candidate.py --port 8020 --card 1 --name $name --out $D/$name.json 2>&1 | tail -30
  [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "!! $name bench failed - server log:"; docker logs --tail 40 llamacpp-bench 2>&1; }
  docker logs llamacpp-bench > $D/$name.serverlog 2>&1
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true
  echo "-- kernel faults since start: $(journalctl -k --since "$T0" --no-pager | grep -cE 'Engine reset|CAT error|Timedout job|GuC')"
}
MEL=$B/mellum2.1-mtp; GRO=$B/groot-19b-a3b; FLI=$B/flint
G_NEXTN=$(nextn $GRO/GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf); M_NEXTN=$(nextn $MEL/Mellum2.1-12B-A2.5B-Thinking-MTP-Q4_K_M.gguf)
echo "nextn layers: groot=$G_NEXTN mellum=$M_NEXTN"
for r in r1 r2; do
  one mellum-q4km-solo-$r  MODEL_DIR=$MEL MODEL=Mellum2.1-12B-A2.5B-Thinking-MTP-Q4_K_M.gguf CTX=131072 MEM=16g
  [[ $M_NEXTN -gt 0 ]] && one mellum-q4km-mtp3-$r MODEL_DIR=$MEL MODEL=Mellum2.1-12B-A2.5B-Thinking-MTP-Q4_K_M.gguf SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" CTX=131072 MEM=16g
  one groot-q4km-solo-$r   MODEL_DIR=$GRO MODEL=GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf CTX=131072 MEM=20g
  [[ $G_NEXTN -gt 0 ]] && one groot-q4km-mtp3-$r MODEL_DIR=$GRO MODEL=GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3" CTX=131072 MEM=20g
  one flint-q4km-$r        MODEL_DIR=$FLI MODEL=Flint-Q4_K_M.gguf CTX=32768 MEM=28g
  one lfm3-1.7b-q4km-$r    MODEL_DIR=$B/lfm3-1.7b MODEL=LFM3-1.7B-Q4_K_M.gguf CTX=131072 MEM=8g
  one spark-x2.6-4b-q4km-$r MODEL_DIR=$B/spark-x2.6-4b MODEL=Spark-X2.6-4B-Q4_K_M.gguf CTX=131072 MEM=8g
done
echo "ALLDONE $(date -u +%FT%TZ)"
