#!/usr/bin/env bash
# 2026-10-31: interleaved 3-arm bench on card 2 :8020, standard 4 models.
#   old   = llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e (prod :8002, b11856)
#   new   = master 5047748f1 (b12017) + 7-commit SYCL series
#   clean = master 5047748f1 stock
# r1,r2 = all 4 models; r3 = ornith + nemotron (prod-relevant). Arm order rotates per round.
# Waits for build-5047748f1.sh. Takes prod nemotron :8011 down for the window; restores it (trap).
set -uo pipefail
until grep -q '^ALLDONE' /data/llm/build/build-5047748f1.log; do
  if ! pgrep -f 'build-5047748f1.sh' >/dev/null; then echo "build died"; tail -20 /data/llm/build/build-5047748f1.log; exit 1; fi; sleep 30; done
OLD=llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e
NEW=llama.cpp:sycl-f16-5047748f1-patched-c9bf2e9a6
CLN=llama.cpp:sycl-f16-5047748f1-clean
D=/data/llm/benchmarks/20261031-5047748f1
T0=$(date '+%F %T')
restore() { /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true; echo "== $(date -u +%FT%TZ) restoring llamacpp-nemotron =="; docker start llamacpp-nemotron; }
trap restore EXIT
echo "== $(date -u +%FT%TZ) stopping llamacpp-nemotron for bench window =="
docker stop -t 120 llamacpp-nemotron; sleep 5
one() { MODELS=$1 IMAGE=$2 LABEL=$3 OUT=$4 /data/llm/benchmarks/run-5047748f1-ab.sh; }
arms() { local m=$1 out=$2 order=$3
  for a in $order; do case $a in old) one $m $OLD old $out;; new) one $m $NEW new $out;; clean) one $m $CLN clean $out;; esac; done
  echo "-- kernel faults since start: $(journalctl -k --since "$T0" --no-pager | grep -cE 'Engine reset|CAT error|Timedout job|GuC')"
}
for m in ornith e2b gemma26b nemotron; do arms $m $D/r1 "old new clean"; done
for m in ornith e2b gemma26b nemotron; do arms $m $D/r2 "new clean old"; done
for m in ornith nemotron;              do arms $m $D/r3 "clean old new"; done
echo "ALLDONE $(date -u +%FT%TZ)"
