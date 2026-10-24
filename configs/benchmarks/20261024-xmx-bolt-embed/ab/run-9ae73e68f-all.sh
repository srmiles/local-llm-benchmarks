#!/usr/bin/env bash
# 2026-10-24: interleaved 3-arm bench on card 2 :8020, isolates upstream #29864 (XMX multi-col MMVQ).
#   old   = prod 9d1878a43-patched-def76ec9a
#   new   = master 9ae73e68f + 6-patch SYCL series (0005 MMVQ-chunk + reorder-gate dropped, inert at MAX_COLS=8)
#   noxmx = new image, GGML_SYCL_ENABLE_XMX_ESIMD=0 (the PR's own off switch)
# Waits for build-9ae73e68f.sh. Takes prod nemotron :8011 + its watchdog down for the window; restores both (trap).
set -uo pipefail
until grep -q '^ALLDONE' /data/llm/build/build-9ae73e68f.log; do
  if ! systemctl is-active -q llama-build; then echo "build died"; exit 1; fi; sleep 30; done
OLD=llama.cpp:sycl-f16-9d1878a43-patched-def76ec9a
NEW=llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e
D=/data/llm/benchmarks/20261024-9ae73e68f
T0=$(date '+%F %T')
restore() { /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true; echo "== $(date -u +%FT%TZ) restoring llamacpp-nemotron =="; docker start llamacpp-nemotron; sudo systemctl start nemotron-wedge-watchdog 2>/dev/null || true; }
trap restore EXIT
echo "== $(date -u +%FT%TZ) stopping llamacpp-nemotron for bench window =="
sudo systemctl stop nemotron-wedge-watchdog 2>/dev/null || true
docker stop -t 120 llamacpp-nemotron; sleep 5
arms() { local m=$1 out=$2
  MODELS=$m IMAGE=$OLD LABEL=old                OUT=$out /data/llm/benchmarks/run-9ae73e68f-ab.sh
  MODELS=$m IMAGE=$NEW LABEL=new                OUT=$out /data/llm/benchmarks/run-9ae73e68f-ab.sh
  MODELS=$m IMAGE=$NEW LABEL=noxmx XMX_ESIMD=0  OUT=$out /data/llm/benchmarks/run-9ae73e68f-ab.sh
  echo "-- kernel faults since start: $(journalctl -k --since "$T0" --no-pager | grep -cE 'Engine reset|CAT error|Timedout job|GuC')"
}
for r in r1 r2 r3; do for m in ornith nemotron; do arms $m $D/$r; done; done
echo "ALLDONE $(date -u +%FT%TZ)"
