#!/usr/bin/env bash
# card2-empty.sh [timeout_s] — exit 0 once card 2 (0000:0b:00.0) holds < 1 GiB and :8011 is not running.
# 2026-10-24: added after a chained bench started on card 2 while llamacpp-nemotron was coming back up
# (two tenants, 23.9 GiB, DEVICE_LOST, host reboot). Source every bench launch through this.
set -uo pipefail
T=${1:-90}
# xpu-smi device id for 0b:00.0, re-derived every call (stale card tables have bitten before)
DEV=$(xpu-smi discovery 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && /Device Name/ {id=$1} /PCI BDF Address: 0000:0b:00.0/{print id; exit}')
[[ "$DEV" =~ ^[0-9]+$ ]] || { echo "!! cannot map 0000:0b:00.0 to an xpu-smi device id"; exit 1; }
for ((i = 0; i < T; i += 5)); do
  used=$(xpu-smi stats -d "$DEV" 2>/dev/null | awk '/GPU Memory Used/{for(i=1;i<=NF;i++) if($i=="current:"){print int($(i+1)); exit}}')
  if ! docker ps --format '{{.Names}}' | grep -qx llamacpp-nemotron && [[ -n "$used" && "$used" -lt 1024 ]]; then
    echo "card2 empty (dev $DEV, ${used} MiB)"; exit 0
  fi
  sleep 5
done
echo "!! card2 NOT empty after ${T}s (dev $DEV, ${used:-?} MiB; nemotron running: $(docker ps -q -f name=^llamacpp-nemotron$ | grep -q . && echo yes || echo no))"
docker ps --format '{{.Names}} {{.Image}}' | grep -E "llamacpp|tei" >&2
exit 1
