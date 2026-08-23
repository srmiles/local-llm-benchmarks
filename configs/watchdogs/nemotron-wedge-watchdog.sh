#!/usr/bin/env bash
# Wedge watchdog for the Nemotron agent slot (:8011, card 2).
#
# WHY THIS IS NOT e2b-wedge-watchdog.sh
# The E2B watchdog detects a wedge as "llamacpp:prompt_tokens_total and
# tokens_predicted_total both frozen while requests_processing > 0". Those
# counters only advance when a request COMPLETES, which is fine for categorise
# (sub-second requests) and wrong here: a single legitimate Nemotron turn can
# generate for minutes, and a cold 131K prefill takes ~127s at 1,030 tok/s, so
# the E2B detector reaches its freeze threshold on perfectly healthy traffic.
# Measured 2026-08-23: a normal 4,000-token generation hit "frozen 3/6".
#
# WHAT THIS USES INSTEAD
# Intra-request progress from the server log. llama-server emits a line every
# ~2-3s while it is actually working:
#     prompt processing, n_tokens = ..., progress = ...     (prefill)
#     n_gen = ..., tg = ... t/s                             (decode)
# A wedge is: a request in flight AND no log line at all for STALL_WINDOW.
# That is exactly the signature of the 2026-08-23 incident - task 10104 stopped
# emitting at n_gen=1211, the client cancel never released the slot, and with
# --parallel 1 every later request queued behind it forever while /health
# happily returned 200.
#
# NOTE: depends on llama-server's default verbosity emitting those progress
# lines. If the launcher ever adds `-lv 0`, this detector goes blind - it will
# read every in-flight request as a stall.
set -uo pipefail

CONTAINER=${CONTAINER:-llamacpp-nemotron}
PORT=${PORT:-8011}
POLL_INTERVAL=${POLL_INTERVAL:-20}
STALL_WINDOW=${STALL_WINDOW:-90}      # seconds of total log silence that counts as "no progress"
STALL_THRESHOLD=${STALL_THRESHOLD:-2} # consecutive polls -> ~2-3 min before acting
UNREACH_THRESHOLD=${UNREACH_THRESHOLD:-3}
COOLDOWN=${COOLDOWN:-90}

LOG=/var/log/nemotron-watchdog.log
touch "$LOG" 2>/dev/null || LOG=/tmp/nemotron-watchdog.log
log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

log "start — container=$CONTAINER port=$PORT poll=${POLL_INTERVAL}s stall-window=${STALL_WINDOW}s threshold=${STALL_THRESHOLD}p"

stalled=0; unreach=0

restart() {
  local reason="$1"
  log "WEDGE DETECTED ($reason)"
  log "  last 15 log lines before restart:"
  docker logs --tail 15 "$CONTAINER" 2>&1 | while read -r l; do log "    $l"; done
  docker restart "$CONTAINER" >/dev/null 2>&1 && log "  restarted $CONTAINER"
  log "  cooldown ${COOLDOWN}s"
  sleep "$COOLDOWN"
  stalled=0; unreach=0
}

while true; do
  m=$(timeout 5 curl -s "http://localhost:${PORT}/metrics" 2>/dev/null)
  if [ -z "$m" ]; then
    unreach=$((unreach + 1))
    log "metrics unreachable ${unreach}/${UNREACH_THRESHOLD}"
    [ $unreach -ge $UNREACH_THRESHOLD ] && { restart "metrics unreachable ${unreach}p"; continue; }
    sleep "$POLL_INTERVAL"; continue
  fi
  unreach=0

  f=$(echo "$m" | awk '/^llamacpp:requests_processing/ {print $2; exit}'); f=${f:-0}
  if [ "${f%.*}" = "0" ]; then
    stalled=0
    sleep "$POLL_INTERVAL"; continue
  fi

  lines=$(docker logs --since "${STALL_WINDOW}s" "$CONTAINER" 2>&1 | wc -l)
  if [ "$lines" -eq 0 ]; then
    stalled=$((stalled + 1))
    log "in-flight ($f) but ZERO log output in ${STALL_WINDOW}s — stalled ${stalled}/${STALL_THRESHOLD}"
    [ $stalled -ge $STALL_THRESHOLD ] && { restart "in-flight with no log progress for $((STALL_WINDOW * stalled))s"; continue; }
  else
    stalled=0
  fi

  sleep "$POLL_INTERVAL"
done
