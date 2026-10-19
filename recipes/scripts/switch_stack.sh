#!/usr/bin/env bash
# Switch the production stack between kindling (docker compose) and v22
# (sparkrun). Run it on the WORKER (spark-276f). This INTERRUPTS SERVING.
#
#   switch_stack.sh kindling [--dry-run] [--yes] [--force]
#   switch_stack.sh v22      [--dry-run] [--yes] [--force]
#   switch_stack.sh status
#
# Refuses to run while requests are in flight unless --force. Without --yes it
# asks for confirmation. --dry-run prints every mutating step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KINDLING="$REPO/recipes/kindling/kindling.sh"
V22_RECIPE="recipes/glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml"
HEAD_IP="10.7.0.87"
WORKER_IP="10.7.0.142"
SECOND_ROOT_IF="enP2p1s0f0np0"
# v22 runs both ConnectX roots addressed inside the first root's 192.168.177.0/24.
HEAD_SECOND_ROOT_ADDR="192.168.177.87/24"
WORKER_SECOND_ROOT_ADDR="192.168.177.142/24"
V22_LOG="/tmp/glm53_v22_launch.log"

TARGET="${1:-}"; shift || true
DRY=0; YES=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;; --yes) YES=1 ;; --force) FORCE=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }
on() { local n="$1"; shift; if [ "$n" = head ]; then ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_IP" "$*"; else bash -c "$*"; fi; }
mut() { if [ "$DRY" = 1 ]; then log "[dry-run] ($1) $2"; else on "$1" "$2"; fi; }
MAINT_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/glm53-watchdog/maintenance"
healthy() { curl -sf -m 10 -o /dev/null "http://${HEAD_IP}:8000/health"; }
running_requests() {
  curl -sf -m 10 "http://${HEAD_IP}:8000/metrics" 2>/dev/null \
    | awk '/^vllm:num_requests_running\{/ {s+=$2} /^vllm:num_requests_waiting\{/ {s+=$2} END{print s+0}'
}

current_stack() {
  if [ -n "$(on head "docker ps -q -f name=glm53")" ]; then echo kindling
  elif [ -n "$(on head "docker ps -q -f name=sparkrun_")" ]; then echo v22
  else echo none; fi
}

guard() {
  ip -4 -o addr | grep -q " ${WORKER_IP}/" || die "run this on the worker (${WORKER_IP})"
  local cur busy; cur="$(current_stack)"
  log "current stack: $cur (healthy: $(healthy && echo yes || echo no))"
  [ "$cur" != "$TARGET" ] || [ "$FORCE" = 1 ] || die "$TARGET is already the running stack (--force to redeploy)"
  busy="$(running_requests)"
  if [ "$busy" != 0 ] && [ "$FORCE" != 1 ]; then
    die "$busy request(s) running or waiting. Wait for idle or pass --force"
  fi
  if [ "$YES" != 1 ] && [ "$DRY" != 1 ]; then
    read -r -p "Switch ${cur} -> ${TARGET}? This interrupts serving. [y/N] " ans
    [ "$ans" = y ] || [ "$ans" = Y ] || die "aborted"
  fi
}

stop_v22() {
  # Stop BOTH ranks and verify (partial-stop split-brain lesson). Never wrap in timeout.
  mut worker "cd $REPO && sparkrun stop $V22_RECIPE || true"
  if [ "$DRY" != 1 ]; then
    for n in head worker; do
      on $n 'docker rm -f $(docker ps -aq -f name=sparkrun_) >/dev/null 2>&1 || true'
      [ -z "$(on $n "docker ps -q -f name=sparkrun_")" ] || die "$n: sparkrun container still running"
    done
  else
    log "[dry-run] (both) remove leftover sparkrun_ containers and verify none remain"
  fi
}

to_kindling() {
  stop_v22
  # kindling.sh start removes the second-root IPv4 itself, so no extra step here.
  if [ "$DRY" = 1 ]; then "$KINDLING" start --dry-run --force; else "$KINDLING" start --force; fi
}

to_v22() {
  "$KINDLING" stop $([ "$DRY" = 1 ] && echo --dry-run)
  log "Restoring second-root addressing for v22"
  mut head "ip -4 addr show dev ${SECOND_ROOT_IF} | grep -q '${HEAD_SECOND_ROOT_ADDR%/*}' || sudo -n ip addr add ${HEAD_SECOND_ROOT_ADDR} dev ${SECOND_ROOT_IF}"
  mut worker "ip -4 addr show dev ${SECOND_ROOT_IF} | grep -q '${WORKER_SECOND_ROOT_ADDR%/*}' || sudo -n ip addr add ${WORKER_SECOND_ROOT_ADDR} dev ${SECOND_ROOT_IF}"
  log "Pre-launch flush"
  if [ "$DRY" = 1 ]; then log "[dry-run] recipes/scripts/prelaunch_flush.sh ${HEAD_IP},127.0.0.1 --during-load"
  else bash "$REPO/recipes/scripts/prelaunch_flush.sh" "${HEAD_IP},127.0.0.1" --during-load; fi
  log "Launching v22 (backgrounded; poll health, do not signal the launch process)"
  if [ "$DRY" = 1 ]; then log "[dry-run] nohup sparkrun run $V22_RECIPE > $V22_LOG"
  else (cd "$REPO" && nohup sparkrun run "$V22_RECIPE" > "$V22_LOG" 2>&1 &); log "launched; log: $V22_LOG. Poll: curl http://${HEAD_IP}:8000/health"; fi
}

pause_watchdog() {  # 75 min; kindling.sh start lifts it early once healthy
  [ "$DRY" = 1 ] && { log "[dry-run] would pause the kindling watchdog for 75 min"; return 0; }
  mkdir -p "$(dirname "$MAINT_FILE")" && echo $(( $(date +%s) + 75 * 60 )) > "$MAINT_FILE"
}

case "$TARGET" in
  status) "$KINDLING" status; log "stack: $(current_stack)" ;;
  kindling) guard; pause_watchdog; to_kindling ;;
  v22) guard; pause_watchdog; to_v22 ;;
  *) echo "usage: $(basename "$0") {kindling|v22|status} [--dry-run] [--yes] [--force]" >&2; exit 2 ;;
esac
