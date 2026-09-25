#!/usr/bin/env bash
# Periodic memory-compaction loop for GB10 hosts.
#
# Root cause this addresses (see recipes/VALIDATION.md, "Root cause found:
# head-node-only medium-order fragmentation, not a memory leak", 2026-09-21):
# the head node (the rank running the APIServer/TCPStore-rendezvous-master/
# EngineCore roles, not just the GPU worker role) develops severe
# medium-order (order-4/64KB, order-5/128KB) free-block fragmentation within
# the first ~20 minutes of a fresh boot -- confirmed via /proc/buddyinfo,
# not a memory leak (proven separately via Prometheus history: API-server
# RSS, host AnonPages, and host Cached all plateaued flat while
# MemAvailable kept eroding through 3 real earlyoom kills). Kernel
# compaction (`echo 1 > /proc/sys/vm/compact_memory`) fixes this
# immediately and safely -- confirmed live against a host serving real
# production traffic with zero disruption (compaction only reorganizes
# already-free pages; it cannot touch memory a running process holds).
#
# The existing prelaunch_flush.sh fragmentation check/compaction only runs
# ONCE, before launch. This script runs compaction periodically for the
# WHOLE serving lifetime of the job, since the fragmentation this addresses
# develops fast (within ~20 min) and a one-time pre-launch pass can't catch
# whatever develops after that.
#
# Lifetime: runs until no sparkrun node container remains on this host
# (i.e. the job stopped or was replaced) -- NOT a fixed duration like
# cache_flusher_remote.sh's 25-minute during-load window, since this needs
# to cover the entire (potentially many-hour) serving lifetime. Checking
# for the container's presence (rather than a pidfile) avoids needing to
# know the job's content-hash-derived name in advance, and avoids becoming
# an orphaned background process if the job is stopped through means other
# than this script's own invoker (a manual `sparkrun stop`, a crash, etc).
#
# Usage: compaction_flusher_remote.sh [interval_seconds]
# (run via ssh on the target host, backgrounded by the caller -- see
# prelaunch_flush.sh)
set -euo pipefail

INTERVAL="${1:-300}"
STARTUP_WAIT_SEC="${GLM53_COMPACTION_STARTUP_WAIT_SEC:-1800}"
LOG="/tmp/glm53_compaction_flusher.log"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"
}

sparkrun_node_running() {
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qE '^sparkrun_.*_node_'
}

log "compaction flusher started, interval=${INTERVAL}s"

# This script is invoked by prelaunch_flush.sh BEFORE the sparkrun container
# exists (that's the whole point of a *pre*-launch flush) -- so the
# lifetime-tracking loop below (which exits once no container is found)
# would otherwise see "no container" on its very first check and exit
# immediately, having compacted nothing. Wait for the container to actually
# appear first, bounded so a failed/abandoned launch doesn't leave this
# running forever.
waited=0
while ! sparkrun_node_running; do
  if [ "$waited" -ge "$STARTUP_WAIT_SEC" ]; then
    log "no sparkrun node container appeared within ${STARTUP_WAIT_SEC}s -- giving up, exiting"
    exit 0
  fi
  sleep 10
  waited=$((waited + 10))
done
log "sparkrun node container detected after ${waited}s -- starting periodic compaction"

# order4_5 reads the actual diagnostic signal from today's investigation
# (order-4/64KB and order-5/128KB free-block counts in zone Normal) rather
# than the nr_free_pages_blocks aggregate the existing preflight check
# uses -- that aggregate is dominated by abundant small-order blocks and
# doesn't move visibly even when these specific orders collapse to ~1.
order4_5() {
  awk '$4 == "Normal" { print $9, $10; exit }' /proc/buddyinfo 2>/dev/null || echo "? ?"
}

while sparkrun_node_running; do
  before="$(order4_5)"
  if echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1; then
    after="$(order4_5)"
    log "compacted (order-4/order-5 free blocks: ${before} -> ${after})"
  else
    log "compaction trigger failed (sudo -n refused or /proc/sys/vm/compact_memory unwritable)"
  fi
  sleep "$INTERVAL"
done

log "no sparkrun node container found -- exiting"
