#!/usr/bin/env bash
# Experimental EXLR8 (EXL3-trellis routed experts, k3) TP=2 arm. Run on the WORKER (10.7.0.142).
# Production is kindling; this script borrows the single serving slot and always gives it back.
#
#   exlr8.sh serve [--force]                                    PRODUCTION: big config (KV 20 GiB, ctx 262,144), no deadman,
#                                                               waits for /health; no-op if already healthy unless --force
#   exlr8.sh stop                                               deliberate stop (watchdog paused 30 min)
#   exlr8.sh start [--kv BYTES] [--ctx TOKENS] [--budget MIN]   trial: stop kindling, boot exlr8, arm the deadman
#   exlr8.sh restore                                            stop exlr8, start kindling (kindling.sh start --force)
#   exlr8.sh extend MIN | disarm | status
#
# The deadman is a detached `sleep N; exlr8.sh restore`, so kindling comes back even if this session dies.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
HEAD_IP=10.7.0.87; WORKER_IP=10.7.0.142
EX_DIR='$HOME/work/exlr8'; KAI_DIR='$HOME/work/kai-main'
HEALTH="http://${HEAD_IP}:8000/health"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/exlr8"; mkdir -p "$STATE"
MAINT="${XDG_STATE_HOME:-$HOME/.local/state}/glm53-watchdog/maintenance"
SECOND_ROOT_IF=enP2p1s0f0np0
READY_TIMEOUT=1800
healthy() { curl -sf -m 10 -o /dev/null "$HEALTH"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
on() { local n="$1"; shift; if [ "$n" = head ]; then ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_IP" "$*"; else bash -c "$*"; fi; }
pause_wd() { mkdir -p "$(dirname "$MAINT")"; echo $(( $(date +%s) + $1 * 60 )) > "$MAINT"; }
arm() {  # arm <minutes>
  disarm
  nohup setsid bash -c "sleep $(( $1 * 60 )); '$HERE/exlr8.sh' restore" > "$STATE/deadman.log" 2>&1 < /dev/null &
  echo $! > "$STATE/deadman.pid"; pause_wd $(( $1 + 45 )); log "deadman armed: restore in $1 min (pid $(cat "$STATE/deadman.pid"))"
}
disarm() { [ -f "$STATE/deadman.pid" ] && { pkill -P "$(cat "$STATE/deadman.pid")" 2>/dev/null || true; kill "$(cat "$STATE/deadman.pid")" 2>/dev/null || true; rm -f "$STATE/deadman.pid"; }; return 0; }
down_all() {
  for n in head worker; do on $n "cd $EX_DIR && ./glm53f down --timeout 60 || true; $KAI_DIR/up.sh down --timeout 60 || true"; done
  for n in head worker; do [ -z "$(on $n "docker ps -q -f name=glm53")" ] || { log "FATAL: $n still has a glm53 container"; return 1; }; done
}

launch() {  # launch <kv> <ctx>: install env, flush, start mentatd, worker then head
  local KV=$1 CTX=$2 n
  for n in head worker; do
    sed -e "s/^KV_CACHE_MEMORY=.*/KV_CACHE_MEMORY=$KV/" -e "s/^MAX_MODEL_LEN=.*/MAX_MODEL_LEN=$CTX/" "$HERE/env/exlr8.$n.env" > "$STATE/env.$n"
    if [ $n = head ]; then ssh "$HEAD_IP" "cat > $EX_DIR/compose/.env" < "$STATE/env.$n"; else bash -c "cat > $EX_DIR/compose/.env" < "$STATE/env.$n"; fi
    on $n "sudo -n ip -4 addr flush dev $SECOND_ROOT_IF; cd \$HOME/work/mentat && docker compose -f mentatd.yaml up -d" >/dev/null
  done
  GLM53_PREFLIGHT_GMU=0.86 bash "$REPO/recipes/scripts/prelaunch_flush.sh" "${HEAD_IP},127.0.0.1"
  log "launching exlr8 (kv=$KV ctx=$CTX) worker then head"
  on worker "cd $EX_DIR && ./glm53f up -d"; on head "cd $EX_DIR && ./glm53f up -d"
}

CMD="${1:-}"; shift || true
case "$CMD" in
  serve)
    FORCE=0; [ "${1:-}" = --force ] && FORCE=1
    ip -4 -o addr | grep -q " ${WORKER_IP}/" || { echo "run on the worker"; exit 1; }
    if [ $FORCE = 0 ] && healthy && on head "docker inspect glm53 --format '{{range .Mounts}}{{.Source}} {{end}}'" 2>/dev/null | grep -q work/exlr8; then
      log "exlr8 already serving and healthy"; exit 0
    fi
    for n in head worker; do on $n "test -f /models/exlr8/glm-5.3-flash/exlr8-release/config.json && test -d $EX_DIR/tree" || { echo "$n: exlr8 weights or checkout missing"; exit 1; }; done
    disarm; pause_wd 60; trap 'rm -f "$MAINT"' EXIT
    log "stopping any running stack"; down_all
    launch 21474836480 262144
    log "waiting up to ${READY_TIMEOUT}s for $HEALTH"
    w=0; until healthy; do sleep 15; w=$((w+15)); [ $w -lt $READY_TIMEOUT ] || { trap - EXIT; echo "not healthy after ${READY_TIMEOUT}s; inspect docker logs glm53 on both nodes"; exit 1; }; done
    log "exlr8 healthy after ${w}s"
    ;;
  stop)
    pause_wd 30; log "stopping exlr8 (watchdog paused 30 min)"; down_all
    for n in head worker; do on $n "docker rm -f mentatd >/dev/null 2>&1 || true"; done ;;
  start)
    KV=6012954214; CTX=163840; BUDGET=150
    while [ $# -gt 0 ]; do case "$1" in --kv) KV=$2; shift 2;; --ctx) CTX=$2; shift 2;; --budget) BUDGET=$2; shift 2;; *) echo "bad arg $1"; exit 2;; esac; done
    ip -4 -o addr | grep -q " ${WORKER_IP}/" || { echo "run on the worker"; exit 1; }
    for n in head worker; do on $n "test -f /models/exlr8/glm-5.3-flash/exlr8-release/config.json && test -d $EX_DIR/tree" || { echo "$n: exlr8 weights or checkout missing"; exit 1; }; done
    arm "$BUDGET"
    log "stopping kindling"; down_all
    launch "$KV" "$CTX"
    log "launched; poll $HEALTH separately (never wrap the launch in timeout)"
    ;;
  restore)
    log "restore: stopping exlr8, starting kindling"; disarm
    down_all
    exec "$REPO/recipes/kindling/kindling.sh" start --force
    ;;
  extend) arm "${1:?minutes}" ;;
  disarm) disarm; log "deadman disarmed" ;;
  status)
    [ -f "$STATE/deadman.pid" ] && kill -0 "$(cat "$STATE/deadman.pid")" 2>/dev/null && log "deadman armed (pid $(cat "$STATE/deadman.pid"))" || log "deadman not armed"
    for n in head worker; do log "$n: $(on $n "docker ps --format '{{.Names}}={{.Status}}' | grep -E '^(glm53|mentatd)' | tr '\n' ' '")"; done
    curl -sf -m 10 -o /dev/null "$HEALTH" && log "health OK" || log "health NOT ok" ;;
  *) echo "usage: exlr8.sh {serve|stop|start|restore|extend|disarm|status}"; exit 2 ;;
esac
