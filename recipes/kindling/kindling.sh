#!/usr/bin/env bash
# Control script for the kindling stack (NVFP4 + DFlash2, docker compose via
# ~/work/kai-main/up.sh + mentatd). Run it on the WORKER (spark-276f, 10.7.0.142);
# the head (spark-2dd4, 10.7.0.87) is driven over ssh.
#
#   kindling.sh status                 read-only state of both nodes
#   kindling.sh start [--dry-run] [--force]
#   kindling.sh stop  [--dry-run]
#
# `start` and `stop` interrupt serving. --dry-run prints every mutating step
# and runs only the read-only checks. See recipes/kindling/README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

HEAD_IP="10.7.0.87"
WORKER_IP="10.7.0.142"
KAI_DIR='$HOME/work/kai-main'
MENTAT_DIR='$HOME/work/mentat'
IMAGE="spark-glm53:v11"
KAI_PIN="4d93ef7"
MENTAT_PIN="717e767"
SECOND_ROOT_IF="enP2p1s0f0np0"
PREFLIGHT_GMU="0.86"
HEALTH_URL="http://${HEAD_IP}:8000/health"
READY_TIMEOUT="${KINDLING_READY_TIMEOUT:-1800}"

CMD="${1:-}"; shift || true
DRY=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }

# on <head|worker> <command string>: run on that node (worker = local).
on() {
  local node="$1"; shift
  if [ "$node" = head ]; then ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_IP" "$*"
  else bash -c "$*"; fi
}
# mut <node> <command string>: a mutating step; printed only under --dry-run.
mut() {
  if [ "$DRY" = 1 ]; then log "[dry-run] ($1) $2"; else on "$1" "$2"; fi
}
healthy() { curl -sf -m 10 -o /dev/null "$HEALTH_URL"; }

# Watchdog maintenance lock (see kindling_watchdog.sh): epoch-seconds expiry in a file.
MAINT_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/glm53-watchdog/maintenance"
maint_set() {  # maint_set <minutes>
  [ "$DRY" = 1 ] && { log "[dry-run] would pause the watchdog for $1 min"; return 0; }
  mkdir -p "$(dirname "$MAINT_FILE")" && echo $(( $(date +%s) + $1 * 60 )) > "$MAINT_FILE"
}
maint_clear() { [ "$DRY" = 1 ] || rm -f "$MAINT_FILE"; }

require_worker() {
  ip -4 -o addr | grep -q " ${WORKER_IP}/" || die "run this on the worker (${WORKER_IP}); head is driven over ssh"
}

status() {
  for n in head worker; do
    log "$n: containers: $(on $n "docker ps -a --format '{{.Names}}={{.Status}}' | grep -E '^(glm53|mentatd|sparkrun_)' | tr '\n' ' '" || true)"
    log "$n: ${SECOND_ROOT_IF} ipv4: $(on $n "ip -4 -br addr show dev ${SECOND_ROOT_IF} | awk '{print \$3}'" || true)"
  done
  if healthy; then log "health: OK (${HEALTH_URL})"; else log "health: NOT healthy (${HEALTH_URL})"; fi
}

preflight() {
  for n in head worker; do
    on $n "docker image inspect ${IMAGE} >/dev/null 2>&1" || die "$n: image ${IMAGE} missing (build: ${KAI_DIR}/image/build.sh)"
    on $n "test -d /models/kindling/glm-5.3-flash-nvfp4 && test -d /models/kindling/dflash2-bf582e4" || die "$n: /models/kindling weights missing"
    on $n "test -d ${KAI_DIR} && test -d ${MENTAT_DIR}" || die "$n: ${KAI_DIR} or ${MENTAT_DIR} missing"
    kp="$(on $n "git -C ${KAI_DIR} rev-parse --short HEAD 2>/dev/null" || echo no-git)"
    [ "$kp" = "$KAI_PIN" ] || [ "$kp" = no-git ] || log "WARN: $n ${KAI_DIR} at $kp, expected pin $KAI_PIN"
    mp="$(on $n "git -C ${MENTAT_DIR} rev-parse --short HEAD 2>/dev/null" || echo no-git)"
    [ "$mp" = "$MENTAT_PIN" ] || log "WARN: $n ${MENTAT_DIR} at $mp, expected pin $MENTAT_PIN"
    if [ "$FORCE" != 1 ] && [ -n "$(on $n "docker ps -q -f name=sparkrun_")" ]; then
      die "$n: sparkrun (v22) containers are running. Use recipes/scripts/switch_stack.sh kindling, or --force"
    fi
  done
}

install_files() {
  # Canonical copies live in this directory; install only when different.
  for n in head worker; do
    local kenv="$HERE/env/kai.$n.env" menv="$HERE/env/mentat.$n.env"
    for pair in "$HERE/up.sh:${KAI_DIR}/up.sh" "$kenv:${KAI_DIR}/compose/.env" "$menv:${MENTAT_DIR}/.env"; do
      local src="${pair%%:*}" dst="${pair#*:}"
      if [ "$n" = head ]; then
        if ssh -o BatchMode=yes "$HEAD_IP" "cat ${dst}" 2>/dev/null | cmp -s - "$src"; then continue; fi
        log "installing $(basename "$src") -> head:${dst}"
        [ "$DRY" = 1 ] || ssh -o BatchMode=yes "$HEAD_IP" "cat > ${dst}" < "$src"
      else
        if bash -c "cat ${dst}" 2>/dev/null | cmp -s - "$src"; then continue; fi
        log "installing $(basename "$src") -> worker:${dst}"
        [ "$DRY" = 1 ] || bash -c "cat > ${dst}" < "$src"
      fi
    done
    mut $n "chmod +x ${KAI_DIR}/up.sh"
  done
}

compose_down() {
  # Stop BOTH ranks, then verify neither is left behind (partial-stop split-brain).
  mut head "${KAI_DIR}/up.sh down --timeout 60 || true"
  mut worker "${KAI_DIR}/up.sh down --timeout 60 || true"
  if [ "$DRY" != 1 ]; then
    for n in head worker; do
      [ -z "$(on $n "docker ps -q -f name=glm53")" ] || die "$n: glm53 container still running after down"
    done
  fi
}

start() {
  require_worker
  if [ "$FORCE" != 1 ] && healthy && [ -n "$(on head "docker ps -q -f name=glm53")" ]; then
    log "kindling is already serving (${HEALTH_URL} healthy); nothing to do. --force to restart."
    exit 0
  fi
  preflight
  install_files
  # Pause the watchdog for the whole (re)start; always lifted on exit so a failed start is noticed.
  maint_set $(( READY_TIMEOUT / 60 + 15 ))
  trap maint_clear EXIT

  log "Stopping any stale kindling containers (restart policy may have revived them after a reboot)"
  compose_down

  log "Removing IPv4 from the second ConnectX root (${SECOND_ROOT_IF}); kindling runs single-root"
  for n in head worker; do mut $n "sudo -n ip -4 addr flush dev ${SECOND_ROOT_IF}"; done

  log "Starting mentatd on both nodes"
  for n in head worker; do mut $n "cd ${MENTAT_DIR} && docker compose -f mentatd.yaml up -d"; done

  log "Pre-launch memory flush/preflight (gmu ${PREFLIGHT_GMU})"
  if [ "$DRY" = 1 ]; then
    log "[dry-run] GLM53_PREFLIGHT_GMU=${PREFLIGHT_GMU} recipes/scripts/prelaunch_flush.sh ${HEAD_IP},127.0.0.1"
  else
    GLM53_PREFLIGHT_GMU="$PREFLIGHT_GMU" bash "$REPO/recipes/scripts/prelaunch_flush.sh" "${HEAD_IP},127.0.0.1"
  fi

  log "Launching worker first, then head"
  mut worker "${KAI_DIR}/up.sh up -d"
  mut head "${KAI_DIR}/up.sh up -d"

  [ "$DRY" = 1 ] && { log "[dry-run] would wait up to ${READY_TIMEOUT}s for ${HEALTH_URL}"; return 0; }
  log "Waiting up to ${READY_TIMEOUT}s for ${HEALTH_URL} (first boot writes weight snapshots and is slower)"
  local waited=0
  until healthy; do
    sleep 15; waited=$((waited + 15))
    [ "$waited" -lt "$READY_TIMEOUT" ] || die "not healthy after ${READY_TIMEOUT}s; inspect 'docker logs glm53' on both nodes (nothing was torn down)"
  done
  log "kindling healthy after ${waited}s"
}

stop() {
  require_worker
  maint_set 30   # deliberate stop: stays in force after exit so the watchdog neither alerts nor restarts
  log "Stopping kindling (head and worker), then mentatd"
  compose_down
  for n in head worker; do mut $n "docker rm -f mentatd >/dev/null 2>&1 || true"; done
  log "stopped"
}

case "$CMD" in
  status) status ;;
  start) start ;;
  stop) stop ;;
  *) echo "usage: $(basename "$0") {status|start|stop} [--dry-run] [--force]" >&2; exit 2 ;;
esac
