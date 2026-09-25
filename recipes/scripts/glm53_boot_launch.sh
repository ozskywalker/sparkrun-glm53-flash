#!/usr/bin/env bash
# Boot-time production launcher for the GLM-5.3-Flash sparkrun cluster.
# Runs once per boot via glm53-boot-launch.service (see that unit for the
# systemd wiring). Waits DELAY_SECONDS before doing anything, specifically
# so a human has a window to `systemctl stop glm53-boot-launch` and
# intervene manually if the reboot landed in a bad state (SIGTERM during
# the sleep aborts this script cleanly, nothing has touched docker yet).
set -euo pipefail

DELAY_SECONDS="${GLM53_BOOT_LAUNCH_DELAY:-180}"
REPO_DIR="/home/luser/spark-glm-5.3-flash-nvfp4-recipe"
RECIPE="recipes/glm-5.3-flash-exl3-v21-b12xmoe-vllm.yaml"
HOSTS="10.7.0.87,127.0.0.1"
HEAD_HOST="10.7.0.87"
SERVE_LOG="/tmp/glm53_boot_launch_serve.log"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "Boot launch triggered. Waiting ${DELAY_SECONDS}s (manual-intervention window -- 'systemctl stop glm53-boot-launch' to abort)."
sleep "$DELAY_SECONDS"
log "Delay elapsed, proceeding."

# Bounded wait for the head node's SSH to actually be reachable -- if both
# hosts rebooted together, the head may still be coming up after our delay.
log "Waiting for head node ($HEAD_HOST) SSH readiness (up to 300s)..."
for i in $(seq 1 60); do
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$HEAD_HOST" true 2>/dev/null; then
    log "Head node reachable."
    break
  fi
  sleep 5
  if [ "$i" -eq 60 ]; then
    log "FATAL: head node still unreachable after 300s. Aborting boot launch -- needs manual attention."
    exit 1
  fi
done

cd "$REPO_DIR"

# Defensive cleanup: containers with RestartPolicy=no persist in an
# Exited state across a reboot rather than disappearing. Clear any stale
# ones before relaunching so sparkrun doesn't collide with old state.
log "Clearing any stale sparkrun containers from before reboot..."
docker rm -f $(docker ps -aq -f name=sparkrun_) 2>/dev/null || true
ssh "$HEAD_HOST" 'docker rm -f $(docker ps -aq -f name=sparkrun_) 2>/dev/null || true'

log "Running prelaunch_flush.sh..."
bash recipes/scripts/prelaunch_flush.sh "$HOSTS"

log "Launching production ($RECIPE)..."
nohup sparkrun run "$RECIPE" > "$SERVE_LOG" 2>&1 &
disown

log "Launch command issued (backgrounded, pid $!). This service's job is done -- health is not monitored here; check $SERVE_LOG or sparkrun status."
