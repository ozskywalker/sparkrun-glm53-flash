#!/usr/bin/env bash
# Install the kindling watchdog on the WORKER (spark-276f) as a systemd timer.
#   install_watchdog.sh [--enable]     without --enable the units are installed but not started
# Needs sudo. Also needs ~/.config/glm53-watchdog.env (0600) with SLACK_WEBHOOK_URL=...
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONF="$HOME/.config/glm53-watchdog.env"
[ -r "$CONF" ] || echo "WARN: $CONF missing; Slack posts will be skipped (Grafana metrics still work)"
[ "$(stat -c %a "$CONF" 2>/dev/null || echo 600)" = 600 ] || { echo "chmod 600 $CONF first"; exit 1; }

# textfile collector file the script updates (the dir is owned by node_exporter)
sudo install -m 0644 -o luser -g luser /dev/null /var/lib/node_exporter/glm53_watchdog.prom
sed "s#@REPO@#$REPO#" "$HERE/systemd/glm53-watchdog.service" | sudo tee /etc/systemd/system/glm53-watchdog.service >/dev/null
sudo install -m 0644 "$HERE/systemd/glm53-watchdog.timer" /etc/systemd/system/glm53-watchdog.timer
sudo systemctl daemon-reload
if [ "${1:-}" = --enable ]; then
  sudo systemctl enable --now glm53-watchdog.timer
  systemctl list-timers glm53-watchdog.timer --no-pager
else
  echo "installed. Enable with: sudo systemctl enable --now glm53-watchdog.timer"
fi
