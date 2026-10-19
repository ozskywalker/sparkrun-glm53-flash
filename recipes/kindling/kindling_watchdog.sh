#!/usr/bin/env bash
# Watchdog for the kindling stack. Run on the WORKER (spark-276f) once a minute
# by glm53-watchdog.timer. One invocation = one check cycle.
#
#   kindling_watchdog.sh [once] [--dry-run]   one cycle (default)
#   kindling_watchdog.sh status               print saved state
#   kindling_watchdog.sh pause [minutes]      suppress checks/restarts (default 60)
#   kindling_watchdog.sh resume
#   kindling_watchdog.sh test-slack           post a test message
#
# Detects: /health down, glm53 container missing/exited on either rank, engine
# stall (requests running but no engine steps), failed real completion, memory
# and swap pressure, kernel OOM/Xid/earlyoom events. After FAIL_THRESHOLD
# consecutive failed cycles it alerts; if still down RESTART_DELAY seconds later
# it runs `kindling.sh start --force` (budget: RESTART_MAX per 6 h).
# Only containers that EXIST but are not healthy are auto-restarted. If glm53 is
# absent on both nodes (kindling.sh stop, compose down) or v22 is running, it
# alerts at most and never restarts. Metrics go to the node_exporter textfile dir
# (Grafana alert rules); alerts go to Slack. Config: ~/.config/glm53-watchdog.env
# (SLACK_WEBHOOK_URL=..., optional overrides of the variables below; mode 0600).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KINDLING="$HERE/kindling.sh"
HEAD_IP="10.7.0.87"
WORKER_IP="10.7.0.142"
SERVED_NAME="glm-5.3-flash-exl3-v2"
CONF="${GLM53_WATCHDOG_CONF:-$HOME/.config/glm53-watchdog.env}"
# shellcheck disable=SC1090
[ -r "$CONF" ] && . "$CONF"

START_CMD="${START_CMD:-$KINDLING start --force}"   # set START_CMD in the conf for another stack (exlr8: <repo>/recipes/exlr8/exlr8.sh serve --force)
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"      # consecutive failed cycles (~1 min each) before "down"
RESTART_DELAY="${RESTART_DELAY:-600}"      # seconds after "down" before auto-restart
RESTART_MAX="${RESTART_MAX:-2}"            # auto-restarts per 6 h
BOOT_GRACE="${BOOT_GRACE:-2100}"           # seconds after a glm53 (re)start before judging health
HOST_BOOT_GRACE="${HOST_BOOT_GRACE:-900}"  # seconds after host boot: boot launcher owns the start
STALL_SEC="${STALL_SEC:-300}"              # requests running, engine steps frozen this long
PROBE_EVERY="${PROBE_EVERY:-300}"          # real completion interval
REMIND_SEC="${REMIND_SEC:-1800}"           # Slack reminder interval while down/warn
SWAP_WARN_GIB="${SWAP_WARN_GIB:-24}"
MEM_WARN_MIB="${MEM_WARN_MIB:-32}"         # MemAvailable sits at ~50-300 MiB by design here (kernel watermark), so only a dive below this...
MEMLOW_CYCLES="${MEMLOW_CYCLES:-5}"        # ...lasting this many consecutive cycles counts
WARN_REPEAT="${WARN_REPEAT:-21600}"        # re-post the same warning category at most this often
PROM_FILE="${PROM_FILE:-/var/lib/node_exporter/glm53_watchdog.prom}"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/glm53-watchdog"
STATE_FILE="$STATE_DIR/state.env"
MAINT_FILE="$STATE_DIR/maintenance"
LOCK_FILE="$STATE_DIR/run.lock"
HEALTH_URL="http://${HEAD_IP}:8000/health"
METRICS_URL="http://${HEAD_IP}:8000/metrics"
CHAT_URL="http://${HEAD_IP}:8000/v1/chat/completions"

CMD="${1:-once}"; [ "$#" -gt 0 ] && shift
DRY=0; ARG=""
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; *) ARG="$a" ;; esac; done
[ "$CMD" = "--dry-run" ] && { CMD=once; DRY=1; }

mkdir -p "$STATE_DIR"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
now=$(date +%s)

on() {
  local n="$1"; shift
  if [ "$n" = head ]; then ssh -o BatchMode=yes -o ConnectTimeout=8 "$HEAD_IP" "$*"
  else bash -c "$*"; fi
}

slack() {  # slack <text>
  local text="[glm53-watchdog @ $(hostname -s)] $1"
  log "SLACK: $1"
  [ "$DRY" = 1 ] && return 0
  if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then log "no SLACK_WEBHOOK_URL in $CONF; not posted"; return 0; fi
  curl -s -m 10 -o /dev/null -X POST -H 'Content-type: application/json' \
    --data "$(jq -n --arg t "$text" '{text:$t}')" "$SLACK_WEBHOOK_URL" || log "slack post failed"
}

# ---- persistent state ------------------------------------------------------
fails=0; down_since=0; last_alert=0; last_state=ok; restart_times=""; restarts_total=0
last_restart=0; last_probe=0; probe_ok=1; probe_latency=0; last_iter=""; stall_since=0
sig_head=""; sig_worker=""; last_kernel=0; warn_posted=""; memlow_head=0; memlow_worker=0; budget_alert=0
load_state() { [ -r "$STATE_FILE" ] && . "$STATE_FILE"; return 0; }
save_state() {
  [ "$DRY" = 1 ] && return 0
  { for v in fails down_since last_alert last_state restart_times restarts_total last_restart \
             last_probe probe_ok probe_latency last_iter stall_since sig_head sig_worker \
             last_kernel warn_posted memlow_head memlow_worker budget_alert; do printf '%s=%q\n' "$v" "${!v}"; done; } > "$STATE_FILE.tmp" \
    && mv "$STATE_FILE.tmp" "$STATE_FILE"
}

maint_active() {
  [ -r "$MAINT_FILE" ] || return 1
  local until; until="$(cat "$MAINT_FILE" 2>/dev/null)"
  [[ "$until" =~ ^[0-9]+$ ]] && [ "$until" -gt "$now" ]
}

# ---- metrics (node_exporter textfile) --------------------------------------
declare -A M_OK MEM_AVAIL SWAP_USED
state_code() { case "$1" in ok) echo 0;; warn) echo 1;; down) echo 2;; restarting) echo 3;; paused) echo 4;; booting) echo 5;; *) echo 6;; esac; }
publish_metrics() {  # publish_metrics <state>
  local st="$1" tmp="$STATE_DIR/prom.tmp" c n
  {
    echo "# TYPE glm53_watchdog_last_run_timestamp_seconds gauge"
    echo "glm53_watchdog_last_run_timestamp_seconds $(date +%s)"
    echo "# TYPE glm53_watchdog_state gauge"
    echo "# HELP glm53_watchdog_state 0 ok,1 warn,2 down,3 restarting,4 paused,5 booting,6 other/stopped"
    echo "glm53_watchdog_state $(state_code "$st")"
    echo "glm53_watchdog_consecutive_failures $fails"
    echo "# TYPE glm53_watchdog_restarts_total counter"
    echo "glm53_watchdog_restarts_total $restarts_total"
    echo "glm53_watchdog_last_restart_timestamp_seconds $last_restart"
    echo "glm53_watchdog_probe_ok $probe_ok"
    echo "glm53_watchdog_probe_latency_seconds $probe_latency"
    for c in "${!M_OK[@]}"; do echo "glm53_watchdog_check_ok{check=\"$c\"} ${M_OK[$c]}"; done
    for n in "${!MEM_AVAIL[@]}"; do
      echo "glm53_watchdog_mem_available_bytes{node=\"$n\"} $(( ${MEM_AVAIL[$n]} * 1024 ))"
      echo "glm53_watchdog_swap_used_bytes{node=\"$n\"} $(( ${SWAP_USED[$n]} * 1024 ))"
    done
  } > "$tmp"
  if [ "$DRY" = 1 ]; then echo "--- metrics ---"; cat "$tmp"; return 0; fi
  if [ -w "$(dirname "$PROM_FILE")" ]; then mv "$tmp" "$PROM_FILE"
  elif [ -w "$PROM_FILE" ]; then cat "$tmp" > "$PROM_FILE"   # file pre-created by install_watchdog.sh, dir is not ours
  else sudo -n install -m 0644 "$tmp" "$PROM_FILE" 2>/dev/null || log "cannot write $PROM_FILE"; fi
}

# ---- collectors ------------------------------------------------------------
rank_info() {  # prints "<exists> <running> <startedAt> <restartCount>"
  on "$1" "docker inspect -f '1 {{.State.Running}} {{.State.StartedAt}} {{.RestartCount}}' glm53 2>/dev/null || echo '0 false - 0'" 2>/dev/null || echo "? false - 0"
}
meminfo() {  # prints "<MemAvailable KiB> <swap used KiB>"
  on "$1" "awk '/^MemAvailable/{a=\$2}/^SwapTotal/{t=\$2}/^SwapFree/{f=\$2}END{print a+0, t-f}' /proc/meminfo" 2>/dev/null || echo "0 0"
}
kernel_events() {  # node since-epoch
  on "$1" "journalctl -k --since @$2 -q --no-pager 2>/dev/null | grep -E 'Out of memory|oom_reaper|oom-kill|Killed process|NVRM: Xid' | tail -3; journalctl -u earlyoom --since @$2 -q --no-pager 2>/dev/null | grep -E 'sending SIG|killed process' | tail -2" 2>/dev/null
}
engine_metrics() {  # prints "<running> <iteration_count>"
  curl -sf -m 10 "$METRICS_URL" 2>/dev/null | awk '
    /^vllm:num_requests_running\{/ {r+=$2}
    /^vllm:iteration_tokens_total_count\{/ {i=$2}
    END {printf "%d %d\n", r, i}'
}
probe() {  # real completion; sets probe_ok/probe_latency
  local t0 t1 out
  t0=$(date +%s.%N)
  out=$(curl -sf -m 90 -H 'Content-Type: application/json' "$CHAT_URL" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8,\"temperature\":0}" 2>/dev/null)
  t1=$(date +%s.%N)
  probe_latency=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  if [ -n "$out" ] && [ "$(echo "$out" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)" -gt 0 ] 2>/dev/null; then probe_ok=1; else probe_ok=0; fi
  last_probe=$now
}

# ---- restart ---------------------------------------------------------------
restart_budget_ok() {
  local kept="" t n=0
  for t in $restart_times; do [ $((now - t)) -lt 21600 ] && { kept="$kept $t"; n=$((n+1)); }; done
  restart_times="${kept# }"
  [ "$n" -lt "$RESTART_MAX" ]
}
do_restart() {
  slack ":arrows_counterclockwise: Auto-restarting the stack now (${START_CMD}; boot takes up to ~30 min). Reasons: ${reasons[*]}"
  if [ "$DRY" = 1 ]; then log "[dry-run] would run $START_CMD"; return 0; fi
  restart_times="${restart_times:+$restart_times }$now"; restarts_total=$((restarts_total + 1)); last_restart=$now
  save_state; publish_metrics restarting
  if $START_CMD >> "$STATE_DIR/restart.log" 2>&1; then
    slack ":white_check_mark: stack restarted and healthy again (restart #$restarts_total; log: $STATE_DIR/restart.log on $(hostname -s))."
    fails=0; down_since=0; last_state=ok; stall_since=0; last_iter=""
  else
    slack ":x: Auto-restart FAILED (see $STATE_DIR/restart.log on $(hostname -s); also 'docker logs glm53' on both nodes). Manual intervention needed."
    down_since=$(date +%s); last_state=down
  fi
  sig_head="$(rank_info head)"; sig_worker="$(rank_info worker)"
  last_alert=$(date +%s)
}

# ---- subcommands -----------------------------------------------------------
case "$CMD" in
  status) [ -r "$STATE_FILE" ] && cat "$STATE_FILE"; maint_active && echo "PAUSED until $(date -d @"$(cat "$MAINT_FILE")")"; exit 0 ;;
  pause) echo $(( now + ${ARG:-60} * 60 )) > "$MAINT_FILE"; echo "paused until $(date -d @"$(cat "$MAINT_FILE")")"; exit 0 ;;
  resume) rm -f "$MAINT_FILE"; echo resumed; exit 0 ;;
  test-slack) slack "test message from kindling_watchdog.sh"; exit 0 ;;
  once) ;;
  *) echo "usage: $(basename "$0") {once|status|pause [min]|resume|test-slack} [--dry-run]" >&2; exit 2 ;;
esac

exec 9>"$LOCK_FILE"; flock -n 9 || { log "another cycle/restart is running; skipping"; exit 0; }
load_state

# ---- one cycle -------------------------------------------------------------
reasons=(); warns=()
M_OK[health]=1; M_OK[ranks]=1; M_OK[stall]=1; M_OK[probe]=1; M_OK[memory]=1

for n in head worker; do
  read -r m s <<<"$(meminfo $n)"; MEM_AVAIL[$n]=$m; SWAP_USED[$n]=$s
done

if maint_active; then
  log "paused (maintenance lock until $(date -d @"$(cat "$MAINT_FILE")")); no checks"
  sig_head="$(rank_info head)"; sig_worker="$(rank_info worker)"; last_kernel=$now
  fails=0; down_since=0; last_state=paused; stall_since=0; last_iter=""
  publish_metrics paused; save_state; exit 0
fi

read -r up_s _ < /proc/uptime; up_s=${up_s%.*}
if [ "$up_s" -lt "$HOST_BOOT_GRACE" ]; then
  log "host booted ${up_s}s ago; boot launcher owns the start; skipping"
  publish_metrics booting; save_state; exit 0
fi

read -r ex_h run_h st_h rc_h <<<"$(rank_info head)"
read -r ex_w run_w st_w rc_w <<<"$(rank_info worker)"
sp_ok=0; { on head "docker ps -aq -f name=sparkrun_ | head -1" 2>/dev/null | grep -q . ; } && sp_ok=1
if [ "$ex_h" = 0 ] && [ "$ex_w" = 0 ]; then
  M_OK[ranks]=0
  if [ "$sp_ok" = 1 ]; then st=other; msg="v22 (sparkrun) stack is present; kindling watchdog idle"
  else st=other; msg="no glm53 container on either node (stopped or never started)"; fi
  log "$msg"
  if [ "$last_state" != "other:$st" ] && [ "$last_state" != other ]; then
    [ "$sp_ok" = 1 ] || slack ":warning: $msg. Not auto-restarting (containers are absent, so this looks deliberate). Start with: recipes/kindling/kindling.sh start"
  fi
  last_state=other; fails=0; down_since=0; publish_metrics other; save_state; exit 0
fi

# ranks
for pair in "head:$ex_h:$run_h:$st_h:$rc_h" "worker:$ex_w:$run_w:$st_w:$rc_w"; do
  IFS=: read -r node ex run st_at rc <<<"$pair"
  if [ "$ex" = "?" ]; then reasons+=("$node unreachable (ssh)"); M_OK[ranks]=0
  elif [ "$ex" = 0 ]; then reasons+=("glm53 container missing on $node"); M_OK[ranks]=0
  elif [ "$run" != true ]; then reasons+=("glm53 not running on $node"); M_OK[ranks]=0; fi
done
cur_sig_h="$st_h|$rc_h"; cur_sig_w="$st_w|$rc_w"
[ -n "$sig_head" ] && [ "$sig_head" != "$cur_sig_h" ] && slack ":information_source: glm53 on head restarted (RestartCount $rc_h, started $st_h)"
[ -n "$sig_worker" ] && [ "$sig_worker" != "$cur_sig_w" ] && slack ":information_source: glm53 on worker restarted (RestartCount $rc_w, started $st_w)"
sig_head="$cur_sig_h"; sig_worker="$cur_sig_w"

# boot grace: newest container start
newest=0
for ts in "$st_h" "$st_w"; do e=$(date -d "$ts" +%s 2>/dev/null || echo 0); [ "$e" -gt "$newest" ] && newest=$e; done
booting=0; [ $((now - newest)) -lt "$BOOT_GRACE" ] && booting=1

# health / engine / probe
if curl -sf -m 10 -o /dev/null "$HEALTH_URL"; then health=1; else health=0; fi
if [ "$health" = 0 ]; then
  M_OK[health]=0
  [ "$booting" = 1 ] || reasons+=("/health not OK")
else
  read -r running iter <<<"$(engine_metrics)"
  if [ "${running:-0}" -gt 0 ] && [ "$iter" = "$last_iter" ]; then
    [ "$stall_since" = 0 ] && stall_since=$now
    if [ $((now - stall_since)) -ge "$STALL_SEC" ]; then
      reasons+=("engine stalled: $running request(s) running, no engine steps for $((now - stall_since))s"); M_OK[stall]=0
    fi
  else stall_since=0; fi
  last_iter="$iter"
  if [ $((now - last_probe)) -ge "$PROBE_EVERY" ] || [ "$probe_ok" = 0 ]; then probe; fi
  if [ "$probe_ok" = 0 ]; then
    M_OK[probe]=0
    # A saturated but advancing engine can legitimately time a probe out; only a probe failure
    # with an idle or frozen engine counts as down.
    if [ "${running:-0}" -gt 0 ] && [ "$stall_since" = 0 ]; then warns+=("probe::probe slow/failed (${probe_latency}s) but engine is advancing ($running running)")
    else reasons+=("real completion probe failed (${probe_latency}s)"); fi
  fi
fi
[ "$booting" = 1 ] && [ "$health" = 0 ] && log "booting (grace ${BOOT_GRACE}s), not judging /health"

# memory + kernel events (warn level: no restart)
for n in head worker; do
  avail_mib=$(( ${MEM_AVAIL[$n]} / 1024 )); swap_gib=$(( ${SWAP_USED[$n]} / 1048576 ))
  ml=memlow_$n
  if [ "$avail_mib" -lt "$MEM_WARN_MIB" ]; then printf -v "$ml" '%d' $(( ${!ml} + 1 )); else printf -v "$ml" '%d' 0; fi
  if [ "${!ml}" -ge "$MEMLOW_CYCLES" ]; then
    warns+=("mem-$n::$n MemAvailable below ${MEM_WARN_MIB} MiB for ${!ml} min (now ${avail_mib} MiB, swap used ${swap_gib} GiB)"); M_OK[memory]=0
  fi
  if [ "$swap_gib" -ge "$SWAP_WARN_GIB" ]; then
    warns+=("swap-$n::$n swap used ${swap_gib} GiB (threshold ${SWAP_WARN_GIB})"); M_OK[memory]=0
  fi
done
since=$last_kernel; [ "$since" -gt 0 ] || since=$((now - 120))
for n in head worker; do
  ev="$(kernel_events $n "$since")"
  [ -n "$ev" ] && warns+=("kernel-$n::$n kernel/earlyoom events: $(echo "$ev" | tr '\n' ';' | cut -c1-300)")
done
last_kernel=$now

# ---- state machine ---------------------------------------------------------
if [ "${#reasons[@]}" -gt 0 ]; then
  fails=$((fails + 1)); log "FAIL ($fails/$FAIL_THRESHOLD): ${reasons[*]}"
else
  if [ "$down_since" -gt 0 ]; then
    slack ":white_check_mark: kindling recovered after $(( (now - down_since) / 60 )) min down."
  fi
  fails=0; down_since=0; budget_alert=0
fi

if [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
  if [ "$down_since" -eq 0 ]; then
    down_since=$now; last_alert=$now; last_state=down
    slack ":rotating_light: kindling is DOWN: ${reasons[*]}. Auto-restart in $((RESTART_DELAY / 60)) min unless it recovers or you run: recipes/kindling/kindling_watchdog.sh pause"
  elif [ $((now - last_alert)) -ge "$REMIND_SEC" ]; then
    last_alert=$now; slack ":rotating_light: kindling still DOWN for $(( (now - down_since) / 60 )) min: ${reasons[*]}"
  fi
  if [ $((now - down_since)) -ge "$RESTART_DELAY" ]; then
    if restart_budget_ok; then do_restart
    elif [ "$budget_alert" = 0 ]; then
      budget_alert=1; slack ":sos: kindling down and auto-restart budget used up (${RESTART_MAX} per 6 h). Human needed: ${reasons[*]}"
    fi
  fi
  state=down
elif [ "${#warns[@]}" -gt 0 ] || [ "${#reasons[@]}" -gt 0 ]; then
  state=warn
  # Post each warning category at most once per WARN_REPEAT, however its numbers change.
  post=(); newposted=""
  for w in "${warns[@]}"; do
    cat="${w%%::*}"; txt="${w#*::}"
    last=$(echo "$warn_posted" | tr ' ' '\n' | sed -n "s/^${cat}=//p" | head -1)
    if [ -z "$last" ] || [ $((now - last)) -ge "$WARN_REPEAT" ]; then post+=("$txt"); newposted="$newposted $cat=$now"
    else newposted="$newposted $cat=$last"; fi
  done
  warn_posted="${newposted# }"
  [ "${#post[@]}" -gt 0 ] && slack ":warning: kindling warning (next repeat of each category in $((WARN_REPEAT / 3600)) h): $(printf '%s; ' "${post[@]}")"
else
  state=ok
fi
[ "$booting" = 1 ] && [ "$state" = ok ] && [ "$health" = 0 ] && state=booting
last_state=$state
log "state=$state fails=$fails health=$health probe_ok=$probe_ok"
publish_metrics "$state"; save_state
