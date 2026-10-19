# kindling stack (NVFP4 + DFlash2) — production 2026-09-30 to 2026-10-19; now the fallback to exlr8 (`../exlr8/exlr8.sh`)

`nvidia/GLM-5.3-Flash-NVFP4` + DFlash2 drafter on vLLM 0.30.1rc1 (image `spark-glm53:v11`),
served by docker compose from the upstream clone `~/work/kai-main` plus `mentatd` per node. Source:
`kindlingai/glm-5.3-flash-gx10` @ 4d93ef7, drafter bf582e4 (same content as `mmastrac/glm-5.3-flash-4x-gx10`),
mentat `mmastrac/mentat` 0.19.0 (717e767). Before/after vs v22:
`../COMPARISON-v22-exl3-mtp2-vs-kindling-nvfp4-dflash2-2026-09-30.md`. Validation record:
`../VALIDATION.md`. A snapshot of these files is kept in the ai repo under
`research/glm-5.3-flash-gb10/kindling-nvfp4-dflash2/`.

## Files

| File | Purpose |
|---|---|
| `kindling.sh` | `start` / `stop` / `status`, `--dry-run`, `--force`. Run on the worker. |
| `up.sh` | compose wrapper: upstream base + 8 experimental overlays. Installed to `~/work/kai-main/up.sh`; delegates to upstream's `./glm53` (all overlays). |
| `env/kai.{head,worker}.env` | installed to `~/work/kai-main/compose/.env` (differ only in `ROLE`) |
| `env/mentat.{head,worker}.env` | installed to `~/work/mentat/.env` (differ only in `MENTAT_PEERS`) |

`kindling.sh start` installs these when they differ, so the repo copies are canonical.

## Prerequisites (per node, one-time)

- `~/work/kai-main` at 4d93ef7 and `~/work/mentat` at tag v0.19.0 / 717e767 (the head's `~/work/kai` is a plain copy
  without `.git`; the pin check tolerates that).
- Image: `spark-glm53:v11` = `FROM spark-glm53:v10` + mentat 0.19.0 binaries/shim (from `mmastrac/mentat-artifacts:0.19.0`) + `image/entrypoint.sh` and `image/fabric_ring.py` from `~/work/kai-main` 4d93ef7; build context `~/work/v11ctx` on each node. `spark-glm53:v10` = `FROM spark-glm53:v9` + `COPY entrypoint.sh`. Rollbacks kept: kai-main c748079 + `spark-glm53:v10` + mentat d805354 / `mmastrac/mentatd:0.17.1` (swap the env IMAGE lines and `kindling.sh` pins with it); older `~/work/kai` (a4671d2) + `v9` + `/models/kindling/{dflash2,cache}`.
- Weights: `/models/kindling/{glm-5.3-flash-nvfp4,dflash2-bf582e4,cache-main,logs}` (NVFP4 190 GiB + DFlash2 drafter,
  CC BY-NC-ND) on each node's local disk.
- Passwordless ssh worker -> head and `sudo -n` for `ip` and the cache flush.

## Current speeds (live, 26 h to 2026-10-10)

Prefill 2,048 tok/s of prefill time (previous build 1,920; v22 1,462). Decode per request, implied by step time:
~41 / 30 / 20 tok/s at 1 / 2 / 4+ running, level with the previous build; draft acceptance 47.0% (50.6% before),
2.60 tokens per step. Controlled single-request decode 41.1 tok/s, aggregate 56.0 / 73.2 tok/s at c2 / c4.
`MemAvailable` reads 0 on both nodes for hours (kernel clamp); 0 earlyoom kills, 0 request errors. Full table:
`../SPEED.md` and `../COMPARISON-v22-exl3-mtp2-vs-kindling-nvfp4-dflash2-2026-09-30.md` section 5.

## Operating

```bash
recipes/kindling/kindling.sh status
recipes/kindling/kindling.sh start --dry-run --force     # prints every mutating step
recipes/kindling/kindling.sh start                       # no-op if already serving
recipes/kindling/kindling.sh stop
recipes/scripts/switch_stack.sh v22 --dry-run            # switch to the v22 fallback (and back)
```

`start`: preflight (image, weights, pins, no running sparkrun) -> install env files -> compose down
on both (docker's restart policy can revive containers after a reboot) -> flush IPv4 from the second
ConnectX root -> mentatd up -> `prelaunch_flush.sh` (gmu 0.86) -> `up.sh up -d` worker then head ->
poll `/health` up to 30 min. Failure leaves everything up for inspection (`docker logs glm53`).

## Things that are different from v22

- **Single ConnectX root.** The second root (`enP2p1s0f0np0`) carries no IPv4 while kindling runs;
  `switch_stack.sh v22` re-adds `192.168.177.{87,142}/24`. Upstream uses both roots in separate
  subnets (+~11% prefill at 126K); not measured here.
- **Memory margin is thin** (MemAvailable 0.3-1.3 GiB under load, clamped to 0 for hours at a time; healthy only because the kernel
  low watermark is subtracted). Swap is 64 GiB/node, which makes earlyoom's `-s 80` rule inert.
- Max context 163,840 (v22: 262,144), `max_num_seqs` 16 (same as v22), stock prefix cache (v22: tail-evict).
- The model always emits a reasoning span, even with `enable_thinking=false`.
- Health/real-request validation rules from the main README still apply: `/health` plus a real
  completion, and check `docker logs glm53` on both nodes.

## Boot

`recipes/scripts/glm53_boot_launch.sh` (systemd `glm53-boot-launch.service`, 180 s manual-intervention
delay) defaults to `STACK=kindling` and runs `kindling.sh start --force`. `GLM53_STACK=v22`
(environment or `/etc/glm53-boot-launch.conf`) boots the v22 sparkrun recipe instead. Unvalidated
against a real reboot; attend the first one.

## Watchdog (auto-restart, Grafana, Slack)

The watchdog is stack-neutral: auto-restart runs `START_CMD` from `~/.config/glm53-watchdog.env` (default `kindling.sh start --force`; since 2026-10-19 `exlr8.sh serve --force`). `kindling_watchdog.sh` runs on the worker once a minute (`glm53-watchdog.timer`, install with
`kindling/install_watchdog.sh --enable`). It checks `/health`, both `glm53` containers, engine
stall (requests running but `iteration_tokens_total_count` frozen 5 min), a real 8-token completion
every 5 min, MemAvailable/swap and kernel OOM/Xid/earlyoom events.

- 3 failed cycles -> Slack "DOWN"; still down 10 min later -> `kindling.sh start --force` (Slack
  before and after). Budget 2 restarts per 6 h, then it only alerts. Reminders every 30 min.
- Warnings (memory, swap, kernel/earlyoom events, slow probe) never restart anything. Each category is
  posted to Slack at most once per 6 h. MemAvailable only warns below 32 MiB for 5 consecutive minutes:
  on these nodes it normally sits at ~50-300 MiB (head) because the kernel watermark is subtracted, so a
  100 MiB threshold (the first version) spammed. Swap warns at 24 GiB.
- Never restarts when no `glm53` container exists on either node (deliberate stop or v22 running),
  during a container boot grace of 35 min, within 15 min of host boot (boot launcher owns that), or
  while paused. A probe timeout while the engine is still advancing is only a warning.
- `kindling.sh start/stop` and `switch_stack.sh` pause it automatically (lock file
  `~/.local/state/glm53-watchdog/maintenance`). Manual: `kindling_watchdog.sh pause [min]|resume|status`.
- Tunables (`FAIL_THRESHOLD`, `RESTART_DELAY`, `RESTART_MAX`, `BOOT_GRACE`, ...) and
  `SLACK_WEBHOOK_URL` go in `~/.config/glm53-watchdog.env` (0600, not in the repo).
- Metrics `glm53_watchdog_*` land in `/var/lib/node_exporter/glm53_watchdog.prom` (job `nodes`,
  10.7.0.142:9100). Grafana folder "LLM Infra", group `glm53-kindling`: serving DOWN (state 2/3 for 2 m),
  watchdog silent >5 min, vLLM scrape down 3 m (independent), WARN 10 m. Test without side effects:
  `kindling_watchdog.sh once --dry-run`.
- Slack: #notify-internal-infra via the incoming webhook that `infrastructure-longbranch-truelabs`
  (`Containers/infra-pykms/standalone/docker-compose.yml`) already uses. Grafana has no contact point yet,
  so Grafana rules are visible in the UI but do not notify until one is attached.
