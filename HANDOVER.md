# Handover: DGX Spark GLM-5.3-Flash cluster

Written 2026-10-14 by the Claude Code agent running on the worker node (spark-276f). It is a snapshot for another agent or person who is documenting and assembling the wider AI-work presentation. Anything marked **[unverified]** or **[gap]** could not be confirmed from here. Secrets (Slack webhook, tokens, `auth.json`, `.env` contents) are deliberately not included.

Source of truth for everything below is `recipes/VALIDATION.md` (dated, append-only). If this file and VALIDATION.md disagree, believe VALIDATION.md.

---

## 1. What this is, in one paragraph

Two NVIDIA DGX Spark (GB10) machines are joined with 200 Gb/s ConnectX links and serve one large language model, GLM-5.3-Flash, as a single OpenAI-compatible API using tensor parallelism (TP=2). Most of the work in this repo is a long series of matched A/B comparisons of serving stacks (quant format, MoE kernel, speculative decoding, memory tuning), each recorded with exact numbers and a verdict. Production is currently the **kindling** stack (NVFP4 weights, DFlash2 speculative decoding). The previous production stack, **v22-cacheevict** (EXL3 4-bit, b12x MoE, MTP-2), is kept as a one-command fallback. An open decision for the owner is whether to move back from NVFP4 to EXL3, because NVFP4 leaves little room for context and KV cache.

---

## 2. Hardware and OS

| | Head | Worker (this machine) |
|---|---|---|
| Hostname | `spark-2dd4` | `spark-276f` |
| LAN IP (enP7s7, 10.7.0.0/24) | 10.7.0.87 | 10.7.0.142 |
| Fabric IP (enp1s0f0np0, ConnectX) | 192.168.177.1 | 192.168.177.2 |
| Role | Serves the API on :8000, runs APIServer, TCPStore master and EngineCore | Headless TP rank 1. Also runs the watchdog and the boot launcher |

Both nodes are identical in hardware:

- **SoC:** NVIDIA GB10 Grace Blackwell (SM121), 20 Arm cores (Cortex-X925 + A725), arm64.
- **Memory:** 121 GiB unified CPU/GPU memory. 64 GiB swap per node.
- **GPU driver / CUDA:** 580.178.04 / CUDA 13.0.
- **OS:** Ubuntu 24.04.5, kernel `7.0.0-1019-nvidia`. The kernel is pinned in GRUB on both nodes (fixed 2026-09-25 after the default silently reverted to an older kernel).
- **Disk:** 3.7 TB NVMe per node, **92% used on the worker** (306 GB free). Model weights live under `/models` on the same filesystem. Disk cleanup is outstanding debt.
- **Network:** gateway and DNS 10.7.0.1, domain `longbranch.lwalker.me`. The fabric link is point-to-point between the two Sparks. The second ConnectX root (`enP2p1s0f0np0`) deliberately has no IPv4 under kindling (single-root NCCL). The v22 stack used both roots in the 192.168.177.0/24 subnet.
- **Host tuning:** `vm.min_free_kbytes` 1 GiB with a large watermark scale factor. This makes `MemAvailable` read near 0 under load (head about 1.7 GiB, worker about 0.8 GiB). That is normal on these nodes, not a leak.
- **earlyoom:** running with `-m 2 -s 80`. With 64 GiB swap the swap rule is effectively inert. This is a deliberate owner decision, so do not re-ask.

---

## 3. Services and containers running right now

Snapshot taken 2026-10-14 from the worker; the head was checked only through its API.

### Containers

| Node | Container | Image | Notes |
|---|---|---|---|
| both | `glm53` | `spark-glm53:v11` | vLLM 0.30.1rc1.dev193 (commit ddd6fbca) |
| both | `mentatd` | `mmastrac/mentatd:0.19.0` | Ray replacement for control plane. Host network, ports 6379/6380, MCP/status 8082 |
| both | `dcgm-exporter` | `nvidia/dcgm-exporter:latest` | GPU metrics on :9400 |

### Host services (worker; the head runs the same exporters)

| Service | Purpose |
|---|---|
| `node_exporter` :9100 | Host metrics, plus the textfile collector for `glm53_watchdog.prom` |
| `earlyoom` | OOM guard (see above) |
| `glm53-watchdog.timer` | Runs `kindling_watchdog.sh` every minute (worker only) |
| `glm53-boot-launch.service` | Relaunches production after a reboot, with a 180 s delay for manual intervention. `STACK=kindling` by default. **Never validated against a real reboot** |
| engram (`serve`, 127.0.0.1:7437) | Agent memory service used by OpenCode |
| Docker | Container runtime |

### Listening ports on the worker

22 (ssh), 53 (DNS stub), 6379/6380 and 8082 (mentatd), 7437 (engram, loopback), 9100 (node_exporter), 9400 (dcgm), plus 11000 and 28028 **[unverified]**. 11000 is probably the NVIDIA DGX dashboard and 28028 is unidentified: identifying the owning process needs `sudo ss -ltnp`, which was blocked by the permission classifier, so it was skipped. The remaining high ports are ephemeral.

### The model server (head, `http://10.7.0.87:8000`)

| Setting | Value |
|---|---|
| API | OpenAI-compatible `/v1/*`, `/health`, `/metrics` (Prometheus) |
| Served model name | `glm-5.3-flash-exl3-v2` (legacy name, kept so clients don't change) |
| Weights | `nvidia/GLM-5.3-Flash-NVFP4` at `/models/glm-5.3-flash-nvfp4`, 190 GiB split over 2 ranks |
| Parallelism | TP=2 across the two Sparks (one GPU each) |
| Max context | 163,840 tokens per request |
| Max concurrent sequences | 16 (`max_num_seqs`) |
| KV cache | Pinned at 6,012,954,214 bytes (5.6 GiB) = 757,304 tokens, a shared pool (about 4.6 full-length sessions at once) |
| GPU memory utilisation | 0.86 (a no-op in practice because KV is pinned) |
| Speculative decoding | DFlash2 drafter (commit bf582e4), k=7, AdaptiveKScheduler. Drafter licence is **CC BY-NC-ND (non-commercial)** |
| MoE / comms | MegaMoE, ARX all-reduce, `VLLM_GLM_SP_TP=1`, single ConnectX root |
| Behaviour note | The model always emits a reasoning span, even with `enable_thinking=false` |

### Performance (live, mixed traffic, from Prometheus)

- Prefill about 2,050 tok/s. Controlled cold prefill 2,500-2,770 tok/s at 16K-122K tokens. v22 was 1,275-1,791.
- Decode per request about 41 / 30 / 20 tok/s at 1 / 2 / 4+ running requests. Controlled aggregate 41 / 56 / 73 tok/s at c1 / c2 / c4.
- DFlash2 draft acceptance 47%, 2.6 tokens per step.
- Image `spark-glm53:v11` was measured as speed-neutral against `spark-glm53:v10`.

### How it is launched

```bash
recipes/kindling/kindling.sh status | start | stop      # run on the worker
recipes/scripts/switch_stack.sh v22 --dry-run           # switch to the fallback (and back)
```

`start` does preflight, installs the env files, runs `compose down` on both nodes, flushes IPv4 from the second ConnectX root, brings up mentatd, runs `prelaunch_flush.sh`, starts the worker and then the head, and polls `/health` for up to 30 minutes. The compose project is the upstream clone `~/work/kai-main` (pinned 4d93ef7) with the repo's `up.sh` overlay wrapper. Env files in `recipes/kindling/env/` are canonical and get installed into `~/work/kai-main/compose/.env` and `~/work/mentat/.env`.

### Watchdog and alerting

- `kindling_watchdog.sh` checks `/health`, both containers, engine stall, a real 8-token completion every 5 minutes, memory/swap, and kernel OOM/Xid/earlyoom events.
- Three failed cycles trigger a Slack "DOWN" post. If still down 10 minutes later it runs `kindling.sh start --force`, with a budget of 2 restarts per 6 hours.
- `kindling.sh start/stop` and `switch_stack.sh` pause it automatically via a lock file under `~/.local/state/glm53-watchdog/`.
- Alerts go to Slack `#notify-internal-infra`; the webhook lives in `~/.config/glm53-watchdog.env` (mode 0600, not in the repo).
- Grafana alert rules exist (folder "LLM Infra", group `glm53-kindling`) but **Grafana has no contact point attached yet**, so they show in the UI without notifying.

---

## 4. Repos and agent setups

### 4.1 This repo

`/home/luser/spark-glm-5.3-flash-nvfp4-recipe`, branch `feat/sparkrun-recipe`, forked originally from `tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` and since diverged. Working tree has many **uncommitted** changes (README, SPEED.md, VALIDATION.md, boot launcher, `recipes/kindling/`, `switch_stack.sh`, comparison report, and others). No commits are made unless the owner asks.

| Path | What it is |
|---|---|
| `README.md` | Current-state table, operational rules, guidance for AI agents |
| `recipes/VALIDATION.md` | The real changelog: every promotion, rejection, root cause and incident, dated |
| `recipes/SPEED.md` | Condensed speed table by version (**lags**: still shows the c748079 + v10 table, not v11) |
| `recipes/kindling/` | Current production runbook: `kindling.sh`, `up.sh`, `env/*.env`, watchdog scripts, `install_watchdog.sh` |
| `recipes/COMPARISON-v22-exl3-mtp2-vs-kindling-nvfp4-dflash2-2026-09-30.md` | Before/after report for the stack change |
| `recipes/glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml` | sparkrun recipe for the v22 fallback |
| `recipes/build/<version>/` | Dockerfiles and vLLM overlay patches (`overlay/patch_*.py`) for every image version |
| `recipes/scripts/` | `prelaunch_flush.sh`, fragmentation check, compaction flusher, boot launcher, `switch_stack.sh`, `legacy/` |
| `recipes/probes/` | Benchmark harnesses. `probe_throughput_ab.py` and `ab_stack.py` are the house A/B methodology |

### 4.2 Other checkouts on the Sparks

| Path | What | Notes |
|---|---|---|
| `~/work/kai-main` | `kindlingai/glm-5.3-flash-gx10` @ 4d93ef7 | Current production compose project (on both nodes) |
| `~/work/mentat` | `mmastrac/mentat` @ 717e767 (v0.19.0) | mentatd compose |
| `~/work/kai` | a4671d2, plain copy on the head | Older kindling rollback (needs `spark-glm53:v9`) |
| `~/work/v11ctx` | Build context for `spark-glm53:v11` | Both nodes |
| `~/work/jl` (head) | jayleaton TensorFold-spark clone | Contents not re-inspected this session **[unverified]** |
| `~/ai` | Owner's personal "AI" monorepo | See 4.3 |

### 4.3 The `~/ai` repo (read-only inspection)

Contains Ansible playbooks and skills for the wider fleet:

- `ansible-dgx-spark/`: how the Sparks are provisioned.
- `ansible-workstation/opencode-setup`: **canonical source** of the OpenCode configuration (the file on the Sparks is rendered from it).
- `skills/` (for example `hermes-config-env`): reusable agent skills.
- `research/glm-5.3-flash-gb10/kindling-nvfp4-dflash2/`: a mirror snapshot of the kindling runbook and env files. **Caveat:** its original `README.md` was overwritten with the runbook text during the last change and is not recoverable (the directory is untracked). If the owner has the original, recreate it.

### 4.4 Agent setups

**Claude Code** (this agent). Runs here with project memory under `~/.claude/projects/.../memory/` (indexed in `MEMORY.md`, roughly 70 dated notes covering incidents and A/B outcomes; the most useful raw material for a presentation). A Grafana MCP server (stdio, `mcp-grafana`) gives read-only access to the monitoring dashboard on portainer1. Its exact registration in `~/.claude.json` was not read **[unverified]**.

**OpenCode.** Config at `~/.config/opencode/opencode.jsonc` with `AGENTS.md`, `agents/*.md` and `tui.json`, ansible-managed from the workstation repo. Tooling around it: the engram MCP memory service (127.0.0.1:7437 per node) and codegraph-tools (`cgg`, `ast-grep`, `ripwire`, `ast-edit`, `tgrep`).

> **Stale config, important:** `opencode.jsonc` on the Sparks still points at **DeepSeek-V4-Flash-0731** on both Sparks and references a nonexistent `~/localcode/nopus` plugin path, while the Sparks currently serve **GLM-5.3-Flash**. OpenCode on these machines would hit the wrong model name. This is a config drift to fix in the ansible source, not a local edit.

**OpenChamber and Hermes.** **[gap]** Neither runs on the Sparks. They are only referenced by playbooks and skills in `~/ai` (for example `skills/hermes-config-env`). Their hosts and current state must be documented from whichever machine actually runs them. A separate host named `ai-assistant-01` (10.7.0.165) appears in the Grafana dashboard; whether that is where they live is **not confirmed**.

---

## 5. How this connects to the other hosts

| Host | Address | Relationship |
|---|---|---|
| spark-2dd4 (head) | 10.7.0.87 | API endpoint for all clients (`:8000`). Scraped by Prometheus (node 9100, dcgm 9400) |
| spark-276f (worker) | 10.7.0.142 | This machine. Passwordless SSH to the head, and to the head's fabric address, is used by `kindling.sh` and the watchdog |
| Fabric link | 192.168.177.1 / .2 | Point-to-point ConnectX RDMA between the Sparks for NCCL/TP traffic. Not routed |
| portainer1 | 10.7.0.10 | Runs Grafana and Prometheus. Dashboard uid `llm-metrics` ("LLM Serving - vLLM & Hosts", folder "LLM Infra", about 59-65 panels). Read-only service account `sa-1-mcp-grafana`, so dashboards cannot be edited from agent sessions. Also a heavy live API client |
| ai-assistant-01 | 10.7.0.165 | Appears on the same dashboard as a third monitored host. Role **[unverified]** |
| llm-server2 | (LAN) | Serves a separate model (`Qwen3.8`) on :8080 |
| 10.7.0.62 and others | LAN | Observed API clients |
| Slack | external | `#notify-internal-infra` via an incoming webhook shared with the infra repo `infrastructure-longbranch-truelabs` |
| Hugging Face | external | Weights and drafter downloads (about 190 GiB per node for NVFP4) |
| GitHub upstreams | external | Tracked weekly. See next section |

Metrics path: vLLM `/metrics` and node/dcgm exporters feed Prometheus on portainer1, and the watchdog writes `glm53_watchdog_*` metrics through node_exporter's textfile collector (job `nodes`, 10.7.0.142:9100).

### Upstreams tracked (details in VALIDATION.md)

`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (base of the EXL3 lane), `Enntity/sparkglm`, `mmastrac/mentat`, `mmastrac/glm-5.3-flash-4x-gx10` (now `kindlingai/glm-5.3-flash-gx10`), `tonyd2wild/...NVFP4-DFlash2...`, `AEON-7/vllm-ultimate-dgx-spark`, `jayleaton` TensorFold-spark and `MiaAI-Lab/...TensorFold`, plus several others. Last full sweep 2026-10-09.

---

## 6. Known issues and what has been tried

Grouped by theme. Each has a detailed, dated write-up in VALIDATION.md or in the memory notes.

### 6.1 Open issues

1. **NVFP4 has too little context and KV headroom.** The 190 GiB of weights in 2x128 GB is the limit. The KV pool is pinned at 5.6 GiB and context at 163,840, against 262,144 on EXL3. The `v11` image and mentat 0.19.0 bundle did not change this (they were neutral on speed and gave no headroom). **Owner decision pending: NVFP4 vs EXL3.** A matched TensorFold (EXL3) A/B needs a production stop and a roughly 176 GB download and has not been started.
2. **Thin host-memory margin.** `MemAvailable` sits near 0 for hours; healthy only because the kernel watermark is subtracted. Earlier gmu 0.86 runs hit earlyoom kills on the EXL3 stack; with kindling there have been zero kills since promotion.
3. **NCCL shm-broadcast stall (vLLM #51921).** Occasional boot-time hang on GB10; tell is one rank at 0% GPU. Fix is stop, flush, retry. Still open upstream.
4. **Split-brain after partial stops.** One rank killed, the other spinning, while `docker ps` still says "Up". Always check `/health` plus a real completion, and look at `docker logs glm53` on both nodes. Never wrap `sparkrun run` in `timeout`.
5. **Unexplained clean API shutdown (2026-09-25).** Root cause never found despite ruling out reboot, kernel, earlyoom, Xid and cron. Watch for recurrence.
6. **Disk pressure.** Worker root is 92% used. Candidates for cleanup: `/models/kindling` copies, `/models/glm53-tf`, `/models/hub`, old images, `~/work/{kai,jl}`.
7. **Boot autolaunch untested.** `glm53-boot-launch.service` and the kernel pin have not been exercised by a real reboot. The first one should be attended.
8. **Grafana contact point not attached**, so alert rules do not notify.
9. **OpenCode config drift** (section 4.4).
10. **Docs lag.** `recipes/SPEED.md` lacks a v11 row.

### 6.2 Resolved root causes worth telling in a talk

- **Head-node memory fragmentation** caused repeated earlyoom kills and Xid 31. Not a leak: medium-order (64/128 KB) block fragmentation unique to the head's roles. Fixed with a periodic kernel compaction flusher and a corrected preflight check. Forty-two hours later: zero kills.
- **TileLang JIT cache** was ephemeral and slowed every boot. Fixed (found via Enntity/sparkglm).
- **Chat-template reasoning leak** (a "Reasoning Effort" line appearing in answers). Fixed.
- **b12x MTP CUDA-graph crash.** Scratch cache was allocated mid-capture; fixed with a fixed-capacity arena.
- **kpool stride bug.** Kernel assumed dense packing while the real stride was padded. Reproduced on GPU, fixed, 18/18 tests.
- **`thinking_token_budget`.** An earlier "not applicable" ruling was wrong (the system does run Model Runner V2); mirrored upstream's fix and verified end to end.
- **Kernel drift.** A manual reboot moved the fleet off the pinned kernel; GRUB default was stale. Fixed on both nodes.

### 6.3 Tried and rejected (do not re-test without a reason)

| Experiment | Result |
|---|---|
| DFlash2 on the EXL3 stack | -19% decode vs MTP-2 (also caused a 10 h production disruption, see lesson below) |
| MTP-3 vs MTP-2 | -4.4 to -4.9% decode |
| NCCL channel pinning (Entrpi) | about -2.3% |
| EXL3 MoE fast path | dead code under b12x; b12x wins by about 4% |
| TP=4 | Rejected; RoCE fabric fault modes in the upstream 4-node experience confirm TP=2 |
| Dense FP8 (v16) | +12% decode but -12/-15% prefill |
| Triton sparse-MLA | 2.1x slower than FlashInfer |
| ARX all-reduce port to EXL3 | about 4% decode, not worth it |
| TensorFold (jayleaton) vs kindling | TensorFold best deployable challenger (about 1,550 prefill, 46-58 decode, more memory margin); kindling fastest prefill but thinner margin. Nothing was promoted at that time |
| DFlash2 hot-protect cache patch (Reederey87 #82) | Worse (hit ratio 0.28 vs 0.42) |

### 6.4 Promoted over time (highlights)

v12 (E3 kernel to 262K context) -> v15 (FlashKDA, MoE gate dedup) -> v19 (indexer compat) -> v20 (upstream sync) -> v21-b12xmoe (+8.2% decode, -9.1% TTFT) -> v22-cacheevict (tail-evict prefix cache: replay hit ratio 0.0 to 0.42, TTFT about 38 s to about 20 s) -> **kindling 2026-09-30** (prefill +45%, decode about -8% live) -> kindling main c748079 + `spark-glm53:v10` (2026-10-09) -> **`spark-glm53:v11` + mentat 0.19.0 + kai-main 4d93ef7** (2026-10-13, speed-neutral, kept for boot robustness).

### 6.5 Process lessons recorded

- Any task that reconfigures production needs a time budget and an unconditional restore-on-exit; there is only one shared serving slot.
- Do not treat a "completed" task notification as proof that its resources are abandonable.
- The auto-mode classifier gates (`sparkrun stop`, earlyoom changes) are real; only explicit user approval works.
- Always use full image names (for example `spark-glm53:v11`, never "v11") when speaking to the owner.

### 6.6 Rollbacks kept on both nodes

1. kai-main c748079 + `spark-glm53:v10` + mentat d805354 (`mmastrac/mentatd:0.17.1`). The env `IMAGE` lines and the pins in `kindling.sh` must be reverted with it.
2. `~/work/kai` (a4671d2) + `spark-glm53:v9` + `/models/kindling/dflash2` and `cache`.
3. The v22 EXL3 sparkrun recipe via `recipes/scripts/switch_stack.sh v22`.

---

## 7. Notes for whoever assembles the presentation

- **Best story material:** the memory index at `~/.claude/projects/-home-luser-spark-glm-5-3-flash-nvfp4-recipe/memory/MEMORY.md` (one line per note, in rough chronological order), and VALIDATION.md for exact numbers. `recipes/COMPARISON-...2026-09-30.md` is the cleanest single before/after.
- **Headline numbers:** controlled cold prefill 1,275-1,791 tok/s (v22) -> 2,500-2,770 tok/s (kindling); live prefill 1,462 -> about 2,050 tok/s; live single-request decode about 41 tok/s; EXL3 262K context vs NVFP4 163K; 2 nodes with 121 GiB each.
- **A published artifact** called "twosparks" (an HTML page summarising the stack, version 8) exists in the owner's Claude artifacts; it holds the speed tables up to the v10 build.
- **Caveats on facts you can't verify from here:** OpenChamber and Hermes hosts, the role of `ai-assistant-01`, ports 11000/28028, the contents of `~/work/jl`, and the Claude MCP registrations in `~/.claude.json`. Two checks were skipped on purpose after a permission-classifier denial (a `sudo` port-owner lookup and reading `~/.claude.json`), and were not worked around.
- **Licence flags for public slides:** the DFlash2 drafter is CC BY-NC-ND (non-commercial); EXL3 checkpoint comes from Mia-AiLab; model is `zai-org/GLM-5.3-Flash`, NVFP4 quant from NVIDIA.
- **Do not publish** the Slack webhook, any `.env` or `auth.json`, or internal addresses beyond what the owner is comfortable showing.
