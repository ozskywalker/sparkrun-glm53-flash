# sparkrun-glm53-flash

GLM-5.3-Flash, served across 2x NVIDIA DGX Spark (GB10, TP=2), via
`sparkrun` (`uv tool install sparkrun`, by scitrera.ai). EXL3 4-bit quant, a
fused-trellis MoE kernel, MTP-2 speculative decoding, and a growing set of
GB10-specific vLLM patches this project maintains itself.

This repo started as a fork of
[tonyd2wild's original GLM-5.3-Flash NVFP4 day-0 deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark).
It has since diverged on every axis that matters — quant format (NVFP4 →
EXL3), orchestration (hand-rolled launch scripts → sparkrun), MoE kernel
(stock → a custom fused-trellis kernel), speculative decoding (that
project runs DFlash2; we measured it against our own MTP-2 twice and
rejected it both times) — so it's now a standalone project.

## What's running right now

**Since 2026-10-19: the exlr8 stack** (`recipes/exlr8/`), promoted over kindling after a matched TP=2 trial
(VALIDATION.md, "EXLR8 ... trial"). Fallbacks: kindling NVFP4 (`recipes/kindling/kindling.sh start`) and
v22-cacheevict (`recipes/scripts/switch_stack.sh v22`).

| | |
|---|---|
| Model | EXLR8 release (EXL3-trellis routed experts, k3 ~3 bpw; dense linears FP8/NVFP4), from `kindlingai/glm-5.3-flash-exlr8-gx10` @ 20b80d3 |
| Launch | `recipes/exlr8/exlr8.sh serve` (docker compose, not sparkrun) |
| Image | `spark-glm53:v11` + patched vLLM tree from `~/work/exlr8/tree` |
| Speculative decoding | DFlash2 drafter, AdaptiveKScheduler |
| Context | 262,144 tokens, `max_num_seqs` 16 |
| KV cache | pinned 20 GiB = 3.05M tokens |
| Hardware | 2x DGX Spark (GB10 / SM121), TP=2, single ConnectX root |
| Controlled decode (aggregate, c1/c2/c4) | 39 / 63 / 88 tok/s (kindling 36 / 53 / 73) |
| Controlled cold prefill | 122k: ~2,290 tok/s; 32k: 2,090-2,340 (kindling 122k: ~2,420) |
| Free host memory under load | 12-14 GiB (kindling 0.3-1.3 GiB) |
| Not served | multimodal images |

Previous production, kindling (`recipes/kindling/`): `nvidia/GLM-5.3-Flash-NVFP4`, DFlash2 k=7, ctx 163,840,
KV 5.6 GiB, `spark-glm53:v11`, prefill ~2,500-2,770 cold, 0.3-1.3 GiB free memory.

Fallback v22-cacheevict (`recipes/glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml`): EXL3 4bpw, b12x
MoE, MTP-2, 262,144 context, fp8 KV, gmu 0.82. Everything below the Quickstart heading's
sparkrun commands describes that path.

That table goes stale the moment the config changes — `recipes/VALIDATION.md`
is the actual source of truth, dated and append-only. If this README and
VALIDATION.md ever disagree, believe VALIDATION.md.

## Repo map

```
recipes/
  exlr8/                                     CURRENT production: exlr8.sh (serve/stop/start/restore), env files
  kindling/                                  fallback stack: kindling.sh, up.sh, env files, watchdog, runbook
  COMPARISON-*.md                            side-by-side reports for stack changes
  scripts/switch_stack.sh                    kindling <-> v22 switch (refuses while requests in flight)
  scripts/glm53_boot_launch.sh               boot launcher (GLM53_STACK=kindling|v22)
  scripts/legacy/                            retired scripts (old TP=4 fleet watchdog)
  glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml  v22 fallback recipe (sparkrun)
  glm-5.3-flash-exl3-v21-b12xmoe-vllm.yaml   older production recipe
  glm-5.3-flash-exl3-*.yaml                  older/experimental recipes (mostly historical)
  build/glm53-exl3-v21-b12xmoe/              production build tree: Dockerfile, vLLM overlay
                                              patches (overlay/patch_*.py), tests
  build/<other versions>/                    prior build trees, kept for rollback/reference
  scripts/
    prelaunch_flush.sh                       run before every boot (see Quickstart)
    frag_check_remote.sh                     host memory-fragmentation check (called by the above)
    compaction_flusher_remote.sh             periodic kernel compaction, runs for a job's lifetime
  probes/
    probe_throughput_ab.py                   the house A/B methodology — use this for any
                                              throughput comparison, not a one-off script
    probe_sanity.py, probe_longctx*.py, ...  other benchmark/diagnostic probes
  VALIDATION.md                              full validation & incident history — the real
                                              changelog of this project. Every promotion,
                                              rejection, root-cause, and production incident
                                              is documented here, dated, in order.
  SPEED.md                                   condensed decode/prefill numbers by version
                                              (may lag VALIDATION.md — check dates)
```

## Quickstart

**Current stack (exlr8):** `recipes/exlr8/exlr8.sh serve|stop|status`. Fallback kindling: `recipes/kindling/kindling.sh start|stop|status`, see
`recipes/kindling/README.md`. The sparkrun commands below are the v22 fallback path (or use
`recipes/scripts/switch_stack.sh v22`).

```bash
# Stop whatever's running (safe even if nothing is)
sparkrun stop <job-id>            # or: sparkrun status, to find the job id

# Preflight: host memory fragmentation + headroom checks on both nodes.
# Not optional — this is how a real, previously-diagnosed OOM class got fixed.
bash recipes/scripts/prelaunch_flush.sh <head-ip>,<worker-ip>

# Launch. Always background it — never wrap sparkrun run in `timeout`,
# that SIGTERMs the CLI mid-boot and kills only one rank (split-brain).
nohup sparkrun run recipes/glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml > boot.log 2>&1 &
disown

# Poll health separately from the launch process
until curl -sf http://<head-ip>:8000/health; do sleep 15; done

# Validate with a REAL request before trusting it — health 200 alone is
# not sufficient (a container can show "Up" for hours after its actual
# server process has died; see VALIDATION.md's incident writeups)
curl http://<head-ip>:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-exl3-v2","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
```

## Operational rules (hard-won — read before touching production)

- **A running container does not mean a running server.** sparkrun
  containers are a `sleep infinity` shell; the actual vLLM process inside
  can die and the container stays "Up" indefinitely. Always check `/health`
  *and* send a real request, not just `docker ps`.
- **Never `timeout`-wrap `sparkrun run`.** It SIGTERMs the CLI mid-boot,
  killing only one rank and reproducing a split-brain hang. Background it
  with `nohup ... &` / `disown` and poll health separately.
- **Run `prelaunch_flush.sh` before every boot.** It catches host memory
  fragmentation that a raw `MemAvailable%` check misses (see
  VALIDATION.md's fragmentation root-cause writeup) and starts the
  periodic compaction flusher for the job's lifetime.
- **Tear down cleanly before relaunching.** `sparkrun stop` first, confirm
  both nodes' containers are actually gone (`docker ps -a` on both), *then*
  relaunch. A stale process from a prior attempt can silently collide with
  a new one.
- **Any task that reconfigures production for testing needs a time budget
  and an unconditional restore-on-exit.** This fleet has exactly one
  shared serving slot; there is no separate test environment. See
  VALIDATION.md's DFlash2 incident writeup for what happens without one.

## For AI agents working in this repo

Read `recipes/VALIDATION.md` before making any production change — it's
dated, append-only, and contains the reasoning behind every current
config value, plus every rejected alternative (so you don't re-test
something already measured and rejected). When you finish a piece of
work that changes production behavior, add a dated section there
following the existing format: what was tested, the exact numbers, the
verdict, and why.

The validation discipline used throughout this project: stop → preflight
→ launch (backgrounded, never `timeout`-wrapped) → poll health → validate
with a real request (not health alone) → check both nodes' logs for
errors → only then consider a boot "done." Skipping any step here is how
past incidents happened.

## Credits

- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- Quant: [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
- Orchestration: `sparkrun` (scitrera.ai)
- Originally forked from [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)'s day-0 NVFP4 deployment; diverged into its own project from there.
