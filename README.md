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

| | |
|---|---|
| Model | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (EXL3 4bpw) |
| Recipe | `recipes/glm-5.3-flash-exl3-v21-b12xmoe-vllm.yaml` |
| Hardware | 2x DGX Spark (GB10 / SM121), TP=2 |
| MoE kernel | b12x fused-trellis (`GLM53_EXL3_B12X_MOE=1`) |
| Speculative decoding | MTP-2 (`{"method":"mtp","num_speculative_tokens":2}`) |
| Context | 262,144 tokens |
| KV cache | fp8 |
| `gpu_memory_utilization` | 0.82 |
| Decode throughput | ~34.2-34.4 tok/s median, single-stream |
| TTFT | ~0.246-0.257s median |

That table goes stale the moment the config changes — `recipes/VALIDATION.md`
is the actual source of truth, dated and append-only. If this README and
VALIDATION.md ever disagree, believe VALIDATION.md.

## Repo map

```
recipes/
  glm-5.3-flash-exl3-v21-b12xmoe-vllm.yaml   current production recipe
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

```bash
# Stop whatever's running (safe even if nothing is)
sparkrun stop <job-id>            # or: sparkrun status, to find the job id

# Preflight: host memory fragmentation + headroom checks on both nodes.
# Not optional — this is how a real, previously-diagnosed OOM class got fixed.
bash recipes/scripts/prelaunch_flush.sh <head-ip>,<worker-ip>

# Launch. Always background it — never wrap sparkrun run in `timeout`,
# that SIGTERMs the CLI mid-boot and kills only one rank (split-brain).
nohup sparkrun run recipes/glm-5.3-flash-exl3-v21-b12xmoe-vllm.yaml > boot.log 2>&1 &
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
