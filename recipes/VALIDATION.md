# GLM-5.3-Flash sparkrun recipes — validation record

**This file holds only the currently-promoted version's validation record.**
Everything before it lives in `recipes/validation-archive/`, one file per
prior version, in the order they were promoted (or, for unpromoted
experiments, the version they were investigated under). Read this file
first; go to the archive only when you need history a specific past version
found, fixed, or ruled out.

## How to use this file (for any future agent/session)

- **Before touching the serving recipe**, read this file's own entry below
  in full — it's the currently-shipped configuration's reasoning, not just
  its flags.
- **Before re-investigating something that looks like a new bug**, grep
  `recipes/validation-archive/*.md` first. This project has hit the same
  handful of failure classes repeatedly (see "Where to look for common
  problems" below) — a new-looking crash is very often an old one.
- **When a new candidate is promoted**, move this file's current content
  into a new `recipes/validation-archive/vNN-<name>.md` (keep the exact
  content, just relocate it — don't summarize on the way out, the raw
  investigation detail is what makes the archive useful later), then replace
  this file's entry with the new version's own record. Keep this stub
  section (everything above and including "How to use this file") unchanged.
- **Auto-memory** (this session's persistent memory, not this repo) tracks
  faster-moving state — which recipe is live right now, open investigation
  threads, incidents in progress. This file is the durable, per-promotion
  record; memory is the current-status layer on top of it. If they disagree,
  trust a fresh `sparkrun status` / `git log` over either.

## Tracked upstream/sibling repos

Checked periodically ("routine upstream check") for anything applicable to
this fork. Add new repos here when the user names one, rather than letting
the rotation live only in memory/session context.

**Last full sweep: 2026-10-21** (all repos below checked to their `HEAD` of that date; see "Upstream sweep, 2026-10-21"). `brandonmmusic-max/glm-5.3-flash-exl3-4bpw` still 404s: stays dropped.

- **MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks** — the actual upstream this
  fork is built from (pinned, currently `c190db1`-lineage, drifted since).
  Source of the base image, most of the `overlay/patch_*.py` mechanisms,
  and the majority of upstream PRs this project backports. Checked to
  `main` @ `1caea9a` (2026-09-11), `HEAD` as of 2026-09-13 (quiet),
  `HEAD` again as of 2026-09-14 (`f906ee990..d8ad18311` -- a vision-
  encoder host-OOM fix, assessed not urgent, see "TP=4 vs. 2x TP=2
  assessment... 2026-09-14" below), and `HEAD` again as of 2026-09-15
  (`d8ad18311..HEAD`, 91 commits/~15 PRs -- two shipped, see "Routine
  upstream check, 2026-09-15" below).
- **Enntity/sparkglm** — sibling project, same checkpoint family, same GB10
  hardware. Source of the grouped-prefill fat-expert kernel (v9-fatfork)
  and the TileLang JIT-cache-persistence fix. Moved its `main` branch to an
  NVFP4-focused research preview; EXL3-relevant work now lives on its
  `exl3` branch specifically — check that branch, not `main`. That branch
  has itself shifted toward a research/experiments structure (qualification
  protocols, rejected-candidate writeups) rather than shipped features as
  of 2026-09-13 -- checked `exl3` @ `a8aaa229..HEAD`, see below.
- **mmastrac/mentat** — a Ray-replacement control-plane project. Tracked
  but not applicable to this fork (uses vLLM's `mp` executor, not Ray) —
  see the mentat-track-closed history if this is ever reconsidered.
- **mmastrac/glm-5.3-flash-4x-gx10** (moved 2026-09-30 to the `kindlingai/glm-5.3-flash-gx10`
  org repo: same history, same author, `mmastrac/` URL still redirects; track under either name)
  — a 4-node/TP4 GLM-5.3-Flash
  deployment. Source of the OOM-observer diagnostic mechanism
  (`TORCH_MEM_FRACTION` + CUDA-allocator observer). Deep-dived 2026-09-14
  for its TP=4-specific experience (RoCE fabric fault modes, GID
  instability, no multi-replica comparison) -- see "TP=4 vs. 2x TP=2
  assessment... 2026-09-14" below. Three secondary backport candidates
  flagged, not yet investigated: a spin-wait CPU patch, a GPU_MEM_UTIL
  ceiling data point, a `thinking_token_budget` bug report.
- **tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark** — surfaced the
  ModelOpt-format NVFP4 token-corruption bug (vLLM #54150) that's part of
  why this fork's NVFP4 lane uses a compressed-tensors checkpoint instead.
- **AEON-7/vllm-ultimate-dgx-spark** — added 2026-09-12. Independent (not a
  fork of MiaAI-Lab/sparkglm/etc.) from-source vLLM build for GB10/sm_121a,
  currently tracking vLLM v0.29.0, very active (pushed same-day as this
  check, real external contributors, 141 stars). Targets NVFP4/ModelOpt/
  compressed-tensors models (Gemma-4, Qwen3.6/3.8) with DFlash/DSpark spec
  decode -- **no EXL3, no GLM-5.3-Flash** as a first-class target, so
  nothing transfers at the quantization-scheme level. AEON's own patches
  are MIT-licensed (clean to port from). First check surfaced real,
  actionable leads on this fleet's two worst unsolved operational problems:
  see "AEON-7 initial triage" below.
- **cbertucci33/vllm-v29-glm53flash-exl3-dgx** — added 2026-09-15. Same
  target as this fork almost exactly: GLM-5.3-Flash EXL3 on 2x DGX Spark
  TP=2, vLLM 0.29.0 base. Single-author, documented as a 23-step
  integration history rather than an ongoing project (2 commits since
  2026-09-12, may go quiet). Companion HF assets: target checkpoint
  `cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced` (an abliterated/
  uncensored variant, **not** stock GLM-5.3-Flash) and a published DFlash2
  drafter `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8` trained against
  that specific target. See "cbertucci33 triage" below.
- **local-inference-lab/b12x** — added retroactively 2026-09-18 (should
  have been added when first investigated; canonical-list purpose of this
  section had drifted). SM120/SM121 CuTe DSL+Triton kernel library
  (formerly "Sparkinfer" -- see the 2026-09-17 correction to the 2026-09-16
  "fabricated" conclusion, which checked the wrong repo). EXL3-format
  compatibility confirmed via real numeric parity test (cosine sim
  0.9999995) against this fork's own production kernel. Full MoE-scale
  integration attempted and found infeasible **as a post-load black-box
  adapter**: the generic `prepare_trellis256_moe_weights()` API forces a
  `.contiguous()` copy of the already-loaded stacked gate+up trellis
  tensor, costing 47.25 GiB/TP-rank across this model's 42 real MoE
  layers -- see `b12x_moe_adapter_infeasible_2026_09_18` memory note. Note
  (2026-09-18, via Entrpi triage below): this blocker is specific to the
  black-box-adapter integration strategy, not to b12x/trellis MoE in
  general -- a from-scratch vLLM fork with a custom load-time converter
  (build the fused layout incrementally as shards stream in, never
  materializing the full standard-layout state first) avoids it entirely.
  Revisit only if this fork is ever willing to take on load-time-loader
  surgery, not a thin adapter. **Archived 2026-10-06, confirmed 2026-10-21:**
  development moved into `flashinfer-ai/flashinfer` (merged
  `flashinfer/experimental/b12x`, PR #5767, commit `fc8fdc17f70`); the
  `b12x` Python import path is preserved as a compatibility shim but there
  will be no more standalone b12x releases. **Track `flashinfer-ai/flashinfer`
  (the `b12x` subtree) going forward, not this repo** -- drop the "wait for a
  release past 1.3.0" framing, it no longer applies.
- **Entrpi/glm-5.3-flash-exl3-2x-spark** — added 2026-09-18 at user
  request. Same target model/checkpoint/hardware as this fork
  (brandonmusic's EXL3 4bpw quant, incoai DFlash2 k=7 drafter, 2x DGX
  Spark), architecturally very different approach: a maintained vLLM
  *fork* (`Entrpi/vllm-glm-5.3-flash-spark`, rebuilt from source each
  release) rather than a patch-overlay on a pinned public image. Exceptionally
  thorough public engineering log (`docs/FINDINGS.md`, 19 dated sections)
  and an explicit head-to-head against MiaAI-Lab (`docs/COMPARISON.md`) --
  the same upstream this fork is built from. See "Entrpi initial triage,
  2026-09-18" below for the full analysis: default KV dtype is bf16 (fp8/
  NVFP4 opt-in) vs our fp8-mandatory lane; DFlash2-primary vs our MTP-2;
  a real, resolved mystery about which "b12x" repo their fused-MoE kernel
  actually comes from (`tpurtell/sparkinfer-glmrt`, a different lineage
  than `local-inference-lab/b12x` above); one memory-floor-erosion finding
  ("floors age ~1.5-2 GiB per day of workload") directly relevant to this
  fleet's recurring earlyoom pattern; one cheap, measured, likely-portable
  win (`NCCL_MIN/MAX_NCHANNELS=8`); and independent confirmation that our
  `max_num_batched_tokens=7168` is safely under the indexer-oversubscription
  ceiling they measured on the same attention backend (they hit it at
  8192, we've run 7168 stably across dozens of boots).
- **brandonmmusic-max/glm-5.3-flash-exl3-4bpw** — added 2026-09-25 at user
  request. Same "brandonmusic" identity referenced elsewhere in this file
  as the HF account behind the EXL3 4bpw checkpoint family (confirmed via
  their own README's licensing back-reference). Off-target on hardware
  (SM120 discrete Blackwell, not our SM121/GB10 unified-memory nodes),
  quant (calibrated NVFP4 MLA KV cache, not our fp8), and parallelism
  (TP2/EP2/DCP2, not our plain TP2). Runtime ships as a sealed, closed
  OCI image (provenance-hashed) -- nothing to backport even in principle.
  Quiet since 2026-08-29. Low priority, tracked passively.
- **gabewillen/GLM-5.3-Flash-EXL3-2x-DGX-Sparks** — added 2026-09-25 at
  user request. First-pass triage 2026-09-27: it's a **fork of MiaAI-Lab/
  GLM-5.3-Flash-EXL3-2x-DGX-Sparks** (this project's own upstream base),
  created 2026-09-19, 0 commits ahead / 91 behind MiaAI-Lab's current
  `main` -- a stale snapshot with zero unique content, falling further
  behind every round. **Dropped from active rotation** -- everything in
  it is already covered by this project's direct MiaAI-Lab tracking above.
- **Reederey87/glm53-flash-exl3-2x-dgx-spark** — spotted 2026-09-21, same
  hardware/topology/checkpoint family as this fork, corroborates our own
  arena fix and earlyoom fight. Checked ad hoc every round since; formally
  added to this list 2026-09-27.
- **r0b0tlab/glm53-flash-exl3-dflash2-sm121** — spotted 2026-09-21, narrower
  scope (single GB10, TP=1, 2.25bpw, DFlash2-primary) -- structurally
  different deployment, not directly comparable, but audited clean against
  our own CUDA-graph capture-safety class. Checked ad hoc every round
  since; formally added to this list 2026-09-27.
- **ashhart/TensorFold** — added 2026-09-27 at user request ("keep an eye
  on it"), after a three-round investigation (see "TensorFold engine A/B"
  and its two follow-up sections above). Not a vLLM-family project --
  a from-scratch exact-decoding serving engine (MLX+CUDA). Verdict so far:
  genuine decode-speed advantage, but a 4.8-8x prefill regression against
  this fleet's actual traffic shape rules it out; not promoted. Worth
  rechecking periodically for a real batched-prefill kernel (their current
  matmul kernel has a hard 128-row bucket ceiling) or evidence of
  production use beyond its single author.
- **jayleaton/glm53-tensorfold-spark** — added 2026-09-29 at user request
  (TensorFold-specific). 57 build-time patches over TensorFold 0.3.4 for
  this exact model on 2x Spark, serving the same TR3 4bpw EXL3 weights we
  run. First pass 2026-09-29: their claimed prefill (~1,600 tok/s cold,
  RigMark 8K-64K) is in our production range (~1,275-1,790), i.e. the
  regression that sank the ashhart/TensorFold test looks largely fixed;
  decode edge (RigMark code 68.6 tok/s) plus session/batching features
  remain. See "jayleaton TensorFold-spark first pass" below. Growing fast as
  of 2026-10-21: external contributors beyond the single author now, 6 PRs
  merged in the two days before this sweep. Our 2026-09-29 recommendation
  (a matched-method A/B against real traffic) was never run.
- **kindlingai/glm-5.3-flash-exlr8-gx10** — added 2026-10-21; this is the
  actual upstream source for our current production stack (EXLR8 TP=2,
  promoted 2026-10-19 at `20b80d3`). Should have been added to this list at
  promotion time rather than only described inline in the EXLR8 section
  below. The companion patched-vLLM-tree repo cited at promotion time,
  `kindlingai/vllm-exlr8`, no longer resolves via the GitHub API or search
  as of this sweep -- likely folded into this repo's own `BASE` re-pin
  commits (e.g. "Re-pin BASE to the preserved mmastrac/vllm-openai copy")
  rather than renamed; unconfirmed, re-check if it matters later. Moving
  fast: 16 commits since our pin, including one that fixes our own
  documented "no multimodal" limitation -- see "Upstream sweep, 2026-10-21"
  below.

## Where to look for common problems

This fleet (2x NVIDIA DGX Spark GB10, unified host/GPU memory, TP=2) has a
small number of failure classes that recur across almost every version.
Check these first before assuming a new bug:

- **A boot or request dies with no Python traceback, no CUDA error, "exit
  code: None," or the server just goes quiet.** Check `journalctl -u
  earlyoom` on both hosts *first*, before anything else. This fleet runs
  close to its memory margin under load (unified host/GPU memory on GB10),
  and earlyoom killing the API server or a worker process produces exactly
  this signature. It is usually **not** correlated with whichever candidate
  is serving — see the v20 entry below for a same-session controlled A/B
  proving this against production. Standard remedy: `sparkrun stop` on the
  job, `recipes/scripts/prelaunch_flush.sh <hosts> [--during-load]`, retry.
  Auto-memory has `earlyoom_root_cause_correction` and
  `gmu_086_earlyoom_regression` with more detail (fleet-wide, not this
  file's concern to re-derive each time).
- **`torch.AcceleratorError: CUDA error: an illegal memory access was
  encountered` during `Capturing CUDA graphs (PIECEWISE)`.** Hit
  intermittently (not deterministically) across multiple versions,
  including on a completely clean, unmodified retry of the exact same
  image. Treated as transient GB10 flakiness, not a code bug, unless it
  starts reproducing deterministically. Standard remedy: same as above
  (stop + flush + retry).
- **The recurring NCCL/shm-broadcast stall** ("No available shared memory
  broadcast block found in 60 seconds," repeating indefinitely instead of
  self-resolving). This is vLLM issue #51921 (GB10/sm_121-specific), still
  open upstream as of this file's writing. A candidate fix exists (PR
  #54929) but was assessed and rejected — see the v20 entry below for why.
  Standard remedy: stop + flush + retry; if a boot is left in this state
  unattended, production stays down until someone intervenes — this has
  actually happened (a ~6.5h incident, see `validation-archive/v19-
  indexercompat.md`). **Never leave a boot in this state unsupervised past a
  sane timeout.**
- **A KV-cache / host-memory-pressure crash on very long context (~240K+
  tokens).** Traced to host memory fragmentation (`nr_free_pages_blocks`,
  not raw `%MemAvailable`), below PyTorch's own allocator layer — see
  `validation-archive/v15-combined.md` and the fragmentation pre-flight
  check already wired into `prelaunch_flush.sh`.
- **`docker ps` shows both containers "Up" but the server is unreachable
  and hasn't logged anything in minutes.** Don't trust container uptime as
  a proxy for the service being alive — check `/health` and the actual log
  tail first. A real incident (2026-09-11/12): the head's API server got an
  external SIGTERM (not a crash — no traceback, no CUDA error, no
  earlyoom, no dmesg event preceding it; looked like a `docker stop`/
  `sparkrun stop` that only reached one host) and shut down cleanly, but
  the *worker* rank never got the same signal — its process stayed alive,
  stuck in a broken NCCL/TCPStore retry loop against a head that no longer
  existed, logging a "Broken pipe" error roughly once a second and
  actively burning CPU (74 minutes of CPU time accumulated) for the
  **entire ~25 hours** until someone checked. `docker exec ... ps aux`
  showed the truth immediately: the head container had nothing left but a
  shell and `sleep infinity`; the worker still had a live but wedged
  `vllm serve --headless` process. Same underlying lesson as the #51921
  entry above (vLLM v1 can't recover a half-dead engine) plus a new one: a
  **partial** stop (one rank only) is worse than no stop at all, since it
  leaves the surviving rank spinning indefinitely rather than exiting.
  Standard remedy: `sparkrun stop` (stops both), flush, relaunch.
- **A boot-time "weights already present" false positive**, or a shard/
  snapshot count that doesn't match reality. `start.sh`'s `count_shards()`
  and `count_dflash_shard()` scope to the active (`refs/main`) snapshot
  specifically (fixed in v20) — if this regresses, it's almost certainly a
  stale-snapshot-directory issue, not a real download failure.
- **Anything touching sparse-MLA / the K-pool indexer / mamba state
  copies.** This subsystem has produced the most distinct production
  crashes in this project's history (four independent bugs across v5-v8,
  all in `validation-archive/v5.md` through `v8.md`). If a new crash looks
  like "silent worker death, no traceback," check whether it matches one of
  those four shapes before assuming it's novel.

## Currently promoted: kindling (NVFP4 + DFlash2, docker compose) (2026-09-30; bumped to main 4d93ef7 + `spark-glm53:v11` + mentat 0.19.0 on 2026-10-13)

*(Update 2026-10-13: image is now `spark-glm53:v11`, kai-main 4d93ef7, mentat 0.19.0 (717e767); see the section "spark-glm53:v11 + mentat 0.19.0" below. The paragraph that follows describes the 2026-10-09 state: `spark-glm53:v10`, c748079, mentat 0.17.1.)*

Live stack since 2026-09-30: `nvidia/GLM-5.3-Flash-NVFP4` + DFlash2 drafter (k=7, AdaptiveKScheduler),
vLLM nightly 0.30.1rc1.dev193 in image `spark-glm53:v10` (updated 2026-10-09; was v9), MegaMoE, VLLM_GLM_SP_TP=1, KV pinned to
5.6 GiB, max context 163,840, single ConnectX root. Source: `kindlingai/glm-5.3-flash-gx10` @ c748079 in `~/work/kai-main`
(same content as `mmastrac/glm-5.3-flash-4x-gx10`), drafter bf582e4. Previous config (a4671d2 in `~/work/kai`, v9,
drafter 7d74cdd, cache `/models/kindling/cache`) is kept on both nodes for rollback.

- Launch / stop / status: `recipes/kindling/kindling.sh {start|stop|status}` (runbook:
  `recipes/kindling/README.md`). Stack switching: `recipes/scripts/switch_stack.sh {kindling|v22}`.
- Before/after comparison with v22: `recipes/COMPARISON-v22-exl3-mtp2-vs-kindling-nvfp4-dflash2-2026-09-30.md`.
- Fallback: v22-cacheevict (`recipes/glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml`). Its full validation
  record (v20 -> v22, 2026-09-11 to 2026-09-28) is archived in `validation-archive/v22-cacheevict.md`;
  section references written as "below" in the tracked-repo list above point there.
- The three sections that follow are the kindling record: the upstream delta that surfaced it, the
  three-way A/B, and the promotion.

## Upstream sweep, 2026-10-09 (kindling is the live stack; production traffic stopped, kindling still up)

Delta since ~2026-09-30, read-only, 16 repos plus HF/vLLM. Nothing was restarted. Staged (additive, prod
untouched): drafter `bf582e4` at `/models/kindling/dflash2-bf582e4` on both nodes (sha256 verified),
image `spark-glm53:v10` (= `v9` + new `entrypoint.sh`, built from `~/work/kai-main`, a fresh clone at
`c748079`; `~/work/kai` is deliberately NOT updated because its overlay files are bind-mounted into the
running containers and a watchdog restart would then mix new overlays with the old image).

**Finding 1 (actionable): we serve a stale DFlash2 drafter.** `/models/kindling/dflash2/model.safetensors`
sha256 `8931dc52...` = incoai revision `7d74cdd` (08-27). HF `main` is `bf582e4` (08-31, "Checkpoint
update", byte-identical safetensors header, different weights). kindling's own tuning/numbers moved to
`bf582e4`. Trap (kindling issue #62): weight snapshots are keyed on the safetensors header hash, so swapping
the drafter directory restores the OLD weights from `/models/kindling/cache/weight-snapshots/v2-draft-*`
unless those are moved aside first. Whether `bf582e4` raises acceptance on our traffic is unmeasured.

**Finding 2: kindling main (`a4671d2..c748079`, 18 commits) changes decode and needs the new image.** Items:
#47 mHC decode weights read as bf16 (exact), #49 two-shot arx all-reduce for decode (`VLLM_ARX_TWO_SHOT_MAX_KB`
2048), #50 draft truncation (drop drafts with predicted survival < 0.3; 6 new overlay files incl. a 2,342-line
`model_runner.py`), #60 `disable_eagle_block_drop` (cache hits no longer recompute the last block; ~1/3 lower agent
TTFT claimed), #38/#41 RecoverSSM KDA-state fix and re-enabled by default, #61 flock around JIT builds, #64 spec config
built only in the entrypoint (`SPEC_EXTRA`), #67 megamoe fix for rows with no routed experts, #58 stock `glm47` tool
parser default. kindling's README TP=2 table moved 60.5/36.4/89.5 to 66.2/42.6/94.7 tok/s (code/prose/structured)
but the benchmark method changed in the same window (gate/bench), so the delta is partly methodology. Prefill
unchanged (2,931 / 2,871 tok/s). **Compatibility trap:** the new `adaptive-k.yaml` sets `SPEC_EXTRA` and no longer
puts `--speculative-config` in `EXTRA_ARGS`; the v9 entrypoint ignores `SPEC_EXTRA`, so new overlays on the old image
silently lose the per-batch draft schedule. Overlays and image must move together (hence v10 + separate clone).
mentat: `0.19.0`/`0.18.0` released (election hold, MCP-only group fixes, ghcr release); kindling main still requires
`0.17.1`, we run `0.17.1`, no action.

**Finding 3: open kindling issues relevant to us (not fixed upstream):** #84 AdaptiveKScheduler under-drafting
equilibrium on a-few-deep-agent-contexts workloads (reporter at TP=2 saw ~31% single-stream decode loss; our shape
is similar, c=2-6 long contexts; could partly explain our live ~23 tok/s per request), #83 vLLM #55960 in the nightly
we run lowers DFlash acceptance upstream (revert PR #58250 open), #86 grammar-constrained requests can be terminated
under DFlash2 (2 in 130k), #82 `tool_choice` required/named loses a call opened inside reasoning, #75 OOM-hardening
overlay for the same memory-margin problem we have (KV pin 7.2 GiB, max seqs 6, batched tokens 4096; benchmarked,
zero earlyoom kills, ~4.7% avail headroom), #81 one very wide image can crash the engine (we serve text).

**Finding 4: structural memory fix exists: kindling-spark-os** (`kindlingai/kindling-spark-os`, v0.9.5, beta). A
read-only OS image booted from GRUB beside DGX OS: 64 KiB-page kernel returns ~2 GiB and `dispramd` lends the
GB10's idle 2 GiB display carveout to CUDA (~4 GB per node). Would address exactly our 0.3 GiB MemAvailable margin.
It replaces our pinned kernel (7.0.0-1019) and boot path, so it is a fleet-level decision, not a drop-in. Trial mode
auto-reboots back to DGX OS unless `sparkos-promote` runs within 10 min. Not done.

**Finding 5: Enntity/sparkglm moved off vLLM to its own Atlas engine** (TP2 over one cable, NVFP4 + DFlash2, 4
concurrent x 512K, prefix cache with a 48 GB disk tier, carveout KV placement). Its published RigMark decode
65.7/36.3/91.6 tok/s is level with kindling's new TP=2 table; cold TTFT 61K/125K = 25.0/49.1 s (about 2,440/2,550
tok/s, below kindling's ~2,870); warm multi-turn TTFT 0.67 s and an evicted 209K conversation back in 1.2-1.5 s are
its strength. Caveats: serves 4 at a time (we run up to 16), separate engine = new operational surface. Candidate for
a matched A/B only if prefix-cache behavior becomes the bottleneck.

**Quiet / not applicable:** `mmastrac/spark-agent` no commits since 09-24. `Entrpi/*`, `r0b0tlab/*` no change.
`tonyd2wild/...NVFP4-DFlash2` is a TP2 port of knapcio's stack (6 GiB KV pin, 560K pool), superseded by kindling.
`ashhart/TensorFold` main is now a native Zig engine (Mac tensor-parallel); `jayleaton` only GPU-watch/idle-spin fixes
and tool-calling patches for its own stack. `MiaAI-Lab` now defaults to and recommends the TensorFold EXL3 pack
(10-04), v22 fallback only. `Reederey87` v1.6.0 (pos-trim, resident-tail scheduling) and `cbertucci33` (MiMo work) are
EXL3-lineage, relevant only if we return to v22. `local-inference-lab/b12x`: README replaced by a FlashInfer
migration guide (b12x is moving into FlashInfer), new NVFP4-CSF W4A16 kernels, aarch64 wheel target; no release tag
past what we track for v22. `AEON-7`: README only. `brandonmmusic-max/glm-5.3-flash-exl3-4bpw`: 404 (repo gone or
private; drop from rotation).
HF: `nvidia/GLM-5.3-Flash-NVFP4` latest `da920bb` (09-29, MTP layer 45 declared BF16 in ignore lists), which is what
we have. `RedHatAI/GLM-5.3-Flash-NVFP4` 09-18, no change. `RedHatAI/GLM-5.3-Flash-speculator.dspark-preview`
(09-25): DSpark = DFlash plus Markov and confidence heads, 5-layer, up to 8 drafts, trained on GLM-5.3-Flash outputs;
needs vLLM DSpark support (not verified in our nightly), watch. vLLM: v0.31.0 (10-05) now contains #58846 (FlashKDA fp32
state), #58454 (kpool corruption), #58720 (expert-map indexing); kindling's image still carries equivalent patches on
nightly `ddd6fbca`, no reason for us to rebase.

### A/B result (2026-10-09 14:00-14:30 EDT): candidate ~= current, no meaningful improvement

Candidate = kindling main c748079 (`~/work/kai-main`, `./glm53`, all overlays incl. draft-trunc/two-shot arx) + image
`v10` + drafter `bf582e4` (`/models/kindling/dflash2-bf582e4`, own hardlinked cache `cache-cand`, draft snapshots
regenerated). Baseline = a4671d2 + `v9` + drafter `7d74cdd`. Both measured on a restore boot, same harness
(`recipes/probes/ab_stack.py`), both served on :8001 so live clients could not contaminate them (the first attempt
on :8000 showed two real clients, 10.7.0.10 and 10.7.0.62, still sending traffic; their requests were cut off with
the user's OK). Boot: candidate first boot ~4 min (wrote new draft snapshot), restore boot ~90 s.

| Metric | Baseline | Candidate | Delta |
|---|---|---|---|
| Decode c1 (per req / agg) | 39.6 / 37.5 | 41.1 / 39.7 | +4-6% |
| Decode c2 agg | 52.2 | 56.0 | +7% |
| Decode c4 agg | 74.8 | 73.2 | -2% |
| Prefill 32K c1 | 2,356 | 1,884 | -20% (single first-request sample, likely warm-up) |
| Prefill 32K c2 agg | 2,368 | 2,380 | flat |
| Prefill 32K c4 agg | 2,038 | 2,260 | +11% |
| Prefill 122K c1 / c2 agg | 2,474 / 2,475 | 2,477 / 2,467 | flat |
| Replay 122K TTFT c1 | 3.0 s | 1.4 s | faster |
| Replay 122K TTFT c2 | 3.0 / 6.4 s | 2.6 / 1.5 s | faster |
| Spec accept rate (accepted/drafted) | 0.577 | 0.539 | lower |
| Tokens/step (1 + accepted/draft) | 2.38 | 2.34 | -2% |
| Per-position accept (0..4) | .706 .464 .095 .062 .040 | .698 .449 .109 .049 .031 | same |

Needle: all pass on both except one baseline 122K replay miss (c2, 64-token cap eaten by the reasoning span).
Verdict: the new commits and drafter change decode by a few percent, no prefill change except a c4 gain, and faster
warm replay, all within plausible run-to-run noise except the replay TTFT. Acceptance did not improve. Draft truncation
(`TAU 0.3`) did not raise acceptance on this synthetic workload; #84 (deep agent contexts) was not tested. The measurement alone did not justify
a change, but **the operator promoted the candidate anyway** (2026-10-09 ~16:25 EDT): live traffic now runs 2-5
concurrent requests, so the small edge at c2-c4 and in cached replay may matter more than in this c1-weighted
suite. Promotion: `kindling.sh` now pins `KAI_DIR=~/work/kai-main`, image v10, c748079; env files point at
`dflash2-bf582e4` and `/models/kindling/cache-main` (renamed from `cache-cand`); `up.sh` delegates to `./glm53`.
Started through `kindling.sh start --force` itself (healthy in 135 s on a restore boot, both snapshots restored),
real completion validated, watchdog live. The candidate c1 prefill dip was not re-measured; re-check live TTFT/acceptance
after a day of traffic and roll back (`~/work/kai` + v9 + old drafter/cache) if acceptance or c1 prefill is worse.

### Live follow-up (2026-10-10 19:41 EDT): 26 h on c748079 + v10 + bf582e4

Prometheus `vllm:*` over the 26 h after promotion (885 requests, avg prompt 45.9K) vs the 8 days before it (a4671d2 + v9 +
7d74cdd, 7,309 requests, avg prompt 66K). Decode-per-request raw numbers are mix-dependent; the like-for-like figure is
tokens per step divided by p50 step time at a fixed number of running requests.

| Metric | Previous build (8 d) | Current build (26 h) |
|---|---|---|
| Prefill, tok/s of prefill time | 1,920 | 2,048 (+7%) |
| Decode per request, raw | 23.6 | 30.6 tok/s |
| Step p50, c1 / c2 / c4+ | 66.6 / 91.3 / 149.8 ms | 62.8 / 86.5 / 130.5 ms |
| Implied decode, c1 / c2 / c4+ | 41.8 / 30.5 / 18.6 | 41.4 / 30.1 / 19.9 tok/s |
| Acceptance (accepted/draft), tokens/step | 0.506, 2.78 | 0.470, 2.60 |
| Per-position acceptance (new) | | .715 .460 .211 .104 .054 .030 .026 |
| Prefix-cache hit | 88.6% | 87.5% |
| TTFT p50 / p90 | 3.29 / 7.14 s | 1.53 / 5.94 s (prompt mix differs) |

Other: ITL p50 75.8 ms; max running 6; request_success since boot stop 734, length 306, error 0, abort 0; containers up 27 h
on both nodes; 0 earlyoom kill lines since 2026-10-09 16:00; swap used 4.4 GiB (head) / 1.9 GiB (worker) of 64. `MemAvailable`
p1 and minimum over the window are 0 on both nodes (kernel clamp under the low-watermark reserve); below 64 MiB for 216 min on
the head and 143 min on the worker; spot readings 719 MiB (head), 534 MiB (worker). Watchdog flags clean.

Verdict on the re-check the A/B asked for: prefill is +7% and step time is 5-13% shorter, acceptance is 3.6 points lower, and
implied decode per request is level, so there is no evidence to roll back and none of a decode gain; the raw 23.6 to 30.6 gap is
traffic mix. The single c1 32K prefill dip from the A/B has not been re-measured. Keep watching acceptance; rollback path is
`~/work/kai` + v9 + `/models/kindling/dflash2` + `/models/kindling/cache`. The side-by-side is in
`COMPARISON-v22-exl3-mtp2-vs-kindling-nvfp4-dflash2-2026-09-30.md` section 5.

### spark-glm53:v11 + mentat 0.19.0 bundled change (2026-10-13 18:04-18:30 EDT): neutral on speed, kept for boot robustness

Why: the operator wanted more context/KV headroom and a check of what kindling had added since c748079. The 4 new commits
(c748079..4d93ef7) are fabric/TP=3/reboot work only; mentat 0.18/0.19 add boot-race fixes (election hold-down until a peer is
heard, stop the daemon when it has no route, MCP cluster graph). The vLLM image embeds the mentatd binary, so bumping only the
host daemon would have broken upstream's "one mentat version everywhere" rule; both moved together.

What changed:
- `spark-glm53:v11` = `FROM spark-glm53:v10` + mentat 0.19.0 binaries (`mmastrac/mentat-artifacts:0.19.0`: `mentatd`,
  `mentatd-probe-machine`, `ray` symlink, Python shim wheel) + `image/entrypoint.sh` and `image/fabric_ring.py` from kindling
  4d93ef7. Built on both nodes from `~/work/v11ctx`; in-image `mentatd --version` and `ray --version` read 0.19.0 on both.
- `~/work/kai-main` c748079 -> 4d93ef7 (compose/, image/Dockerfile unchanged, experimental overlays only), `~/work/mentat` d805354 ->
  tag v0.19.0 (717e767), host `mmastrac/mentatd:0.19.0`. `kindling.sh` pins and `env/*.env` IMAGE lines updated.
- Cutover with production idle: `kindling.sh stop` (both ranks verified gone), checkout, `kindling.sh start`. Healthy in 135 s
  (restore boot). Real completion correct. `docker logs glm53` clean on both nodes (only the expected single-root WARNs: the
  entrypoint's "one PCIe root tops out near 110 Gb/s" and arx's "needs two RDMA devices"; both are the deliberate single-root setup).

Matched A/B, same harness and sizes as the 2026-10-09 table (`ab_stack.py`, :8000, no live traffic during the run):

| Metric | v10 / c748079 / mentat 0.17.1 | v11 / 4d93ef7 / mentat 0.19.0 |
|---|---|---|
| Decode c1 (agg) | 39.7 | 38.2 |
| Decode c2 agg | 56.0 | 55.3 |
| Decode c4 agg | 73.2 | 74.1 |
| Prefill 32K c1 / c2 agg / c4 agg | 1,884 / 2,380 / 2,260 | 1,696 / 2,368 / 2,370 |
| Prefill 122K c1 / c2 agg | 2,477 / 2,467 | 2,461 / 2,345 |
| Replay 122K TTFT c1 | 1.4 s | 1.4 s |
| Replay 122K TTFT c2 | 2.6 / 1.5 s | 2.6 / 1.4 s |
| Needle codes | all pass | all pass |

KV pool 757,304 tokens (max concurrency 4.62x at 163,840), unchanged: it is pinned at 5.6 GiB. Under load MemAvailable reads
1.7 GiB (head) and 0.8 GiB (worker), the same near-zero margin as before. Spec acceptance over the run 3,129/6,346 = 0.49.

Verdict: no speed change beyond run-to-run noise (c1 32K prefill sits low again, as it did on 10-09, consistent with a first-request
warm-up effect), and **no context/KV headroom**: the KV size is a pinned setting and the memory margin did not move, so the
limit is the 190 GiB of NVFP4 weights in 2 x 128 GB unified memory, not anything in this change. Kept as production
because nothing regressed, the mentat versions are now consistent and the boot-race fixes cost nothing. Rollback: kai-main
c748079 + `spark-glm53:v10` + mentat d805354 / `mmastrac/mentatd:0.17.1` (images and checkouts are still on both nodes; env
IMAGE lines and `kindling.sh` pins must go back with them).

## Upstream delta: mmastrac/glm-5.3-flash-4x-gx10, 2026-09-28/29

Repo went from a TP=4 notes repo to a full tuned serving stack in ~3 days (PR #4 merged 09-28, TP=2 support #11,
RecoverSSM #9/#13, TP=2 memory work #15/#16; open: #20 draft-width graph sizes, #21 TP=3, #22 TP=RING4). It is a
**different stack from ours**: NVFP4 checkpoint (dense layers FP8/NVFP4), DFlash2 drafter (CC BY-NC-ND), vLLM
nightly ddd6fbca, mentat control plane, whole-file vLLM replacements mounted over its own image.

Their TP=2 numbers (two GB10, defaults after #15/#16): prefill 32k/128k cold 2,916 / 2,830 tok/s (ours ~1,275-1,524);
RigMark decode code/prose/structured 60.5 / 36.4 / 89.5 (RecoverSSM on); 1.1M-token KV pool (8 GiB pin,
`max_num_batched_tokens=8192`); `max_model_len` 160k; 16 seqs; head node idles at 1.3-3.5 GiB free. **Not like-for-like
with our benchmarks** (different checkpoint/quant, drafter, harness) -- do not quote as a delta against production.

Applicability to us:
- **N/A:** issue #7 (`glm47_failclosed` streaming tool calls) -- we run plain `glm47`. Drafter pool paging (#16) and
  DFlash2 adaptive-k -- we run MTP-2. TP=3 / RING4 -- topology we don't have.
- **Weak fit:** RecoverSSM (one KDA state per request instead of 1+k) -- at MTP-2 we keep 3 states not 8, so the
  saving is much smaller; their own TP=4 gain was +1.5% at 16 streams, large only past ~50 streams.
- **Candidates worth a look, none tried:** (1) `experimental/fixes/gb10_sparse_mla.py` (6 KB Triton sparse-MLA, claims
  FlashInfer's runs ~10 TFLOPS on GB10 and this is 3.3x faster) -- first confirm which sparse-MLA backend our
  prefill actually uses; (2) `arx` RDMA all-reduce for decode (13-27 us vs NCCL ~80 us; needs both ConnectX roots,
  which we already pin) -- but it ships as custom CUDA + a 40 KB replacement `cuda_communicator.py`, so a real port;
  (3) capping CUDA-graph capture at `max_num_seqs x (1+spec)` (#12) freed ~1-1.5 GiB/rank -- cheap, relevant to our
  memory-pressure history.
- **Whole-stack swap question** (their TP=2 stack vs ours on our prefill-dominated ~122K-token traffic): prefill
  looks ~2x ours on their numbers, but 160k max len vs our 262k, tighter memory, non-commercial drafter, and we
  measured DFlash2 -19% vs MTP-2 on our own stack. Same shape as the TensorFold verdict: needs a matched-method
  A/B before anything is claimed. Not started.

### Tried the three mmastrac/4x-gx10 candidates, 2026-09-29 (none adopted)

Production was stopped for the window (single-GPU microbenchmarks + two-node NCCL probe inside the v22 image),
then restored unchanged. Probes: `probes/gb10_sparse_mla_fp8ds.py`, `probes/bench_sparse_mla.py`,
`probes/bench_sparse_mla_graph.py`, `probes/bench_allreduce.py`.

**1. Triton sparse-MLA: rejected, slower than what we run.** Our backend is `FLASHINFER_MLA_SPARSE_SM120`
(FlashInfer's own SM120 sparse kernel, `GLM_NSA` model type, kernel page size 64), not the FA2/SM90 path their
kernel replaces, so "3.3x vs ~10 TFLOPS" doesn't transfer. Adapted their kernel to our packed 656-byte `fp8_ds_mla`
layout (per-128-channel fp32 scales) and benchmarked at our per-rank shape (32 heads, 512 latent, top-k 2048):
- Correctness: OK. Rel error vs fp32 reference 2.7e-3 (Triton) vs 3.4e-3 (FlashInfer); Triton vs FlashInfer 4.3e-3.
- Prefill (per layer-call): T=1024 13.2 ms vs FlashInfer 6.2 ms; T=7168 90.4 ms vs 42.5 ms. Triton ~11 TFLOPS,
  FlashInfer ~24 TFLOPS. Triton is **2.1x slower**.
- Decode inside CUDA graphs (how decode runs): T=3 42 us vs 32 us; T=16 248 vs 157 us; T=48 649 vs 585 us.
  FlashInfer wins at every size (eager timing showed Triton ahead at T=3, but that was Python launch overhead).
- Probe gotcha: FlashInfer only takes its decode path when the KV page size is 64; with 3584-token pages every
  call under 65 tokens fails, so a bench must view the cache with page size 64 to match production.

**2. arx RDMA all-reduce: not ported, estimated gain ~4% decode for a large, risky port.** Measured NCCL all-reduce
across the two nodes (bf16, hidden 4096, in a CUDA graph): 1 row 50 us, 3 rows (MTP-2 step) 52 us, 16 rows 78 us,
48 rows 140 us; 128 rows 466 us; 7168 rows (prefill chunk) 3.07 ms. At ~90 all-reduces per forward (45 layers x 2)
that is ~4.5 ms of a ~65 ms decode step. arx claims 13-27 us for small payloads, so ~2.5 ms/step saved, i.e. roughly
+4% decode, nothing for prefill (payload is above arx's 256 KB cutoff). Cost: custom CUDA (~46 KB) + a 40 KB
replacement `cuda_communicator.py`, on a stack that already has the open NCCL shm-broadcast stall (vLLM #51921).
Not worth it unless decode becomes the priority.

**3. CUDA-graph capture cap: not booted, no benefit path.** Under a fixed `gpu_memory_utilization`, memory freed
from the graph pool just becomes extra KV blocks; our KV pool is already 831K tokens against a 262K max length, and
host-memory headroom is set by gmu, not by what the graphs use. Would only matter if gmu were lowered, which is a
separate decision. Reasoned, not measured.

Production restored on v22-cacheevict after the window.

### jayleaton/glm53-tensorfold-spark first pass, 2026-09-29 (read-only review, nothing run)

Added to the tracked list at user request. Single author, ~3 days old, WIP. Reviewed README/AGENTS/RESULTS/PREFILL-ANALYSIS
and the RigMark baseline; no code executed on our hardware.

**What it is:** 57 build-time patches over ashhart/TensorFold 0.3.4 (MIT), targeting GLM-5.3-Flash EXL3 TR3 4bpw on 2x
Spark: RoCE all-gather (b12x-derived), fat-expert MoE (from Reederey87), FP8 latent KV, a shared 1M-token KV pool for 4
requests, NVMe session tier, prompt-lookup drafting, MTP or DFlash2 (DFlash2 is CC BY-NC-ND, optional).

**Why it matters for us:** the reason TensorFold was rejected (prefill 195-230 tok/s, 6-8x behind) is what this repo
targets. Their RigMark baseline (protocol 1.1.0, unmodified suite): cold prefill 8K/32K/64K = 1,598/1,641/1,621 tok/s
(vLLM TP2 NVFP4 in Alex Ellis's receipts: ~1,813-1,922), code decode 68.6 tok/s, prose 43.2, C1/C2/C4 aggregate
54.3/65.5/82.2. Immediate-replay prefill (6.3K vs 11K tok/s) and C4 TTFT (1.9 s vs 0.8 s) are still behind vLLM; fixes
are said to be in progress. RoCE all-gather A/B: +10.8% decode single stream, +4.3% at 4 streams; independently
consistent with our own ~4% estimate for the arx all-reduce port.

**Caveats before trusting the numbers:**
- Their weights are the abliterated `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, described as a byte-identical mirror of our
  `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`; unverified here. Our earlier TensorFold test used the MLX 4-bit checkpoint,
  not EXL3, so it is not a matched comparison either way.
- FP8 latent KV diverges from bf16 KV; they report byte-identical drafted-vs-serial outputs, not equivalence to vLLM.
- Their RigMark vs Alex's is explicitly a non-strict comparison (different weights, drafter, context, day, machines).
- Worker-node memory is tight (TensorFold rank holds ~90 GB; cannot coexist with vLLM).
- No production track record; the ashhart engine has no vLLM-style process/log/metrics surface, so our ops tooling
  (Prometheus dashboard, preflight, boot autolaunch) would need re-plumbing.

**Recommendation (not started):** a matched-method A/B is now justified: same EXL3 checkpoint, our real ~122K-token
traffic shape (probe_longctx), production-down window (~2-3 h), decode + cold/warm prefill + C1/C2/C4. Promotion bar:
prefill within ~10% of production on long context, and decode gain >= 20%.

## Three-way A/B on the real traffic shape: all three arms measured; kindling fastest prefill but zero memory margin (2026-09-30)

Driver: `recipes/probes/ab_stack.py` (stack-agnostic, streaming, HF tokenizer on the client, documents with 4 planted codes). Same
client for every arm. Cluster was handed over with production traffic stopped.

**Arm C, jayleaton TensorFold (our Mia TR3 checkpoint, DFlash2 drafter, default prod.env, 4 slots / 1M pool)**

| metric | result |
|---|---|
| cold prefill 16K / 64K / 122K | 1,534 / 1,620 / ~1,550 tok/s (TTFT 10.4 / 39.5 / 78.8 s) |
| 4x32K concurrent aggregate | 1,506 tok/s (C1 1,595, C2 1,434) |
| 122K x2 concurrent aggregate | 1,392 tok/s (174 s each) |
| decode, essay prompt | C1 48.3, C2 agg 82.6, C4 agg 129.2 tok/s; probe_throughput_ab median 57.7 |
| warm replay of 122K | TTFT 0.3 s (121,920 cached), C2 0.5-0.6 s |
| needle 16K/64K/122K | 4/4 each, all concurrent codes ok |
| boot | 594 s first (auto second internal restart), then 29 s |
| memory | head MemAvailable ~9-13 GiB while serving |

**Arm A, production v22-cacheevict, same driver (CAVEAT: not a clean baseline)**
Production came back up at ~06:49 and a real client (10.7.0.10) reconnected on :8000 (5 established connections,
`num_requests_running` was already 1-2 before our probes). Everything below shared the engine with real traffic.
- decode C1 33.1 tok/s (probe_throughput_ab median 33.9); C2 agg 47.4, C4 agg 46.4 (likely contaminated)
- cold prefill 32K: C1 1,328, C2 1,471, C4 1,449 tok/s; 122K C1 1,461, C2 1,437 (plausible, in the known 1,275-1,791 band)
- warm 122K replay: C1 TTFT 2.9 s, C2 2.8 s / 78.1 s (tail-evict working, one entry evicted by the concurrent pair)
- probe_longctx 16K/64K/122K: TTFT 116 / 213 / 441 s, INVALID (queued behind real traffic; a rerun at 16K gave 140 s)
Use the earlier uncontended baseline (prefill 1,275-1,791, decode 26-33 tok/s) for the promotion-bar comparison.

**Arm B, kindling/mmastrac NVFP4 + DFlash2, TP=2 (nvidia/GLM-5.3-Flash-NVFP4, spark-glm53:v9, vLLM nightly ddd6fbca, mentat 0.17.1, all overlays)**
Boot needed three deviations from stock: GPU_MEM_UTIL 0.86 (stock 0.88 fails, only 106.9 GiB free on our hosts); a second boot after the first
one wrote weight snapshots; and **earlyoom stopped on both nodes** (user did it manually). With earlyoom running, MemAvailable falls 111 GiB -> 0
during weight load and earlyoom SIGTERM'd both workers on every attempt (03:26, 06:20, 06:27, 06:35). Healthy at 08:13; KV pool 1,095,907 tokens, max_model_len 163,840.
Client note: the model emits a short reasoning span even with enable_thinking=false; ab_stack counts those tokens toward TTFT/decode.

| metric | kindling |
|---|---|
| cold prefill 16K / 64K / 122K (probe_longctx) | 2,671 / 2,767 / 2,589 tok/s (TTFT 6.0 / 23.1 / 47.1 s) |
| 32K x1 / x2 / x4 aggregate | 1,170 (first request, includes warm-up) / 2,021 / 2,435 tok/s |
| 122K C1 / C2 aggregate | 2,489 / 2,570 tok/s (TTFT 48.9 s; C2 91-95 s) |
| decode, essay prompt | C1 41.8, C2 agg 59.4, C4 agg 73.2 tok/s (probe_throughput_ab median 44.4) |
| warm 122K replay | TTFT 3.0 s (C2 3.2 / 6.3 s) |
| needle | 16K/64K 4/4; 122K 3/4 and 32K C1 miss are max_tokens truncation (answer cut mid-code 'CODE-4-', 64-token cap eaten by the reasoning span), not retrieval failure; all replays and concurrent runs ok |
| memory while serving | MemAvailable 1.3 GiB head / 0.16 GiB worker (earlyoom off) |

**Three-way summary (same client; prod = uncontended-band baseline plus contaminated ab_stack run)**

| | prod v22 | TensorFold | kindling |
|---|---|---|---|
| cold prefill long ctx (tok/s) | ~1,275-1,791 (ab_stack 1,461 @122K) | ~1,550 | ~2,500-2,770 |
| decode C1 (tok/s) | 26-33 | 46-58 | 42-44 |
| decode C4 agg | ~46 (contaminated) | 129 | 73 |
| warm 122K replay TTFT | 2.9 s | 0.3 s | 3.0 s |
| max context | 262,144 | ~1M pool | 163,840 |
| memory margin | earlyoom-safe | ~9-13 GiB | ~0.2-1.3 GiB, earlyoom must be OFF |

**Verdict against the promotion bar (prefill within ~10% of prod on long context, decode gain >= 20%)**
- TensorFold clears it (prefill in band, decode ~1.6-2.0x, C4 ~2.8x, near-free warm replay). Best all-round and safest of the two challengers.
- kindling clears it on paper and dominates prefill (~1.8x), decode C1 +30-45%, but is NOT deployable here: needs earlyoom disabled, leaves 0.16 GiB
  on the worker (any host-memory blip is an uncontained OOM), caps context at 163,840 vs our 262,144 (typical ~122K prompts fit, headroom for longer ones does not), emits reasoning with
  thinking off, and its DFlash2 drafter is CC BY-NC-ND 4.0. Its 2,900 prefill claim is roughly reproduced (2,500-2,770 measured).
- Nothing promoted; production stayed on v22. Candidate next steps: TensorFold quality eval beyond needle (q4mse requant), memory-margin soak; for kindling only
  the NVFP4 prefill lessons are worth porting, not the stack.
Cleanup state (2026-09-30): kindling and mentatd containers removed, second-ConnectX addressing back to 192.168.177.x, earlyoom restarted on both nodes
(config unchanged), production v22 relaunched. Left on disk: /models/kindling (191G + dflash2 + ~90G snapshot cache/logs), /models/glm53-tf,
/models/hub symlink, images glm53-tensorfold:dev, spark-glm53:v9, vllm nightly, mmastrac/*, ~/work/{kai,jl,mentat}.

### Kindling memory follow-up: where the memory actually goes (2026-09-30, prod down again, earlyoom left running)

Question: can kindling TP=2 be given more headroom? Runs on both nodes with a 5 s /proc/meminfo + per-process sampler, single ConnectX root
(second root's address removed, mentat announces only enp1s0f0np0), snapshots already present.
- Run 1: KV pin 8 -> ~5.6 GiB (6,012,954,214 B, ~768K tokens), GPU_MEM_UTIL 0.86. Booted healthy in ~3 min WITH earlyoom running, no kills.
  Run 2: same + GPU_MEM_UTIL 0.84. Healthy in ~2.5 min, no kills. Both passed 122K C1/C2 needle.
- 122K cold prefill unchanged by the smaller KV pin: C1 2,454 tok/s (TTFT 48.9 s), C2 agg 2,483 (run 1's first C1 was 1,970, first-request warm-up).
- GPU_MEM_UTIL is NOT a lever with a pinned KV: torch reserved was 99.05 GiB in both runs (allocated 96.4 -> 96.7 GiB). KV 8 -> 5.6 GiB freed 2.4 GiB of torch memory (98.8 -> 96.4 allocated).
- Nothing is pinned by the weight snapshots: Mlocked 24 MiB, Unevictable 24 MiB, Shmem 0.4 GiB, Cached 1-5 GiB. The "pinned snapshot page cache" hypothesis is refuted.
- Per-node breakdown at steady state: torch reserved ~99 GiB; host anon 8 GiB head (API server 2.4 + EngineCore 1.3 + Worker 6.1 RSS) vs 5.4-5.8 GiB worker;
  slab ~2.6 GiB; cache 1-5 GiB; MemFree only 2.8-4.4 GiB.
- MemAvailable stayed 0-2.8 GiB (min 0 MiB head in run 2, 0.8 GiB worker) and did not improve measurably with the smaller pin.
  Part of the gap is the host's own /etc/sysctl.d/99-zz-dgx-oom.conf (min_free_kbytes 1 GiB, watermark_scale_factor 100): low watermark 2.2 GiB, high 3.4 GiB,
  which is subtracted from MemAvailable. Real MemFree is ~3-4 GiB; that reserve is deliberate OOM hardening, not something to trade away.
- earlyoom (-m 2 -s 80) only fires when available < 2% AND swap free < 80%. Swap stayed 84-98% free, so it never fired even at 0 MiB available. That is a
  4-19% margin, not a safety guarantee: earlier first-boot kills happened once swap was in use.
Conclusion: kindling TP=2 is structurally ~99 GiB torch + ~9 GiB host (head) + kernel reserve on a 121.7 GiB box; KV/gmu tuning buys ~2 GiB at best.
Remaining levers, untested: drop DFlash2 drafter (weights + KV + graphs, costs decode), smaller CUDA-graph range / MAX_NUM_SEQS, KV pin ~3 GiB.
A third node is not needed for correctness but does not obviously help either (TP=3 needs a padded checkpoint per their tp3 README).

## Kindling PROMOTED to live production (2026-09-30)

Follow-up to the three-way A/B above. Decision by the operator on the strength of prefill/decode
results; earlyoom was left ON at its original config (`-m 2 -s 80`) and the original objection
(memory margin) was tested empirically rather than assumed.

- Config: kindling stack (`spark-glm53:v9`, NVFP4 + DFlash2, MegaMoE, SP), KV pinned 5.6 GiB,
  max ctx 163,840, single ConnectX root, mentatd 0.17.1 per node. Serving since 09:36; real
  traffic (c=2) from ~11:35. Serving files: `~/ai/research/glm-5.3-flash-gb10/kindling-nvfp4-dflash2/serve/`.
- Swap raised 16 -> 64 GiB per node (extra 48 GiB `/swap2.img`, fstab). Side effect: earlyoom's
  `-s 80` needs ~12.8 GiB swap used before firing, so it is effectively inert; `-s 95` would
  restore the old ~3.2 GiB trigger (not applied, needs operator decision).
- Soak (synthetic): 48 batches over ~80 min, 0 errors, needle 129/132 (misses = 64-token cap eaten by
  the reasoning span), prefill 2,580-2,607 tok/s (122K x3), decode 68.6-72.4 tok/s (x4).
- Live Prometheus vs 11 h of v22: prefill 2,127 vs 1,462 tok/s (+45%); TTFT p50 4.1 vs 4.0 s,
  p90 9.1 vs 13.7 s (mix differs: avg prompt 56K vs 117K); decode 32.0 vs 34.9 tok/s (-8%, explained
  below: concurrency mix); prefix hit 83.1% vs 92.8%.
- Stability over ~4 h of serving incl. soak: 0 earlyoom kills, 0 restarts, 0 log errors; min
  MemAvailable 0.29 GiB head / 0.38 GiB worker (healthy only because the ~2.2 GiB kernel watermark
  is subtracted); peak swap 3.3 GiB head / 1.9 GiB worker.
- Side-by-side tables, caveats, rollback: `recipes/COMPARISON-v22-exl3-mtp2-vs-kindling-nvfp4-dflash2-2026-09-30.md`.
- v22-cacheevict recipe/script untouched and remains the fallback.
- Structural follow-up (2026-09-30, same day, nothing run against production): added
  `recipes/kindling/` (`kindling.sh` start/stop/status with --dry-run, canonical env files, runbook),
  `recipes/scripts/switch_stack.sh` (kindling <-> v22, refuses while requests are in flight),
  and made `recipes/scripts/glm53_boot_launch.sh` stack-selectable (default kindling,
  `GLM53_STACK=v22` or `/etc/glm53-boot-launch.conf` for v22; v22 path also moved from the stale
  v21 recipe to v22). Only `--dry-run` paths were exercised; a real boot and a real switch are
  still unvalidated. Legacy `fleet_watchdog.sh` (TP=4, another cluster's hosts) moved to
  `recipes/scripts/legacy/fleet_watchdog_tp4.sh`; the kindling watchdog (delayed auto-restart, Grafana rules, Slack) is described in
  `recipes/kindling/README.md`; only its read-only `--dry-run`/`once` paths and the systemd timer were
  exercised, no fault injection, and the Slack post path is untested (see README).
- Second ConnectX root: removed only while kindling runs (kindling.sh flushes its IPv4, switch_stack.sh
  v22 restores 192.168.177.{87,142}/24). Effect on kindling prefill is unmeasured: upstream's numbers use
  both roots (+~11% prefill at 126K, decode flat) but require each root in its own subnet, which our
  v22 addressing (both roots inside 192.168.177.0/24) does not satisfy. Dual-root A/B needs a
  maintenance window.
- Live decode gap DIAGNOSED (2026-09-30, Prometheus, per-step ITL at matched concurrency): not a slower
  decoder. p50 step time at c=1 is 67 ms (kindling) vs 63 ms (v22), ~2.70 vs 2.66 tokens/step. The
  11 h v22 window never had c>=2 (serial client); kindling live runs c=2-6, where step time grows
  with long-context sequences in the batch (p50 93 ms at c=2, 126 at c=3, 167 at c>=4). Aggregate
  decode ~40 tok/s at c=1, ~58 at c=2, ~64 at c>=3. Explains "feels more responsive": requests no
  longer queue behind each other. No matched-concurrency v22 live data exists. Correction: kindling
  runs `max_num_seqs` 16 (not 4), so `num_requests_running` of 5-6 is legitimate.
- Open: torch reserved stayed 99.05 GiB after the KV pin reduction. earlyoom stays at `-s 80`
  (operator decision 2026-09-30; inert with 64 GiB swap, accepted).

## EXLR8 (EXL3-trellis routed experts) TP=2 trial vs kindling NVFP4 (2026-10-19)

Source: `kindlingai/glm-5.3-flash-exlr8-gx10` @ 20b80d3 (patched vLLM tree from `kindlingai/vllm-exlr8` a2e163cb05),
checked out at `~/work/exlr8` on both nodes. Routed experts are EXL3-trellis units decoded by a JIT-compiled kernel
(`exlr8_moe.cu`); dense linears stay FP8/NVFP4. TP=2 recipe = width `k3` (~3 bpw), `EXLR8_TAG=k3-tp2`, image
`spark-glm53:v11`, same DFlash2 drafter, `--quantization exlr8 --load-format exlr8`. Launcher/env/deadman:
`recipes/exlr8/` (`exlr8.sh start|restore|extend|disarm|status`; deadman restores kindling unconditionally).
Budgeted production outage with the operator's approval; kindling restored at the end (see the end of this section).

Matched arm A (KV pinned 6,012,954,214 B = 757,304 tokens, ctx 163,840, gmu 0.86) vs a same-day kindling baseline:

| Metric | kindling (v11) | exlr8 arm A | exlr8 arm B (KV 20 GiB, ctx 262,144) |
|---|---|---|---|
| Decode agg tok/s, c1 / c2 / c4 | 35.6 / 53.4 / 73.0 | 41.7 / 63.7 / 84.7 | 39.2 / 62.5 / 88.4 |
| Cold prefill 32k tok/s, c1 / c2 / c4 | 2,166 / 2,334 / 2,540 | 2,170 / 2,077 / 2,363 | 2,243 / 2,089 / 2,338 |
| Cold prefill 122k tok/s, c1 / c2 | 2,414 / 2,430 | 2,285 / 2,278 | 2,291 / 2,271 |
| Weights per rank | ~95 GiB | 62.5 GiB (+2.2 GiB drafter) | same |
| KV pool | 757,304 tok | 757,304 tok | 3,050,402 tok (11.6x at 262k) |
| MemAvailable under load | 0.3-1.3 GiB | ~27 GiB | ~12-14 GiB |
| Max context | 163,840 | 163,840 | 262,144 |

- Quality: NLL on three texts (notes, arx.cu, snap.py) within 0.012 of kindling and within 0.004 of bf16, identical in
  arms A and B. Smoketest 7/8 on both exlr8 arms vs 8/8 kindling: only `image_question` fails because exlr8 sets
  `LIMIT_MM images=0` (multimodal not served). Needle 6/6 up to 136k tokens (arm A); arm B 3/3 at 135k and 3/3 at
  251,643 tokens (TTFT 60-108 s). The GSM8K/HumanEval gate tasks were skipped (`quality-data` not present).
- Stability: arm A 16 soak rounds, arm B 8 rounds (c8 decode ~120-125 tok/s plus 3x16k prefill), 0 errors, no earlyoom
  or Xid events. Arm B stress: 6 concurrent 100k-token prompts (~600k KV tokens) 0 errors in 257 s (2,332 tok/s
  aggregate prefill). One of six failed `codes_ok` at `--max-tokens 64`; re-run alone at 64 and 512 it returned all
  four codes, and a repeat of the 6x100k run at `--max-tokens 512` scored 6/6 (2,312 tok/s). Attributed to the 64-token
  cap eaten by the reasoning span (same cause as the kindling soak's needle misses), not a quality fault.
- Boot: ~9 min the first time (includes a one-time ~130 s weight-snapshot write), ~4 min with the snapshot.
- Not measured: the same 6x100k stress on kindling (kindling cannot hold it: KV 757k tokens), multimodal, a long
  (hours-to-days) soak, a reboot-resume, support/maintenance risk of a custom CC-licensed weight format plus a
  JIT-compiled kernel and a patched vLLM tree.
- HF `nvidia/GLM-5.3-Flash-NVFP4`: the claimed "updated 3 days ago" is not visible at the file level; the last HF
  commit is da920bb0 (2026-09-29) and our local snapshot is already that commit. No NVFP4 re-test was needed.
- Reading: exlr8 = ~+12-19% decode, prefill flat to -9% under concurrency, ~equal quality, 4x the KV and the full
  262k context with an order of magnitude more host-memory headroom, at the cost of multimodal and a heavier
  dependency on one upstream. This answers the context/KV objection to NVFP4 directly. Promotion is the
  operator's decision; nothing was promoted by this trial.

### EXLR8 PROMOTED to live production (2026-10-19)

Operator decision after the trial above ("promote exlr8 big config"). Production is now exlr8 TP=2 k3, arm B config:
KV pinned 20 GiB (3.05M tokens), ctx 262,144, `max_num_seqs` 16, gmu 0.86, served name unchanged
(`glm-5.3-flash-exl3-v2`), image `spark-glm53:v11`, same DFlash2 drafter. Kindling NVFP4 (`recipes/kindling/kindling.sh start`)
is the fallback; v22 remains the second fallback.

- `recipes/exlr8/exlr8.sh serve [--force]` is the production entry point (no deadman, waits for /health, no-op if healthy);
  `stop` is a deliberate stop. `start`/`restore` remain the budgeted-trial pair.
- Watchdog auto-restart now runs `START_CMD` (set in `~/.config/glm53-watchdog.env` to `exlr8.sh serve --force`); the default
  stays `kindling.sh start --force`. Boot launcher default stack is now `exlr8` (`GLM53_STACK=kindling|v22` to override).
  Both unvalidated against a real reboot / real auto-restart.
- `switch_stack.sh` still only knows kindling<->v22 and does not detect exlr8; use `exlr8.sh stop`, then the target stack's own start.
- To roll back to kindling: `recipes/exlr8/exlr8.sh restore` (disarms, stops exlr8, starts kindling), then remove `START_CMD`
  from the watchdog conf and set `GLM53_STACK=kindling`.
- Known losses vs kindling: no multimodal (images disabled), custom weight format and kernel with one upstream.

## Upstream sweep, 2026-10-21 (exlr8 TP=2 is the live stack; read-only, nothing restarted)

Delta since the last full sweep (2026-10-09) plus the exlr8 promotion (2026-10-19). Checked via `gh api`
(pushed_at, releases, commits, PR bodies) against each repo's last-known commit/date in this file.

**Finding 1 (actionable, resolves a known limitation): exlr8-gx10 upstream now serves images.** Our EXLR8
PROMOTED section above lists "no multimodal (images disabled)" as a known loss vs kindling. Upstream PR #11
(`kindlingai/glm-5.3-flash-exlr8-gx10`, merged 2026-10-19, two commits after our `20b80d3` pin) adds a
`dense/vision-bf16.safetensors` file to the HF checkpoint (`kindlingai/glm-5.3-flash-exlr8-k2-k3-k4` commit
`6425b75`, the 347 `model.visual.*` tensors byte-identical to the NVFP4 checkpoint's tower) plus loader/snapshot
changes (`exlr8_loader.py`, `weight_snapshot.py`) and drops the forced `LIMIT_MM=0` in `exlr8.yaml`. Same PR also
fixes a real problem the old flat 38 GiB KV pin had at TP=3 (left the head 0.5 GiB free, 4 GiB into swap before
any load): the entrypoint now sizes the pin per TP, with separate DGX-OS/spark-os tables. Smoke test (image
question) passes at TP=1-4; TP=4 gates on v28 (vision on, 40 GiB pin) show NLL +0.0000, GSM8K 98.0%, HumanEval
157/164, decode 124.5/72.1 tok/s, prefill 4,408/4,437 tok/s (close to their README's 128.0/72.5 and 4,393/4,190).
Default image moved to v28. **We are mid-download of the exlr8 checkpoint right now (started 2026-10-21 08:32,
`kindlingai/glm-5.3-flash-exlr8-k2-k3-k4`) -- the current `main` revision should already include the vision
tower file, no separate checkpoint pull needed.** Pulling this upstream delta into `~/work/exlr8` and rebuilding
would both restore multimodal and fix the TP=3 KV-pin-at-load-time margin; not done, worth a real trial given it
directly answers our own documented gap. The repo moved 16 commits past our pin in 2 days (PR #12 ring/cable-order
fix, #13/#104 fabric-test fix, #1/#100 arx two-shot kernel up to 16 MB -- same optimization landing in kindling's
own repo in parallel, see Finding 2) -- treat the whole range as one rebuild, not just PR #11.

**Finding 2: kindling (our fallback stack) also moved fast, including a direct fix to a trap we hit and
documented ourselves.** `kindlingai/glm-5.3-flash-gx10` main went `4d93ef7..f14b1fe` (our v11 pin is `4d93ef7`,
2026-10-13). The headline item is PR #62/#73, "check weight snapshots against sampled tensor bytes, so a
same-shape checkpoint update loads fresh" -- this is exactly the trap our 2026-10-09 sweep's Finding 1 above
describes (kindling issue #62: safetensors-header-keyed snapshots silently served the stale `7d74cdd` drafter
instead of `bf582e4` unless the snapshot dir was moved aside by hand). Upstream has now fixed the root cause;
we worked around it manually. Also landed: the same arx two-shot kernel as exlr8 (#92/#93/#94/#100), the same
ring/fabric fixes (#103/#104), a new `gate/longreply.py` diagnostic ("flags decode drifting from prefill as a
long reply grows," #74/#96), and a `torch-profiler` compose overlay. Default image moved to v28 here too. Not
urgent to pull (we're on exlr8, not kindling, in production) but relevant if we ever roll back, and the snapshot
fix is a general correctness improvement worth carrying into any future kindling rebuild.

**Finding 3: local-inference-lab/b12x archived -- the "wait for a release past 1.3.0" question is now moot.**
Confirmed archived 2026-10-06. Development folded into `flashinfer-ai/flashinfer` (PR #5767, merged commit
`fc8fdc17f70`, lives at `flashinfer/experimental/b12x`); the `b12x` Python import path is kept as a compatibility
shim (`pip install 'flashinfer-python[b12x]'`, `import b12x` unchanged) but there will be no more standalone
`local-inference-lab/b12x` releases, tags, or issues. VALIDATION.md's tracked-repo entry updated above to point
future checks at FlashInfer's own release cadence instead.

**Finding 4: Enntity/sparkglm shipped a major unification release same-day as this sweep** (`unified-20261021`,
published 2026-10-21 08:14 UTC). One image now serves 2 or 4 DGX Sparks with "Strict" (preserves checkpoint
weight formats, FP8 KV, no cross-engine bitwise promise) and "Spark" (measured corrected-rounding FP4 + DFlash2)
modes, prefix caching, optional native MTP in Strict, and TP4 chain/ring. Published Spark-mode TP2 decode: prose
37.5, code 67.2, structured 89.8 tok/s (two-run medians, uncapped clocks, low reasoning effort, <4,096 output
tokens -- not our methodology, not directly comparable to exlr8's 39.2/62.5/88.4 c1 numbers above). The 3,008
tok/s cold-64K-prefill figure is explicitly re-used from an earlier qualification run, not new. Still a different
engine (Atlas, not vLLM) with its own release/support surface and "two-run medians... do not establish
statistical significance" caveat stated in their own notes. Verdict: interesting if the prefix-cache or TP4 story
ever becomes relevant, but no matched-method comparison exists -- same bar as every other engine-swap candidate
in this file (TensorFold, kindling's own three-way A/B). Not actionable without a real A/B.

**Finding 5: ashhart/TensorFold v1.1.1 (2026-10-19) touches GLM-specific prefill batching directly.** "GLM CUDA
stepped serving interleaves fresh prompt chunks with running decode... batch eligible fresh text prompts across
ranks... at most 16 prompts and 2,048 rows per pass" -- this is in the area of the 128-row bucket ceiling that
sank our own TensorFold A/B (see "TensorFold: 3 rounds" history). Also: GB10 efficiency-core pinning now
documented in their runbook (#1033, our hardware specifically), and a known-issues note that someone got the
ModelOpt NVFP4 checkpoint loading and serving on two GB10s at TP=2 on 1.1.0 (#1040, community-reported, "scored
prompts and more than two ranks are untested"). Needs a closer look at whether the stepped-serving change
actually lifts the 128-row ceiling before concluding anything -- flagging, not re-testing this round.

**Finding 6: jayleaton/glm53-tensorfold-spark growing past single-author.** 6 PRs merged 2026-10-18 to 2026-10-20
from 4 distinct external contributors (ryanmitchellneely, sayyidfareed, kuhlsnu, plus jayleaton's own), including
a tool-call-closing fix, an NVMe session-tier index-memory fix, and "ops/prod-hardening." Our 2026-09-29
recommendation for a matched-method A/B (same EXL3 checkpoint, our real ~122K traffic shape) was never run and
is now slightly stale as the patch set has moved; re-baseline before running it rather than using the 2026-09-29
read.

**Quiet / no actionable change:** `MiaAI-Lab` -- docs/credit commits only since 2026-10-09. `mmastrac/mentat`,
`tonyd2wild/...NVFP4-DFlash2`, `Entrpi/glm-5.3-flash-exl3-2x-spark` (quiet since 09-02), `r0b0tlab`, `AEON-7` --
no push since before the last sweep or docs-only. `Entrpi/vllm-glm-5.3-flash-spark` pushed 2026-10-17 per repo
metadata but its commit history is unchanged since 2026-09-02 -- the push likely touched a non-default ref; not
chased further this round. `cbertucci33` release-9 (2026-10-11) is loading/cleanup hardening (streamed DFlash
checkpoint tensors, iterator-close-on-error, InstantTensor cleanup-on-failure) plus a same-deployment regression
table (65,580-token warm-cache request: cold 41.35 s, warm 3.43 s) -- EXL3-lineage, relevant only if we return to
v22, nothing new architecturally. `Reederey87` no new tag past `v1.6.0` (already noted 2026-10-09), minor commits
since. `brandonmmusic-max/glm-5.3-flash-exl3-4bpw` still 404s, stays dropped. No new repos found worth adding
this round (general GitHub search for GLM-5.3-Flash/EXL3/DGX-Spark/GB10 turned up only forks/mirrors of repos
already tracked here).

**Net for us:** Finding 1 (exlr8 multimodal + per-TP KV pin fix) is the one item worth a real trial -- it
directly reverses a known production limitation and we're already downloading the checkpoint revision it needs.
Everything else is tracking-only this round.

## exlr8-gx10 PR#11 trial: two deadman incidents before any real data (2026-10-21/22)

Attempting to trial the upstream delta from Finding 1 above (`kindlingai/glm-5.3-flash-exlr8-gx10` `origin/main`
`6f5ede7`, 16 commits past our `20b80d3` pin -- vision tower + per-TP KV pin). Setup: isolated `~/work/exlr8-pr11`
worktree on both nodes (code tree rsync'd worker->head since head had no git checkout, only a files-only deploy),
image built as `spark-glm53:pr11` on both nodes (worker: cache-hit, ~8s; head: real build, ~6.3 min for the
flashkda CUDA extension alone), and the full assembled release (`kindlingai/glm-5.3-flash-exlr8-k2-k3-k4` @
`6425b75`, all three widths, 336 GiB on disk -- bigger than the HF API's 223 GiB estimate) materialized via
`hf download --local-dir` on the worker, then **rsync'd worker->head over the cluster network** (~300 MB/s,
~19 min) since each node has independent local storage, not shared NFS -- `/models/exlr8`-style paths looked
like they might be symlinked/shared from a stray `docker inspect` mount-format misread; they are not. The vision
tower file (`dense/vision-bf16.safetensors`) is confirmed present in the downloaded checkpoint, and `MODELS_ROOT`
(not `MODEL_DIR`) is the host-side knob that needs to point at the new release (`exlr8.yaml` bind-mounts it to
`/models/nas` in-container; `MODEL_DIR`'s default in-container path is unchanged).

Rather than reuse `recipes/exlr8/exlr8.sh` as-is (it assumes kindling is production and always restores to it),
a one-off `trial_pr11.sh` was written mirroring its arm/disarm/down_all/launch pattern, pointed at the isolated
tree/image/release, with its `restore` case calling the **real** `exlr8.sh serve --force` (today's actual
production, `20b80d3`) instead of kindling. **Two separate bugs in that script each defeated the deadman safety
net before any test battery data was collected:**

1. **Attempt 1 (launched 15:36:42, budgeted 75 min):** the deadman command was built as
   `bash -c "sleep N; '$0' restore"`. `$0` at that point was the bare relative invocation string
   (`trial_pr11.sh`, no path -- the script was run as `bash trial_pr11.sh ...` from its own directory), not a
   path reference. When the timer fired (~16:52), `bash -c` tried to run `trial_pr11.sh` as a **command** (PATH
   lookup, not a file reference) and failed instantly: `bash: line 1: trial_pr11.sh: command not found`, logged
   once, then the deadman process exited. The trial ran **completely unsupervised for ~4.5 hours past when the
   safety net should have fired (~5h46m total outage, 15:36->21:22)** before the operator noticed via an external
   check (StartedAt timestamp + health probe) that production had never come back. The maintenance-pause file the
   watchdog reads had also lapsed ~3.7h earlier with no visible separate intervention from that side -- not
   chased further here, but worth checking independently. Manually restored via `exlr8.sh serve --force`
   (absolute path), verified with a real inference completion, not just a health ping.
2. **Attempt 2 (launched 22:57:42, budgeted 90 min, deadman rewritten to resolve an absolute `$SELF` path via
   `readlink`-equivalent logic, checked executable, syntax-checked, and dry-run status-tested before arming --
   confirmed via `/proc/<pid>/cmdline` to actually carry the absolute path this time):** the path bug is fixed
   and the deadman **did** fire correctly on schedule (00:27:42, exactly 90 min later) and did invoke `restore`
   via the right path. But `restore`'s first line calls `disarm()`, which runs `pkill -P "$(cat deadman.pid)"`.
   `deadman.pid` holds the PID of the detached `sleep N; ... restore` wrapper, and the `restore` invocation is
   **itself a child of that same wrapper process** -- so `pkill -P <wrapper-pid>`, called from inside the restore
   script, matches and kills the restore script's own process. One line is logged
   (`"restore: stopping pr11 trial, starting ACTUAL production"`), then the process is SIGTERM'd by its own
   cleanup step before `down_all`/relaunch ever run. The trial ran unsupervised for a further ~2 hours
   (00:27->01:16) before the operator caught it via the same external StartedAt/health check pattern. Manually
   restored the same way, verified the same way.

**Root cause in common:** both bugs are reentrancy/self-reference problems in a deadman script that both arms
*and is* the thing being disarmed -- a generic arm/disarm/restore pattern copied from `exlr8.sh` (which only ever
restores to a *different* stack, kindling, and so never hits this) applied to a script whose own restore path
calls back into itself. **Fix adopted for the next attempt:** the external safety net is a plain detached
`sleep N; <absolute-path>/recipes/exlr8/exlr8.sh serve --force` armed independently (not through any
`trial_pr11.sh`/disarm/pkill chain at all), extended manually by the operator on a cadence tied to real check-ins
rather than one long unattended timer. Both outages: production itself recovered cleanly both times (`exlr8.sh
serve --force` is unaffected by either bug, having run successfully 3 times total through this incident: 2
emergency restores plus the original promotion). **No PR#11 test-battery data exists yet** -- multimodal smoke
test, quality gates, decode/prefill benchmarks, and the TP=2 KV-pool-size question are all still open; see the
next section once a clean run produces them.
