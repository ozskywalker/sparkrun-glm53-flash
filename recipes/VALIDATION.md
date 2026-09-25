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
- **mmastrac/glm-5.3-flash-4x-gx10** — a 4-node/TP4 GLM-5.3-Flash
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
  surgery, not a thin adapter.
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

## `v20-upstreamsync` promoted (2026-09-11) — SUPERSEDED 2026-09-19 by `v21-b12xmoe`

See "`v21-b12xmoe` PROMOTED to production (2026-09-19)" further down for
the current pointer. This section stays as the historical record of
`v20`'s own promotion -- every fix it documents remains in `v21` too
(`v21` is a direct descendant, identical except the MoE path).

Routine-upstream-check candidate, built on `v19-indexercompat` (previous
default). Sourced from a triage across MiaAI-Lab's
`GLM-5.3-Flash-EXL3-2x-DGX-Sparks` fork (`9c0794b..1caea9a`, 65 commits),
Enntity/sparkglm's `exl3` branch, and vLLM main since v0.29.0. Full build
detail: `recipes/build/glm53-exl3-v20-upstreamsync/NOTES.md`.

**Four items shipped:**

1. **Chat template `None` leak fixed** (`files/chat_template.jinja`,
   upstream MiaAI-Lab#156). `visible_text()`'s catch-all branch rendered the
   literal string `None` for an assistant turn with `content: None`
   (tool-calls-only, common in agent traffic) — now guarded with
   `elif content is not none`. Only this hunk was ported; #156's other
   hunks target sort-loop structure that's already diverged here from
   independent local changes. New test: `test_chat_template.py::
   NullContentTests` (mutation-verified against the pre-fix template).
2. **`count_shards()` false-completeness bug fixed** (`start.sh`, upstream
   MiaAI-Lab#153). Was counting `*.safetensors` recursively across *every*
   cached snapshot instead of just the active (`refs/main`) one — stale
   shards from an old build could satisfy the check while the real
   snapshot is incomplete. Fixed for both the shared target-model path and
   this fork's own separate DFlash2 call site (same bug, different code,
   not covered by the upstream PR). New test: `test_snapshot_scoped_count.py`.
3. **`LONG_PREFILL_TOKEN_THRESHOLD` opt-in plumbed** (upstream
   MiaAI-Lab#157), **left empty/off by default**. This fork already answers
   the identical "long prefill freezes a warm session" symptom via its own
   `overlay/patch_scheduler_decode_floor.py` gate — running both together
   is untested and could double-throttle or conflict. Do not enable without
   an A/B against that existing gate first. New test:
   `test_long_prefill_threshold.py`.
4. **`BUILD_MIN_MEM_GIB` build-host guard** (ported from Enntity/sparkglm,
   their own original work). Refuses `docker build` under 32 GiB
   `MemAvailable` by default — build-host-side only, no runtime effect. New
   test: `test_build_headroom.py`.

**vLLM PR #54929 assessed and NOT vendored.** It looked like a candidate
fix for the fleet's recurring #51921 shm-broadcast stall, but on
investigation: targets `flashmla_sparse.py`, a backend enum this fork
doesn't use at all (production runs `FlashInferMLASparseSM120Impl`) — zero
overlap with this fork's existing sparse-MLA/indexer patches, meaning
adopting it would require re-deriving this fork's entire bespoke
NoPE-on-SM120 workaround from scratch against 6,848 lines of unreviewed
code. The PR itself is unmerged, has known open correctness bugs
(CodeRabbit-flagged), was never tested against a NoPE/zero-padded,
`index_kpool>1` config like this one, and its root-cause match to #51921
is contested even among the issue's own commenters. Revisit only after
upstream review/merge or a confirmed root-cause match.

**Build**: PASSED, 63/63 steps incl. full build-time self-test suite.

**tinyGLM gate**: PASSED after one transient CUDA-graph-capture crash
(unrelated to any of the four items — none touch that code path); clean
retry matched v19-indexercompat's dispatch-correctness confirmation lines
exactly, determinism confirmed.

**Real-checkpoint validation**: PASSED. `probe_sanity.py` all passed,
decode 28.17-29.99 tok/s (matches v19's 27.4-29.28 tok/s, no regression).
Item 1 verified live with a real tool-calling round-trip (`content: null` +
`tool_calls`) — coherent reply, no `None` artifact. `probe_longctx.py`
@16K: two of the first three v20 attempts hit the ambient earlyoom pattern
described above — more than this project's usual one-retry precedent, so
it was run to ground with a **direct controlled A/B against unmodified
v19-indexercompat** rather than dismissed. v19 hit the identical earlyoom
signature too (mid-request, on a different process, zero effect on the
outcome), and a third v20 attempt with zero code changes passed clean.
**Confirmed ambient/fleet-wide, not a v20 regression.** TTFT@16K across all
clean runs, both versions: 9.6-11.9s, consistent with prior baselines. Full
performance numbers: `recipes/SPEED.md`.

**Verdict: promoted.** No blocking issues. `v19-indexercompat`,
`v18-gb10gemm`, `v15-combined`, `v12-combined`, and `v9-fatfork` all remain
as known-good rollback targets, in that order of preference.

## Pipeline re-measurement: decode kernel trace + dense-FP8 A/B, both re-run against v20 (2026-09-11)

Follow-up to a data-sufficiency question about the pipeline: is the
2026-09-02 deep-instrumentation-night kernel trace still accurate, and does
the v16-densefp8 prefill/decode tradeoff (measured against the stale
v15-combined baseline) still hold on the current kernel stack (E3, GB10
router-GEMM, FlashKDA, MoE-gate-dedup all postdate both original
measurements)? Two fresh measurements, both against production v20.

**Decode kernel trace (batch=1, prose, CUDA graphs on, MTP-2) — composition
is essentially unchanged from 2026-09-02:**

| category | v20 today (rank0/rank1) | 2026-09-02 (rank0/rank1) |
|---|---|---|
| gemm | 53.0% / 52.6% | 52.5% / 49.8% |
| moe_exl3 | 34.1% / 32.8% | 32.8% / 31.3% |
| comms | 5.0% / 6.6% | 7.2% / 11.9% |
| attention | 0.5% / 0.5% | 0.3% / 0.3% |
| mamba_ssm (KDA) | 0.3% / 0.3% | 0.2% / 0.2% |
| GPU busy | 97.0% / 96.5% | 96.0% |

**The single largest kernel is still the same undersized-tile Ampere WMMA
GEMM** (`cutlass_80_wmma_tensorop_bf16_s161616gemm...`, 36.2-36.3% of GPU
time, 16x16 tiles at decode's M=3) -- the custom-kernel opportunity
identified in the original trace is confirmed still live, unaddressed by
any promoted kernel work since. One real change: an SM120-family cutlass
GEMM now appears (4.5-4.8%, absent from the original trace) -- some
Blackwell-native GEMM is engaging, plausibly from the GB10 router-GEMM fix
-- but it's a small slice; the bulk of decode GEMM time still runs the
legacy path. Cross-node comms asymmetry is smaller now (~1.7% delta vs. the
original ~5%). CPU/IPC-spin profile (py-spy) also unchanged: EngineCore
99.9% ipc_spinwait (was 99.9%), Worker_TP0 97.7% (was 96.6%) -- confirms
decode at batch=1 is still GPU-bound, not CPU-orchestration-bound.

**Dense-FP8 A/B, re-run against v20 instead of v15-combined.** Built a
scratch measurement image (`recipes/build/glm53-exl3-v20-densefp8-ab`,
NOT a promotion candidate) by porting v16-densefp8's `overlay/exl3.py`
diff (the `Glm53DenseFp8Method` dispatch class -- `patch_dense_fp8.py`
alone is only half the mechanism, confirmed the hard way: first boot had
zero runtime confirmation lines, because the dispatch logic that routes
`dense`/`kda`/`mla` prefixes to the FP8 path lives in `exl3.py` itself in
v16's tree, not in the separately-portable patch script) onto v20's
`exl3.py`. Same `GLM53_DENSE_FP8=dense,kda` config as the original test.

| Metric | v20 baseline (this session) | v20+densefp8 (this session) | delta | original (v15->v16) |
|---|---|---|---|---|
| Decode | 27.96-29.90 tok/s | 32.59-34.15 tok/s | **+14-17%** | +12% |
| Prefill @16K | ~1,375 tok/s (TTFT 11.6s) | ~1,437 tok/s (TTFT 11.1s) | +4.5% (noise) | **-12.4%** |
| Prefill @64K | ~1,791 tok/s (TTFT 35.7s) | ~1,512 tok/s (TTFT 42.3s) | **-15.6%** | -14.7% |

**Verdict: the tradeoff holds, essentially unchanged in magnitude.** The
16K prefill hit didn't clearly reproduce (within this fleet's normal
9.6-11.9s TTFT@16K noise band, even slightly favorable) but the 64K
regression reproduces almost exactly (-15.6% vs. the original -14.7%),
consistent with the original's own two-point trend (64K's regression
slightly worse than 16K's). Decode's win is, if anything, a bit larger now
(+14-17% vs. the original +12%). **None of the kernel work landed since
the original A/B (E3, GB10 router-GEMM, FlashKDA, MoE-gate-dedup) moved
this tradeoff's economics in decode's favor** -- the prefill cost at
realistic (64K+) context is materially unchanged, and this fleet's one
real-traffic sample (`validation-archive/v9-fatfork.md`, ~101,500 avg
prompt tokens vs. ~181 avg output tokens, 93.4% cache-hit) is far past the
64K point where the regression is clearly visible, not the ~16K point
where it washes out in noise. **The missing piece is still the workload
question, not the kernel-tradeoff question**: that one real-traffic sample
is 5 days old at the time of this re-measurement and was never repeated --
confirming or updating it (has traffic actually shifted toward shorter,
decode-heavier requests) is the one measurement that would actually change
this recommendation, and it's still not done. Absent that, the original
v16-densefp8 rejection reasoning stands on the same evidence it always
did, now confirmed current rather than 9 days stale.

Scratch artifacts kept in the tree for reproducibility:
`recipes/build/glm53-exl3-v20-densefp8-ab/`,
`recipes/glm-5.3-flash-exl3-v20-densefp8-ab-vllm.yaml`,
`recipes/glm-5.3-flash-exl3-v20-profiling-vllm.yaml` (adds
`--profiler-config` for `/start_profile`, matching `v6-profiling.yaml`'s
config) -- none are promotion candidates.

## Dense-FP8 promoted to default, maxprefill sibling added (2026-09-11)

Given the numbers above, the user's own framing: prefill throughput has
grown substantially since v1 (~860-900 -> ~1,500-1,700+ tok/s across the
version history in `recipes/SPEED.md`), and this fleet runs a mixed
~85/15 heavy/short traffic split with growing agentic use -- worth
sacrificing some of that prefill headroom for a real decode win, contrary
to this project's longstanding prefill-first bias but not a bad trade
given how much headroom now exists. Decision: **ship
`GLM53_DENSE_FP8=dense,kda` on by default**, with a sibling recipe for
workloads that still want maximum prefill.

**Folded the dense-FP8 support into the actual `v20-upstreamsync` build
tree** (not just the scratch A/B image) -- `overlay/patch_dense_fp8.py`
plus the `Glm53DenseFp8Method`/`_glm53_dense_fp8_group()` dispatch logic
in `overlay/exl3.py` (see the porting gotcha above: the patch script alone
is inert, the dispatch lives in `exl3.py` itself). `docker build` rebuild
reused the exact same layer digest as the scratch image
(`sha256:62049b...`), confirming byte-identical content -- `glm53-exl3-
v20-upstreamsync:local` is now one shared image for both recipes below,
the toggle is purely the `GLM53_DENSE_FP8` env var, matching the feature's
original runtime-gated design intent.

**Coherence/accuracy check run before shipping this as default** -- the
one validation step this project's own rules required and the original
v16-densefp8 investigation never completed (it was throughput-only,
explicitly flagged as a gap). Six representative prompts (short factual,
a reasoning riddle, code generation, summarization, a CJK translation --
this project's own known corruption-risk case from the ModelOpt checkpoint
history -- and a multi-turn memory check), same image, temp=0, dense-FP8
on vs. off, side by side:

- All six: coherent, correct, no garbling, zero U+FFFD or corrupted CJK
  output.
- Every divergence between on/off was the ordinary paraphrase-level drift
  any precision change causes under greedy decoding at a branch point
  (e.g. "so 9 survive" vs. "the other 8 died" -- same correct answer,
  different phrasing after the tokens diverge) -- never a wrong answer,
  never incoherent, never a different final conclusion.

**Two recipes now share the one image**, toggled by one env var:

- `glm-5.3-flash-exl3-v20-upstreamsync-vllm.yaml` -- **the default**,
  `GLM53_DENSE_FP8=dense,kda`. Re-confirmed end-to-end on the actual final
  recipe file (not just the scratch build): `probe_sanity.py` ALL PASSED,
  decode 31.54-36.76 tok/s, 210 `[glm53-dense-fp8]` per-layer confirmation
  lines fired across both ranks.
- `glm-5.3-flash-exl3-v20-upstreamsync-maxprefill-vllm.yaml` -- sibling,
  identical in every other respect, `GLM53_DENSE_FP8` unset. For workloads
  that are consistently long-context/prefill-dominated and want the old
  tradeoff back.

Still PROVISIONAL per upstream's own commit message ("changes target
numerics; needs a KLD panel") -- the coherence check above is a real,
meaningful check but is not a substitute for one. Revisit if a future
session has the tooling to run an actual KLD panel, or if real production
traffic surfaces a quality regression this spot-check didn't catch.

`v19-indexercompat` remains available as a same-family rollback target if
either v20 recipe needs to be backed out; it predates this change entirely.

## AEON-7/vllm-ultimate-dgx-spark initial triage (2026-09-12)

Added to the tracked-repo rotation at the user's request; first pass
found real, specific leads on two of this fleet's worst unsolved
operational problems. Investigation-only so far -- nothing below has been
tested against this fleet yet.

**Most actionable: `patches/patch_cudagraph_align.py` (their main tree).**
Fixes a real vLLM gap -- spec-decode capture-size alignment (rounding
capture sizes to multiples of `1+num_speculative_tokens`) is gated to
`cudagraph_mode==FULL` only in stock vLLM, silently skipped under
`PIECEWISE` (upstream vLLM #28015/#28207/#29091, fixed by #29102/#23679
upstream but not yet in whatever base we're on). Their own characterization
of the resulting symptom -- `cudaErrorIllegalAddress` mid-decode on
partial-acceptance steps under PIECEWISE -- matches this fleet's own
recurring, never-root-caused `torch.AcceleratorError: CUDA error: an
illegal memory access was encountered` during CUDA-graph capture *exactly*
in class, though not yet confirmed to be the same mechanism. This fleet
runs MTP-2 (num_speculative_tokens=2, so partial acceptance is a real,
frequent runtime state) under PIECEWISE-capable graphs. **Worth a direct
test**: check whether our installed vLLM already has #29102/#23679, and
if not, whether backporting closes the gap. This is the single most
promising lead this project has had on this crash class since it was
first observed.

**Second lead, same repo, DeepSeek-V4-Flash GB10 branch
(`deepseek-v4-gb10`, commit `5e2420e5e0c8d5034aa728965b04f1b11eb55adf`,
open PR, external contributor `gilby`).** DeepSeek-V4-Flash is
architecturally the closest model in that whole repo to GLM-5.3-Flash
(sparse-MLA hybrid MoE). Three findings:
- Documented root cause for a *different* NCCL failure class on 2-node
  GB10 TP2 (`c10::DistBackendError`, not our `#51921` shm_broadcast stall,
  but the same "NCCL + CUDA graphs on multi-node GB10" failure family):
  full decode-graph replay desyncs NCCL between ranks. Fix: force
  `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` -- collectives
  must stay uncaptured on this fabric. We already don't force FULL
  unconditionally, but worth confirming our actual capture mode.
  Corroborated: a version-pin landmine (`tilelang==0.1.12` silently aborts
  with a duplicate type-attr registration on their sparse-MLA decode path;
  pin `0.1.11`) -- worth checking our own pinned TileLang version against
  this if any TileLang-related instability recurs.
- `overlay-deepseek-v4-gb10/vllm/model_executor/layers/sparse_attn_indexer.py`
  (~lines 337-370) patches the same `sparse_attn_indexer.py` file family
  our own `glm5_next` sparse-indexer touches, for a `cooperative_topk`
  landmine: GB10/sm12x has **no thread-block-cluster launch support**
  (`cooperative_topk` fails "invalid argument"), must fall back to
  `persistent_topk`. Hardware-general fact worth confirming we already
  handle correctly (our own K-pool/indexer patches suggest we do, but
  cross-check against this exact anchor).
- Cites DeepGEMM's `nv_dev` fork (`deepseek-ai/DeepGEMM#324`, commit
  `a6b593d`) as shipping genuine native SM120 kernels (vendored DeepGEMM in
  most trees is sm90/sm100-only) -- the closest thing found anywhere to
  real Blackwell-native GEMM work, directly relevant to this project's own
  still-unaddressed decode-time Ampere-fallback GEMM bottleneck
  (`cutlass_80_wmma_tensorop_bf16_s161616gemm...16x16`, ~36% of decode GPU
  time, identified 2026-09-02, confirmed unchanged 2026-09-11). Worth a
  look before considering any from-scratch custom kernel effort.

**Third lead, worth an A/B, not yet a fix**: their documented "dual-Spark
TP=2 over RoCE" stability recipe (`capture_error_mode="thread_local"` at
all `torch.cuda.graph` sites, `fuse_allreduce_rms:false`,
`--disable-custom-all-reduce`, `VLLM_ALLREDUCE_USE_FLASHINFER=0` --
they found v0.29's FlashInfer-all-reduce-on-by-default causes garbled
output, not a hang, on RoCE Sparks specifically). Different symptom than
our shm_broadcast stall, but the closest available "someone else
stabilized dual-GB10-TP2-CUDA-graphs" playbook found anywhere.

**Fourth, worth tracking**: their issue #9 (closed) on GB10 unified-memory
pressure closely parallels this project's own MemFree/MemAvailable-gap and
silent-kill investigation tracks -- two real mechanisms found (`--kv-
cache-memory-bytes` skips vLLM's own UMA/cudagraph memory-profiling clamp
via an early-return path; Docker cgroup memory limits don't police
`cudaMalloc` on GB10's unified pool, so container RSS looks compliant
right up until the driver fails to service an allocation). Does **not**
mention host memory fragmentation (`nr_free_pages_blocks`) specifically --
that finding still appears to be unique to this project.

**Not applicable**: no EXL3/exllamav3/trellis quantization anywhere in the
repo (nothing transfers at the quant-scheme level); no GLM-5.3-Flash as a
target model; no mention of `earlyoom` specifically.

## cudagraph_align hardening ported; 2026-09-12 23:41 UTC incident root-caused

Following up on the AEON-7 triage above: the user asked to pursue the
`patch_cudagraph_align.py` lead specifically because production had just
crashed (a ~3-hour uptime run, after a period of heavy agentic traffic).
Two separate pieces of work came out of this.

**1. cudagraph_align ported and shipped.** Read this image's own
`vllm/config/compilation.py` (commit `g487ecf187`) directly and confirmed
the gap AEON-7 described is real here too:
`resolve_cudagraph_mode_and_sizes` only rounds `cudagraph_capture_sizes` to
multiples of `uniform_decode_query_len` (3, for this fleet's MTP-2) when
`cudagraph_mode.decode_mode() == CUDAGraphMode.FULL`; every other decode
mode silently skips it, and `cudagraph_dispatcher.py`'s
`_create_padded_batch_descriptor` then asserts
`num_tokens_padded % uniform_decode_query_len == 0` for any uniform decode
batch. AEON-7's own patch text didn't apply byte-for-byte (this image adds
a `not use_v2_model_runner` clause their version predates), so
`recipes/build/glm53-exl3-v20-upstreamsync/overlay/patch_cudagraph_align.py`
is a from-scratch port matching our actual anchor, broadening the
condition from `decode_mode() == FULL` to `!= NONE`. New
`tests/test_cudagraph_align.py` covers both an unpatched and
already-patched source (host fixture + a real extracted `compilation.py`
from this image); full docker rebuild passed all self-checks including the
new one. Shipped directly into `glm53-exl3-v20-upstreamsync:local` (no new
version number -- same build tree, patch added). See
`recipes/build/glm53-exl3-v20-upstreamsync/NOTES.md`, "Amendment
2026-09-12: cudagraph_align hardening" for full detail. **This is a
confirmed-real, independently-justified gap closure, not a fix verified
against a reproduction** -- our own boot log shows
`cudagraph_mode=FULL_AND_PIECEWISE` resolving normally, so the FULL-only
gate is very likely already satisfied in normal operation; this change is
a no-op today and a hardening against any future config/backend-support
path that leaves decode mode at PIECEWISE.

**2. The actual 2026-09-12 23:41 UTC incident -- root-caused, and it is
NOT the cudagraph_align gap.** The user supplied the actual crash log
(`Worker proc VllmWorker-0 died unexpectedly (exit code: None)`, no
traceback of its own, followed by the downstream `shm_broadcast`
"cancelled" error and `EngineDeadError`). The dying step's
`dump_input.py` output showed `num_scheduled_tokens=5832` for a single
request already at `num_computed_tokens=179200` -- a large one-shot
continuation chunk for a ~180K-token-deep request, not a small MTP-2
decode step. That's far above this config's `max_cudagraph_capture_size=96`,
so the step ran eager -- ruling out cudagraph capture/replay
(and therefore cudagraph_align) as the mechanism for this specific crash.

Host-level investigation (dmesg + `journalctl -u earlyoom`, both nodes,
which the user's own earlier `sparkrun stop` at 19:43 had left no other
trace of -- containers were already destroyed) found the real cause:

- **Both hosts hit severe, simultaneous host memory exhaustion** starting
  ~19:40:45 EDT: this host (node_1/rank1) pinned at **3600 MiB avail out of
  124610 (2.89%)** continuously for 29+ seconds; the head (node_0/rank0,
  10.7.0.87) dropped to **2508 MiB (2.01%)** at the same time.
- **19:40:51.14 EDT**, head node: earlyoom crossed its SIGTERM threshold
  and sent SIGTERM to `1712017 uid 1000 "VLLM::Worker_TP"` (badness 984,
  VmRSS 2320 MiB).
- **19:41:01.22 EDT** (10s later): earlyoom logged **`kill failed: Timer
  expired`** -- the SIGTERM did not take effect within earlyoom's wait
  window. Plausible explanation: the worker was mid-flight on the large
  5832-token batch, blocked in CUDA/NCCL work and unable to act on the
  signal promptly.
- **19:41:06-09 EDT**: memory on both hosts suddenly recovered to
  86-92% avail -- consistent with the worker actually dying/releasing its
  memory around then, matching the `23:41:06 UTC` (=19:41:06 EDT) `Worker
  proc ... died unexpectedly` timestamp in the user's log almost to the
  second.
- This host (node_1/rank1) never logged its own SIGTERM/SIGKILL line in
  this window despite being equally starved (2.89% avail) -- it just sat
  at the low-memory warning level without earlyoom escalating to a kill
  here; the head's kill (successful or not) was enough to end the episode
  for both ranks (TP=2 -- losing either rank kills the whole engine).

**This extends, rather than replaces, the project's existing "earlyoom
root-cause correction" finding** ([[earlyoom_root_cause_correction]]):
previously documented as "check earlyoom first, it's usually the silent
killer." New wrinkle, not previously seen: **earlyoom's own kill attempt
can itself fail/time out** against an unresponsive GPU/NCCL-blocked
target, which likely explains why some past "silent kill" incidents were
hard to pin to a specific earlyoom action from timestamps alone -- the
SIGTERM fires, doesn't land immediately, and the process dies later for
reasons that then look uncorrelated unless you specifically check for a
"kill failed" line.

**New data point for the standing host-memory-pressure track**
([[host_fragmentation_xid31_reboot_risk]], the 240K-context fragmentation
finding): this episode happened at ~185K total tokens (179200 computed +
5832 scheduled), well under the ~240K neighborhood previously implicated.
The common factor isn't a fixed context-length threshold -- it's **a
single large one-shot continuation/recompute batch** (5832 tokens in one
step, vs. MTP-2's normal 1-3-token decode steps) for a request already
deep in a long context. Worth checking whether the scheduler's chunking
policy (`patch_scheduler_decode_floor.py`'s mixed-prefill-decode policy)
can be made to cap continuation-chunk size for already-long-context
requests specifically, rather than only floor-ing decode-side batching --
not yet investigated further; flagging for a future session.

**Not yet done**: identifying what specifically was consuming host RAM in
that window (vLLM's own host-side buffers scaling with the 5832-token
batch vs. something fragmentation-related vs. an unrelated host process);
no host-side memory profiler was attached at the time and the containers
are gone. Production was relaunched on the patched image
(`glm53-exl3-v20-upstreamsync:local`, now carrying `patch_cudagraph_align.py`)
after this investigation; see the top of this file for the current state.

## Routine upstream check, 2026-09-13 -- quiet round, nothing folded in

MiaAI-Lab main `1caea9a..HEAD` (32 commits, 6 PRs), Enntity/sparkglm `exl3`
branch `a8aaa229..HEAD` (15 commits), plus pulse checks on the other three
tracked repos and vLLM itself. Honest result: **nothing directly
actionable for this fork's production recipe this round** -- every
substantive item is either launcher plumbing this fork bypasses (sparkrun
runs `vllm serve` directly, not MiaAI-Lab's `start.sh`), a DFlash-specific
feature this fork doesn't ship (MTP-2 stays the production speculator),
or a correctness fix already present in our own vLLM base. Detail:

**MiaAI-Lab, by PR:**
- **#130** (merged): opt-in per-KV-cache-group prefix-cache retention +
  safe replay for DFlash's sliding-window drafter cache. DFlash-only
  (`GLM53_APC_RETENTION_INTERVAL_SWA` requires `SPEC_METHOD=dflash`) --
  not applicable, we run MTP-2.
- **#169** (merged): computes the CUDA-graph capture-size list for
  DFlash's "adaptive-k" verification-length feature from
  `GLM53_ADAPTIVE_K_SET`/`DFLASH_TOKENS`/`MAX_NUM_SEQS` instead of a fixed
  list. Same *class* of bug this fork just spent a session on
  (cudagraph capture sizes not accounting for variable spec-decode token
  counts -- see the cudagraph_align entry above) but the feature itself
  (DFlash adaptive-k) is DFlash-only -- not applicable. Worth noting as
  corroboration that this bug class is real and recurring industry-wide,
  not specific to us or to MTP.
- **#172** (merged): launcher's RoCE GID preflight only checked the first
  HCA on a dual-rail (`HEAD_CX7_IB=dev1,dev2`) kit, silently skipping
  validation on the second -- an unpopulated GID there kills that rank
  ~60s into a run. Fix + new test. Their own measurement: **dual-rail NCCL
  all-reduce hit 20.9 GB/s peak busbw vs. 12.8 GB/s single-rail** on their
  2x GB10 kit. The preflight fix itself is start.sh-only, not applicable
  (sparkrun handles our own networking setup, abstracted behind
  `transfer_interface: cx7` in `~/.config/sparkrun/clusters/default.yaml`)
  -- but that ~63% bandwidth number is worth checking against our own
  setup: **not yet verified whether sparkrun is using both RoCE rails on
  our CX7 cards or just one.** Flagging for a future session -- if we're
  single-rail today, this could be a real, free TP=2 communication
  bandwidth win.
- **#175** (merged): makes `start.sh`'s hardcoded
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` overridable (for
  derived images with a KV connector that can't tolerate expandable
  segments). Not applicable as a fix -- we already set this directly in
  our own recipe YAML, unconditionally, bypassing `start.sh` entirely.
  Confirms our existing setting matches upstream's own recommended
  default.
- **#136, #173, #176** (merged): benchmark-script bearer auth, launcher
  Jinja2-interpreter discovery, a docs typo. Tooling/docs only, no
  runtime-behavior relevance.

**Enntity/sparkglm (`exl3` branch)**: this branch has shifted into a
heavier "research/experiments" structure (qualification protocols,
rejected-candidate writeups) rather than shipped features. Two things
worth recording:
- **`research/experiments/exl3-direct-epilogue`**: a candidate
  micro-optimization to the fat-expert grouped-MoE kernel's epilogue
  (keep the transformed row in its owning warp, skip a shared-memory
  round-trip). **Rejected by their own performance screen** -- compiled
  and passed correctness, but was slower than the reference. Nothing to
  port; recorded for completeness (same rigor this project applies to its
  own rejected experiments).
- **`research/experiments/exl3-e3/UPDATE_REVIEW.md`** flagged **vLLM PR
  #55234** as "especially easy to backport incorrectly" (MLA cache-group
  capability handling for non-causal draft paths, plus a `MambaSpec.merge`
  assertion-preservation fix under `python -O`). Checked directly against
  our own vendored vLLM (`g487ecf187`,
  `vllm/v1/kv_cache_interface.py:487-488`): **already present** (the `any()`
  aggregation the fix restores is exactly what our source has). Nothing to
  do. Their update review also independently confirms two things this
  project already concluded on its own: their reported E3-vs-E2 speedup
  "is not transferable" since SparkGLM (and this fork) already has grouped
  M64 prefill + cooperative K4 decode, and "replacing DFlash2 with MTP"
  is flagged as its own separate hypothesis requiring independent
  measurement, not something to blend silently -- same conclusion this
  fork reached independently months ago.

**Pulse-checked, no new relevant activity**: mmastrac/glm-5.3-flash-4x-gx10
(last push 2026-09-07, before this round's window), tonyd2wild/GLM-5.3-
Flash-NVFP4-DFlash2-2x-DGX-Spark (2026-09-02), AEON-7/vllm-ultimate-dgx-
spark (2026-09-12, before yesterday's full triage -- nothing since).
mmastrac/mentat had a burst of activity (a 0.10.0 release) but it's all
within its existing Ray-replacement scope -- track stays closed, see
[[mentat_track_closed]].

**vLLM upstream**: still v0.29.0 (published 2026-09-09), no new release
since the backport-candidates round closed 2026-09-10. Nothing new to
triage there.

**Net**: a genuinely quiet round. One follow-up flagged (verify RoCE
dual-rail usage in our own sparkrun cluster config) but nothing shipped or
changed in this fork as a result of this check.

## TP=4 vs. 2x TP=2 assessment, and routine upstream check, 2026-09-14

Prompted by the user contemplating scaling from 2 to 3 or 4 DGX Sparks.
Conversational recommendation for both cases: don't extend the TP group
(TP=3 hits an odd-sharding-degree risk class this project has never
validated at any TP degree; TP=4 doesn't have that specific risk but
multiplies this fleet's single most-proven fragility, the RoCE/NCCL
fabric, across twice the physical link count). Prefer independent
replicas -- a 3rd node as its own TP=1 lane for the ~15% short-traffic
slice, or 2x TP=2 pairs instead of one TP=4 group -- both reuse the
entire already-validated TP=2 stack with zero new sharding risk and, more
importantly given this project's own incident history, halve blast
radius instead of concentrating it.

**mmastrac/glm-5.3-flash-4x-gx10 deep-dive (fanned out to a subagent,
production untouched throughout)**: confirms the TP=4 concern directly.
Their repo is genuine flat TP=4 across 4 separate physical GB10 boxes
over RoCE (not PP, not DP -- some unused PP plumbing exists in the repo
but isn't what's actually invoked). Their own troubleshooting docs
describe a RoCE fabric fault mode independent of anything we've hit:
NCCL all-reduce silently crawling to ~12 Gb/s (vs. 196 Gb/s healthy)
after a cable hot-plug, with **zero error-counter signal** -- `ib_write_bw`
reads fine, only a Ring-vs-Tree NCCL microbenchmark catches it -- and
requiring a full power-off (not a reboot) to clear. Their own words: "the
existing 4x recipes each solve a different subset and none of them
mention the fabric fault, which is the one that costs half your prefill
throughput while every metric reads healthy." Second independent fragility
axis: the RoCE GID index is per-node-*and-per-boot*, not a stable value --
they derive it dynamically every boot rather than trust a pinned config,
since a stale pin silently breaks TP init. A useful, independently-derived
data point: their own note that per-token all-reduce payload (~720KB) is
"a fraction of a millisecond against a 45ms token -- weigh it only at
TP>2," and that dual-rail RoCE "bought nothing" for them at TP=2 --
direct confirmation the fabric-dependency cost is roughly free at TP=2
and non-trivial past it. GLM-5.3-Flash's NoPE sparse MLA (compressed
latent KV, not per-head) means the "replicate KV heads when TP exceeds
head count" bug class other models hit doesn't apply here regardless of
TP degree. They never evaluated or discuss a multi-replica alternative
anywhere in the repo -- so no counter-evidence to our thesis, only more
confirmation of the exposure. **Not adopted as a topology**; recorded as
the deciding evidence for staying off TP=4.

Secondary backport candidates surfaced by the same dig, not yet acted on:
a spin-wait tuning patch (`busy_loop_s` 1.0->0.002s cut their vLLM CPU
185%->109% and *raised* decode throughput on GB10 by exploiting the
unified CPU/GPU power budget -- this project already has its own
spinwait patch, `patch_spinwait.py`, worth comparing constants);
`GPU_MEM_UTIL=0.90` silently wedges their box hours later while passing
every startup check, same shape as this project's own 0.85/0.86 GMU
regressions -- worth comparing their observed ceiling against our 0.84
one; a report that `thinking_token_budget` is silently ignored at
temp 0/1 -- worth checking against our own build. None investigated
further this round.

**Rest of the tracked-repo rotation** (all fanned out in parallel,
production untouched):
- **MiaAI-Lab main**, `f906ee990..HEAD` (2 commits, one PR, #183):
  "fix(vision): cap per-image tokens so a chat video cannot OOM the
  host." Real incident on their side 2026-09-14: an 11.9MB chat video
  decomposed into ~33 full-res images, `SKIP_MM_PROFILING=1` (mandatory
  on their UMA kit) meant nothing was ever reserved for the vision tower,
  prompt hit 236,544 vision-encode tokens, host OOM-killed the worker.
  Fix: `LIMIT_MM` image cap 100->48, new `MM_IMAGE_TOKENS`/`--mm-
  processor-kwargs max_image_tokens` knob, new `MM_PROCESSOR_CACHE_GB`
  (vLLM defaults to reserving 4 GiB host RAM for processed media --
  real cost on UMA regardless of per-prompt limits). **Assessed: not
  urgent for us.** Our own `limit_mm: {"image":4,"video":1}` is already
  far tighter than even their patched 48-image cap, so we're not exposed
  to their crash scenario. We do NOT explicitly set `--mm-processor-
  cache-gb` or `max_image_tokens`, so we're on vLLM's stock 4 GiB/8000
  defaults -- a real but currently-inert standing host-memory cost, worth
  revisiting alongside the broader host-memory-pressure track
  ([[cudagraph_align_shipped_earlyoom_timeout_found]]) rather than
  urgently now.
- **Enntity/sparkglm `exl3`**: still `2da6a1c31`, confirmed no drift/
  force-push. Quiet.
- **tonyd2wild**: still last-pushed 2026-09-02. Quiet.
- **AEON-7/vllm-ultimate-dgx-spark**: still last-pushed 2026-09-12,
  no new commits/issues/PRs touching CUDA-graph capture, attention
  kernels, Blackwell/SM12x, or RoCE/NCCL. Quiet.
- **vLLM upstream**: still v0.29.0, no new release -- no full backport-
  candidates round warranted. Two things worth tracking: **PR #54929**
  ("Portable Triton sparse-MLA fallback for SM12x") explicitly claims
  `Fixes #51921 (on SM12x)` -- our long-standing open shm_broadcast-stall
  issue. Root cause per the PR: DSA models (DeepSeek-V3.2, GLM-5.2/5.3)
  have no working native sparse-attention kernel on SM12x, the native
  extension livelocks under sustained load (GPUs pin 100% until the NCCL
  watchdog kills the server -- matches our own stall signature), and this
  PR binds a portable Triton fallback (derived from #49026, already
  validated on sm_121/GB10) instead. **Still open, has merge conflicts as
  of 2026-09-14** -- not yet mergeable, nothing to backport yet, but this
  is the most promising #51921 lead found to date; recheck next round.
  Separately, **PR #55737** ("Use FlashKDA for KDA chunked prefill")
  merged 2026-09-14 upstream -- the same approach this fork's own
  `v15-combined` already shipped; not new work for us, just confirms
  upstream converging on what we already have. Worth a quick diff-check
  next backport round, not urgent.

**Net**: the TP=4 question is answered (don't); nothing else this round
needs immediate action. Two things to revisit next time someone's in
here: PR #54929's merge-conflict status (the #51921 fix candidate), and
whether the mm-processor-cache-gb/max-image-tokens defaults are worth
pinning explicitly given the standing host-memory-pressure track.

## Routine upstream check, 2026-09-15 -- two backports shipped

Second fan-out round (6 parallel agents again, production untouched
throughout). MiaAI-Lab main had genuinely moved this time
(`d8ad18311..HEAD`, 91 commits, ~15 real PRs) -- the rest of the rotation
was quiet (sparkglm exl3 still `2da6a1c31`, tonyd2wild still silent since
09-02, AEON-7 still silent since 09-12 -- its 09-13 `pushed_at` bump
turned out to be a `WatchEvent`/star, not a push, confirmed by checking
every branch's actual commit history).

**vLLM PR #54929 re-checked** (the #51921 stall fix candidate): still not
safe to pull. `mergeable: MERGEABLE` (conflict-free this hour, after the
author merged `main` in again just ~2h before this check) but
`mergeStateStatus: BLOCKED`, `reviewDecision: REVIEW_REQUIRED`, and
`pre-run-check` CI is currently failing. Zero maintainer review despite
the author pinging the relevant code owners on 2026-09-04. Scope has also
grown since first seen (a second commit added a "7.4x decode attention"
split-K path, unvalidated by anyone but the author). A week of
conflict/rebase churn, not a stable target. Recheck again in a few days.

**Two items shipped this round** (both in `recipes/build/glm53-exl3-
v20-upstreamsync/`, see NOTES.md's "Amendment 2026-09-15" for full
technical detail):
- **`patch_default_max_new_tokens.py`** (MiaAI-Lab PR #51): omitted-`max_tokens`
  decode-hygiene default. `DEFAULT_MAX_NEW_TOKENS=65536` now set in both
  the default and maxprefill recipes' `env:` blocks. Without this, a
  client that never sets `max_tokens` gets vLLM's own fallback of
  `max_model_len - prompt_len` -- up to our full 262144-token ceiling on
  a short prompt -- large enough for one such client to decode until it
  preempts every other session via KV pressure. Explicit client
  `max_tokens` is completely unaffected. All three patch anchors matched
  our vLLM byte-for-byte; ported near-verbatim from an already
  well-built, tested upstream implementation.
- **`prelaunch_flush.sh`'s `check_memory_available()`** (MiaAI-Lab PR
  #39): pre-boot check comparing host `MemAvailable` against
  `gpu_memory_utilization x MemTotal + headroom` on both nodes, run right
  after the existing drop_caches + fragmentation steps. Catches a
  host-memory hold (not a container -- distinct from any existing
  container-level check) BEFORE the image pull and weight load, instead
  of dying mid-bring-up with a `ValueError: Free memory on device ...
  less than desired` deep in a worker log. FATAL by default
  (`GLM53_PREFLIGHT_SKIP_MEMORY_CHECK=1` to override) -- deliberately
  different policy from the fragmentation check next to it, which stays
  advisory-only, because this failure mode is a guaranteed deterministic
  hard-stop rather than a probabilistic risk. Directly relevant to this
  project's own MemFree/MemAvailable-gap and earlyoom tracks. Verified
  via `bash -n` and a standalone arithmetic check against real host
  numbers; not yet exercised on a live boot (would require running it
  against a host with production traffic, not done this round).

**`thinking_token_budget` bug (flagged via the mmastrac/4x-gx10 dig,
2026-09-14) -- checked and confirmed NOT applicable.** The bug lives in
vLLM's V2 model runner's sampler
(`vllm/v1/worker/gpu/sample/sampler.py`'s `_requires_logits_processing()`
gate, which never checks thinking-budget state, silently skipping budget
enforcement whenever temperature is 0 or 1.0 and no other sampling knob
is active). Read our own vendored vLLM directly: GLM-5.3-Flash+EXL3
resolves to the V1 model runner (established during the cudagraph_align
work), and V1's own sampler (`vllm/v1/sample/sampler.py`) has a
structurally different implementation -- `apply_logits_processors`
applies thinking-budget logic unconditionally whenever there are tracked
requests, with no temperature-based short-circuit gate at all. The buggy
code technically exists in our installed vLLM package (both V1 and V2
ship unconditionally) but is dead code for our runtime config. Confirmed
by reading the actual code, not assumed from the upstream report.

**Also assessed, not adopted this round** (all from the MiaAI-Lab range):
PR #170 (long-prefill boot-warmup ladder extension to 3584/7168/14336/
65536-token rungs -- we're prefill-heavy, plausibly worth it, just not
done yet), PR #94 (`GLM53_KV_CAPACITY_LOG`, informational-only logging
clarifying the boot "GPU KV cache size" line for hybrid MLA+mamba+drafter
models), PR #70 (`spec-accept-gate.sh` -- a diagnostic script checking
for vLLM #53030, CUDA graphs pinning per-position spec-decode acceptance
at exactly 1.00; useful given we run MTP-2, but tooling not a serving-path
change), PR #41 (`spark_doctor.sh`, ops diagnostic tooling). PR #186/#187
(fair-v5 mixed-prefill scheduler, now MiaAI-Lab's own TP=2 default,
measured real TTFT/throughput tradeoffs on their kit) is explicitly
**not** a drive-by candidate -- a real scheduling-behavior change on our
exact topology, needs its own deliberate A/B before ever being
considered, not bundled into a routine-check round. Not applicable at
all: PR #31/#37 (bench convenience endpoint), PR #75 (EXL3 SM121 kernel
lab, dev-only/no GPU tested), any TP=3/TP=4-specific commits (we run
TP=2 only), PR #129 (docs-only), PR #189 (cosmetic).

**mmastrac/glm-5.3-flash-4x-gx10 secondary candidates, resolved**: the
spin-wait patch (`busy_loop_s` 1.0->0.002, same lever as this fork's own
`patch_spinwait.py`, +5.5% decode / 20C cooler on their kit) is worth a
constant-diff check against our own value -- not yet done. The
`GPU_MEM_UTIL=0.90` wedge report is informative but not actionable as a
config change (their shipped ceiling is 0.88, higher than our 0.84; their
failure mode -- total unresponsive wedge, no SSH, OOM killer can't even
intervene on UMA -- is corroborating evidence for this project's own GMU
root-cause understanding, not a new lever). The `thinking_token_budget`
item is covered above.

**Net**: a real, productive round -- two backports shipped (memory
preflight + max-tokens hygiene), one flagged bug ruled out with actual
code verification rather than assumption, several more items identified
and deliberately deferred rather than rushed. Production was never
touched; the image was rebuilt under the same tag (`glm53-exl3-
v20-upstreamsync:local`) but not relaunched.

## cbertucci33/vllm-v29-glm53flash-exl3-dgx triage (2026-09-15)

User-directed look at a single-author repo targeting the same hardware/
model combination as this fork almost exactly. Read-only research: repo
tree via `gh api`, README, and a direct file-existence diff against
`vllm-project/vllm@v0.29.0` to separate genuinely custom work from stock
vLLM. No production access needed or used.

**First-pass mistake, corrected by the user**: initially read the
DFlash2 drafter as unpublished (the README says "model weights are
published separately" without a link). The user supplied the actual HF
links -- both the target (`cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-
Sliced`) and the drafter (`local-inference-lab/GLM-5.3-Flash-DFlash2-
MXFP8`) are public. The material point survives anyway: the target is an
**abliterated/uncensored** checkpoint, not stock GLM-5.3-Flash, and the
drafter was trained against that specific target's activations. Spec-
decode drafters are distilled against one target's output distribution --
there's no reason to assume this drafter's acceptance rate transfers to
our stock (non-abliterated) GLM-5.3-Flash EXL3 checkpoint, and abliteration
is known to shift a model's logit distribution in exactly the ways that
matter for draft-token acceptance. Unverified either direction; flagging
the mismatch rather than assuming it's fine or assuming it's broken.

**What's stock vLLM 0.29.0, not their invention** (confirmed via
`gh api repos/vllm-project/vllm/contents/<path>?ref=v0.29.0`, checking for
a 404): DFlash2 (`vllm/v1/spec_decode/dflash.py`), B12X
(`vllm/model_executor/layers/fused_moe/b12x.py`), and the TopK CUDA
kernel family (`csrc/libtorch_stable/persistent_topk.cuh`) all resolve
in stock v0.29.0. Their README's step-by-step framing implies these are
their own additions; they aren't -- this ecosystem's PRs land upstream
fast. `vllm/models/glm5next/` (their path) is genuinely absent from
v0.29.0 (only `vllm/models/{common,deepseek_v32,deepseek_v4,dots3_note,
hy_v4,inkling,kimi_k3,minimax_m3,qwen4_exp}` exist at that tag) -- so
their GLM-5.3-Flash model support (imported from vLLM PR #53906 per
their README) is a real backport of not-yet-released upstream work, same
category of thing this fork's own overlay patches do.

**What's genuinely custom to them, in priority order**:
1. **Sparkinfer** (`github.com/gittensor-ai-lab/sparkinfer`, real
   separately-maintained project, corroborated by multiple unrelated
   repos in this ecosystem citing it) -- a native EXL3 Trellis execution
   engine used *instead of* stock ExLlamaV3 for the actual quantized
   matmul path. The single most architecturally distinct choice in the
   repo. We (`overlay/exl3.py`) run stock ExLlamaV3 directly, same as
   most EXL3-on-vLLM efforts. Real potential upside if Sparkinfer's
   kernels are faster on GB10 specifically, but unproven against our
   workload and a nontrivial integration cost (own CUTLASS DSL pin, a
   custom ARM64 build with x86 AVX units stripped). Research lead, not a
   backport candidate yet.
2. ~~A GB10-safe exact TopK kernel for MoE expert routing~~ **CORRECTED
   2026-09-16, see below**: this is not the MoE router's TopK at all --
   it's FlashInfer's own JIT topk module
   (`flashinfer.jit.topk.gen_topk_module`, built by their
   `build/build_flashinfer_topk.sh`), used by their custom
   `FLASHINFER_MLA_SPARSE_SM120` attention backend for sparse-MLA
   page/candidate selection. Same subsystem as the already-tracked
   `cooperative_topk`->`persistent_topk` indexer finding, not a second,
   independent call site. See "cbertucci33 items 1/2 resolved" below for
   the actual verified answer.
3. **A lossless EXL3 checkpoint pre-slicer**
   (`tools/slice_exl3_checkpoint.py`) -- splits routed-expert tensors
   per-TP-rank offline, verifies bit-for-bit reconstruction, before
   deploy. Relevant to this project's own boot-time host-memory-pressure
   track (`memfree_memavailable_gap`, the new `check_memory_available()`
   preflight) -- pre-sliced per-rank checkpoints could lower the
   per-node peak memory during weight load, which is exactly the phase
   the new preflight check guards. **Not yet evaluated against our own
   checkpoint format/loader.**
4. Their own EXL3 quantization layer (`vllm/model_executor/layers/
   quantization/exl3.py`, confirmed absent from stock vLLM) -- same
   category as our own `overlay/exl3.py`, expected to exist independently
   in any EXL3-on-vLLM effort since vLLM doesn't ship EXL3 natively.

**Their claimed performance, treated skeptically**: 24.5 tok/s weighted
decode, +13.3% over a "previous runtime, same hardware" baseline of 21.62
tok/s, from a 197-request live sample with DFlash2 (7 proposals). Two
reasons not to read this as "DFlash2 beats MTP-2": (1) it's unclear what
the 21.62 tok/s baseline actually was -- no spec decode, MTP-2, or
something else isn't stated; (2) their own "improved" 24.5 tok/s sits
inside the range this fleet is *already* measuring for stock MTP-2 on the
same hardware class (p50 26.7 tok/s, p95 20.5 tok/s decode-only,
`request_time_per_output_token_seconds` over the last 3h as of this
check -- see the mcp-grafana probe from the same session). More likely
their baseline was weaker than what we already run, not evidence DFlash2
itself is faster.

**Net, and what's queued for the next routine round** (production
untouched this session; these are research-only next steps against our
own build tree, not against a live host):
- Check our MoE router's TopK path for the same GB10 shared-mem-limit
  exposure as items 2 above -- cheap, do this first.
- Evaluate `tools/slice_exl3_checkpoint.py` against our own checkpoint
  layout for the boot-memory-pressure angle.
- Sparkinfer stays a flagged research lead (real, active, corroborated
  project) -- not actionable without a dedicated eval, given the native
  dependency cost.
- DFlash2/their published drafter: not adoptable as-is (trained against
  an abliterated target we don't run); would need our own DFlash2
  drafter trained against stock GLM-5.3-Flash to be a fair comparison at
  all, which is out of scope for a routine check.
- Added `cbertucci33/vllm-v29-glm53flash-exl3-dgx` to the tracked-repo
  rotation above, flagged as likely low-activity (single-author,
  integration-history framing, may not get further commits).

## Routine upstream check, 2026-09-16 -- and items 1/2/3 from the cbertucci33 triage

Fan-out round on a production traffic break (6 parallel agents, read-only
research, production never touched). Also closed out the three action
items queued from the 2026-09-15 cbertucci33 triage.

**MiaAI-Lab main, `d8ad18311..HEAD` (155 commits, mostly merge noise from
a 09-15 PR-cleanup burst).** Two items need a dedicated A/B, not a
routine-round adoption:
- **PR #186/#194/#198 -- "fair v5" mixed-prefill scheduler is now
  MiaAI-Lab's own TP=2 default** (`GLM53_MIXED_PREFILL_CHUNK`:
  `skip`->`fair`, `GLM53_FAIR_PREFILL_SHARE=0.30`, `MAX_STEP_MS=1000`).
  A real decode/prefill-interleaving policy change on our exact topology.
  Not adopted -- same standing rule as PR #186/#187 from the 09-15 round.
- **PR #200/#201 -- default image now `:exl3-instanttensor`,
  `LOAD_FORMAT=instanttensor`**: a new third-party direct-I/O safetensors
  loader (`instanttensor==0.2.0`, installed `--no-deps` to dodge an NCCL
  conflict) replacing the core weight-loading path. New dependency in a
  correctness-critical path -- needs its own validation before we'd point
  at it. Not adopted.

Low-risk, applicable backport candidates identified (not shipped this
session -- see "Net" below for why): PR #37 (`/reset_prefix_cache` admin
endpoint, opt-in), PR #42 (pipefail-safe container health checks), PR #81
(`GLM53_EXTRA_ENV` diagnostic passthrough, opt-in), PR #94
(`GLM53_KV_CAPACITY_LOG`, boot-time log line only, default-on, useful
given our memory-pressure tracks), PR #95 (`GLM53_APC_NO_STORE`
per-request prefix-cache skip, inert unless a caller opts in), PR #170
(long-prefill boot-warmup metadata-kernel fix). Not applicable: PR #184
(TP=3), #115/#187/#188 (TP=4, or fair-scheduler-off confirmation for
TP=3/4 -- confirms the TP=2 fair-default is specifically what needs our
A/B), #137 (abliterated-weights preset, different model), #75 (EXL3
SM121 kernel lab, still dev-only, passively watching), #129/#189/#192
(docs/cosmetic/test-path).

**Enntity/sparkglm `exl3` branch, `a8aaa229..2da6a1c3` (3 commits).**
Nothing to backport -- every substantive item is the sibling project
re-deriving conclusions we already reached independently (E3-vs-E2 not
transferable once you already have grouped M64 + cooperative K4, matches
our own v12-combined-era finding; cooperative-decode-32 not worth it,
matches our serving limit of 16; MXFP8 DFlash2 draft still fails their
own arithmetic semantic gates). One operational signal: the branch's
README now explicitly marks it "retained" (frozen/archival), read
together with the 09-13 shift toward a research/experiments structure --
EXL3-relevant activity from this sibling may be tapering off. Flag for
future rounds: check whether relevant work migrates to `main` (now
NVFP4-focused, likely irrelevant) or simply stops.

**mmastrac/glm-5.3-flash-4x-gx10.** No repo activity since 2026-09-14
(last push still 2026-09-07). Resolved the deferred spin-wait diff-check:
their value is `busy_loop_s=0.002` (2ms), applied via a `sed` bind-mount
override targeting `shm_broadcast.py`, reporting CPU 185%->109%, ~20C
cooler, decode 66.9->70.6 tok/s on their kit. **No action needed on our
side** -- our own `patch_spinwait.py` docstring already records that we
tested 2ms ourselves in our own frozen TP=2/MNBT=2048 sweep and it *lost*
1.68% decode versus our chosen 16ms, which beat stock on both decode
(+0.95%) and CPU (-85.3%). Their optimum for their workload isn't ours;
we'd already tested their exact candidate and rejected it with real data.

**tonyd2wild and AEON-7.** Both confirmed still silent (tonyd2wild: no
commits past 09-02 on either branch; AEON-7: no commits past 09-12 on any
of its four branches). The AEON-7 `updated_at` bump noted in the 09-15
round remains a non-push event. Nothing to review.

**cbertucci33/vllm-v29-glm53flash-exl3-dgx, `29640cb..96483c3`.** One new
commit, README-only (+2 lines, notes on base-model provenance and
cross-quant-variant compatibility). Confirms the "low-activity,
single-author" read from the initial triage.

### cbertucci33 items 1/2 resolved

**Item 1 (GB10 TopK exposure) -- corrected and closed, no action
needed.** Direct verification against our own running production
container (`docker exec`, read-only) rather than assumption:
- Our MoE expert router (`vllm/model_executor/layers/fused_moe/router/
  grouped_topk_router.py`, the path `glm5next`'s `use_grouped_topk=True`
  config selects) calls `ops.grouped_topk(...)`, a dedicated small-k
  CUDA kernel built for the "top-8-of-288-experts" routing problem. It
  never touches `cooperative_topk`/`persistent_topk` at all -- confirmed
  by grepping every file in our installed vLLM package for those two
  symbols: they appear ONLY in `sparse_attn_indexer.py` and
  `sparse_attn_indexer_kpool.py`. **There is no "MoE router TopK GB10
  exposure" -- that framing in the 2026-09-15 entry was wrong, based on
  an unverified assumption about which subsystem cbertucci33's fix
  targeted.** Their fix (confirmed by reading their
  `build/build_flashinfer_topk.sh`, which calls
  `flashinfer.jit.topk.gen_topk_module`) is inside FlashInfer's own JIT
  module for their custom `FLASHINFER_MLA_SPARSE_SM120` backend -- a
  third code path, distinct from both of vLLM's own.
- More importantly: **our actual sparse-indexer TopK gate already
  handles GB10 correctly**, read directly from
  `sparse_attn_indexer.py`:
  ```
  use_cooperative_topk = (
      current_platform.is_cuda()
      and topk_tokens in (512, 1024, 2048)
      and num_rows <= 32
      and logits.stride(0) % 4 == 0
      and current_platform.has_device_capability(90)
      and not current_platform.is_device_capability_family(120)
  )
  use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
      512, 1024, 2048,
  )
  ```
  `not current_platform.is_device_capability_family(120)` explicitly
  excludes GB10 (family 120) from the cooperative path; GB10 falls
  through to `use_persistent_topk`, which has no such exclusion. This
  **directly confirms**, for the first time by reading the actual gate
  rather than inferring from patch presence, the item the AEON-7 triage
  left open ("worth confirming we already handle correctly"). Closed,
  no code change needed.
- We don't use FlashInfer's native sparse-MLA backend at all (our
  FlashInfer is stock 0.6.17, not their custom-built 0.6.18 with the
  `GLM53_NOPE` SM120/SM121 kernels from FlashInfer PRs #4802/#4947), so
  we were never exposed to whatever shared-mem issue exists in
  FlashInfer's own topk JIT module either. Not applicable to us on two
  independent grounds.

**Item 2 (checkpoint pre-slicer) -- real, plausible lever, not yet
trialed.** Read our own `overlay/exl3.py`'s weight-loading path:
`_narrow_tp()` (and `shard_exl3_col`/`shard_exl3_row`) slice each routed-
expert tensor by `tp_rank`/`tp_size` via `.narrow(dim, ...).contiguous()`
-- applied AFTER vLLM's standard loader has already materialized the
FULL, un-sharded tensor from the checkpoint into host memory (vLLM's
default safetensors loader path calls `safe_open(...).get_tensor(name)`,
which copies the complete tensor out of the mmap rather than returning a
view). That means, per large routed-expert tensor, each rank transiently
holds both the full tensor AND its own narrowed half at once, before the
full one is freed -- a real, avoidable peak-memory spike during the load
phase specifically. This lines up directly with this project's own
`memfree_memavailable_gap` and boot-time earlyoom tracks. A pre-sliced
checkpoint (`slice_exl3_checkpoint.py`'s approach: split per-TP-rank
offline, each rank's file only ever contains its own half) would remove
this transient entirely -- each rank's `get_tensor()` call only ever
materializes already-halved data.
Caveat: we run TP=2 across 2 *separate* hosts (one rank per host), so we
don't have the "N ranks competing for one host's RAM simultaneously"
multiplier a single-host TP=2/4 setup would -- the exposure here is the
single-rank-per-host transient 2x-on-the-largest-tensor, not a
cross-rank pile-up. Real, but its actual magnitude (how big the largest
single routed-expert tensor actually is, and whether it's the dominant
term in our known boot-memory pressure or a minor contributor next to
other allocations) is **not yet quantified** -- would need a scratch
trial: slice our own checkpoint with their tool (or an equivalent), boot
against it, and diff peak host RSS during load against our current
un-sliced boot. Not done this session (production was on a brief break,
not available for a full boot trial). Queued as a real candidate for the
next dedicated (not routine) round.

**Item 3 (Sparkinfer dedicated eval) -- downgraded from "research lead"
to "unverifiable dependency," not worth a trial.** Dedicated research
pass materially overturns the 2026-09-15 framing:
- Sparkinfer (`gittensor-ai-lab/sparkinfer`) is real and very active
  (1,568 commits, pushed same-day, MIT-licensed) but tied to Bittensor
  subnet SN74 -- contributors paid in a crypto-incentive token for
  verified speedups, judged by the project's own automated eval bot. A
  full-repo code search for `exl3`/`trellis` returns **zero hits**. It
  supports GGUF and NVFP4/ModelOpt for Qwen3.8/3.6 only -- no GLM, no
  EXL3, nothing resembling cbertucci33's claimed integration surface.
  `sm_121` is a genuine build target, but every published benchmark
  (+86% decode/+127% prefill vs llama.cpp, DSpark speedups) is measured
  on RTX 5090; DGX Spark/GB10 is listed only as an unstarted roadmap
  item, and the "verifiable" eval log is self-hosted/self-reported, not
  third-party audited -- structurally exactly the setup where narrow
  overfitting to the pinned incentive-eval hardware is a real risk, not
  a hypothetical one.
- Inside cbertucci33's own `exl3.py`: generic/dense EXL3 matmuls already
  run through stock `exllamav3_ext.exl3_gemm` -- the same path we use.
  Sparkinfer is invoked ONLY for the routed-MoE-expert path, gated on
  pre-sliced checkpoints, and the module's own docstring calls the
  relevant API "Sparkinfer's **planned** full-rotation Trellis MoE API"
  -- their own word, "planned." Their pinned Sparkinfer commit
  (`d4438d490691f79022fdfc8149e1c5f161d15445`) returns 404 against the
  real public repo; no fork or mirror containing it is findable anywhere
  on GitHub. Their own test for this path monkeypatches
  `_load_sparkinfer_trellis()` rather than exercising a real build. This
  dependency is not obtainable -- their MoE-path performance claims rest
  on something that, as far as can be verified from outside their own
  machine, doesn't exist in public form.
- If it did exist, integration cost would actually be modest (the
  Sparkinfer-specific glue in their `exl3.py` is a small, isolated
  slice, not smeared through the file) -- the blocker is entirely the
  missing artifact, not architectural invasiveness. And yes, it would
  force our checkpoint onto their pre-sliced-per-rank schema, layered on
  top of (not replacing) the current dense-tensor path.
- **Verdict: watch, don't act.** Re-check `gittensor-ai-lab/sparkinfer`
  for `exl3`/`trellis` on the next routine rotation (cheap grep); re-open
  only if that appears or cbertucci33's repo gets a resolved pin. No
  trial possible today -- there's nothing installable to trial against.

### Net

Two real MiaAI-Lab (c)-category items flagged for future dedicated A/Bs
(fair-v5 scheduler, instanttensor loader) and six low-risk (b)-category
backport candidates identified but **not shipped this session** --
volume (6+ new patches) and the higher-priority live decode-speed
investigation reported by the user this same session took precedence;
queued for the next implementation pass. The cbertucci33 GB10-TopK item
is now fully resolved (corrected framing, verified we're safe on both
counts). The checkpoint pre-slicer stays a real, well-reasoned but
unquantified lever. The Sparkinfer lead is downgraded to a cheap watch
item, not a live research thread -- its load-bearing dependency doesn't
verifiably exist. Production untouched throughout.

## Long-context decode-speed investigation, 2026-09-16 (5-9 tok/s at c=1/c=2)

User report: after ~3 days of stability (1 reboot, 1.2-1.3B input tokens
served), long-context decode has "crawled to 5-9 tok/s decode even on
c=1 and c=2" specifically during subagent-heavy sessions (openchamber +
opencode harness). Investigated via mcp-grafana (Prometheus), DCGM host
metrics, and re-reading this project's own prior profiling work --
production traffic was on a brief break, no new live capture was
triggered against it (see "not done this session" below).

**GPU hardware ruled out.** `DCGM_FI_DEV_SM_CLOCK` held a steady
2400-2540 MHz on both nodes across the full 3-day window (no throttling,
running at/above base clock throughout). `DCGM_FI_DEV_GPU_TEMP` cycled
39-79C with load, well within normal range, no thermal runaway or
sustained high-temp plateau. Clean on both counts -- this is not a
hardware degradation story.

**Aggregate server-side metrics don't show a broad regression, but the
tail does.** `request_time_per_output_token_seconds`-derived decode
throughput: p50 over the full 3-day window held flat at ~26.4-26.6 tok/s
the entire time (barely moved). But the **p95 over just the last 6h
dropped to 10.2 tok/s**, versus ~20.5 tok/s measured in yesterday's probe
of this same metric. Typical/short requests are unaffected; the WORST
requests in the distribution have gotten meaningfully worse recently.
That's consistent with a problem concentrated in the long-context tail
specifically, diluted away in the aggregate p50 -- exactly where
subagent-harness traffic (deep, growing multi-turn context, per the
mcp-grafana probe from two sessions ago: median prompt 122,860 tokens,
95.1% prefix-cache hit rate) would land.

**This project already has an unresolved, matching finding from before
this session.** `docs/DESIGN-indexer-workspace.md` (`patch_indexer_
workspace.py`'s design doc, written during the v19-indexercompat work) 
states directly: *"the research train's arithmetic put the indexer's
entire context-proportional decode cost at ~1 ms/step at 100K KV, ~0.5%
of the measured 200 ms long-context delta... roofline arithmetic cannot
prove the indexer is not a contributor -- so this PR simply does not
make a latency claim in either direction."* Read plainly: at 100K KV
context, someone already measured a **~200ms/step decode delta** versus
short-context decode, confirmed the sparse indexer's own theoretically-
expected linear-scaling cost is not the cause (only ~1ms of the 200ms),
and then **explicitly left the actual cause unresolved** -- this was
never root-caused, just ruled out as "not the indexer." **The magnitude
matches the user's report almost exactly**: a healthy short-context
decode step (~26 tok/s implies ~38ms/token) plus a +200ms/step delta at
100K KV gives ~238ms/token, i.e. **~4.2 tok/s** -- squarely inside the
user's observed 5-9 tok/s range. This was not surfaced or connected to
the current complaint until now; it was written up as an aside in a
memory-focused design doc and never cross-referenced into VALIDATION.md
or SPEED.md.

**Corroborating, not yet conclusive: a stray profiling trace.** Two
untracked artifact directories already existed in the working tree
before this session (`recipes/probes/cpu_profiles_v20/`,
`recipes/probes/traces_v20_b1/`, both `git status`-untracked, no
README/manifest, no cross-reference anywhere in VALIDATION.md/NOTES.md/
SPEED.md -- an orphaned capture from an earlier ad-hoc session, unclear
what context length it was captured at). Parsed the rank0 PyTorch trace
directly (gzipped Chrome-trace JSON, 632,699 events):
- Confirms the already-separately-tracked Ampere-fallback GEMM finding
  is still present and substantial: `cutlass_80_wmma_tensorop_bf16_
  s161616gemm...16x16` totals 2.74M us across 19,866 calls -- this is
  the same kernel flagged in the AEON-7 triage entry above as "~36% of
  decode GPU time, identified 2026-09-02, confirmed unchanged
  2026-09-11." Still unaddressed as of this trace. This is a constant
  per-active-token cost (MoE routed-expert GEMM), not itself obviously
  context-length-scaled, so it's a separate, compounding inefficiency,
  not the specific explanation for the *long-context-specific* delta.
- One suspicious data point: `cudaEventSynchronize` totals 7.24M us
  across only 85 calls -- an average of **~85ms per sync event**, large
  enough to be a real contributor to a per-step delta in this range. Not
  conclusive on its own without a paired short-context trace to diff
  against (this file has no companion short-context capture, and no
  metadata confirming what context length it was captured at) -- flagged
  as the most promising lead for a follow-up trace comparison, not a
  confirmed cause.

**Not investigated this session, deliberately**: did not trigger a new
profiling capture (py-spy/torch-profiler) against the live production
containers -- that requires either a longer safe window than a brief
traffic break, or explicit go-ahead, given capture overhead and the
project's standing rule not to disturb production without confirmation.
Also did not instrument the openchamber/opencode harness side (no access
from this environment) -- can't fully rule out a compounding harness-side
effect (e.g. how it paces/batches subagent calls), but the magnitude
match to an already-documented, unresolved SERVER-side ~200ms/step
long-context delta is strong enough that harness-side causes should be
considered secondary, not primary, until this is re-checked.

**Recommended next step** (not started, needs a dedicated window):
capture a fresh, labeled paired trace -- one short-context (a few K
tokens) and one long-context (100K+) decode step, same request shape
otherwise -- and diff them directly for what actually grows with context
length. Candidates worth checking first, in order: the `cudaEventSynchronize`
count/duration, any O(context) CPU-side Python work per decode step
(block-table/seq_lens/position_ids construction), and the indexer's
*scoring* pass specifically (distinct from its already-ruled-out
`cooperative_topk`/`persistent_topk` selection cost) since indexer
scoring over full KV history is inherently O(context) by construction
in this architecture family, unlike selection itself.

**Status**: root cause not found, but the search space is now much
narrower and grounded in this project's own prior (undocumented-until-
now) measurement rather than a fresh guess. Not a GPU hardware issue,
not fully explained by the already-tracked GEMM-kernel inefficiency
alone, most likely the same ~200ms/100K-KV-step delta this project
measured and set aside months ago without root-causing it. Production
untouched throughout this investigation.

## Correction: controlled short/long-context test overturns the 200ms-delta hypothesis (2026-09-16)

Direct follow-up to "Long-context decode-speed investigation, 2026-09-16"
above. With production traffic paused (a real window, not simulated),
sent three controlled, isolated requests directly against the live
`glm53-exl3-v20-upstreamsync:local` server (`c=1`, nothing else running
concurrently) and read each request's own contribution to the
`vllm:request_decode_time_seconds_sum` / `vllm:time_to_first_token_
seconds_sum` / `vllm:request_generation_tokens_sum` counters via tight
before/after deltas -- a clean, direct measurement, no profiler needed:

| Request | Prompt tokens | Completion tokens | Decode-phase (measured) | TTFT (measured) |
|---|---|---|---|---|
| Baseline | 5 | 5 | 42.4 ms/token (23.6 tok/s) | 0.80s |
| Short | 37 | 30 | 29.7 ms/token (33.6 tok/s) | 0.30s |
| Long, cold (0% cache) | 108,001 | 30 | **28.1 ms/token (35.6 tok/s)** | 69.67s |
| Long, repeated (cache hit) | 108,001 | 30 | **28.0 ms/token (35.7 tok/s)** | 2.77s |

**Decode-phase per-token cost did not degrade with context length in
this controlled test** -- 28-30 ms/token essentially flat from 37 to
108,001 tokens, cached or not. This directly contradicts reading the
old `DESIGN-indexer-workspace.md` "~200ms/step long-context delta" note
as the explanation for the current complaint. That old figure either
measured something materially different (concurrent load, a different
vLLM/patch state, real multi-turn conversational KV structure rather
than a single long synthetic prompt) or no longer reproduces on the
current build -- either way, **it does not explain today's user report**,
and citing it as the leading hypothesis in the entry above was wrong.
Retracting that as the primary lead.

**What the same test data actually explains cleanly**: TTFT. Cold (0%
cache) 108K-token prefill took 69.67s -- a real, expected cost at roughly
1,550 tok/s raw uncached prefill throughput, nothing wrong with it, just
the honest cost of genuinely new tokens. The *identical* prompt repeated
immediately after collapsed to 2.77s TTFT (25x faster) via prefix-cache
reuse, confirming caching itself works correctly. **This is the same
mechanism identified in this engagement's very first mcp-grafana probe**:
apparent "tokens/sec" as commonly computed by a client
(`output_tokens / total_wall_clock_time`, TTFT included) is dominated by
prefill/TTFT whenever output is short relative to prompt size --
median output length for this traffic class was ~40 tokens. Do the
arithmetic on a partially-cached long-context turn: even a modest
cache-miss fraction of a 100K+-token context, at ~1,550 tok/s raw prefill,
adds seconds to tens of seconds of TTFT that a client-side "tok/s" counter
will fold into the completion-token denominator, producing exactly the
kind of single-digit "tok/s" the user is seeing -- without decode itself
ever slowing down.

**Revised leading hypothesis**: the user's subagent harness (openchamber
+ opencode) most likely isn't getting the same ~95% prefix-cache hit
rate this fleet's *aggregate* traffic sees. Plausible mechanisms, neither
confirmed yet: (a) many parallel/sequential subagents with large but
mutually-diverging contexts (shared system prompt, divergent task
content) evicting each other's cached prefixes under KV-cache capacity
pressure: (b) the harness's own "tokens/sec" reporting counts TTFT in
the denominator, making a client-side measurement artifact look like a
server-side regression. Neither ruled in nor out this session -- would
need either real (not synthetic) subagent-shaped multi-branch context
traffic replayed against a monitored server, or the harness's own timing
methodology, to settle definitively.

**Status**: the ~200ms/100K-KV-step decode delta from `DESIGN-indexer-
workspace.md` stays on record as a real, previously-measured, still-
unexplained data point from past work -- but it is NOT the explanation
for this session's user report. Root cause of *that* old number remains
genuinely open; root cause of *the current complaint* is now most likely
prefix-cache-hit-rate/TTFT-related, not a decode-speed regression at all.
Production traffic resumed normally after this test (both hosts healthy,
sanity completion request verified correct output before and after).

## RoCE dual-rail verification (2026-09-16)

Resolves the follow-up flagged in "Routine upstream check, 2026-09-13"
(MiaAI-Lab measured +63% busbw dual-rail on their kit; never checked
whether our own cluster config uses both rails). Read-only host
inspection, both nodes, production untouched.

**Physical hardware: both rails already exist and are cabled/connected
on both hosts.** Each node has two separate physical ConnectX-7 cards
(`rocep1s0f0`/`enp1s0f0np0` and `roceP2p1s0f0`/`enP2p1s0f0np0`), each
with a live link-up port on the `192.168.177.0/24` RoCE subnet.
Confirmed with a direct ping across the second rail specifically
(`192.168.177.87` from the other node): sub-millisecond RTT, 0% loss,
same as the first rail. **No new cable needed** -- this is a stock
dual-CX7 DGX Spark configuration, already fully wired, just not
exploited.

**Current config: single-rail.** `start.sh` pins exactly one HCA device
per rank (`HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"`,
`WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"`), passed to NCCL as a
single `NCCL_IB_HCA=<one device>` value. The second card on each host
(`roceP2p1s0f0`/`enP2p1s0f0np0`) sits live and reachable but genuinely
idle from NCCL's point of view.

**What dual-rail would and would not help, and why**: TP=2 across two
*separate hosts* (no NVLink) needs an NCCL all-reduce over the network
at every layer, for every token, during both prefill and decode --
that's the only place RoCE bandwidth matters for this deployment.
Checkpoint loading and container/image distribution do NOT use this
fabric in our setup (weights come from HF per-host independently;
`sparkrun`'s image distribution goes over the separate management
network, `10.7.0.x` -- confirmed via the `ib:`/`mgmt:` address split
`sparkrun status` already shows per node).
- **Prefill**: processes a large batch of tokens per forward pass, so
  each all-reduce message is large -- genuinely bandwidth-bound.
  Doubling available bandwidth via a second rail should give a real,
  proportional benefit here. This lines up well with this fleet's own
  workload shape (median prompt ~122K tokens per the mcp-grafana probe
  earlier this engagement) -- unlike the dense-FP8 kernel work rejected
  earlier for favoring decode at prefill's expense, this lever favors
  the phase that actually dominates our traffic.
- **Decode**: one token (or a handful, at our typical c=1/c=2) per step
  -- the all-reduce message is tiny, and tiny-message collective time is
  dominated by per-message latency/overhead, not raw bandwidth. A
  second rail mainly helps here only if the first rail is congested
  (unlikely at low concurrency); expect little to no decode-throughput
  change from this change alone.
- MiaAI-Lab's own +63% figure (20.9 vs. 12.8 GB/s peak busbw) is a raw
  NCCL all-reduce microbenchmark, not an end-to-end inference-throughput
  measurement -- a reasonable proxy for the prefill story above, but the
  real serving-throughput gain is unmeasured and would need our own A/B.

**Known pitfall if this is ever turned on**: MiaAI-Lab's own PR #172
(referenced in the 2026-09-14 upstream-check entry above) fixed a bug
where their launcher's RoCE GID preflight only checked the FIRST HCA on
a dual-rail config (`HEAD_CX7_IB=dev1,dev2`), silently skipping
validation and failing ~60s into a real run instead of at boot. Our own
`start.sh` GID-preflight logic (around line 584-603) would need the
same fix before a dual-rail config could be trusted -- not yet checked
whether ours has this exact gap, since we've never run dual-rail.

**Status**: verified feasible (hardware ready, zero cabling cost),
config change identified (`NCCL_IB_HCA` needs both devices, comma-
separated, on both ranks), expected benefit is prefill-specific and
plausible-but-unquantified for us specifically. Not enabled this
session -- this touches core NCCL fabric behavior and should get a
deliberate test (and the GID-preflight dual-rail check ported first),
not a drive-by flip in production.

## InstantTensor checkpoint loader validation (2026-09-16)

Follow-up to the "MiaAI-Lab's instanttensor checkpoint loader" item
flagged in the 2026-09-16 upstream-check round. Read-only research +
container inspection, no scratch trial run this session (recommended as
the next step, not completed).

**Architecture is safe for our EXL3 format by design, verified not
assumed.** InstantTensor (`scitix/InstantTensor`, Apache-2.0, ScitiX AI +
Peking University, real test suite) is a generic, dtype-agnostic
`safe_open`-compatible reader -- it has zero EXL3/quant-specific code,
by design, and doesn't need any: vLLM's native support (merged upstream
via PR #36139, refined #46868/#52801 -- our base image postdates all
three, **no vLLM-side patch needed**) wires it in at
`default_loader.py`'s iterator-selection level, strictly below every
per-parameter `weight_loader` hook. Our own `overlay/exl3.py`'s
`shard_exl3_col`/`shard_exl3_row`/`_load_exl3` dispatch is completely
untouched either way -- architecturally about as low-risk as a loader
swap could be built.

**Our own guardrails would catch most corruption loudly**: `_load_exl3`
raises on any shape mismatch; `process_weights_after_loading` checks
every expert's MCG marker (`0xCBAC1FED`) and raises if it doesn't match
-- a real per-expert integrity check. **Real residual gap**: neither
check validates the trellis payload bytes themselves (the bulk of the
checkpoint) -- a silent bit-flip inside a correctly-shaped trellis
tensor would pass both checks and only show up as degraded output
quality, not a crash.

**We already have real production evidence it round-trips correctly on
this exact hardware** -- the NVFP4 lane (`glm-5.3-flash-nvfp4-vllm.yaml`,
a different non-EXL3 checkpoint format) has already run `--load-format
instanttensor` and passed a 249,951-token needle-in-haystack test, 4/4
codes retrieved, zero corruption. Doesn't cover EXL3's int16-packed
trellis format specifically, but is real, not hypothetical, evidence on
TP=2/GB10.

**Two concrete open risks, not resolved**:
1. MiaAI-Lab installs it `--no-deps` "to dodge an NCCL version
   conflict," despite upstream's own PR #52801 stating it only depends
   on Torch as of >=0.1.9 -- something doesn't match on their end, and
   pip's resolver isn't verifying it. Worth resolving before trial given
   this fleet already tracks an unrelated but real open NCCL issue
   (vLLM #51921).
2. **`scitix/InstantTensor#13` (open)**: loading a second, small model
   after the main one (their case: a speculative-decode draft model)
   fails with a host-staging-allocator OOM, even with the loader's own
   memory-budget knob turned down. This maps directly onto our own MTP
   drafter load pattern, on a fleet already flagged across half a dozen
   memory-pressure tracks as earlyoom-sensitive. `#19` (open) documents
   a related CUDA host-registration failure on some driver/kernel
   combos.
3. Silver lining: both known failure modes are LOUD (process
   termination / registration abort before reading), not silent
   corruption -- meaningfully de-risks the worst-case scenario, though
   it doesn't close the trellis-payload gap above.

**What the actual upside is, and isn't**: boot/weight-load time only --
upstream reports 10-32x load-time speedups on H200/H20; MiaAI-Lab
measured ~35s for their 164 GiB checkpoint. **Explicitly no decode/
prefill throughput change.** Boot time is not currently a pain point
this project has flagged anywhere -- we already tolerate up to a
1-hour boot timeout and it's never come up as an active complaint.

**Recommendation**: not a priority. The upside doesn't address any
current pain point, and the one open, concrete risk (#13, OOM on a
second small model load) lands exactly on this fleet's known weak spot
(MTP drafter load + earlyoom sensitivity) rather than somewhere neutral.
If ever revisited: a throwaway scratch image with the dependency
resolved WITHOUT `--no-deps` (to see what it actually wants and whether
that's a problem), then a real TP=2 boot watching specifically for the
MCG-marker check and for the drafter-load OOM pattern, then a logit-
level A/B (not just "did it boot") before any production consideration.

**Status**: architecture validated as safe-by-design; two concrete,
unresolved risk flags identified; recommendation is to deprioritize
given low payoff and risk concentration on an existing weak spot. No
scratch trial run this session.

## RoCE single-rail utilization measured directly (2026-09-16)

Direct follow-up to the dual-rail verification above, prompted by the
user asking how much of the single active rail we're actually using.
Grafana ships `node_exporter`'s real InfiniBand port counters
(`node_infiniband_port_data_transmitted_bytes_total` and friends) --
didn't need to guess.

**Link capacity, confirmed**: `rocep1s0f0`/`roceP2p1s0f0` (the two active
rails) both report `node_infiniband_rate_bytes_per_second` = 25,000,000,000
(200 Gb/sec NDR) -- this is the theoretical per-rail ceiling.

**Actual measured usage, right now (idle)**: ~0 -- expected, no traffic
in flight between requests.

**Actual measured usage during real load**: queried the exact window
around this session's own 108K-token cold-prefill test (the heaviest,
most network-intensive single request we could construct -- a full,
uncached TP=2 all-reduce workload). Peak observed: **~4.9 Gbit/s**
(`max_over_time` across the full 3-day production window at 30s
granularity). Against the 200 Gbit/s per-rail ceiling, that's **~2.45%
utilization at the heaviest moment we could find or construct.**
Typical (3-day average) usage sits far lower still, under 1 Gbit/s
(~0.4%).

**This revises the earlier dual-rail assessment.** The prior entry
above reasoned that dual-rail bandwidth "should give a real,
proportional win" for our prefill-heavy workload -- that was a
first-principles argument (large all-reduce messages are
bandwidth-bound in general), not yet a measurement. Now measured
directly: **we are nowhere near saturating even the single active
rail**, even at the heaviest realistic load. Doubling available
bandwidth via a second rail is very unlikely to move real throughput
under CURRENT traffic patterns -- if the link were the bottleneck,
we'd expect this fleet's own 3-day peak to already be pushing toward
the 200 Gb/s ceiling, and it isn't within two orders of magnitude.
Whatever governs actual prefill/decode speed on this fleet, it is not
network bandwidth. This is also consistent with the fleet's own
observed concurrency (per earlier mcp-grafana probes: c=0-1 almost
always, rarely c=2) -- the traffic pattern that would actually stress
the link (many large concurrent prefills at once) essentially never
occurs here.

**Revised recommendation**: dual-rail stays technically free (hardware
already cabled, zero cost to try) but should NOT be expected to move
real throughput given current traffic patterns -- deprioritize versus
the original "plausible real win" framing. Worth revisiting only if
concurrency/traffic shape changes meaningfully (e.g. many simultaneous
long-context requests becoming routine, which isn't the case today).

## Ampere-fallback GEMM deep-dive: closed on the LM head, a real new lead elsewhere (2026-09-16)

Deep research pass on the ~36%-of-decode-GPU-time `cutlass_80_wmma`
kernel, tracked since 2026-09-02. Read-only research + read-only
container inspection, no production changes.

**The original framing (an SM80/Ampere kernel shape running on
Blackwell hardware because no native kernel was ever selected) is
CLOSED, negative, and reconfirmed current.** This exact question was
already investigated 2026-09-02 (`recipes/validation-archive/v4.md`):
the single biggest contributor -- the LM head GEMM (605 MB/rank, the
largest dense weight in the model) -- is genuinely memory-bandwidth-
bound, not tensor-core-bound (M=3 and M=8 measured identical latency).
It already achieves 245.6 GB/s, which *exceeds* this exact hardware's
own measured raw bandwidth ceiling (~220-231 GB/s from independent
copy/reduction microbenchmarks on the same node). **No kernel --
Blackwell-native or otherwise -- can move bytes faster than the memory
subsystem allows, and this one already does.** Re-verified this
session: current toolchain (CUDA 13.0, cuBLAS 13.1.1.3, PyTorch
2.13.0+cu130) is byte-identical to what was tested in September, so
this isn't stale. Not worth pursuing further.

**A real, different, untested lead exists in the same kernel bucket.**
The LM head benchmarks that established "already at bandwidth ceiling"
were run OUTSIDE CUDA graph capture (an isolated repro, chosen at the
time as a profiler-correlation workaround). Production always runs
INSIDE captured CUDA graphs. Whether cuBLAS behaves differently under
graph capture specifically, for the *mid-sized* dense GEMMs that share
this kernel family (attention projections, MLP gate/up -- still a real
slice of that same GEMM bucket, separate from the LM head), was never
tested. Evidence this is plausible: `vllm-project/vllm#35467` documents
cuBLAS auto-tuning picking a suboptimal tile config for medium-batch
bf16 GEMMs on B200/SM100 (16-30% left on the table, open/unfixed,
general not GB10-specific); more concretely, `gabrielolympie/
sglang-flashnext-sm120` (a sibling SM120 project, RTX PRO 6000
Blackwell) directly measured and fixed exactly this: **cuBLAS-under-
graph-capture running mid-sized bf16 dense projections (512 KB-128 MB
weight range -- matches our o_proj/MLP-gate-up/q_b_proj, NOT the 605 MB
LM head) at only 20-75% of DRAM bandwidth**, versus ~90% with a
purpose-built Triton split-K kernel. Their own docs confirm the LM-
head-sized case is already fine under cuBLAS (~94% bandwidth) --
consistent with, not contradicting, our own LM-head finding. Their
measured end-to-end win: +3.6% decode tok/s. Caveat: their kernel's
tuning table is hardcoded for RTX PRO 6000's much-higher-bandwidth
GDDR7 memory -- the *technique* is portable, the *tuned numbers* are
not; GB10 would need its own autotuning pass, and since our dense-GEMM
numbers already sit closer to our own (lower) bandwidth ceiling than
theirs did, the realistic gain here may be smaller than their headline.
No LICENSE on that repo -- reimplement the documented technique, don't
copy code verbatim.

**Recommended next step, cheap and concrete**: a few hours of
`docker exec`-based benchmarking, same methodology as the original
September investigation, but run INSIDE an actual captured CUDA graph
this time (not eager/isolated) -- to see whether GB10 shows the same
graph-capture-specific degradation before committing to any kernel
work. If confirmed, the fix is a **Triton-level kernel port/autotune**
(comparable scope to the already-shipped `patch_gb10_router_gemm.py`,
days not weeks), not new CUDA/CUTLASS authorship from scratch.

**DeepGEMM's SM120 port**: `vllm-project/DeepGEMM` PR #4 ("Port SM120
kernels from nv_dev") merged 2026-09-14, two days before this
investigation -- but the PR's own text states no SM120 kernel in that
branch has ever actually executed; validation is compile+SASS-opcode
only, developed on SM100 hardware. Its featured ops (MQA logits,
hyperconnection pre-norm, FP8/FP4 grouped GEMM) target DeepSeek-V4-
Flash-style architectures and don't clearly match our dense-skinny-BF16
shape either. Watch-item, not adoptable now -- revisit once it has real
hardware validation.

**Upstream CUTLASS**: recent SM120/121 additions found are FP8/
blockscale/grouped-GEMM (MoE-oriented), not BF16 dense skinny-GEMM --
not relevant to this specific gap.

**Status**: original framing closed (LM head is fine, bandwidth-bound,
already optimal). New, narrower, evidence-backed lead identified
(graph-capture cuBLAS inefficiency on mid-sized dense GEMMs) with a
cheap, concrete validation step queued but not yet run this session.

## CUDA graph-capture cuBLAS benchmark: negative result, GEMM investigation fully closed (2026-09-16)

Direct test of the lead from the previous entry. Production confirmed
idle (0 running requests, 0% GPU util both nodes) before starting;
single brief `docker exec` benchmark run, no container/config changes,
no restarts.

**No graph-capture-specific cuBLAS penalty on GB10.** Tested 8 real
weight shapes (pulled from the actual model's attention/MLA/MLP/KDA
projections, TP=2-local sizes, 1.5-97.5 MB) x 3 decode-realistic batch
sizes (M=1,4,8) = 24 combinations, eager vs. `torch.cuda.graph()`-
captured-and-replayed. Mean graph-captured bandwidth was **96.8% of
eager** (range 76.7-142.2%, noise-dominated in both directions -- no
systematic regression). For the shapes that are actually memory-bound
(48-97.5 MB; smaller ones showed L2-cache-resident inflated numbers,
consistent with the original LM-head investigation's own caveat about
sub-50MB tensors), both eager (169-235 GB/s) and graph-captured
(155-220 GB/s) land in the same band as this hardware's own ~220-231
GB/s raw ceiling -- same conclusion as the already-closed LM head
finding, just confirmed across the smaller dense shapes too.

**This closes the Ampere-fallback GEMM investigation with no available
kernel-level win.** The sglang-flashnext-sm120 project's reported gap
plausibly doesn't transfer here because their hardware (RTX PRO 6000)
has far higher HBM bandwidth than GB10's unified LPDDR5x -- their
graph-capture penalty may be specific to that memory subsystem or their
own plumbing, not a general Blackwell/cuBLAS trait. GB10 simply doesn't
reproduce it. Combined with the already-closed LM-head result, the
entire dense-GEMM slice of the ~36% decode-time bucket is now
confirmed bandwidth-bound at the hardware ceiling regardless of eager
vs. graph-captured execution -- there is no native-kernel or dispatch-
level fix available.

**The only remaining lever is quantization coverage** (bf16 -> fp8 for
attention projections / LM head, halving bytes moved) -- which is not
new: this is the same dense-FP8 tradeoff already investigated and
deliberately rejected (`v16_densefp8_prepped` / "Dense-FP8 promoted to
default" history above) specifically because it favors decode at
prefill's expense, the wrong tradeoff for this prefill-heavy fleet.
Nothing new to promote from this line of investigation.

**Status**: CLOSED. Both the LM head and the smaller dense GEMMs are
confirmed bandwidth-bound at the hardware ceiling in both eager and
graph-captured execution -- no kernel-dispatch bug exists to fix. The
only lever (dense-FP8) is already known and already rejected for this
fleet's workload shape. Not worth further investigation unless the
fleet's prefill/decode balance changes materially.

## Backport deployment incident, 2026-09-16: a real bug shipped, a real ambient crash hit on rollback, memory-preflight finally genuinely validated

User granted standing permission for `sparkrun stop`/`run` and
`prelaunch_flush.sh` this session specifically so the queued backport
deployment and the long-blocked memory-preflight test could finally
happen. Both did -- with two real incidents along the way, both now
understood and one fixed.

**Attempt 1: the 6 backports, real bug found.** Stop -> `prelaunch_
flush.sh` (both hosts, clean pass) -> `sparkrun run` with the newly
backported image. The container came up but the actual vLLM server
process crashed on launch: `PermissionError: [Errno 13] Permission
denied: '/usr/local/lib/python3.12/dist-packages/vllm/v1/request.py'`.
Root cause, found by reading `overlay/patch_apc_no_store.py`'s
`atomic_write()` directly: `tempfile.mkstemp()` creates its temp file
mode `0600` (owner-only) regardless of the target file's real
permissions; `os.replace()` is a rename, not a content copy, so the
target inherited that restrictive mode. The Docker build applies this
patch as root, so the bug was invisible there -- the container runs the
server as a non-root user (`luser`), which lost read access to a core
vLLM source file and crashed the whole engine at first launch. **This
project's own `patch_spinwait.py` already has the fix for this exact
class of bug** (`os.chmod(temp, stat.S_IMODE(target.stat().st_mode))`
before the replace) -- the new patch just didn't carry it over. Fixed
in `overlay/patch_apc_no_store.py` this session (added `import stat`,
preserve `path.stat().st_mode` before `os.chmod(tmp, orig_mode)` prior
to `os.replace`). The other two new patches (`patch_cache_reset.py`,
`patch_kv_capacity_log.py`) use `Path.write_text()` directly (in-place
overwrite, not a temp-file+replace) and do NOT share this bug --
checked directly, confirmed.
**Operational lesson**: this project's local Docker build validation
runs entirely as root (build stage + the self-check `RUN` step), so it
cannot catch a non-root runtime permission regression like this one --
the build passed all 14 self-checks and still shipped a launch-blocking
bug. **Any future patch touching an installed system file should be
smoke-tested as the actual runtime user** (check the Dockerfile/
container for `USER`, currently a non-root `luser`), not just validated
at build time as root. Not yet added as an automated check -- worth
doing before the next patch round.
**Immediate recovery**: retagged the previous known-good image
(`sha256:4ae0df077e70...`, the exact image that had run stably for 3
days pre-incident) back onto the `:local` tag and redeployed --
straightforward since Docker doesn't delete an image just because its
tag moves.

**Attempt 2 (the rollback): a real, ambient NVRM allocation failure,
unrelated to the backport bug.** Stop -> fresh `prelaunch_flush.sh`
pass (clean on both hosts again) -> `sparkrun run` with the rolled-back
image. This time the server booted completely successfully -- weights
loaded, CUDA graphs captured, engine initialized, API server started,
even answered a real request (`GET /metrics 200` from the Prometheus
scraper at `10.7.0.10`) -- then crashed **12 seconds later**:
`Worker proc VllmWorker-0 died unexpectedly (exit code: None)`, the
same uninformative-exit-code signature this project has chased
before. Cross-referenced the head host's kernel log directly
(`dmesg -T`, timezone-adjusted): `NVRM: nvCheckOkFailedNoLog: Check
failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from
_memdescAllocInternal(pMemDesc)`, timestamped within ~2 minutes of the
crash. This is the exact NVRM allocation-failure signature this
project's `host_fragmentation_xid31_reboot_risk` track already
documents -- a host can show healthy `MemAvailable`/fragmentation
numbers at one instant (both preflight checks passed cleanly
immediately before this boot) and still hit a large-contiguous-
allocation failure moments later once new activity (first live
request, first fresh CUDA-graph-adjacent allocation) demands one. This
is an ambient, previously-known, still-not-fully-resolved risk class --
**not** caused by the backport bug (this was the pre-backport image),
not new. Confirmed `journalctl -u earlyoom` showed nothing (earlyoom
itself didn't fire this time -- a raw NVRM allocation failure inside
the CUDA driver, upstream of anything earlyoom watches).
**Recovery**: stopped, re-ran `prelaunch_flush.sh` (clean again), retried
the identical `sparkrun run`. Succeeded cleanly this time -- healthy in
~9 minutes, verified with a real completion request, no repeat NVRM
error. Consistent with this failure class being probabilistic/timing-
dependent, not deterministic -- a bare retry has a real chance of
working, as it did here.

**Memory-preflight check: finally genuinely validated, twice, on real
hardware.** This was the entire point of today's `sparkrun stop`/`run`
permission grant -- previously blocked by tooling permissions across two
prior sessions. Ran for real, both attempts, both hosts, both times
clean: `memory preflight: <N> KiB available >= <threshold> KiB needed
(gmu=0.84 + 2 GiB headroom), OK`, alongside the neighboring
fragmentation check. The mechanism works exactly as designed. Closes
the long-open `check_memory_available()` validation item.

**Status**: production restored and confirmed healthy on the known-good
image. The backport bug is fixed in the working tree but **the fix has
not yet been rebuilt or redeployed** -- next attempt should include a
non-root smoke-test step before ever touching production again. The
NVRM ambient-fragmentation risk remains open and unresolved (as it has
been for weeks) -- this incident is a fresh, well-documented data point
for that existing track, not a new investigation.

## Backport rebuild + non-root validation (2026-09-16, post-incident)

Rebuilt `glm53-exl3-v20-upstreamsync:local` with the `patch_apc_no_
store.py` permission fix from the incident above. Build succeeded,
all 14 existing root-context self-checks passed (unchanged from
before -- these never would have caught the bug, that's the whole
point of what follows). New image: `sha256:a78a2d958fbc247b...`.

**New: a real non-root validation, closing the gap the incident
exposed.** Added `tests/check_nonroot_permissions.sh` -- runs the built
image with `docker run --user 1000:1000` (the actual runtime UID:GID
this fleet's containers use, confirmed via `docker inspect <image>
--format '{{.Config.User}}'` returning empty -- it's supplied by
`docker run --user` at launch time, not baked into the image) and
verifies every file any `overlay/patch_*.py` in this build tree is
known to touch is still readable by that user, plus does a full Python
`import` of the specific modules the just-fixed patch touches. Ran it
against the rebuilt image: **all 12 tracked files pass, mode 0644 on
every one** -- including `vllm/v1/request.py`, the exact file that
broke production. A full `import vllm.v1.request`, `vllm.sampling_
params`, `vllm.v1.core.block_pool`, and the two `completion/*` modules
also succeeded cleanly as the non-root user, and confirmed the actual
new feature (`SamplingParams.skip_writing_prefix_cache`) is present and
attached.

**Not baked into the Dockerfile itself** -- deliberately kept as a
separate, explicit post-build script rather than a `USER`-directive
switch inside the build, since flipping the image's default user
mid-build risks unintended side effects on later build steps and isn't
how this project's containers actually get their runtime UID today.
Run `tests/check_nonroot_permissions.sh [image_tag]` after any build
that touches an `overlay/patch_*.py`, before ever deploying -- add new
files to its tracked list whenever a new patch writes one.

**Status**: rebuild validated at both the root (build-time) and
non-root (runtime-identity) level for the first time in this project's
history. **Not yet redeployed to production** -- this was a build+
validate pass only, matching what was asked; deployment is a separate,
deliberate next step.

## Backports deployed to production, confirmed stable (2026-09-16)

Deployed the fixed, rebuilt, non-root-validated image
(`sha256:a78a2d958fbc...`) via the now-standard cycle: stop ->
`prelaunch_flush.sh` (both hosts, clean) -> `sparkrun run`. Healthy in
~9.5 minutes. Verified: a real completion request returned correct
output; checked again 15s later (the exact window the prior rollback
crashed in) -- still healthy, no NVRM signature in the head's kernel
log. Confirmed the new features are genuinely active, not just present:
the `[glm53-kv-capacity-log]` per-group breakdown (PR #94 backport)
appears in the real boot log with real numbers (5 KV-cache groups,
usable block accounting, the "394/87 = 4.53x" cross-check line matching
its own design).

**Production now runs all 6 of today's backports**: `/reset_prefix_
cache` admin endpoint (off by default), the pipefail-safe health-check
+ cluster-lock fix, `GLM53_EXTRA_ENV` diagnostic passthrough, KV-
capacity boot logging (on by default, log-only), per-request prefix-
cache skip (`GLM53_APC_NO_STORE`, inert unless a caller opts in), and
the long-prefill warmup-ladder fix -- on top of the same known-good
v20-upstreamsync base that's been running for days.

**Status**: DEPLOYED, stable, confirmed. This closes out the entire
2026-09-16 backport-deployment thread: bug found, fixed, rebuilt,
validated (root AND non-root), deployed, confirmed. The
`tests/check_nonroot_permissions.sh` gate added during this incident
should be run before any future patch-touching rebuild, going forward.

## PR #215 backport: tool_choice:"none" decode-time fix (2026-09-18)

One item from the 2026-09-17 upstream-check round against MiaAI-Lab/
GLM-5.3-Flash-EXL3-2x-DGX-Sparks: PR #215, field-validated on their own
2x-Spark deployment before this backport. New
`overlay/patch_tool_choice_none.py` overrides `Glm47MoeParser.adjust_request`
(a hook it didn't previously implement) to append the tool-call opener
token to `request.bad_words` whenever `tool_choice=="none"` and `tools` is
set -- the sampler then masks that opener at every decode position
(including spec-decode draft positions) instead of letting the model open
a `<tool_call>` block that the API layer then strips with no answer text
behind it. Correctness fix, no kill switch, inert for every other
`tool_choice` value. Live only because `--reasoning-parser glm45` is set
in every recipe here (confirmed).

Verified during implementation, not assumed: `ChatCompletionRequest.
bad_words` defaults to `Field(default_factory=list)` (never `None`, so
`.append()` is safe on every request); `Glm47MoeParser`'s only base class
is `ParserEngine` (`vllm/parser/engine/parser_engine.py`), whose own
`adjust_request` sets `request.skip_special_tokens = False` -- a real
effect, not a no-op -- and the patch correctly chains through it via
`super()` before appending. (The patch's own docstring originally
mis-cited this as living in `abstract_parser.py`; corrected after tracing
the actual MRO.)

Rebuilt clean: 83/83 Docker build steps, all 15 self-checks including the
new `test_tool_choice_none.py` (apply/idempotent/fail-closed, same
convention as every sibling patch test). Non-root permission gate
(`check_nonroot_permissions.sh`) passes on all 13 tracked files including
the newly-added `vllm/parser/glm47_moe.py`. Image:
`sha256:a85b2e8844cda7cd1105b1f15fac14332e3903c23d1c4cd6beaf56b816de074f`.

**Deployed and validated against production traffic pause (2026-09-18)**:
stop -> `prelaunch_flush.sh --during-load` (both hosts, clean -- first use
of the during-load continuous cache-flusher this fleet actually owns,
per the 2026-09-17 upstream-check's tonyd2wild field-report cross-check)
-> `sparkrun run`. Healthy in ~9 minutes (7 min to container-up, health
green 3 checks later). Real-request test matching the exact bug scenario
(`tools` declared, `tool_choice:"none"`, "what is 12*7, don't use a
tool") returned `"content": "84"` -- a real answer, not an empty
tool-call-only response. `probes/probe_sanity.py` full pass: no
`<think>` leakage, TTFT 0.18-0.26s, decode 31.74/33.33/32.31 tok/s --
inside the documented 31.54-36.76 tok/s baseline (see `SPEED.md`), no
regression.

**Status**: DEPLOYED, validated, stable. Working tree not yet committed.

## PR #217 backport: opt-in SM121 EXL3 thin-decode kernel (2026-09-18)

Second item from the 2026-09-17/18 upstream-check rounds against MiaAI-Lab.
PR #217 is the maintainer-merged "thin-only" split of PR #182 -- the KDA
FP8 large-M dispatch half was deliberately excluded upstream after its own
numerical study found reproducible shifts at the geometry it engages; none
of that excluded half is ported here. New
`overlay/patch_exl3_decode_pipeline.py` (native CUDA/C++ source patch,
ported against the exact pinned exllamav3 commit
`c5d9c657966ffeeaa9353f0cc899f18629da4a13` this fork already builds -- no
drift to reconcile) adds a K4/N256 thin-decode kernel (1 fragment / 8
shared-memory stages vs. stock 3/3) that skips the redundant gate/up input
Hadamard when the gate/up SUH pointer tables are the identical allocation
(proven per-expert via `torch.equal` at load, only in fast mode). Host
dispatch requires K==4, hidden/intermediate both %256==0,
`GLM53_EXL3_MOE_FAST=1`, and a live SM121 check via
`cudaDeviceGetAttribute` -- not just the env var. `overlay/exl3.py` gets
the gating helper (`exl3_moe_fast_requested()`, strict "0"/"1" validation)
and fails closed (raises, not silent fallback) if FAST=1 is requested on an
image without the native symbol or version match.

**PR #217's own numerical-parity verdict is INCONCLUSIVE, not a pass** --
its predeclared stock-vs-stock control gate failed (an instrument-
calibration property of their test material per their own write-up, not
evidence against the kernel itself). Their descriptive serving A/B on
current-main measured +7.71% to +14.59% median decode tok/s in the
range-resolved cells (two cells withheld by their own overlap rule) --
real and encouraging, but not re-run against this fork's checkpoint (288
experts) or TP=2 topology. **Shipped opt-in, default OFF
(`GLM53_EXL3_MOE_FAST=0`), available-but-off, not benchmarked-and-
recommended** -- do not flip this on in production without a dedicated
GPU trial first.

Rebuilt clean: full Docker build, all self-checks pass including the new
`test_exl3_decode_pipeline.py` (10 CPU checks). Confirmed at the deepest
level available without a GPU: `exllamav3_ext`'s compiled SM121a cubin
list includes `glm53_exl3_moe_fast.sm_121a.cubin` -- the native kernel
genuinely compiled for our target, not just Python-side wiring. Checked
for collision against this fork's other GB10-specific routing
(`_disable_gb10_persistent_topk`, `GLM53_DENSE_FP8`) -- none, different
code paths. Image: `sha256:d99bfefa5bf699eb2e68cf5710409567ec8bc787da1901331d7d0f87038670e1`.

**Deployed and validated (2026-09-18, same production-traffic-pause
window as PR #215)**: stop -> `prelaunch_flush.sh --during-load` (both
hosts, clean) -> `sparkrun run`. Healthy in ~12 minutes. Confirmed
`GLM53_EXL3_MOE_FAST` unset in the running container (defaults to off, as
designed) -- this deploy is expected to be behaviorally inert, purely
validating the available-but-off packaging, not the new kernel. Real
completion request clean. `probes/probe_sanity.py` full pass: no
`<think>` leakage, TTFT 0.25s flat, decode 31.83/32.82/34.94 tok/s --
inside the documented 31.54-36.76 tok/s baseline, no regression, as
expected for an inert flag.

**Status**: DEPLOYED (flag off), validated, stable. **Open decision, not
made here**: whether to run a dedicated `GLM53_EXL3_MOE_FAST=1` GPU trial
(coherence + decode-speed A/B, same method as the dense-FP8/PR #215
validation) at this fork's actual scale. Working tree not yet committed.

## Entrpi/glm-5.3-flash-exl3-2x-spark initial triage (2026-09-18)

Added to the tracked-repo rotation at the user's request; first pass,
read-only (their repo cloned to scratch, nothing installed or run against
production). Same target model, same brandonmusic EXL3 4bpw checkpoint,
same incoai DFlash2 k=7 drafter family, same 2x DGX Spark GB10 class of
hardware as this fork -- and, notably, the same upstream (MiaAI-Lab) this
fork is itself built from, which they benchmark head-to-head in their own
`docs/COMPARISON.md`. Exceptionally well-documented: 19 dated sections in
`docs/FINDINGS.md` including negative results, plus `docs/BUILD.md` giving
exact image provenance down to source commit hashes.

**Architecturally a different animal, not a sibling of this fork.** They
maintain `Entrpi/vllm-glm-5.3-flash-spark`, a full vLLM fork rebuilt from
source each release, vs. this fork's (and MiaAI's) patch-overlay on a
pinned public image. Meaningfully higher engineering investment, buys them
things a patch-overlay architecturally cannot: a custom attention backend
(`B12X_MLA_SPARSE`, their own GLM_NEXT-specific MLA contract), a
transplanted non-native drafter (DFlash2) with bespoke KV handling, and a
from-scratch fused-MoE loader. Nothing here is a drop-in backport
candidate -- it's a different foundation, not a patch on the same one.

**The b12x mystery, resolved.** Their README credits their fused expert
kernels to `tpurtell/sparkinfer-glmrt`, "the b12x GPU runtime" -- alarming
on first read, since it sounds like the exact `local-inference-lab/b12x`
this fork spent 2026-09-18 validating and then ruling out (see the
tracked-repos entry above). `docs/BUILD.md`'s source-commit table clears
it up: their b12x is `Entrpi/sparkinfer-glmrt` (fork of
`tpurtell/sparkinfer-glmrt`, itself fork of `lukealonso/b12x`) -- a
**different lineage** from `local-inference-lab/b12x`, hacked specifically
for GLM-5.3 serving (`ModelType.GLM_NEXT`, a first-class NoPE-MLA
contract) rather than the generic pip-installable kernel library this fork
tested. More importantly, `docs/FINDINGS.md` §1 describes how they get
fused-trellis MoE serving from the *standard* checkpoint: "A load-time
TP-slicing port (`standard_fused_moe`) builds the b12x fused runtime
directly from the standard checkpoint -- no offline conversion." That's a
custom loader integrated into their vLLM fork's weight-loading pipeline,
converting each shard's layout incrementally *as it streams in* -- not a
post-load black-box adapter calling a self-contained `prepare_*()`
function on an already-fully-materialized standard-layout state. That
distinction is exactly why they never hit this fork's 47.25 GiB/rank
contiguity-copy wall: there's no point where a full stacked gate+up
tensor exists in both the original and target layout simultaneously.
**This confirms our finding was correct for the approach we tried, and
reveals a real but far more invasive alternative** (fork vLLM's own
loader, don't bolt a library on afterward) -- not in scope unless this
fork is ever willing to take on that level of engineering.

**Default KV dtype is the opposite of ours.** Their lane defaults to bf16
attention memory, with fp8 (`KV_DTYPE=fp8_e4m3`, +48% pool) and NVFP4
(`KV_DTYPE=nvfp4_ds_mla`, +28.6% pool over their fp8 baseline, 304 B/token)
as opt-in options with measured quality deltas at each step. This fork
(and MiaAI's) mandates fp8_ds_mla -- our `FLASHINFER_MLA_SPARSE_SM120`
backend has no bf16 sparse kernel at all. Their bf16-default posture is a
consequence of running a different attention backend family
(`B12X_MLA_SPARSE`) with more format flexibility than ours has, not
evidence our fp8-mandatory choice is wrong for our stack.

**Speculative decode: DFlash2-primary vs. our MTP-2.** Their production
default is DFlash2 k=7 (a transplanted, non-native drafter with a custom
"spec-state ring" to avoid pool-tax problems, measured +21% single-stream
over their own MTP-4 baseline). This fork ships `overlay/dflash2_speculator.py`
already but runs native MTP with `num_speculative_tokens=2` in production
-- weaker than even their MTP-4 comparison point. Worth a dedicated look
independent of this triage: is there headroom in bumping MTP's spec-token
count, or in actually exercising the DFlash2 path this fork already
carries? Not actioned here -- flagging as a real, separate backport
candidate.

**Two directly actionable items:**

1. **`max_num_batched_tokens` safety, checked and confirmed fine.** Their
   §10 notes "MNBT 8192 survives 133k prefill on this lane. The MiaAI
   recipe's 1024-chunk cap is a FlashInfer-SM120-lane constraint" --
   worrying on first read since we run `FLASHINFER_MLA_SPARSE_SM120` at
   7168, well above their cited MiaAI figure of 1024. Checked against our
   own boot logs: this fork's indexer-workspace rightsizing (`[glm53-
   indexer-workspace] rightsize: ... max_num_batched_tokens=7168 ->
   1048592 entries`) has fired cleanly across dozens of boots this session
   with zero oversubscription errors -- we already carry the same MiaAI-Lab
   #86 right-sizing fix Entrpi also credits and adopted. 7168 sits safely
   under whatever ceiling their 8192 figure was warning about on the same
   backend family; nothing to change, but don't raise past 7168 without
   testing (never validated on this fork).
2. **`NCCL_MIN/MAX_NCHANNELS=8`** -- a measured, mechanism-explained win on
   their 200GbE RoCE link (default channel selection 3.38ms vs. pinned-8
   2.96ms for a 37.7MB all-reduce; decode 6.2->5.5ms/step in-serve). Cheap,
   env-only, plausibly transfers to our identical link topology. Not
   tested against this fork -- a real, low-effort candidate for the next
   backport round.

**One finding relevant to this fleet's own memory instability, not yet
acted on:** their §5 memory methodology notes "floors age ~1.5-2 GiB
per day of workload. Fresh-boot floors flatter." If real and transferable,
this would help explain why node_0 sat at a chronic 4.3-4.9% mem-avail
margin for hours before actually crashing today (2026-09-18, see the
gmu-0.83 conversation) with no single acute trigger -- gradual floor
erosion under sustained serving, not a step-change. Also notable: their
§5 "head/worker swap is impossible... the box holding weights on local
NVMe dies at ~92% of shard load when it also runs the head -- local read
outruns page-cache reclaim" independently reproduces the exact mechanism
category behind this fork's own `prelaunch_flush.sh`/`--during-load`
cache-flushing discipline. Neither finding is actioned here; both are
worth a dedicated look if the earlyoom pattern recurs.

**Not investigated this pass**: their `tools/memlog.sh` continuous
memory-floor sampler (a real methodology upgrade over this fork's
single-point preflight check -- could validate gmu changes like today's
0.82->0.83 against actual under-load floors instead of boot-time
snapshots only); their mixed-prefill decode-starvation fix
(`MIXED_PREFILL_CAP`/`MIXED_PREFILL_DECODE_WEIGHT`) looks structurally
identical to this fork's own `GLM53_MIXED_PREFILL_CHUNK` mechanism --
likely inherited from the same MiaAI-Lab source, not a new idea to port.

## NCCL_MIN/MAX_NCHANNELS=8 A/B: tested, rejected (2026-09-18)

Followed up on the Entrpi triage's NCCL channel-pinning candidate with a
real A/B on this fleet, using `recipes/probes/probe_throughput_ab.py`
(n=20 runs/arm, identical fixed prompt, `max_tokens=400`, temperature 0 --
the house methodology built specifically to resolve small effects against
this cluster's own run-to-run noise).

**A real bug found along the way**: `NCCL_MIN_NCHANNELS`, set in the
recipe's environment block identically to its `NCCL_MAX_NCHANNELS`
sibling, silently did not survive into the `Worker_TP` process. Confirmed
via `/proc/<pid>/environ` on a freshly spawned worker on both ranks:
every other NCCL_* var from the same container environment (`NCCL_MAX_
NCHANNELS`, `NCCL_CUMEM_ENABLE`, `NCCL_CROSS_NIC`, the `NCCL_IB_*` family,
`TORCH_NCCL_ASYNC_ERROR_HANDLING`) survived intact; `NCCL_MIN_NCHANNELS`
(and, separately, `NCCL_DEBUG`) did not. Same unexplained env-stripping
class as `_fix_glm53_mixed_prefill_env`'s already-documented mystery
(previously only seen on `GLM53_*`-prefixed vars), now confirmed to hit a
bare `NCCL_*` name too. Checked vLLM source directly: the only code
touching `NCCL_MIN_NCHANNELS` at all is `batch_invariant.py`'s
`override_envs_for_invariance()`, gated behind `envs.VLLM_BATCH_INVARIANT`
-- confirmed not set here, so not the cause. Root cause not chased further
(the GLM53_* case already burned effort on this without resolution).
Worked around with the identical fix shape (capture-on-first-execution,
set `os.environ` directly in every re-executing process before NCCL's own
native `getenv()` call) to get a clean, actually-pinned-to-8 test at all;
without it, the "B arm" would have silently been "capped at 8, floor
unset" rather than "pinned to exactly 8," a different and untested
condition.

**Result**: baseline (no pinning) n=20, median 31.767 tok/s (stdev
0.697); with both channels pinned to 8, median 31.07 tok/s (stdev
0.819) -- a ~2.3% *decrease*, not the improvement Entrpi measured on
their own hardware (whose own data, separately, says the channel default
alone should be decode-neutral end-to-end -- see the Entrpi triage above).
The per-arm spread is tight enough that this isn't just overlapping
noise, though a single-boot A/B can't fully rule out an unrelated
boot-to-boot confound (this fleet had a real host-memory incident earlier
today; some residual variability between arbitrary boots is possible).

**Decision: rejected, not shipped.** Landing this permanently would mean
carrying a second env-stripping workaround (this fix's own file-state
mechanism, alongside the mixed-prefill-env one) for a change that didn't
demonstrate a benefit and, on this one measurement, trended slightly
negative. Reverted both the env vars and the workaround function from the
recipe. See `nccl_channel_ab_2026_09_18.md` for the durable memory record
of the env-stripping finding, kept for its own sake independent of this
specific rejection.

## b12x fused-trellis MoE via native `[2,E,...]` weight layout: validated, `v21-b12xmoe` (2026-09-18)

Supersedes the "infeasible" verdict from earlier today (see the
tracked-repos entry above and `b12x_moe_adapter_infeasible_2026_09_18`
memory) -- that verdict was correct for the approach tested (a post-load
black-box adapter calling `b12x`'s generic `prepare_trellis256_moe_weights()`
on an already-loaded standard-layout tensor, forcing a `.movedim(1,0)
.contiguous()` copy costing 47.25 GiB/rank). The idea itself wasn't dead;
the integration strategy was. Following the Entrpi triage's insight (their
vLLM fork avoids the same wall via a load-time converter, not a post-load
adapter), this fork found and validated a much lighter third path: change
`w13_trellis`/`w13_suh`/`w13_svh`/`w13_mcg`'s axis order at allocation time
from expert-major `[E,2,...]` to projection-major `[2,E,...]` -- exactly
what b12x's `w13_layout="trellis_t256_proj"` wants -- so this fork's own
existing incremental per-expert `weight_loader` (`_load_exl3`, unchanged
in its calling convention) naturally produces b12x's desired contiguous
layout as a side effect of normal checkpoint loading, with no separate
conversion step ever needed.

Built as a new sibling recipe, `recipes/build/glm53-exl3-v21-b12xmoe/`
(copy of `v20-upstreamsync`, own `overlay/exl3.py` and test file; `v20`
confirmed untouched throughout via `git diff`). Two Opus-planned,
advisor-reviewed phases of implementation (Phases 0/A/B/1, then 2-6),
each executed by a Sonnet fork and reported back before the next began.

**Phase 0/A/B/1 (memory-residency gate -- the gate that killed the
earlier approach)**: full audit of all ~24 `w13_*`/`w2_*` references in
`exl3.py`, categorized by axis-sensitivity, all sensitive sites fixed
consistently (Parameter allocation shape, `_load_exl3`'s destination
indexing, the per-expert view construction in
`process_weights_after_loading`, a `.reshape` load-time initializer, an
`E`-count read). A genuinely silent test-file bug was found along the
way (`tests/test_exl3_overlay.py`'s dummy-init helper copied the wrong
expert's data under the new layout -- shape-valid, semantically wrong,
would have silently downgraded the production fused-kernel tier from
"grouped" while the whole suite stayed green) -- proven to actually trip
a guard assertion (red on the broken form, green on the fix), not just
patched and assumed caught. A narrower broadcast bug at exactly `E=2`
was empirically demonstrated, not just reasoned about. **Measured on
real layer-10 checkpoint weights, TP=2 rank-0**: `data_ptr()` equality
between b12x's prepared weight and this fork's own loaded tensor holds
(zero-copy confirmed, not assumed) -- total new residency **~71 MiB/rank**
(vs the 47.25 GiB/rank that killed the earlier approach -- a 665x
reduction), matching the plan's predicted number to within 0.1 MiB.

**Phase 2 (b12x liveness)**: resolved an open question from the earlier
investigation -- a synthetic smoke test's NaN/Inf output was hypothesized
to be an artifact of feeding raw, unrotated input under b12x's
`full_rotation=False` mode. Confirmed: under `full_rotation=True` (fp16
prepared weights, fp32 output, real per-expert rotation tables), the
same kind of input produces finite output. **New finding, not previously
known**: `full_rotation=True` explicitly rejects a SwiGLU activation
clamp ("intermediate_rotation requires unclamped gated silu or situ").
This fork's own `apply_exl3_python_loop`/`apply_exl3_fused_moe` both
clamp gate/up to +/-10.0 before SiLU as a stability guard; the b12x path
cannot replicate that clamp. **This is a real, uncorrected semantic gap**
for out-of-range activations, not something routed around -- the parity
ladder below used realistic-magnitude test data and would not
necessarily exercise this gap. Worth resolving before any real shipping
decision.

**Phase 3 (permutation sweep on `intermediate_rotations`'s three
blocks)**: position 0 (`gate_svh`) is strongly discriminating (moving it
collapses cosine similarity to 0.64-0.79); positions 1/2 (`up_svh`/
`down_suh`) were **not** independently discriminating at the specified
tolerance -- 2 of 6 orderings passed, not the expected 1. Investigated
rather than accepted at face value: confirmed via bit-non-identical
outputs and an adversarial re-test (artificially scaling `down_suh` 60x)
that this is a genuine low-sensitivity property of the kernel for this
checkpoint's data, not a test-discrimination failure. Adopted the
source-documented order (`gate_svh, up_svh, down_suh`) since it's
authoritative and empirically at least as good as any alternative --
documented as a real nuance rather than claimed as a clean proof.

**Phase 4 (parity ladder vs. `apply_exl3_python_loop`)**: 30/30 cases
pass (3 layers x 7 token counts + 3 adversarial routing cases x 3
layers), cosine similarity 1.000000 to displayed precision, maxabs diffs
0.0005-0.0018, no tolerance-loosening needed anywhere (the anti-goal was
respected). A 3-way check (loop reference / real `grouped`-tier fused
path / b12x) initially looked concerning (4/6 cases showing b12x
disagreeing with the fused path more than either disagreed with the
loop reference) -- root-caused to the test harness not exercising the
real tier-resolution logic (a fake layer stuck at `tier=legacy`), not a
real discrepancy; fixed and re-confirmed the disagreement sits at the
1e-7-to-1e-3 noise floor (5-6 nines of cosine similarity).

**Phase 5 (dispatch wiring)**: added `GLM53_EXL3_B12X_MOE` (opt-in,
default off), following this fork's established fail-closed pattern
(`GLM53_EXL3_MOE_FAST`/`exl3_moe_fast_requested()`): missing b12x, a
b12x version/signature mismatch (b12x's `prepare_trellis256_moe_weights`
is a private API, pinned at `1.3.0` with a load-time signature
assertion), or a failed `data_ptr()` zero-copy invariant all `raise` at
load time rather than silently falling through. New tests confirm
default-off byte-identical behavior and fail-closed behavior on bad
config. `v21`'s existing test suite re-run clean.

**Phase 6 (throughput)**: the isolated single-GPU test harness measured
the real `grouped`-tier fused path itself running ~2x faster than the
documented TP=2 production numbers (6.48ms->3.3ms at 16 tokens,
17.41ms->9.2ms at 256 tokens) -- flagged explicitly as an apples-to-oranges
artifact (no cross-host NCCL in this harness), not a real production
speedup. The fair, same-harness comparison is b12x vs. re-measured-grouped:
**0.98x at 16 tokens (parity), 0.93x at 256 tokens (b12x ~7% faster)**.
b12x's JIT warmup cost is real (~30ms/call during a 10-call warmup at 16
tokens, dropping to 3.2ms/call warm) and must be budgeted for (a
pre-warm step, matching this fork's existing JIT-warmup patterns for
other kernels) before any CUDA-graph-captured production path could use
it -- not a free swap-in.

**Status: validated at the correctness/memory/throughput level in an
isolated harness. NOT deployed, NOT built into any image.** Two real
open items before a shipping decision, not blocking this validation
pass but real: (1) the SwiGLU-clamp incompatibility under
`full_rotation=True` -- needs either a resolution or an explicit,
understood scope limitation; (2) the permutation-sweep's non-clean
discrimination on two of three block positions, which weakens (without
invalidating) confidence that the adopted order is uniquely correct
rather than merely adequate. Per the original plan's scope, Dockerfile/
production-image changes and any subjective generation-quality
assessment remain explicitly deferred pending a separate go-ahead.

## SwiGLU-clamp real-traffic measurement: real but rare, resolved (2026-09-19)

Chased down the one open correctness item from the previous entry: does
this checkpoint's real activity ever actually approach the +/-10.0 SwiGLU
clamp that b12x's `full_rotation=True` mode can't replicate? Built
`glm53-exl3-v21-b12xmoe:local` (image, not just source) with a shadow
gate/up-recompute diagnostic (`GLM53_EXL3_CLAMP_DIAG`, default off,
zero-overhead when disabled) and a new candidate recipe,
`recipes/glm-5.3-flash-exl3-v21-b12xmoe-vllm.yaml` (b12x itself OFF for
this measurement -- only the diagnostic, on top of the already-shipping,
already-validated MoE path).

**Three real bugs found and fixed getting to a trustworthy number, all
worth recording:**

1. **The diagnostic's first design crashed both ranks at boot.**
   `_exl3_clamp_diag_sample`'s `torch.unique(ids)` call requires a host
   sync, which CUDA forbids while a stream is capturing --
   `breakable_cudagraph.py`'s warmup/capture phase runs this exact code
   path with synthetic data before any real request exists. Fixed with an
   explicit `torch.cuda.is_current_stream_capturing()` guard (also
   correctly excludes capture-phase synthetic data from the sample by
   construction, not just avoiding the crash). Cost one down/restore
   cycle on production while diagnosing.
2. **The fixed diagnostic still measured nothing real.** Deployed clean,
   served real requests correctly, but the counter froze the instant boot
   warmup finished and never advanced across 12 subsequent real requests.
   Root cause: production's decode path uses a *captured* CUDA graph for
   supported batch shapes -- graph replay re-executes the recorded GPU op
   sequence directly, bypassing the Python-level `apply_exl3_experts`
   dispatch (and this diagnostic hook inside it) entirely. Every sample
   collected in the first deploy was boot-time synthetic warmup data, not
   real content -- confirmed empirically, not assumed. Fixed by adding
   `--enforce-eager` to the candidate yaml for this measurement pass only
   (disables graph capture/replay, forces every forward through Python).
3. **The `--enforce-eager` edit itself broke multi-node launch.** Placing
   an explanatory comment block after the new flag, without a trailing
   backslash, silently terminated the shell command's line-continuation
   chain -- sparkrun's own appended `--nnodes/--node-rank/--master-addr`
   flags then ran as a separate, disconnected command instead of reaching
   `vllm serve`, producing a confusing single-host rank-count
   `AssertionError` with no connection to the actual cause. Fixed by
   keeping the new flag as a clean last line (matching the original
   file's structure exactly) and moving all rationale out of the command
   block. Cost a second down/restore cycle.

**The measurement** (8 diverse real requests under `--enforce-eager`:
prose, code, math, JSON, an extreme-number-computation prompt, a
150x-repetition prompt, a two-equation word problem, and a deliberately
extreme creative-writing prompt -- ~3,150 completion tokens total, ~8.03
billion gate/up activation values sampled per rank):

| | Node 0 (head) | Node 1 (worker) |
|---|---:|---:|
| max \|gate\| | 23.03 | 17.73 |
| max \|up\| | 28.18 | 20.71 |
| gate >10.0 | 1,161 (~1 in 3.46M) | 273 (~1 in 14.7M) |
| gate >12.0 | 327 (~1 in 12.3M) | 31 |
| up >10.0 | 834 (~1 in 4.82M) | 672 (~1 in 6.0M) |
| up >12.0 | 144 (~1 in 27.9M) | 207 |

**Verdict: real, not theoretical -- but rare, not dominant.** The clamp is
not dead code for this checkpoint: real content (including some
unremarkable prompts, not only the deliberately adversarial ones --
measurement doesn't have per-request attribution to separate the two)
produces activations that exceed +/-10.0, sometimes by a wide margin (up
to ~2.8x the threshold). But the per-value rate is genuinely low, on the
order of 1 in a few million to 1 in ten-plus million individual
activations -- within this modest 8-request test battery alone, the event
happened hundreds of times in absolute terms, which is a rate that scales
with real request volume rather than a one-in-a-trillion curiosity, but
is nowhere near common enough to be a dominant driver of output quality
across normal usage.

**Not resolved by this measurement**: whether "ordinary" traffic alone
(excluding the three deliberately-stressing prompts in this battery)
triggers this at a meaningfully different rate -- would need a larger,
purely-typical-traffic run with per-request attribution to separate
cleanly. Also not attempted: a mitigation (e.g. detecting and
special-casing the rare tokens that would exceed the clamp). Given the
measured rate and magnitude, this fork's judgment is that the finding is
real and worth carrying as a documented, accepted limitation of the b12x
path, not a blocker to promotion -- but it is the user's call, not a
default engineering decision, given this directly affects production
output correctness on some rare fraction of real requests.

Production restored and validated stable after both diagnosis cycles;
`v21-b12xmoe`'s candidate yaml reverted to its non-diagnostic state
(`--enforce-eager` removed; `GLM53_EXL3_CLAMP_DIAG` turned back off ahead
of promotion below -- its one job was done).

## Currently promoted: `v21-b12xmoe` (2026-09-19)

Per explicit user direction ("push forward to get v21-b12xmoe to be our
next candidate to promote" ... "promote v21"), following the SwiGLU-clamp
measurement above landing on "real but rare, not disqualifying."
`GLM53_EXL3_B12X_MOE` flipped from off to `"1"` in the candidate yaml --
this is the actual promotion, not just a container-tag swap: the b12x
fused-trellis MoE path is now live production traffic, not a dormant
opt-in.

**Two more real bugs found getting this live, neither related to the
b12x design itself:**

1. **The production image never had `b12x` installed.** All of today's
   GPU validation (Phases 0-6) ran in isolated venvs that separately
   `pip install`'d it; nobody had added it to the actual Dockerfile.
   First promotion attempt crashed both ranks immediately:
   `ModuleNotFoundError: No module named 'b12x'` -- the fail-closed design
   worked exactly as intended (loud crash, not silent fallback), but
   production was down for the length of one diagnose-and-fix cycle.
   Checked every one of b12x's declared dependencies
   (`torch>=2.12`, `cuda-python`, `nvidia-cutlass-dsl==4.6.2` + its libs,
   `apache-tvm-ffi`, `transformers>=4.51`, `safetensors>=0.6`, `rich>=13`)
   against what's already pinned in the image before fixing --
   `nvidia-cutlass-dsl` is already exactly `4.6.2`, an exact match, and
   everything else already present too -- so the fix is `pip install
   --no-deps b12x==1.3.0` in the Dockerfile, not a dependency-resolution
   change that could disturb anything else pinned. Rebuilt, verified the
   import directly (`docker run --entrypoint python3 ... import b12x`)
   before redeploying.
2. **A pre-existing, already-tracked ambient GB10 crash hit the very next
   attempt** -- a `c10::DistNetworkError` (repeated distributed-backend
   network error) cascaded into `RuntimeError: Executor failed.` and a
   full clean shutdown, moments after health had already passed. This is
   the same class of "torch.AcceleratorError... intermittently... on a
   completely clean, unmodified retry" flakiness this file's own "Where
   to look for common problems" section already documents -- not a new
   bug, not b12x-related (nothing in that traceback touches MoE code at
   all). Standard remedy applied: stop, flush, retry. Second retry booted
   clean, zero errors, confirmed via a real generation request
   immediately (not just `/health`, given the previous attempt's
   health-then-crash sequence).

**Confirmed b12x is genuinely active, not silently falling through**:
`GLM53_EXL3_B12X_MOE=1` verified present in the actual worker process's
exec-time environment (`/proc/<pid>/environ`, the same verification
method developed earlier today for the NCCL env-stripping investigation).
`build_b12x_prepared_state()` logs nothing on success (only on
fail-closed failure) -- but its own load-time checks (the `data_ptr()`
zero-copy invariant, the b12x version/signature assertion) would have
raised and crashed the boot had anything been wrong, and the boot didn't
crash. This is treated as sufficient confirmation; a follow-up could add
an explicit success log line if this ambiguity recurs.

**Real-production performance delta**, `probe_throughput_ab.py`
(n=20, identical fixed prompt, `max_tokens=400`, temp 0 -- same
methodology as every A/B this session, run today against both versions
at identical `gpu_memory_utilization=0.83`):

| | v20-upstreamsync (prior baseline, same day) | v21-b12xmoe (promoted) | Delta |
|---|---:|---:|---:|
| decode tok/s (median) | 31.767 (stdev 0.697) | 34.38 (stdev 0.082) | **+8.23%** |
| TTFT (median) | 0.274 s | 0.249 s | **-9.12%** |

Both the throughput gain and the much tighter run-to-run variance (stdev
dropped ~8.5x) held up in real TP=2 production serving, not just the
isolated single-GPU test harness that originally measured "0.93x at 256
tokens (~7% faster)" -- real production numbers came in even slightly
better than that same-harness estimate.

**Verdict: promoted.** `v20-upstreamsync` remains the rollback target
(swap `container:` back and unset `GLM53_EXL3_B12X_MOE` -- or just
relaunch the v20 yaml directly, which is what "rollback" has meant all
session). The SwiGLU-clamp limitation (see above) is accepted as a known,
documented, rare-but-real tradeoff, not silently ignored.

## v21-b12xmoe did not survive a plain restart: illegal-memory-access during MTP speculator CUDA graph capture, root-caused and fixed (2026-09-19/20)

The very next day, with zero code changes, a routine relaunch of the
promoted `v21-b12xmoe` job hit a real split-brain outage first (head
node's engine died to a `c10::DistNetworkError`-derived `EngineDeadError`;
the worker rank's `Worker_TP1` process kept spinning for 5+ hours in a
broken-pipe reconnect loop without ever tripping its own container's
shutdown watchdog -- `docker ps` showed both containers "Up 8 hours" the
whole time, which was stale/misleading). Recovering via the usual
stop/flush/relaunch discipline hit a second, DIFFERENT crash:
`torch.AcceleratorError: an illegal memory access was encountered` on
rank 1, at `capture_end()`, during `Capturing prefill CUDA graphs
(PIECEWISE)` under `Capturing model for speculator...` -- i.e. during the
MTP draft model's own CUDA graph capture, not the main model's (which had
already captured cleanly, both PIECEWISE and FULL). Given production had
been down 2.5+ hours at that point, rolled back to `v20-upstreamsync`
(verified in known-good state first: gmu 0.83, no stray NCCL vars) to
restore service, then investigated with no traffic on the line.

**Root cause, isolated by A/B on real hardware, not assumed:** relaunched
the same v21-b12xmoe image with `GLM53_EXL3_B12X_MOE` flipped `1`->`0`
(same weight layout, b12x kernel path disabled, falls back to the
pre-existing `apply_exl3_fused_moe`/`apply_exl3_python_loop` path). The
exact capture step that had crashed twice completed 100% cleanly, zero
errors either node, real generation succeeded. This ruled out ambient
GB10 flakiness, the weight-layout change, and the Dockerfile dependency
addition as causes, and pointed squarely at the b12x kernel dispatch
itself. (Also checked, and ruled out: `patch_cudagraph_align.py` --
existing hardening for a different, already-tracked spec-decode
capture-alignment class -- is present and byte-identical between v20 and
v21's overlays; this was a genuinely new bug, not a missing patch.)

**Mechanism, confirmed by reading both sides of the interaction:**
`_b12x_scratch_for()`'s scratch-buffer cache was keyed on the exact token
count `m`; on a cache miss it called b12x's `make_w4a16_packed_buffers()`,
a live `torch.empty()` allocation, with no guard against doing this while
a CUDA graph is capturing. This was safe for the main model only by
accident: vLLM runs a real "Profiling CUDA graph memory" forward pass
(outside any capture context) before real capture, for every one of its
~23 batch-size buckets, which happens to pre-warm this cache. Confirmed
directly in vLLM source (`vllm/v1/worker/gpu/spec_decode/autoregressive/
speculator.py:147`, `capture()`) that the MTP speculator's capture path
has no such profiling pass -- it calls straight into real graph capture
with synthetic dummy data. Its MoE layer (`Glm5NextMTP`, confirmed via
`vllm/models/glm5next/nvidia/mtp.py`, builds from the identical
`Glm5NextMoE` class the main model uses) therefore hit the scratch cache
cold, for the first time ever, while literally inside the capture region
-- exactly the risk this file's own original comment on
`_b12x_scratch_for` had flagged as unaddressed ("a production hardening
pass should size one upper-bound arena instead, to avoid... any CUDA-
graph-capture incompatibility of allocating inside a captured region").

**Fix implemented: fixed-capacity arena, matching this file's own
existing `_FUSED_TEMP_CACHE` pattern** (used successfully by the
pre-existing fused MoE path: one persistent buffer sized to a fixed cap,
never reallocated, tiled into for anything larger, rather than a fresh
allocation per shape). Replaced the `m`-keyed lazy design with:

- `_b12x_topk_and_m_max_from_vllm_config()`: reads `num_experts_per_tok`
  (8, a fixed model constant) and `max_num_batched_tokens` (7168, this
  deployment's scheduler cap) from the live vLLM config -- raises loudly
  rather than guessing a default, since an undersized arena from a wrong
  guess is a silent-corruption risk, not a crash.
- `_b12x_build_arena()`: sweeps b12x's own `plan_w4a16_buffers()` --
  the authoritative sizing function, not a hand-derived formula -- across
  every `m` from 1 to `max_num_batched_tokens` (an exhaustive sweep, not
  a monotonicity assumption; several of the plan's fields are non-
  obviously monotonic in `m` because of `block_size_m`'s discrete bucket
  selection and an SMS-count-based `min()` cap on the GEMM scratch
  fields), takes the elementwise max across the whole sweep, and
  allocates one arena at that true worst case.
- Built eagerly inside `build_b12x_prepared_state()` -- load time, during
  `process_weights_after_loading`, unconditionally before any graph
  capture for either the main model or the speculator. Not built lazily
  on first apply-time use; that "hope profiling runs first" assumption is
  exactly what broke.
- `apply_exl3_b12x_moe()` now takes `.narrow().view()` slices into the
  persistent arena per call (sized by that call's own real `m`, via the
  same `plan_w4a16_buffers()`, always <= the swept worst case by
  construction) instead of allocating. No `torch.empty()` occurs on this
  path ever again after the one-time load-time build.
- Defense-in-depth guards, fail-loud not fail-silent: a topk mismatch
  between a real call and the arena's build-time constant raises; `m`
  exceeding the arena's swept `m_max` raises rather than silently
  reallocating; reaching a cold cache during
  `torch.cuda.is_current_stream_capturing()` raises rather than
  allocating or silently falling back to a different numerical path
  (a silent fallback would permanently and invisibly downgrade that
  graph's MoE tier forever -- the same blind-spot class as the Phase 4
  test-file bug from earlier this week).
- Incidentally also fixes an unrelated, lower-severity issue the old
  code's own comment already flagged: the `m`-keyed cache grew one entry
  per distinct token count ever seen, for the life of the process --
  unbounded memory growth under workloads with varying prefill-chunk
  sizes. The arena is now bounded once, permanently.

**Validated, not just implemented:**
- New tests in `tests/test_exl3_overlay.py`
  (`_check_b12x_arena_stability`, `_check_b12x_arena_capture_guard`),
  wired into `main()`'s real invocation chain (checked explicitly, given
  this project's own past incident where new tests were written but never
  wired in): interleaved/repeated calls at varied `m` (1, 2, 8, 16, 33,
  8, 1, 33) produce deterministic results on repeats, the arena's
  `data_ptr()`s never change across any of them, `m > m_max` raises, and
  a cold-cache-during-capture simulation raises. All existing checks in
  the suite still pass -- 17/17 on real GPU hardware.
- One honest finding surfaced along the way, not hidden: a correctness-
  parity check against `apply_exl3_python_loop` using this suite's
  `_tiny_layer` fixture (i.i.d. random trellis/suh/svh weights) shows
  weak agreement (cosine ~0.6-0.7, far below the ~1.0 the real-checkpoint
  validation found). Confirmed this is pre-existing and unrelated to this
  fix by reproducing the identical weakness on the *unmodified* pre-fix
  code against the same fixture, and separately confirmed the rewrite
  itself is purely a memory-management change by diffing its output
  directly against the original on identical inputs -- bit-for-bit
  identical everywhere finite. b12x's `full_rotation=True` correctness
  depends on realistically-conditioned (real, trained) rotation matrices;
  random weights aren't a fair correctness test for it. Real-checkpoint
  parity (cosine 1.0, 30/30) was already established separately and
  wasn't touched by this change.
- Real hardware, same production yaml, `GLM53_EXL3_B12X_MOE=1`, fresh
  boot: the exact capture step that crashed twice (`Capturing prefill
  CUDA graphs (PIECEWISE)`/`(FULL)` under `Capturing model for
  speculator...`) completed 100% cleanly. Zero errors in either node's
  full log. Health 200, confirmed with a real `/v1/chat/completions`
  request, not just health.
- Throughput re-measured (`probe_throughput_ab.py`, same n=20
  methodology): decode median 34.221 tok/s, TTFT median 0.254s -- both
  within noise of the original promotion's 34.38 tok/s / 0.249s,
  confirming the arena rewrite is throughput-neutral as expected for a
  pure memory-management change.

**Re-promoted (2026-09-20), per explicit user direction.** The fixed
`v21-b12xmoe` (already running from the fix's own validation pass) is
the deployed job again.

## Post-re-promotion validation, benchmark, and concurrency sweep (2026-09-20)

**Validation**: `probe_sanity.py` full suite, clean pass -- served-model
id, metrics endpoint (413 `vllm:*` series), non-empty/coherent chat,
finish_reason, no reasoning-leak into content, streaming TTFT/usage/
counts, 3 bench runs all passing (35.8/36.4/37.1 tok/s decode).

**Throughput** (`probe_throughput_ab.py`, n=20, same methodology as
every prior A/B): decode median 34.226 tok/s (mean 34.24, stdev 0.133),
TTFT median 0.246s. Matches both the original pre-crash promotion
baseline (34.38 tok/s / 0.249s) and the fix's own validation pass
(34.221 tok/s / 0.254s) within noise -- three independent measurements
of the same number now, across three different points in this saga.

**Concurrency sweep** (`probe_concurrency_pipeline.py`, levels
1/2/4/8/16 -- 16 is this deployment's `max_num_seqs` ceiling --, 3 waves,
prose workload):

| concurrency | aggregate tok/s | client e2e p50 | server TTFT avg | accept rate | preemptions | KV cache peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.35 | 3.96s | 0.192s | 0.614 | 0 | 2.3% |
| 2 | 51.26 | 4.83s | 0.252s | 0.655 | 0 | 4.7% |
| 4 | 60.71 | 8.63s | 0.306s | 0.625 | 0 | 9.3% |
| 8 | 135.03 | 7.08s | 0.743s | 0.761 | 0 | 18.6% |
| 16 | 190.91 | 9.71s | 0.729s | 0.726 | 0 | 37.3% |

Zero preemptions and zero request queueing (`waiting_max=0`) at every
level, including at 16 -- this deployment's exact concurrency ceiling.
KV cache usage scales sensibly with load and stays well under
saturation even at max concurrency (37.3% peak). Zero errors in either
node's log across the entire sweep. Spec-decode acceptance rate actually
improves under load (61% at c=1 up to 73-76% at c=8/16) rather than
degrading -- a bonus finding, not a concern. Aggregate throughput scales
sub-linearly but healthily (~5.9x at 16x concurrency), with a notably
good jump from c=4 to c=8 that tracks the acceptance-rate improvement.

Host memory during the sweep sat at ~2.8-5.2 GiB available on both
nodes -- tight, but consistent with this fleet's already-documented
standing margin under load (see the earlyoom/gmu-tuning history
elsewhere in this file), not a regression introduced by this fix or by
b12x specifically.

**Verdict: re-promotion confirmed healthy under concurrent load, not
just idle.** No new concerns found; the fixed-arena change holds up
identically to the pre-bug baseline across single-stream throughput and
now, additionally, across a full concurrency sweep to the deployment's
own ceiling.

## Two real earlyoom kills at gmu=0.83 within one afternoon; reverted to 0.82 (2026-09-20)

Not a b12x issue -- this fleet's already-documented earlyoom-margin
fragility, recurring. Within ~3 hours of the concurrency-sweep
validation above (run at `gpu_memory_utilization=0.83`), the head node
(`10.7.0.87`, the fleet's chronically thinner-margin host per its own
long-standing flag) took two independent earlyoom SIGTERM kills:

1. **19:11:40 UTC** -- the API server process itself (`badness 984,
   VmRSS 3353 MiB`). Looked like a clean shutdown in the vLLM log (no
   traceback, `EngineDeadError` only as the downstream effect of the
   forced EngineCore kill) because earlyoom's SIGTERM let it unwind
   normally -- this is the well-known "silent kill" signature this
   project has hit many times before. Confirmed via `journalctl -u
   earlyoom` directly, not inferred. The worker rank never received an
   equivalent signal and spun on its dead peer -- the same partial-stop
   split-brain pattern seen earlier this week, with `docker ps` again
   misleadingly showing both containers "Up" the whole time.
2. **17:55:00 local, the very next boot** (post stop/flush/relaunch
   recovery, same 0.83) -- a `VLLM::Worker_TP` process this time
   (`badness 985, VmRSS 2367 MiB`), plus a `kill failed: Timer expired`
   on top of it.

The concurrency sweep (real KV cache growth to 37.3% at concurrency 16)
run immediately before the first kill is the likely proximate trigger,
not a steady-state idle problem -- but two kills on the same host in one
afternoon means the margin at 0.83 genuinely cannot absorb real
concurrent load on this kernel, not just "still on the knife-edge, not
tipped over yet" as hoped when 0.83 was set. **Reverted
`gpu_memory_utilization` 0.83 -> 0.82** (recipe yaml, with the incident
documented inline in the same comment block as the rest of this
setting's history). Recovered via the standard stop/flush/relaunch
discipline both times; validated the final 0.82 boot with a real
generation request (not just health). Post-relaunch memory headroom
visibly improved: ~9.1 GiB / 13 GiB available on head/worker
respectively, vs. ~2.8-5.2 GiB observed under the 0.83 sweep.

**Not yet re-validated under concurrent load at 0.82** -- deliberately
did not immediately re-run the same concurrency sweep that plausibly
triggered the first kill, to avoid risking a third crash in one session
before the lower setting has had any soak time. If 0.82 needs its own
load-bearing confirmation, that's a follow-up, not assumed clear from
one real request succeeding.

## Upstream check, 2026-09-21

Routine triage round (read-only, no changes) of all 7 tracked repos plus
a scan for new ones, specifically pointed at anything touching CUDA-
graph-capture safety or memory-margin tuning given this fork's last two
incidents.

- **MiaAI-Lab (main upstream)**: 38 commits since 09-18, all TP3/TP4-
  specific feature work, not applicable (this fork runs TP2). No hits
  searching for b12x/cudagraph/earlyoom/gmu/MTP-speculator/illegal-
  memory-access topics. Two small generic launcher fixes not yet looked
  at (`#197` USER-unbound-in-non-login-shells, `#168` env-override
  notice) -- optional, low priority. **Nothing new.**
- **Enntity/sparkglm**: `exl3` branch's last commit is 09-08, predates
  the prior 09-13 check. **Fully quiet.**
- **mmastrac/glm-5.3-flash-4x-gx10**: one commit (`a30d4b24f`) bakes a
  `thinking_token_budget` fix into the build. Resolves one of three
  previously-flagged-but-uninvestigated items from the 09-14 deep-dive.
  **Possible tension with this fork's own history**: memory records this
  exact bug class as already "ruled out (dead code, we run V1 not V2)"
  -- but the fix touches `vllm/v1/worker/gpu/sample/sampler.py`'s
  `_requires_logits_processing()` guard, which the fix's own docstring
  calls "the V2 sampler" despite the `v1/` path. The bug itself: the
  guard never checks the `thinking_token_budget` flag, so temp=0
  (greedy) and temp=1.0 (this model's own default) both silently skip
  budget enforcement -- and since the guard is `np.any` over the whole
  batch, one co-scheduled request at a different temperature can
  randomly "turn on" the cap for everyone else that step, which would
  look intermittent rather than simply broken. **Worth a direct check**
  against this fork's own installed `sampler.py` before trusting either
  the old ruling or this new finding -- not yet done.
- **AEON-7/vllm-ultimate-dgx-spark**: 2 doc-only commits since 09-12 (a
  release announcement). **Quiet.**
- **cbertucci33/vllm-v29-glm53flash-exl3-dgx**: did NOT go quiet as
  previously guessed -- 29 commits since 09-12, 5 numbered releases with
  real production telemetry (Release 5: 922 completed requests, 0
  errors, 800K-token-limit/971K-token-cache DFlash2 run -- different,
  abliterated target checkpoint, not directly comparable throughput-wise,
  but a real long-run reliability data point). **The actionable
  finding**: commit `111425cfc` "Fix GLM hybrid prefix-cache corruption"
  (09-18) -- `_kpool_tail_seed_kernel`'s tail-cache write offset assumed
  a dense `[2, KPOOL, HEAD_DIM]` stride, but the tail cache actually
  aliases the indexer cache and uses the indexer's own *padded* block
  stride. Silent layout-mismatch corruption, no crash -- the same class
  of bug this fork just spent a week hunting in the b12x integration.
  Fix passes real tensor strides as kernel constexpr args instead of
  assuming dense packing, plus adds stride assertions. Touches
  `vllm/models/glm5next/nvidia/ops/kpool_compress.py` and
  `vllm/v1/core/sched/scheduler.py` -- the same GLM5Next/kpool
  architecture this fork's own `patch_kpool_tail_slotmap.py`/
  `patch_hybrid_prefix_hit.py` touch. (A second, bundled fix in the same
  commit -- `use_eagle_block_drop` wrongly applying to DFlash/DSpark
  drafters -- confirmed NOT relevant: MTP was already correctly included
  both before and after.) **Worth checking whether this fork's own kpool
  tail-seeding code makes the same dense-stride assumption** -- not yet
  done.
- **local-inference-lab/b12x**: 14 commits since 09-18 on `master`, all
  past the pinned PyPI `1.3.0` (still the latest release, uploaded
  08-25 -- none of this shipped yet). Most relevant:
  `8783519a3` "Pass constexpr arguments to prepared W4A16 route pack
  launchers," touching `b12x/moe/_shared/kernels/w4a16/route_pack.py` --
  the exact kernel family `apply_exl3_b12x_moe` calls
  (`pack_topk_routes_by_expert`). Commit message describes new test
  coverage for "graph replay after poisoning buffers... with fixed
  output addresses" -- **b12x upstream is independently hardening this
  same kernel family for CUDA-graph-capture/replay safety**, a similar
  "fixed address" philosophy to the arena fix this fork just shipped.
  Their own commit notes GPU correctness is "unverified by this commit
  pass" (hit OOM during their own GB10 validation) -- in-flight,
  self-reported-unverified, not adoptable yet. **Track, revisit when a
  release past 1.3.0 lands** -- check then whether it changes or
  obsoletes this fork's own arena fix.
- **Entrpi (both repos)**: zero new commits since 09-18, `FINDINGS.md`
  still ends at section 19 (09-02) -- nothing new on memory-floor-
  erosion or gmu-tuning specifically. **Quiet.**

**New repos spotted, not yet tracked**: `Reederey87/glm53-flash-exl3-2x-
dgx-spark` (65 stars -- second only to MiaAI-Lab in this niche, pushed
09-20, claims 1M context + 97%+ multi-session prefix caching + DFlash2)
and `r0b0tlab/glm53-flash-exl3-dflash2-sm121` (brand new, created 09-18,
unstarred). Neither investigated yet -- candidates for the tracked list
if worth a first look.

**Open follow-ups from this round, none yet done**: (1) check this
fork's `sampler.py` against the mmastrac thinking_token_budget finding;
(2) check this fork's kpool tail-seeding code against the cbertucci33
dense-stride finding -- this one in particular matches this fork's own
recent bug class closely enough to be worth prioritizing; (3) revisit
b12x's PyPI releases past 1.3.0 when one lands; (4) decide whether to
add Reederey87 and/or r0b0tlab to the tracked rotation.

## kpool tail-cache stride bug: confirmed live in this fork's own production code (2026-09-21)

Chased the cbertucci33 finding above directly against this fork's real
running image (`glm53-exl3-v21-b12xmoe:local`), not just the upstream
diff. Confirmed the identical bug is present and reachable in our own
deployment, via numbers derived from our own real boot logs and vLLM's
own authoritative sizing formulas, not assumption:

- `_kpool_tail_seed_kernel`
  (`vllm/models/glm5next/nvidia/ops/kpool_compress.py:449`, vendored
  inside our image) computes the tail-cache write offset as
  `base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM`, where `blk` is the
  **global** block index (confirmed by reading the caller,
  `sparse_attn_indexer_kpool.py:416`, which passes the full shared
  `tail_kv_cache` tensor and a global `slot_mapping`, not a per-request
  view). This assumes each block occupies exactly
  `2 * KPOOL * HEAD_DIM = 2*4*128 = 1024` bf16 elements = 2048 bytes.
- That 2048B figure exactly matches `SlidingWindowSpec.real_page_size_bytes`'s
  formula (`vllm/v1/kv_cache_interface.py:596-614`:
  `block_size * num_kv_heads * (head_size+head_size_v) * dtype_size`) for
  this spec -- self-consistent with the kernel's own assumption.
- But our own production boot logs show the group's **actual allocated**
  `page_size_bytes` -- the value that governs the real tensor's per-block
  stride -- is **118,272 bytes, 57.75x bigger**
  (`[glm53-kv-capacity-log] group 1: KpoolTailSpec ... page_size=118,272 B`,
  identical to the co-located MLA attention group's own page_size,
  confirming padding/aliasing rather than dense packing).
- Net effect: for any block index `blk >= 1` -- i.e. any request beyond
  whichever one lands in the very first block, which under real
  concurrent serving is almost every request -- the kernel writes
  `blk*2048` bytes into a tensor whose real per-block stride is
  `blk*118272` bytes. Silent corruption, no crash, on every prefill
  request with an incomplete trailing kpool (common).

**Fix in progress** (delegated, in flight as of this writeup): a
build-time source patch (`patch_kpool_tail_seed_stride.py`, mirroring
this fork's existing `patch_kpool_tail_slotmap.py` pattern) to pass the
tensor's real strides into the kernel instead of the dense-packing
constant formula, matching cbertucci33's validated fix direction. See
the follow-up entry below once that work reports back for the actual
diff, ground-truth stride values, and real-hardware validation outcome.

## thinking_token_budget: prior "ruled out, dead code" verdict was WRONG -- we run Model Runner V2 (2026-09-21)

Re-chased the mmastrac/4x-gx10 finding directly against this fork's own
installed vLLM, resolving the ambiguity the upstream-check round left
open. **Definitive, not inferred**: our own real boot log's line numbers
for `model_runner.py` messages (`:353` "Loading model from scratch",
`:952` "Estimated CUDA graph memory", `:1012` "Graph capturing
finished") match `vllm/v1/worker/gpu/model_runner.py` -- **Model Runner
V2** -- exactly. The V1 candidate (`vllm/v1/worker/gpu_model_runner.py`)
has its equivalent log lines at `:6960`/`:7067`, nowhere close. This
fork's prior ruling ("dead code, we run V1 not V2") conflated "vLLM v1
engine" (which we do run, correctly) with "Model Runner V1" (a
*different*, unrelated versioning axis *within* the v1 engine -- vLLM's
own `VLLM_USE_V2_MODEL_RUNNER` mechanism, `vllm/config/vllm.py:628-674`).
Confirmed why V2 is selected for us specifically: our model's
architecture string, `Glm5NextForConditionalGeneration`, is explicitly
listed in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`
(`vllm/config/vllm.py:73-84`), and nothing in our config trips the
V2-unsupported-feature fallback.

**The bug itself is real and live in our production code**:
`_requires_logits_processing()`
(`vllm/v1/worker/gpu/sample/sampler.py:220`) -- the guard that decides
whether `apply_sampling_params` does any per-token logits work at all --
checks logit_bias, penalties, bad_words, non-default temperature, min_p,
top_k, and top_p, but never checks
`thinking_budget_state`/`use_thinking_budget`. Yet
`thinking_budget_state.apply(...)` is called unconditionally further
down inside `apply_sampling_params`, which short-circuits with a bare
`return logits` before ever reaching that call whenever the guard
returns `False`. Net effect: a request with `thinking_token_budget` set
but otherwise-default sampling params (temperature 0 or 1.0, no
penalties/bad_words/logit_bias) silently never gets its budget
enforced. Confirmed our deployment actually exercises this code path,
not just theoretically: `--reasoning-parser glm45` is configured, which
feeds real (non-empty) reasoning start/end/natural-end token IDs into
`ThinkingBudgetState`, so `self.enabled` is `True` here, not
short-circuited off.

**Not yet fixed** -- this round was a reverification, not a fix pass.
mmastrac/4x-gx10's own commit (`a30d4b24f`) already has a working
one-line fix (add a `use_thinking_budget` check to the guard) available
to mirror if this is worth patching. Flagged to the user; fix scope and
priority is their call, not decided here.

## Two new repos scouted: Reederey87 (high value), r0b0tlab (narrower) (2026-09-21)

Full findings in memory (`new_repos_scouted_2026_09_21.md`); summary
here for the canonical-repo record.

**Reederey87/glm53-flash-exl3-2x-dgx-spark** -- same hardware/topology/
weight lineage as this fork, DFlash2 instead of MTP-2, mature disciplined
engineering (~75 commits, bit-exactness test gates, dated benchmark
receipts, rejected experiments recorded alongside adopted ones). Two
concrete techniques worth a direct diff against this fork's own overlay:
a per-group KV-retention fix (`overlay/patch_apc_per_group_retention.py`,
fixes a drafter sliding-window group starving the shared LRU pool for
every other KV group) and a sub-page fine-grained APC exemption
(`overlay/patch_fine_grained_apc.py`, fixes 0% cache hits below one full
page for prompts under 3584 tokens). Also independently corroborates two
of this fork's own recent findings: their own lazy-scratch-buffer
experiment for MoE kernels was tried and reverted for the same reason
this fork's arena fix exists (CUDA graph capture pre-commits to the
worst-case shape regardless), and their own GB10 fleet hits the identical
narrow earlyoom-margin signature (`GPU_MEM_UTIL=0.85` flagged as
"0.87 crash-loops" in their config). **Recommended addition to the
tracked rotation** -- real ongoing engineering, not a one-off.

**r0b0tlab/glm53-flash-exl3-dflash2-sm121** -- narrower scope (single
GB10, TP=1, 32k max context, own from-scratch vLLM plugin with stock
exllamav3 kernels, no b12x overlap), too young to compare directly
(created 09-18, single-author, 0 stars). Two CUDA-graph-capture-safety
bugs in their own EXL3 MoE path worth pattern-matching against this
fork's own b12x integration: `torch.bincount` (not stream-capturable,
replaced with `zeros + scatter_add_`) and a per-expert fallback loop
using `torch.unique`/`.nonzero().tolist()` for batches above the fused
row capacity (silently broke any capture size above ~32 tokens before
the fix). **Not yet checked against our own `apply_exl3_b12x_moe`/
`apply_exl3_fused_moe` dispatch paths** -- worth a grep audit given this
fork just spent a week on a structurally similar capture-safety bug.

## kpool tail-cache stride bug: fixed, validated, deployed (2026-09-21)

Follow-up to the confirmation above. Fixed via
`recipes/build/glm53-exl3-v21-b12xmoe/overlay/patch_kpool_tail_seed_stride.py`
(new, mirrors `patch_kpool_tail_slotmap.py`'s idempotent/fail-closed
build-time source-patch pattern).

**Ground truth, traced through vLLM's own allocation code and confirmed
live on GPU** (more precise than the initial estimate above):
`tail_kv_cache`'s logical shape stays `(num_blocks, 2, KPOOL=4,
HEAD_DIM=128)` -- only `stride(0)` (the block dimension) is padded.
Confirmed directly in `vllm/v1/worker/gpu/attn_utils.py`'s
`page_size_padded` strided-view branch, whose own comment states "the
only stride that must change is the block stride." Real `stride(0) =
page_size_bytes // dtype_size = 118272 // 2 = 59,136` elements on this
deployment, vs. the buggy kernel's hardcoded `2 * KPOOL = 1024`. Every
other stride (the intra-block `[2, KPOOL, HEAD_DIM]` layout) is
unaffected -- only the `blk *` multiplier was wrong.

**The decode-side companion kernel was already correct** --
`_kpool_decode_update_batched_kernel` already receives the real stride
as an explicit `TAIL_BLOCK_ELEMS` constexpr via
`tail_kv_cache.stride(0)` at its call site. Only the prefill-seed
kernel had the bug -- an internal inconsistency within the same file,
not a systemic pattern across the whole cache.

**Reproduced the corruption live on GPU before fixing**, not just
reasoned about it: seeded block 1 (an ordinary case under real
concurrency) with the unpatched kernel -- block 1's real memory stayed
untouched (zeros) while the data landed 1024 elements in, inside block
0's region instead. After the fix, block 1 was written correctly and
neighboring blocks stayed clean. Same repro pattern is now a permanent
regression test (`test_live_kernel_writes_at_strided_offset_if_gpu` in
the new `tests/test_kpool_tail_seed_stride.py`, GPU-gated, using a
`torch.as_strided` synthetic tensor with a deliberately padded stride).

**Fix**: adds `TAIL_BLOCK_ELEMS: tl.constexpr` to
`_kpool_tail_seed_kernel`, changes `base = (blk*2*KPOOL + t%KPOOL) *
HEAD_DIM` -> `base = blk*TAIL_BLOCK_ELEMS + (t%KPOOL)*HEAD_DIM`, and
adds two runtime assertions at the call site
(`tail_kv_cache.stride(1)`/`.stride(2)` match the expected unpadded
intra-block values) so a future change to the padding contract fails
loudly instead of silently corrupting again. Both assertions passed
against the real production tensor on redeploy -- positive confirmation
the real layout matches what the fix assumes, not just a passing test
in isolation.

**Validated**: 18/18 suites pass in the full Docker build chain on both
hosts (16 pre-existing + 2 new), wired into the real test-invocation
chain (`test_recipe_wiring_if_present` explicitly checks the Dockerfile
`COPY`/`RUN` lines exist and are correctly ordered -- this project has
been burned before by tests that were written but never wired in).
Rebuilt, stopped, flushed, relaunched `v21-b12xmoe` with
`gpu_memory_utilization: 0.82` unchanged. Health 200, real chat
completion succeeded, zero errors in either node's log. Exercised the
actual buggy code path with `probe_concurrency_pipeline.py --levels 4,8
--waves 2` (24 real concurrent requests) -- zero errors, zero
preemptions, accept_rate 0.68-0.76. Independently re-confirmed
afterward (not just trusting the fix report): grepped the actual live
running containers on both hosts for `TAIL_BLOCK_ELEMS` and found the
patched code genuinely loaded and executing, not just present in the
built image; re-ran a real chat completion myself; confirmed
`gpu_memory_utilization` was not accidentally reverted during the
rebuild/redeploy cycle.

**Current state**: both hosts running the fixed `v21-b12xmoe`, healthy.
Other build trees (`v20-upstreamsync` etc., not currently serving) were
not checked or patched -- out of scope, since they're not in production.

## r0b0tlab capture-safety pattern audit against our own b12x dispatch: clean (2026-09-21)

Chased the open follow-up from the new-repo scouting above:
r0b0tlab/glm53-flash-exl3-dflash2-sm121 found and fixed two CUDA-graph-
capture-safety bugs in their own EXL3 MoE path (`torch.bincount` for
expert-routing counts, and a per-expert fallback loop using
`torch.unique`/`.nonzero().tolist()` for large batches -- both
host-syncing calls that silently broke capture above a threshold).
Grepped this fork's own b12x dispatch path for the same pattern
(`torch.bincount`, `torch.unique`, `.nonzero()`, `.tolist()`, `.item()`)
end to end:

- `apply_exl3_b12x_moe` (the actual b12x call site, `exl3.py:2177-2414`):
  zero hits.
- `_b12x_arena_for`/`_b12x_build_arena`/
  `_b12x_topk_and_m_max_from_vllm_config` (the arena machinery from the
  CUDA-graph-capture fix earlier this week, `exl3.py:1986-2176`): zero
  hits.
- `apply_exl3_experts` (the dispatch wrapper that routes into b12x when
  `GLM53_EXL3_B12X_MOE=1`, `exl3.py:2716` onward, up through and
  including the b12x branch): zero hits. The one host-sync-shaped call
  reachable in this function, the SwiGLU-clamp diagnostic's
  `torch.unique(ids)`, is already explicitly guarded by
  `torch.cuda.is_current_stream_capturing()` -- the exact fix this fork
  applied to itself after finding that same crash on 2026-09-19.
  `pin_exl3_expert_map` (called unconditionally before the b12x branch)
  already documents and defends against exactly this class of bug in
  its own docstring ("CUDA graph capture forbids a CPU->GPU copy"), and
  returns `None` immediately for b12x's current configuration (which
  doesn't yet support TP expert-map remapping) before reaching any
  `.to()` call.
- The external b12x library modules actually called from this path
  (`b12x/moe/_shared/kernels/w4a16/route_pack.py`,
  `.../kernel.py`, installed in the production image): zero hits.

**Verdict: clean, no action needed.** This fork's b12x integration does
not share the pattern that bit r0b0tlab, in either the code this fork
wrote or the specific library modules it calls into. Worth remembering
as a standing thing to re-check if the b12x pin is ever bumped past
1.3.0 (see the earlier "track, not actionable yet" entry on upstream
b12x's own unreleased CUDA-graph hardening work) -- a library version
bump could introduce a new capture-unsafe call this audit wouldn't
catch retroactively.

## thinking_token_budget: fixed, validated end-to-end, deployed (2026-09-21)

Mirrors mmastrac/glm-5.3-flash-4x-gx10's fix (`a30d4b24f`) for the
`_requires_logits_processing` guard confirmed live in our own production
code above. Fixed via
`recipes/build/glm53-exl3-v21-b12xmoe/overlay/patch_thinking_budget_guard.py`
(new, mirrors the kpool patches' idempotent/fail-closed build-time
source-patch pattern).

**`ThinkingBudgetState`'s actual attributes matched the predicted shape
exactly**, confirmed by reading the real installed class before writing
the fix: `self.enabled: bool` and `self.use_thinking_budget: np.ndarray`
-- the latter is only set as an attribute at all inside the
`if not self.enabled: return` early-return's complement, so the fix
checks `.enabled` first (`if self.thinking_budget_state.enabled and
np.any(self.thinking_budget_state.use_thinking_budget[idx_mapping_np]):
return True`, added to the existing guard's chain of checks) rather than
assume the attribute always exists.

**Tests**: new `tests/test_thinking_budget_guard.py`, wired into the
Dockerfile's COPY/RUN/test-chain (mirroring `test_recipe_wiring_if_present`'s
self-check pattern). Includes a genuine RED-then-GREEN proof at the unit
level, not just an integration smoke test: a duck-typed fake `self`
constructs the exact bug scenario (only `use_thinking_budget` set,
everything else at its default value) and confirms the *unpatched*
guard returns `False` (the bug, reproduced), then confirms the *patched*
guard returns `True` for the identical input (the fix). Also covers: the
`.enabled=False` case doesn't raise `AttributeError`; a
budget-not-requested case is unaffected; and the guard's pre-existing
checks (e.g. non-default temperature) still fire correctly post-fix. All
pass, plus the patch's own fixture/idempotency/fail-closed tests and a
check against the real installed source.

**Real-hardware validation, including the actual end-to-end
budget-enforcement behavior the fix exists for** (not just "doesn't
crash"): rebuilt on both hosts (this session runs on the worker node,
`127.0.0.1`; the head node, `10.7.0.87`, doesn't share a filesystem or
registry, so the build context was rsynced over and built there too --
same two-host build discipline the kpool fix established). Stopped,
flushed, relaunched with `gpu_memory_utilization: 0.82` confirmed
unchanged before and after. Health 200, zero errors either node's log, a
real chat completion succeeded.

Then exercised the actual fixed code path with two real requests, same
prompt/temperature(0)/max_tokens(200), differing only in whether
`thinking_token_budget` was set:
- **Without** a budget (control): reasoning consumed the entire
  200-token limit and never reached `content` at all (0 chars of
  content, still mid-reasoning when cut off).
- **With** `thinking_token_budget: 50` (temperature 0, nothing else
  non-default -- exactly the case the bug silently skipped before this
  fix): reasoning was cut short around the budget and the model moved on
  to produce real content (600+ chars) within the same 200-token limit.

This is a genuine behavioral difference on real hardware, not just a
passing unit test -- positive, direct confirmation the fix changes real
serving behavior the way it's supposed to. (Note for future reference:
the OpenAI-compatible response field is `reasoning`, not
`reasoning_content` -- easy to get wrong when scripting a check like
this.)

Independently re-verified afterward: grepped both live running
containers directly for the patch's marker/guard code (present and
identical on both), confirmed `gpu_memory_utilization` was not
disturbed by the rebuild, and confirmed the earlier kpool stride fix
(`TAIL_BLOCK_ELEMS`) survived the rebuild intact -- no regression from
combining both patches in the same image.

**Current state**: both hosts running `v21-b12xmoe` with both fixes
(kpool stride + thinking-budget guard) applied, healthy, `gpu_memory_utilization=0.82`.

## Third earlyoom kill on the head node, 0.82 has not resolved this (2026-09-21)

~1h10-1h20m after the thinking-budget-fix redeploy above, the head node
(`10.7.0.87`, spark-2dd4 -- the same chronically-thinner-margin host as
both of the 09-20 kills) took a third earlyoom SIGTERM kill at
`gpu_memory_utilization=0.82`:

```
Sep 21 17:54:14 spark-2dd4 earlyoom: low memory! at or below SIGTERM limits: mem 2.00%, swap 80.00%
Sep 21 17:54:14 spark-2dd4 earlyoom: sending SIGTERM to process 3253687 "python3": badness 984, VmRSS 3390 MiB
```

Same split-brain shape as before (head's engine died cleanly via
SIGTERM, worker rank kept spinning on its dead peer, `docker ps` showed
both containers "Up" throughout). **Notably different from the prior
two kills**: the visible log window before this crash shows only
routine `/metrics` polling, no real chat-completion traffic and
certainly nothing resembling the concurrency sweep that plausibly
triggered the first kill. This crash happened under light/idle
conditions, purely from uptime -- memory crept up over roughly
1-1.5 hours and crossed the threshold on its own.

Checked for a scheduled-task explanation (the second and third kills
landed within a minute of the same wall-clock time 24 hours apart,
09-20 21:55 UTC and 09-21 21:54 UTC) -- no systemd timer, cron job, or
other journalctl-visible event around that window on either occurrence.
Treated as coincidence, not a confirmed lead, absent further data.

Recovered via the standard stop/flush/relaunch/validate-with-a-real-
request discipline. **Verdict: 0.82 has not actually solved this fleet's
earlyoom fragility** -- it bought a somewhat longer runway than 0.83 (a
few hours instead of under one), but the head node still crosses the
threshold under essentially idle conditions within roughly an hour or
two of uptime. This is now 3 kills on the same host in about 30 hours.
Not yet resolved; options going forward (not decided here): drop gmu
further (unclear how much runway that actually buys given even idle
memory growth seems to be the driver, not concurrent-request KV cache),
investigate the actual source of the memory growth over time on this
specific host directly (rather than continuing to treat gmu tuning as
the only lever), or accept periodic restarts as an operational reality
on this host pending a real root cause.

## Root cause found: head-node-only medium-order fragmentation, not a memory leak (2026-09-21)

Investigated why ONLY the head node (`10.7.0.87`/spark-2dd4) has hit
earlyoom 3 times while the worker (`127.0.0.1`/spark-276f) has never
once been killed, with production traffic live during the
investigation (read-only diagnostics only, one safe live kernel
operation -- see below -- nothing else touched).

**Ruled out via real historical Prometheus data from the actual third-
kill incident** (`process_resident_memory_bytes{job="vllm-spark-2dd4"}`,
`node_memory_AnonPages_bytes`, `node_memory_Cached_bytes`,
`vllm:kv_cache_usage_perc`, all queried for the exact 20:34-21:56 UTC
window bracketing the 21:54:14 kill): the API server's own RSS
plateaued at ~3.39 GB and stayed completely flat for the final 34
minutes before the kill; host-wide AnonPages and Cached both plateaued
too; `kv_cache_usage_perc` was **flat zero the entire window** -- no
real requests were being served. Yet `MemAvailable` kept eroding through
that same flat-everything window, in a step-wise pattern (not smooth),
until it crossed earlyoom's combined mem+swap threshold. This is
inconsistent with a classic memory leak in any single process.

**Found via direct `/proc/buddyinfo` comparison** (live, same moment,
both hosts, same ~20 minute uptime): the head node's "Normal" zone had
only **1 free block each at order-4 (64 KB) and order-5 (128 KB)**,
while the worker had **16,408 and 11,597 respectively** at the same
uptime -- a dramatic, specific, medium-block-size collapse unique to
the head node. This develops fast (within the first ~20 minutes of
boot), not gradually over hours -- what erodes gradually afterward is
`MemAvailable`'s own reclaim-cost estimate continuing to degrade on top
of this already-fragmented baseline (matching the step-wise Prometheus
pattern above), not the fragmentation itself building slowly.

**The structural reason it's head-node-only**: only the head node runs
the APIServer/FastAPI/uvicorn process, the TCPStore rendezvous master
role, and EngineCore/the scheduler, *in addition to* the same
Worker_TP GPU-worker role the other host has running headless. That
additional process diversity -- HTTP connection handling, JSON
(de)serialization, asyncio task churn, streaming-response buffering --
is the most likely driver of the medium-order-specific fragmentation;
the worker's simpler, single-role boot doesn't produce the same
allocation pattern.

**A real gap found in this project's own existing fragmentation
preflight check**: `recipes/scripts/frag_check_remote.sh` (used by
`prelaunch_flush.sh`, added after the 2026-09-08 NVRM/Xid-31 incident)
reads `nr_free_pages_blocks` from `/proc/zoneinfo` -- an **aggregate
count across every block order**, currently ~115,712-20,909,568
depending on when sampled. That aggregate is dominated by abundant
small-order (4-32 KB) blocks and stays comfortably above the check's
own 2,000-block threshold even while medium orders are individually
down to 1 block each. The check isn't wrong -- it was correctly
calibrated against the prior incident's failure signature (aggregate
dropping to exactly 0, a much more total exhaustion) -- but it has a
real blind spot for this partial, medium-order-specific pattern, and
has reported "OK" on every boot this whole session, including
immediately before boots that later hit this exact failure mode.

**Confirmed fixable, live, non-disruptively**: triggered kernel
compaction directly (`echo 1 > /proc/sys/vm/compact_memory`) on the
live head node, with production traffic running. Order-4 blocks went
from 1 -> 582; order-5 from 1 -> 655 -- immediate, dramatic
improvement. Verified production was completely unaffected
afterward (health 200, container uptime undisturbed, no restart).
Compaction only reorganizes free pages; it cannot touch or corrupt
memory already in use by a running process, which is why this was safe
to test directly against a live traffic-serving host.

**Not yet implemented as a permanent fix** -- this round was root-cause
diagnosis, confirmed via a real but manual live test. Options for
closing this properly (none decided or applied beyond the one manual
compaction trigger above):
1. Fix the existing fragmentation check to also look at medium orders
   specifically (e.g., check order-4/order-5 counts directly from
   `/proc/buddyinfo`, not just the `nr_free_pages_blocks` aggregate),
   so it actually catches this pattern at the next preflight instead of
   reporting false-OK.
2. Add a periodic (not just pre-launch) compaction trigger while a job
   is live serving, since this fragmentation develops within the first
   ~20 minutes of boot and the existing check only ever runs once,
   before launch.
3. Investigate whether reducing the head node's own process/allocation
   diversity is practical (unlikely, given a TP=2 deployment
   structurally needs one rank to run the API/scheduler layer).

## Both mitigations implemented and deployed (2026-09-21)

Options 1 and 2 above, both done, per explicit user direction ("add a
periodic compaction trigger, then fix the preflight check").

**Periodic compaction trigger** (option 2): new
`recipes/scripts/compaction_flusher_remote.sh`. Unlike the existing
`cache_flusher_remote.sh` (fixed 25-minute during-load window), this
runs for the WHOLE serving lifetime of the job -- it waits (bounded,
30 min default) for a `sparkrun_*_node_*` container to appear, then
loops `echo 1 > /proc/sys/vm/compact_memory` every 300s (default,
`GLM53_COMPACTION_INTERVAL_SEC` overridable) for as long as that
container exists, exiting cleanly once it's gone rather than running as
an orphan. Logs before/after order-4/order-5 free-block counts (the
actual diagnostic signal from this investigation) to
`/tmp/glm53_compaction_flusher.log`, not the blind aggregate. Wired into
`prelaunch_flush.sh` unconditionally (not gated behind `--during-load`
like the cache flusher, since this needs to cover the whole session) --
runs on every host passed in, harmless on the worker where it's not
strictly needed. Killed-then-restarted via the same pidfile pattern the
cache flusher already uses, so re-running `prelaunch_flush.sh` doesn't
accumulate duplicate flushers.

Deployed retroactively to the already-running job too (editing
`prelaunch_flush.sh` only protects future launches): started manually
via ssh+nohup on both live hosts with zero disruption -- confirmed via
health check, container status, and uptime immediately after, all
unaffected, while real production traffic was live.

**Preflight check fix** (option 1): `frag_check_remote.sh` now reports
two lines -- the existing `nr_free_pages_blocks` aggregate (line 1,
unchanged) plus `order4_count order5_count` from `/proc/buddyinfo`
(line 2, new). `prelaunch_flush.sh`'s `check_fragmentation` now
evaluates both independently: the existing aggregate check (catches
near-total exhaustion, the 2026-09-08 NVRM/Xid-31 failure class) and a
new medium-order check (`FRAG_MEDIUM_ORDER_WARN_THRESHOLD=50` per
order, chosen the same conservative way as the original threshold --
above the observed-collapsed range of 0-8, below both the healthy
worker's 11,597-16,408 and one manual compaction's achieved 564-793;
unvalidated against a large sample same as the original threshold
initially was). Either check failing triggers the existing compaction
remedy and reports post-compaction values for both. Tested directly
against the live head node (isolated function test, not the full
script, to avoid an unnecessary `drop_caches` against live traffic):
correctly caught a real borderline case (order-4=67 fine, order-5=19
below threshold, aggregate=12288 fine) that the old aggregate-only
check would have silently passed, attempted compaction, and reported
honestly that it was still below threshold afterward rather than
claiming false success.

**Current state**: both fixes deployed and running against the live
production job with zero disruption (health 200, container uptime
undisturbed, verified after every step of this work, including the two
live compaction triggers this testing performed against the head node
while it served real traffic).

## 42-hour real-world confirmation: mitigations holding, zero new kills (2026-09-23)

Checked real metrics rather than just assuming the deploy above worked.
Production has been up continuously for **42 hours** since the fix
deployed (2026-09-21 23:44 UTC), the same job, no restart. `journalctl
-u earlyoom` across the full 48-hour window shows exactly one kill --
the pre-fix third kill (21:54:14 UTC, already documented above) --
and **zero kills since**.

**Compaction flusher log** (head node, 497 entries over 42 hours):
order-4/order-5 free-block counts settled into a stable steady state
around **~3,900/~3,440 blocks** -- two full orders of magnitude above
the `FRAG_MEDIUM_ORDER_WARN_THRESHOLD=50` floor, and comfortably above
even the worker's original healthy reference (11,597/16,408 at a fresh
boot). The only low readings in the entire log are the first two
entries, before the very first compaction pass had run. Recent entries
show before/after values nearly identical (e.g. "3908 3442 -> 3906
3439") -- the periodic cadence is now keeping fragmentation from
accumulating meaningfully between runs at all, not just cleaning up
after the fact.

**MemAvailable trend** (Prometheus, `node_memory_MemAvailable_bytes`,
30-min steps across the full 42-hour window): a smooth, gentle
oscillation between ~3.6% and ~5.5%, with a slow gradual overall decline
(started ~5.3%, now ~3.6-3.8%) but **no step-wise collapses** -- the
signature that preceded all 3 real kills (sudden multi-percentage-point
drops holding at a new, lower floor) is completely absent from this
window. Never approached the ~2% SIGTERM danger zone.

**Verdict: real, sustained improvement, not just "no crash yet."** The
slow gradual decline (~1.7pp over 42h, roughly 0.04pp/hour) is worth
continuing to watch -- if that rate holds it would take many more days
to become a concern, a fundamentally different risk profile than the
pre-fix pattern (hours, not days, to crash), but it's not yet a long
enough baseline to rule out a slower residual drift entirely. No action
needed now; revisit if the decline rate itself increases or approaches
the danger zone over a longer observation window.

## Upstream check, 2026-09-23 (delta since 09-21)

Triage round across all 8 tracked repos, read-only, scoped specifically
to what changed since the 09-21 round. Framed as candidates for the
next production maintenance window, not immediate action -- production
is currently stable (see the 42-hour confirmation above) and nothing
here is urgent enough to justify a restart on its own.

**Cheapest possible win, verify before the other two**:
`GLM53_EXL3_MOE_FAST` (MiaAI-Lab PR #217) is **already backported into
this fork's own overlay** (`overlay/patch_exl3_decode_pipeline.py`,
present in both `v20-upstreamsync` and `v21-b12xmoe` build trees) --
confirmed by reading the file directly, not just the upstream check's
claim. It is NOT set in the current production yaml, so it's sitting
present-but-off. MiaAI-Lab's new TP=2 reference profile (PR #255,
`examples/tp2-long-coding.env`, merged 09-23) reports **+7.7% to
+14.6% decode tok/s specifically on TP=2** for this exact mechanism --
this fork's own PR #217 backport docstring already flagged the
original numerical study as "formally INCONCLUSIVE" and untested
against this fork's own checkpoint/workload, so this is a real,
actionable, zero-new-code opportunity: flip the flag, A/B it with
`probe_throughput_ab.py` (same methodology as every other A/B this
project has run), and decide. No new engineering required at all.

**Three other candidates, ranked by the checking agent**:
1. **mmastrac's `glm53_reasoning_always_parsed.py` root-cause** (new
   09-23 commit) -- directly adjacent to this fork's own pending,
   still-undeployed "Chat template reasoning-leak fix" (fixed in build
   tree 2026-09-09, never shipped). Their diagnosis: stock vLLM's
   `glm47_moe` parser sets `self.thinking_enabled = False` whenever
   `enable_thinking` is false, which stops it from tracking `<think>`
   and starts it in `CONTENT` state -- so the reasoning block (including
   literal "Reasoning Effort" text) lands unparsed directly in the
   visible `content` field. Their fix forces `thinking_enabled = True`
   unconditionally, since the template always opens `<think>` regardless
   of the kwarg. **Worth cross-checking against this fork's own 09-09
   fix before deploying it** -- may be the identical root cause, or may
   reveal the existing fix only patched the leaked-text symptom rather
   than the parser-state cause. Needs the same downtime already pending
   for the original fix, so no additional cost to bundle this check in.
2. **MiaAI-Lab `GLM53_DRAFT_KV_COMPACT`** (PR #238/#255, merged 09-23) --
   real TP=2 numbers from the same reference profile above: 32.78% fewer
   reserved DFlash2 cache IDs, a 28,672-token-prefix-plus-64-token-tail
   repeat 90% faster, costs ~1% slower prose decode / 1-3% slower cold
   long prefills. Default off upstream. Requires the FLASH_ATTN backend
   at the tested 896-token derived block size -- verify this fork's own
   backend selection before assuming compatibility. Worth an isolated
   A/B, moderate effort (new cache-layout mechanism, not a flag on
   existing code like MOE_FAST above).
3. **Reederey87's two APC patches** (`patch_apc_per_group_retention.py`,
   `patch_fine_grained_apc.py`) -- real, mature, fail-closed, numbers
   behind both, but **neither is production-default even in Reederey87's
   own deployment** (per their own repo's header comments). Per-group
   retention patches `vllm/v1/core/kv_cache_coordinator.py` internals --
   real engineering review needed before considering, not a quick
   backport. Lower priority than the two above given the extra review
   cost.

**Confirms prior conclusions, no action**: mmastrac deleted their own
`gb10_topk_fallback.py` and `thinking_budget_guard.py` patches this
round, replaced by a stock flag and an upstream fix respectively --
independent confirmation this fork's own "GB10 TopK: already safe"
(09-16) and the thinking_token_budget fix chain
([[thinking_budget_v2_confirmed_2026_09_21]]/[[thinking_budget_fixed_2026_09_21]])
landed on the right answer. `local-inference-lab/b12x` has shipped **no
PyPI release past 1.3.0** (confirmed directly against `pypi.org/pypi/
b12x/json`, still dated 2026-08-25) -- this fork's fixed-size-arena
workaround for the CUDA-graph-capture bug remains un-obsoleted by
upstream. b12x's other ~20 commits since 09-21 are all in W4A8/MXFP4
dense-GEMM, a new RoCE comms proxy, and context-parallel attention --
none touch the W4A16/EXL3 trellis path this fork actually uses.

**Quiet, nothing to do**: Enntity/sparkglm's `exl3` branch (still last
touched 09-08; their `main` branch had real 09-22 activity but it's
their NVFP4/MXFP8 stack, off-target for this EXL3 fork -- one concept
worth remembering if host memory pressure ever worsens further: an
NVMe-backed prefix-cache offload, `overlay/patch_nvme_prefix_cache.py`).
AEON-7 (quiet since 09-12). cbertucci33 (real pace change -- went from
29 commits/5 releases in 10 days to a hard stop at 09-20, nothing since
-- worth knowing the project may have gone quiet, no content to act on
either way). Entrpi (both repos quiet since 09-02, confirmed no new
memory-fragmentation/buddyinfo/compaction content to compare against
this fork's own recent root-cause work). r0b0tlab (nothing since
09-21, same snapshot already scouted).

## GLM53_DRAFT_KV_COMPACT: NOT APPLICABLE, no backport (2026-09-23)

Chased this candidate from the 09-23 upstream check and found it
doesn't transfer. Read MiaAI-Lab PR #238's actual body/diff directly
(not the secondhand summary from the triage pass): the mechanism is
exclusively about **DFlash2's** sliding-window drafter KV pages --
every description in the PR is scoped to "DFlash2's shared MLA pages,"
and its largest diff hunk is in `overlay/patch_glm5_drafter_group.py`.

This fork's own copy of that exact file
(`recipes/build/glm53-exl3-v21-b12xmoe/overlay/patch_glm5_drafter_group.py`)
is already 100% DFlash2-specific in its own right -- its docstring
says "teach the GLM-5-Next KV layout about the DFlash2 drafter's
SlidingWindowSpec layers," every internal reference is DFlash2-only,
and it has zero mentions of MTP anywhere. This fork runs MTP-2, whose
speculator uses a structurally different `KpoolTailSpec`-style KV
group (see `patch_kpool_tail_slotmap.py`/`patch_kpool_tail_seed_stride.py`),
plumbed through entirely separate overlay code. There is no DFlash2
sliding-window drafter group in this deployment for PR #238's
compaction to act on -- the code path it optimizes is simply never
exercised at runtime here.

Second, independent disqualifier: PR #238's own numbers were measured
requiring the FLASH_ATTN attention backend at the tested block size.
This deployment uses FlashInfer (`FLASHINFER_MLA_SPARSE_SM120` at
runtime, `--no-enable-flashinfer-autotune`), not FLASH_ATTN.

**Verdict: not applicable, no backport, no further action.** A clean
negative result -- not every real upstream mechanism transfers to
every downstream deployment, and forcing this one in would mean
patching code for a code path (DFlash2 drafter KV groups) that
doesn't exist in this fork's own runtime at all.

## Reederey87's two APC patches: neither worth adopting, two different reasons (2026-09-23)

**`patch_apc_per_group_retention.py` -- NOT APPLICABLE**, same class of
reason as `GLM53_DRAFT_KV_COMPACT` above: gated entirely on an
"EAGLE-exempt drafter group," a separate SlidingWindowSpec KV group
DFlash2/EAGLE-style speculators register but MTP does not. Confirmed
via this fork's own `patch_glm5_drafter_group.py` (explicitly named
for the DFlash2 drafter, not MTP) and by grepping this file's entire
own incident history in this document for that patch's own log message
("DFlash2 drafter KV: ...") -- zero occurrences, meaning that code path
has never once fired under this fork's real `method:"mtp"` config. MTP
shares the main model's MLA/KV structure directly rather than
registering a separate drafter group, so there's no "drafter group
evicting the shared LRU pool" phenomenon here to fix. Confirmed
directly from Reederey87's own source that this patch is explicitly
marked "NOT APPLIED" in its own first docstring line -- their own
`docs/DESIGN-apc-per-group-retention.md` reference 404'd (likely
renamed/moved), so their own reasoning for keeping it staged-off
couldn't be confirmed directly, but the patch's fail-closed design
(validates unconditionally at coordinator init, boot-fails rather than
silently no-ops on a bad value) reads as "real but wants more soak,"
not "known broken."

**`patch_fine_grained_apc.py` -- applicable-but-currently-a-no-op,
not a quick pickup**. The underlying vLLM bug it fixes is real and
confirmed present, verbatim, in this fork's own vendored source:
`kv_cache_coordinator.py`'s `unsupported_partial_hit_managers` veto
check exactly matches Reederey87's own `VETO_OLD` anchor (checks
`manager.block_size != hash_block_size` with no
`participates_in_prefix_caching` exemption) -- this fork has not
received upstream PR #125's fix. This fork's own `KpoolTailSpec.
participates_in_prefix_caching = False` property already exists with a
docstring explaining it exists precisely to avoid this class of
problem -- but the veto logic doesn't currently consult that property,
so the existing intent is technically undermined by unpatched code.
**However**: the veto block only executes when
`self.enable_partial_hash_hits` is `True`, which requires
`mamba_cache_mode == "align"` -- checked directly against
`vllm/config/cache.py` and this fork's own yaml/overlay, neither of
which ever sets this (vLLM's own default is `"none"`). So the buggy
veto code is dead in this fork's current production config today; this
patch would only become worth adopting if this fork separately decided
to adopt `mamba_cache_mode="align"` for some other reason -- a distinct,
unexplored architectural change entirely out of scope for this review.
**Verdict: note the gap, don't backport in isolation now** -- there's
nothing live for it to fix yet.

## GLM53_EXL3_MOE_FAST A/B: tested, rejected -- b12x wins (2026-09-24)

Production traffic was off for the night, so tested this directly on
real hardware rather than just flag-flipping blind. `GLM53_EXL3_MOE_FAST`
cannot be tested as an additive flag on top of the current production
config -- confirmed by reading `apply_exl3_experts`'s dispatch order
first: it checks `exl3_b12x_moe_requested()` and returns immediately via
`apply_exl3_b12x_moe` if true, never reaching the `fused`/`MOE_FAST`
branch below. Since production runs `GLM53_EXL3_B12X_MOE=1`, MOE_FAST is
dead code in the current config. Tested the only meaningful comparison
instead: `GLM53_EXL3_B12X_MOE=0` + `GLM53_EXL3_MOE_FAST=1` (falls back
to the old exl3 fused pipeline with the fast-decode kernel) against the
current b12x baseline, same image (`glm53-exl3-v21-b12xmoe:local`, same
weights, same everything else).

**Hit one transient crash getting there**: first boot attempt died with
`torch.AcceleratorError: illegal memory access` during
`self.speculator.capture()` -> `prepare_inputs_to_capture` ->
`<kpool_tail_patch:prepare_attn>` -> `sparse_mla_attention.py`'s
`_build_req_id_per_token` -> `pin_memory()`. Took this seriously rather
than dismissing it immediately, since the traceback passes through this
fork's own recently-added `kpool_tail_seed_stride` patch -- but a clean
retry (same config, same image, no changes) booted and served correctly
with zero errors, confirming this was the same already-documented class
of ambient, non-deterministic GB10 illegal-memory-access flakiness this
project has hit repeatedly for unrelated reasons, not a real bug in the
kpool patch. Crash logs saved to scratchpad for reference if this
signature ever needs to be revisited with more evidence.

**Result** (`probe_throughput_ab.py`, n=20, identical methodology):
MOE_FAST-without-b12x median decode **32.913 tok/s** (stdev 0.721) vs.
the current b12x production baseline's **34.226-34.38 tok/s** (stdev
0.082-0.133, from repeated measurements this week) -- b12x is ~4%
faster AND 5-9x less noisy run-to-run. **Verdict: rejected, keep b12x
as production.** Makes sense in hindsight: MOE_FAST's upstream-reported
+7.7% to +14.6% gain was measured against the *stock* exl3 fused
kernel, not against b12x -- an entirely different, independently-built
kernel approach this fork integrated separately. Beating stock and
losing to b12x are not in tension; b12x remains the better choice for
this fleet specifically.

Production restored to the standard `v21-b12xmoe` yaml
(`GLM53_EXL3_B12X_MOE=1`) unchanged, validated with a real request
before considering this closed.

## Reasoning-leak cross-check: already resolved, no action needed (2026-09-24)

Chased the mmastrac `glm53_reasoning_always_parsed.py` candidate against
this fork's own long-pending, memory-flagged "not yet deployed" 09-09
chat-template fix -- turned out the memory note itself was stale, and
the deeper question mmastrac's fix raises doesn't apply here either.

**First correction**: this fork's own 09-09 template fix (gating the
`Reasoning Effort: {{ effort }}` header line on `thinking_enabled` in
`chat_template.jinja`) IS already present in the current production
build tree (`recipes/build/glm53-exl3-v21-b12xmoe/files/chat_template.jinja`),
confirmed by reading the file directly -- byte-identical to the fixed
version in the old `v15-combined` tree the memory note referenced. It
propagated forward naturally as new build trees were created by
copying the latest one at each point; the memory note's "not yet
booted/deployed" status was accurate on 09-09 but went stale without
being corrected as later build trees shipped it.

**Second, deeper question -- does mmastrac's parser-level root cause
apply here too?** Confirmed the underlying vLLM parser code IS shared:
this fork's `--reasoning-parser glm45` resolves to the exact same file
mmastrac's fix targets (`vllm/parser/glm47_moe.py`, confirmed via
`docker run --rm --entrypoint bash glm53-exl3-v21-b12xmoe:local`), and
it has the identical `thinking_enabled`-gated behavior mmastrac
described (`extract_reasoning`/`extract_content_ids`/`is_reasoning_end`
all short-circuit to "treat everything as content, no reasoning
extraction" when `thinking_enabled` is false). **But this fork's own
chat template structurally prevents the scenario that would trigger
it**: when thinking is disabled, this fork's template inserts an
already-closed, empty `<think></think>` pair into the prompt *before*
the model's turn begins generating (`'<think>' if thinking_enabled else
'<think></think>'`) -- there is no open, unclosed reasoning span left
for the model to fill or for the parser to mishandle. Mmastrac's own
template, per their diagnosis, always opens `<think>` unconditionally
regardless of the kwarg, leaving a real reasoning span for their
parser's identical "just treat everything as content" behavior to
mishandle. Same shared bug in the code, different template design
means it's structurally unreachable here.

**Verified empirically, not just from code reading**: sent a real
default-config (thinking disabled) request after restoring production
-- clean output (`"content": "Paris"`, `"reasoning": null`), no leaked
text of any kind.

**Verdict: already correctly resolved, no fix needed.** Nothing to
deploy, nothing to change. Worth remembering the general shape of this
outcome: a shared upstream bug can be present in code two projects both
run, yet only actually manifest in one of them depending on how a
completely different piece (the chat template) is designed -- worth
checking the whole interaction, not just whether the buggy function
exists.

## DFlash2 vs MTP-2 A/B: rejected -- and a real production incident (2026-09-24)

Tested whether to switch this fork's speculative-decoding method from
MTP-2 to DFlash2/EAGLE, using a genuinely compatible drafter checkpoint
(`incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd`). The question itself is
now closed; getting the answer cost roughly 10 hours of real production
disruption on this fleet's single shared serving cluster, which is
documented here in full because the process failure is at least as
important as the result.

**The result**: DFlash2 rejected, keep MTP-2. Throughput A/B
(`probe_throughput_ab.py`, same methodology as every other A/B this
project runs): DFlash2 decode median **27.656 tok/s** (steady-state
runs 27.603-27.810 tok/s, n=9 of a planned 20 -- see incident narrative
below for why it stopped short) vs. this week's MTP-2 production
baseline of **34.226-34.38 tok/s** -- a **-19.2% to -19.6%** regression.
TTFT was at parity (0.2469s vs. 0.246-0.257s baseline), so the
regression is decode-only, not a context-length or cold-start artifact
-- DFlash2's test config used a smaller `max_model_len` (24576 vs.
production's 262144, forced by a real KV-budget constraint, see below),
which if anything favors DFlash2 on a decode benchmark, so -19% is a
generous reading, not an inflated one. Correctness was clean on 3
diverse real requests (factual, code-gen, prose). Accept rate was not
captured (the test deployment was replaced before `/metrics` could be
queried -- see incident narrative). vLLM's own startup logging flagged
the config as suboptimal (`max_num_scheduled_tokens=7168`, suggesting
either a higher `max_num_batched_tokens` or fewer than the drafter's
default `num_speculative_tokens=7`) -- so the correct framing is "not
promotable as configured," not a blanket "DFlash2 is categorically
slower." This is directionally consistent with the third-party evidence
already on file (cbertucci33's own DFlash2 numbers measured worse than
their MTP-2, in the 2026-09-21 upstream check) and inconsistent with
Entrpi's own +21% DFlash2-vs-their-MTP-4 claim -- different baselines,
different fleets, not a contradiction requiring further chasing.

**A real KV-budget finding, worth remembering on its own**: DFlash2's
drafter model needs its own KV-cache group (structurally, per
[[draft_kv_compact_not_applicable_2026_09_23]]'s finding that this is a
DFlash2/EAGLE-specific mechanism MTP-2 doesn't have), and at this
fleet's production `gpu_memory_utilization=0.82`, that extra group does
not leave enough KV budget to support the full `max_model_len=262144`
production context -- boots at full context threw a hard
`ValueError: ... 7.61 GiB KV cache is needed ... available (6.2 GiB)`.
Forcing `max_model_len` down to 24576 fixed it. Whether the full
context is reachable via an explicit `--kv-cache-memory` override was
flagged as an untested lever, not chased further since the throughput
result alone was already decisive against promotion.

**The incident**: this A/B was delegated to a fork with instructions to
wire up, boot, validate, and A/B test DFlash2 against the cluster's
single shared production serving slot (the same two-host TP=2 cluster
that serves real user traffic -- this fleet has no separate test
environment). What actually happened, reconstructed from both the
fork's own draft report and direct process/log inspection after the
fact:

1. The fork's first two boot attempts crashed -- one ambient GB10
   flakiness (ordinary, clean retry), one the real KV-budget
   `ValueError` above. Both expected, unremarkable in isolation.
2. The task went quiet for hours, reporting only terse "Waiting."
   status updates with little visible progress. A "completed"
   task-notification arrived at one point with a thin, clearly-interim
   result body ("Waiting for health check to complete in the
   background.") -- this was read as evidence the fork had stopped and
   its resources (the running test container, its launcher process)
   were abandoned.
3. Direct verification at that point found real production had not
   been serving correctly since roughly 02:52 UTC (~10 hours): the
   test container showed `docker ps` status "Up" throughout (as
   always -- sparkrun containers are a `sleep infinity` shell
   regardless of whether the actual server process inside is alive),
   but the vLLM process itself had exited hours earlier on the
   KV-budget crash and nothing had restarted it. This part of the
   finding was correct and the fix (stop the dead job, verify the
   production yaml on disk was untouched, run the standard preflight
   checks, relaunch production, validate with a real request) was the
   right call regardless of what happened next.
4. What happened next: the fork, unaware of this intervention, had
   independently retried with the corrected 24576-ctx config (its own
   "attempt 3"/"attempt 4" in its report), booted clean, validated
   correctness, and had a real 20-run throughput A/B already 9 runs
   in -- genuinely live, genuinely producing the data above -- at the
   same time production was being manually restored. The two actions
   collided on the one shared cluster slot: what looked, from outside,
   like an "orphaned retry process" (a `sparkrun run` pointed at the
   test yaml, reparented to init after its parent shell exited) was
   in fact the fork's own live, in-progress, legitimate test. It was
   killed, along with its throughput-probe process, believing both to
   be abandoned zombies. This is exactly the event the fork's own
   report flags as an unexplained anomaly ("container vanished with
   zero local crash evidence, replaced by production config under the
   same reused job ID") -- it has a complete, mundane explanation, and
   the fork's own report was right to flag it as suspicious rather
   than attribute it to DFlash2 itself.
5. Net effect: the DFlash2 A/B stopped 11 runs short of its planned 20
   and lost its shot at capturing accept-rate metrics, but the
   steady-state spread already collected (~0.2 tok/s across 7 runs)
   was tight enough that the 19% gap was already decisive -- the
   throughput conclusion stands. Production ended up in the correct,
   verified, stable MTP-2 state either way, confirmed via a real
   completion request and several minutes of repeated health +
   container-identity checks (not just a single health-check pass) to
   rule out a further silent swap.

**Process lesson, the important part**: a task-notification's
"completed" status, especially with a thin or clearly-interim result
body, is not sufficient grounds to treat a delegated task's resources
(running containers, background launcher processes) as safely
abandoned -- verify the underlying work actually looks finished, or at
minimum check for genuinely live background children, before taking
destructive action on shared production infrastructure. More broadly:
any future delegated task that temporarily reconfigures this fleet's
one shared production cluster for testing needs an explicit wall-clock
budget and an unconditional "restore production" step on every exit
path (success, failure, or budget exhaustion) baked into the task
itself -- this run went unattended for 10+ hours on the single
prod-serving slot with no such guardrail, which is the real root
enabler of the incident, independent of which specific action
ultimately disrupted which specific attempt. See
`dflash2_ab_incident_2026_09_24` in memory for the fuller process
writeup.

## MTP-3 vs MTP-2 A/B: rejected, keep MTP-2 (2026-09-25)

Follow-up to the DFlash2 rejection above: this fleet's own Entrpi
triage (2026-09-18) had flagged bumping MTP's `num_speculative_tokens`
from 2 to 3 as a real, unactioned backport candidate, on the reasoning
that Entrpi's own DFlash2-vs-MTP comparison used their MTP-4 as the
baseline -- weaker (fewer proposed tokens) than even their own
production choice, let alone worth leaving untested here. Tested it
directly: same production yaml, same checkpoint, single-line config
change (`num_speculative_tokens: 2` -> `3`), no new drafter model, no
extra KV-cache group (this is still native MTP, not DFlash2 -- the
model's own next-token-prediction head just runs one more forward
pass), so none of the KV-budget complications from the DFlash2 test
applied here.

**Result**: rejected. Throughput A/B (`probe_throughput_ab.py`, n=20,
identical methodology): MTP-3 decode median **32.707 tok/s** (stdev
0.28, full 20/20 clean run, no crashes) vs. this week's MTP-2
production baseline of **34.226-34.38 tok/s** (stdev 0.082-0.133) --
a **-4.4% to -4.9%** regression. TTFT at parity (0.244s median vs.
0.246-0.257s baseline). Correctness clean.

**Real acceptance-rate data captured this time** (via `/metrics`,
`vllm:spec_decode_num_accepted_tokens_per_pos_total`): of 3168 draft
attempts, per-position acceptance was position 0 = 74.1% (2347/3168),
position 1 = 51.2% (1623/3168), position 2 = 29.1% (921/3168) --
overall 51.5% of proposed tokens accepted (4891/9504), averaging 1.544
accepted tokens per speculative step out of 3 proposed. This is a
direct, measured confirmation of the exact tradeoff vLLM's own startup
warning describes for `num_speculative_tokens > 1`
("running multiple times of forward on the same MTP layer... may
result in lower acceptance rate"): the third speculative position adds
real compute cost (proposing + verifying a token) but only pays off
29% of the time, and the net effect nets negative on this hardware/
workload. **Verdict: not worth it, keep MTP-2 (num_speculative_tokens=2)
as production.** Both MTP-3 and DFlash2 are now closed questions for
this fleet -- MTP-2 remains the right choice on both counts, for
related reasons (diminishing per-position acceptance as more tokens
are proposed per step).

Production restored to the standard `v21-b12xmoe` yaml unchanged,
validated with a real request and clean logs on both nodes before
considering this closed. This test was driven directly rather than
delegated to an unattended fork, given the process lesson from the
DFlash2 incident immediately above -- single boot cycle, real-time
supervision throughout, no incident.
