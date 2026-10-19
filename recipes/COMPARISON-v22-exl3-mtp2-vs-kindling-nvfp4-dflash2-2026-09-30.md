# v22-cacheevict (EXL3 + b12x + MTP-2) vs. kindling (NVFP4 + DFlash2): before/after

**Date:** 2026-09-30 · **Host(s):** spark-2dd4 (head, 10.7.0.87) / spark-276f (worker, 10.7.0.142) · **Status:** kindling serving on :8000 since 2026-09-30 09:36 (real production traffic from ~11:35); promotion decision made by the operator on the strength of the prefill/decode numbers below. v22 stays in the repo as a fallback. **Updated 2026-10-10:** the live build is now kindling main c748079 (promoted 2026-10-09 16:25 EDT); see section 5 for its 26 h live numbers. Sections 1-4 describe the 2026-09-30 build (a4671d2) unless marked.

Both stacks serve the same model family (GLM-5.3-Flash, TP=2 over two GB10 nodes) under the
same served name `glm-5.3-flash-exl3-v2` on :8000, so clients did not change.

The per-request story is in `VALIDATION.md` ("Three-way A/B ..." and "Kindling memory
follow-up"); this file is the side-by-side summary.

## 1. Configuration: before vs. after

| Area | Before: v22-cacheevict | After: kindling | Changed? |
|---|---|---|---|
| Kernel / driver | 7.0.0-1019-nvidia, driver 580.178.04 | same | no |
| Host sysctls | `vm.min_free_kbytes` 1 GiB, `watermark_scale_factor` 100, swappiness 1 | same | no |
| earlyoom | `-m 2 -s 80`, original prefer/avoid lists | same args | no (but see swap) |
| Swap | 16 GiB `/swap.img` per node | 64 GiB per node (added 48 GiB `/swap2.img`, in fstab) | yes |
| Inter-node network | both ConnectX roots addressed (192.168.177.x), NCCL pinned to both HCAs | one root only (`enp1s0f0np0`); second-root addresses removed | yes |
| Extra services | none | `mmastrac/mentatd:0.17.1` container on each node | yes |
| Image | `glm53-exl3-v22-cacheevict:local`, built from Mia's c190db1 base (vLLM ~0.29 lineage, not re-verified) | `spark-glm53:v10` (v9 until 2026-10-09), vLLM 0.30.1rc1.dev193 (nightly ddd6fbca), torch 2.13.0+cu130, triton 3.7.1, Python 3.12.3 | yes |
| Weights | Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw, quantization `exl3` | nvidia/GLM-5.3-Flash-NVFP4 (190 GiB per node, local disk) | yes |
| Speculative decoding | MTP-2 (2 tokens) | DFlash2 drafter (incoai/GLM-5.3-Flash-DFlash2, revision bf582e4 since 2026-10-09; 7d74cdd before), up to 7 tokens, per-batch-size schedule, TRITON_ATTN drafter backend | yes |
| Scheduler | stock | `AdaptiveKScheduler` (vllm.v1.core.sched.adaptive_k) | yes |
| MoE path | b12x fused-trellis MoE (`GLM53_EXL3_B12X_MOE=1`) | vLLM MegaMoE (`VLLM_MEGAMOE=1`, variant 20, prefill Y8, prefill min 1024 tokens) | yes |
| Attention / SP | our GLM53_* patch set: FlashKDA prefill, FP8 LM head, dense FP8, indexer workspace "rightsize", FlashInfer | `VLLM_GLM_SP_TP=1` with fused + quantized MoE gather, draft pool | yes |
| KV cache | FP8, sized by `gpu_memory_utilization` 0.82 | pinned 5.6 GiB (`KV_CACHE_MEMORY=6012954214`); gmu ignored once KV is pinned (set 0.86) | yes |
| Max context | 262,144 | 163,840 | yes |
| `max_num_seqs` / batched tokens | 16 / 7,168 | 16 / 8,192 | only batched tokens |
| Prefix caching | depth-first tail eviction (`GLM53_CACHE_TAIL_EVICT=1`) | stock | yes |
| Reasoning output | thinking off, reasoning-leak fix in our chat template | emits a reasoning span even with `enable_thinking=false` | yes |
| Served name / port | `glm-5.3-flash-exl3-v2` on :8000 | same | no |
| Launch | `sparkrun run recipes/glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml` after `prelaunch_flush.sh` | `recipes/kindling/kindling.sh start` (runs `~/work/kai-main/glm53 up -d`, worker first then head, after `mentatd` is up) | yes |
| Licence | n/a | DFlash2 drafter is CC BY-NC-ND | yes |

Kindling source: `github.com/kindlingai/glm-5.3-flash-gx10` @ c748079 in `~/work/kai-main` (2026-09-30 build: a4671d2 in `~/work/kai`; same content as
`mmastrac/glm-5.3-flash-4x-gx10`), submodule `mmastrac/spark-agent` @ 8ee80bf, mentat 0.17.1
(`mmastrac/mentat`). Serving files live in `recipes/kindling/` (copies in
`~/ai/research/glm-5.3-flash-gb10/kindling-nvfp4-dflash2/serve/`).

## 2. Performance: before vs. after

### 2a. Live production traffic (Prometheus, `vllm:*` metrics)

Before = 11 h of v22 production traffic (945 requests, avg prompt 117K tokens).
After = 100 min of live traffic on kindling (~75 requests, avg prompt 56K tokens, c=2).
The mixes differ, so per-token figures are more reliable than TTFT.

| Metric | Before (v22) | After (kindling) | Change |
|---|---|---|---|
| Prefill, tokens per second of prefill time | 1,462 | 2,127 | +45% |
| TTFT p50 | 4.0 s | 4.1 s | flat (prompts half the size) |
| TTFT p90 | 13.7 s | 9.1 s | -33% (mix differs) |
| Decode per request | 34.9 tok/s | 32.0 tok/s | -8% |
| Inter-token latency | 76 ms | 81 ms | +6% |
| Prefix-cache hit ratio | 92.8% | 83.1% | -10 points |

### 2b. Controlled A/B (same harness, `recipes/probes/ab_stack.py`, distinct ~16K-122K documents)

| Metric | v22 (prod) | kindling | TensorFold (3rd arm, for reference) |
|---|---|---|---|
| Cold prefill 16K-122K | 1,275-1,791 tok/s | 2,500-2,770 tok/s | 1,534-1,620 tok/s |
| 32K x4 concurrent, aggregate | n/a | 2,435 tok/s | ~1,500 tok/s |
| 122K x3 concurrent, aggregate (soak) | n/a | 2,580-2,607 tok/s | n/a |
| Decode, 1 request | 26-33 tok/s | 42-44 tok/s | 46-58 tok/s |
| Decode, 4 concurrent, aggregate | n/a | 68.6-72.4 tok/s | 129 tok/s |
| Warm 122K replay TTFT (single) | n/a | ~3.0 s | 0.3 s |
| Needle retrieval | pass | pass (few misses from 64-token cap eaten by the reasoning span) | pass |

The v22 prod baseline numbers in this table come from the earlier measurement band
(prod was contaminated by live traffic during the matched run).

Soak: 48 batches over ~80 minutes, 0 request errors, needle 129/132, prefill 2,580-2,607 tok/s
(122K x3), decode 68.6-72.4 tok/s (x4).

### 2c. System utilisation

| Metric | Before (v22) | After (kindling) |
|---|---|---|
| Min MemAvailable, worker / head | 8.6 / 5.1 GiB | 0.38 / 0.29 GiB |
| Avg MemAvailable, worker / head | 8.9 / 5.5 GiB | 1.25 / 1.03 GiB |
| Peak swap used, worker / head | 3.2 / 3.5 GiB of 16 | 1.9 / 3.3 GiB of 64 |
| earlyoom kills | not checked for that window | 0 (~3.5 h serving incl. soak) |
| Request errors / restarts | not checked for that window | 0 / 0 |
| KV cache usage | n/a | 8-16% at c=2 |

## 3. Caveats that matter

- **Memory margin is gone.** Kindling runs with ~0.3-1.3 GiB MemAvailable. The kernel low
  watermark (~2.2 GiB with our sysctls) is subtracted, so the node is healthy at low
  MemAvailable, but there is no slack for a CUDA allocation spike. With 64 GiB swap,
  earlyoom's `-s 80` rule needs ~12.8 GiB of swap in use before it can act (was ~3.2 GiB),
  so earlyoom is effectively inert; `-s 95` would restore the old trigger.
- **The live decode gap (-8%) is a concurrency artifact, not a slower decoder.** Step time
  (inter-token latency, which vLLM records per engine step, ~2.7 tokens/step on both stacks) at
  equal concurrency is the same: p50 at c=1 is 67 ms on kindling vs 63 ms on v22. The 11 h v22
  window never had a moment with 2 or more requests running (clients were serial); kindling live
  traffic runs c=2-6, where step time rises with the number of long-context sequences in the
  batch (p50: c=2 93 ms, c=3 126 ms, c>=4 167 ms). Per-request speed drops with concurrency while
  aggregate rises: ~40 tok/s at c=1, ~58 at c=2, ~64-65 at c>=3 (kindling). No matched-concurrency
  v22 live data exists. The earlier "32.0 vs 34.9" also drifted with mix: the last 2 h show 23 tok/s
  per request (avg 2,078 generated tokens, 44K prompts, c~2+).
- **Context ceiling is 163,840** vs 262,144. Traffic seen so far (~122K prompts) fits.
- **Single ConnectX root (effect unmeasured).** Kindling runs with the second root
  (`enP2p1s0f0np0`) carrying no IPv4; it is re-addressed only when switching back to v22.
  Upstream's numbers use both roots (126K prefill 2,412 -> 2,680 tok/s, about +11%; decode
  unchanged) but require each root in its own subnet, MTU 9000 and the same RoCE GID index.
  Our v22 addressing puts both roots inside 192.168.177.0/24, which does not satisfy that. Our
  own data is confounded: the first kindling run (second root temporarily in 192.168.178.0/24)
  measured 2,500-2,770 tok/s vs. 2,454-2,607 at 122K on later single-root runs, with the KV pin
  and other changes in between. A proper dual-root A/B needs a maintenance window.
- **Drafter licence is non-commercial, no-derivatives.**
- **Reasoning span is always emitted**, so short `max_tokens` caps can be eaten by reasoning.
- **Not yet validated:** a real reboot (the boot launcher now defaults to kindling but only its
  `--dry-run` paths were exercised), a real `switch_stack.sh` run, multi-day stability,
  behaviour at c>6. A kindling watchdog exists (auto-restart after a 10 min delay, Slack, Grafana rules) but its restart and Slack paths have not been exercised.

## 4. Rollback (to v22)

`recipes/scripts/switch_stack.sh v22` does all of the following (use `--dry-run` first; it refuses
while requests are in flight unless `--force`). Manual equivalent: stop kindling
(`recipes/kindling/kindling.sh stop`), restore the second-root addresses
(`sudo ip addr add 192.168.177.87/24 dev enP2p1s0f0np0` on head, `192.168.177.142/24` on worker),
`recipes/scripts/prelaunch_flush.sh 10.7.0.142,10.7.0.87 --during-load`, then
`nohup sparkrun run recipes/glm-5.3-flash-exl3-v22-cacheevict-vllm.yaml > <log> 2>&1 &`
and poll `/health` (never wrap the launch in `timeout`).

## 5. Update 2026-10-10: kindling main c748079, 26 h of live traffic

Build change on 2026-10-09 16:25 EDT: kindling main `c748079` (`~/work/kai-main`, `./glm53`), image `spark-glm53:v10`,
drafter `bf582e4` (`/models/kindling/dflash2-bf582e4`), cache `/models/kindling/cache-main`. The matched A/B beforehand
read as roughly equal (details in `VALIDATION.md`, "A/B result (2026-10-09 ...)"); the operator promoted it expecting
2-5 concurrent requests to favor it. Numbers below are Prometheus `vllm:*` over the 26 h after promotion (queried
2026-10-10 19:41 EDT) against the 8 days before it.

| Metric | Previous build (a4671d2 + v9 + 7d74cdd), 8 d, 7,309 req, avg prompt 66K | Current build, 26 h, 885 req, avg prompt 45.9K | Read |
|---|---|---|---|
| Prefill, tok/s of prefill time | 1,920 | 2,048 | +7% |
| Decode per request, raw | 23.6 tok/s | 30.6 tok/s | prompt mix and concurrency, not a speedup |
| Step time p50, 1 / 2 / 4+ running | 66.6 / 91.3 / 149.8 ms | 62.8 / 86.5 / 130.5 ms | -6% / -5% / -13% |
| Tokens per step | 2.78 | 2.60 | lower |
| Draft acceptance (accepted/drafted) | 0.506 | 0.470 | lower; per position .715 .460 .211 .104 .054 .030 .026 |
| Implied decode per request (tokens per step / step time), c1 / c2 / c4+ | 41.8 / 30.5 / 18.6 | 41.4 / 30.1 / 19.9 | about equal |
| TTFT p50 / p90 | 3.29 / 7.14 s | 1.53 / 5.94 s | prompt mix differs (66K vs 46K), not comparable |
| Prefix-cache hit ratio | 88.6% | 87.5% | flat |

For reference, v22 live (11 h, 945 req) was prefill 1,462 and decode 34.9 tok/s per request at c=1 (section 2a), so the
cumulative live prefill gain from v22 to the current build is about +40%.

Reliability over the window: containers up 27 h on both nodes, request_success since boot stop 734 / length 306 /
error 0 / abort 0, 0 earlyoom kill lines since 2026-10-09 16:00, swap used 4.4 GiB (head) and 1.9 GiB (worker) of 64.
**Memory:** `MemAvailable` p1 and minimum over the window are 0 on both nodes (the kernel clamps it to 0 under
low-watermark pressure; it sat below 64 MiB for 216 min on the head and 143 min on the worker). Spot readings now: head
719 MiB, worker 534 MiB. Watchdog flags clean. The margin caveat in section 3 therefore holds in a stronger form:
healthy, not killed, but with no measurable slack.

What the data supports: the bump delivered prefill (+7%) and a shorter engine step at 4+ running (-13%), offset by lower
draft acceptance, leaving decode per request level. What it does not support: any claim that decode per request rose
(the 23.6 to 30.6 raw gap is traffic mix). Open check from the A/B: the single c1 32K prefill dip (1,884 vs 2,356
tok/s) was never re-measured; acceptance trails by 3.6 points. If either persists, roll back with the paths below.

Rollback to the previous kindling build (kept on both nodes): `~/work/kai` (a4671d2) + image `spark-glm53:v9` +
drafter `/models/kindling/dflash2` (7d74cdd) + cache `/models/kindling/cache`. Edit `recipes/kindling/kindling.sh`
(`KAI_DIR`, `IMAGE`, `KAI_PIN`, drafter preflight) and the `env/kai.{head,worker}.env` files back to those values, then
`kindling.sh stop` and `kindling.sh start --force`. Do not run the old overlays on image v10 or the new overlays on v9;
overlays and image must move together.
