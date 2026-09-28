#!/usr/bin/env python3
"""Flood-then-replay prefix-cache eviction pressure test.

Mirrors Reederey87/glm53-flash-exl3-2x-dgx-spark's own methodology for
measuring GLM53_CACHE_TAIL_EVICT / GLM53_CACHE_HOT_PROTECT (two ~46K-token
paused "agents", eighteen ~31K-token one-shot floods, then replay the
agents' follow-up turn). With only ~275 usable KV blocks on this fleet's
recipe (~3,023 tokens/block, ~831,629-token pool), this volume is large
enough to create real eviction pressure without needing hundreds of
thousands of tokens.

Sequence:
  1. Send N_AGENTS cold-start turns (long document + question), each a
     distinct document -- these are the "paused" conversations. Record
     their full message history for the replay step.
  2. Flood N_FLOOD one-shot chats (distinct large documents, single turn
     each) to consume/churn the free-block pool.
  3. Replay each agent's SAME history plus a follow-up question. Compare
     the replay's TTFT (near-instant on a cache hit, ~document-length/
     prefill-rate seconds on a cache miss) and the /metrics prefix-cache
     hit-ratio delta around the replay window.

Usage:
  probe_cache_eviction_pressure.py [--base-url http://10.7.0.87:8000]
                                   [--agents 2] [--agent-tokens 46000]
                                   [--flood 18] [--flood-tokens 31000]
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

SUBJECTS = ["the migration patterns of arctic terns", "the invention of the printing press",
            "deep-sea hydrothermal vents", "the history of the Suez Canal",
            "how bilingualism affects cognitive aging", "the geology of the Deccan Traps",
            "the domestication of horses", "ant colony optimization algorithms",
            "the restoration of wetlands in Florida", "medieval guild regulations",
            "the discovery of penicillin", "monsoon dynamics in South Asia",
            "the construction of Gothic cathedrals", "fermentation in food preservation",
            "the evolution of the metric system", "coral bleaching events",
            "the Silk Road trade in lapis lazuli", "how sonar was developed",
            "the physiology of hibernation", "the design of Roman aqueducts"]
ASPECTS = ["economic consequences", "technical challenges", "political context",
           "environmental impact", "key historical figures", "measurement methods",
           "common misconceptions", "notable failures", "modern legacy", "early records"]


def make_sentence(rng):
    s = rng.choice(SUBJECTS)
    a = rng.choice(ASPECTS)
    n = rng.randint(1732, 1989)
    return (f"Archival note {n}, source {rng.randint(100000, 999999)}: the committee "
            f"reviewed {s} with particular attention to {a}, cross-referencing field "
            f"measurements against the registry copies and recording dissenting "
            f"opinions in the annex for later ratification.")


def build_document(target_tokens, base_url, model, seed):
    rng = random.Random(seed)
    target_chars = int(target_tokens * 5.7)
    paras, chars = [], 0
    while chars < target_chars:
        para = " ".join(make_sentence(rng) for _ in range(rng.randint(4, 8)))
        paras.append(para)
        chars += len(para) + 1
    return "\n".join(paras)


def tokenize_count(base_url, model, text):
    payload = json.dumps({"model": model, "prompt": text, "add_special_tokens": False}).encode()
    req = urllib.request.Request(base_url + "/tokenize", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.load(r)["tokens"])


def chat(base_url, model, messages, max_tokens, timeout):
    payload = {"model": model, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": messages}
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    pieces = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            chs = ev.get("choices") or []
            if chs:
                piece = (chs[0].get("delta") or {}).get("content") or ""
                if piece and first is None:
                    first = time.perf_counter() - t0
                pieces.append(piece)
    return {"ttft": first, "total": time.perf_counter() - t0, "answer": "".join(pieces), "usage": usage}


def metrics_snapshot(base_url):
    with urllib.request.urlopen(base_url + "/metrics", timeout=30) as r:
        text = r.read().decode()
    queries = hits = None
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total{"):
            queries = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:prefix_cache_hits_total{"):
            hits = float(line.rsplit(" ", 1)[1])
    return queries, hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--agent-tokens", type=int, default=46000)
    ap.add_argument("--flood", type=int, default=18)
    ap.add_argument("--flood-tokens", type=int, default=31000)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    print(f"seed={seed}", flush=True)

    agents = []
    for i in range(args.agents):
        doc = build_document(args.agent_tokens, base, args.model, seed=seed * 1000 + i)
        q = f"Summarize the above in one sentence. This is agent {i}."
        messages = [{"role": "user", "content": doc + "\n\n" + q}]
        t0 = time.perf_counter()
        r = chat(base, args.model, messages, args.max_tokens, args.timeout)
        print(f"agent {i} cold: ttft={r['ttft']:.1f}s prompt_tokens="
              f"{(r['usage'] or {}).get('prompt_tokens')}", flush=True)
        messages.append({"role": "assistant", "content": r["answer"]})
        agents.append({"messages": messages, "doc_tokens": (r["usage"] or {}).get("prompt_tokens")})

    q_before, h_before = metrics_snapshot(base)

    for i in range(args.flood):
        doc = build_document(args.flood_tokens, base, args.model, seed=seed * 2000 + i)
        messages = [{"role": "user", "content": doc + "\n\nWhat is the third word of the document?"}]
        r = chat(base, args.model, messages, args.max_tokens, args.timeout)
        print(f"flood {i}/{args.flood}: ttft={r['ttft']:.1f}s prompt_tokens="
              f"{(r['usage'] or {}).get('prompt_tokens')}", flush=True)

    q_after_flood, h_after_flood = metrics_snapshot(base)

    replay_results = []
    for i, agent in enumerate(agents):
        followup = list(agent["messages"]) + [{"role": "user", "content": "Now name one aspect mentioned."}]
        r = chat(base, args.model, followup, args.max_tokens, args.timeout)
        prompt_tokens = (r["usage"] or {}).get("prompt_tokens")
        prefill_tps = (prompt_tokens / r["ttft"]) if r["ttft"] and prompt_tokens else None
        print(f"replay agent {i}: ttft={r['ttft']:.2f}s prompt_tokens={prompt_tokens} "
              f"prefill~{prefill_tps if prefill_tps is None else round(prefill_tps)} tok/s "
              f"(low ttft + high prefill-rate = cache hit; ~real prefill rate = miss)", flush=True)
        replay_results.append({"ttft": r["ttft"], "prompt_tokens": prompt_tokens})

    q_after_replay, h_after_replay = metrics_snapshot(base)

    flood_hit_ratio = ((h_after_flood - h_before) / (q_after_flood - q_before)
                        if q_after_flood > q_before else None)
    replay_hit_ratio = ((h_after_replay - h_after_flood) / (q_after_replay - q_after_flood)
                         if q_after_replay > q_after_flood else None)
    print(f"\nflood-phase hit ratio: {flood_hit_ratio}", flush=True)
    print(f"replay-phase hit ratio: {replay_hit_ratio}  <-- the number that matters", flush=True)
    print(f"replay TTFTs: {[r['ttft'] for r in replay_results]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
