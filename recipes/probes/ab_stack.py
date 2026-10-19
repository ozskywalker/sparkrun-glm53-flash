#!/usr/bin/env python3
"""Stack-agnostic A/B driver (OpenAI-compatible /v1/chat/completions, streaming).

Modes:
  prefill  N-token distinct documents, C concurrent, retrieval question -> per-request TTFT / prefill tok/s, aggregate
  warm     same as prefill but each doc sent twice sequentially; reports cold vs replay TTFT
  decode   short prompt, long essay, C concurrent -> per-request and aggregate decode tok/s
Documents are built with a local HF tokenizer (no /tokenize dependency) and cached on disk.

Usage: ab_stack.py MODE --base-url URL --model NAME --tokenizer-dir DIR [--tokens N] [--conc 1,2,4] [--label X]
"""
import argparse
import concurrent.futures as cf
import json
import os
import pickle
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_longctx_tensorfold import build_document, plant_codes  # noqa: E402

QUESTION = ("The document above contains four CLASSIFIED VERIFICATION MARKERS, each with a code of the form "
            "CODE-<digit>-<4 letters>. List all four codes in the order they appear, nothing else.")
ESSAY = ("Write a detailed, technical essay of about 600 words on how modern CPUs execute instructions out of "
         "order, covering register renaming, the reorder buffer, and speculation.")


def stream(base, model, content, max_tokens, timeout=3600):
    payload = {"model": model, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    out = []
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning") or d.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - t0
                        out.append(piece)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    total = time.perf_counter() - t0
    ct = (usage or {}).get("completion_tokens") or 0
    pt = (usage or {}).get("prompt_tokens") or 0
    dec = (ct - 1) / (total - first) if first and ct > 1 and total > first else None
    return dict(ttft=first, total=total, prompt_tokens=pt, completion_tokens=ct, decode_tps=dec,
                text="".join(out), err=err,
                cached=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))


def get_doc(tok, n, seed, cache_dir):
    p = os.path.join(cache_dir, f"doc_{n}_{seed}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    text, _ = build_document(n, lambda t: len(tok(t, add_special_tokens=False).input_ids), seed=seed)
    text, codes = plant_codes(text)
    pickle.dump((text, codes), open(p, "wb"))
    return text, codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill", "warm", "decode"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--tokens", type=int, default=32000)
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--label", default="run")
    ap.add_argument("--cache-dir", default="/tmp/ab_docs")
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    os.makedirs(args.cache_dir, exist_ok=True)
    tok = None
    if args.mode != "decode":
        from tokenizers import Tokenizer
        _t = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
        tok = lambda t, add_special_tokens=False: type("E", (), {"input_ids": _t.encode(t, add_special_tokens=add_special_tokens).ids})()  # noqa: E731

    for c in [int(x) for x in args.conc.split(",")]:
        if args.mode == "decode":
            jobs = [(ESSAY, max(args.max_tokens, 400))] * c
            seeds = [None] * c
        else:
            seeds = [args.seed_base + i + 100 * c for i in range(c)]
            docs = [get_doc(tok, args.tokens, s, args.cache_dir) for s in seeds]
            jobs = [(d[0] + "\n\n" + QUESTION, args.max_tokens) for d in docs]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda j: stream(base, args.model, j[0], j[1]), jobs))
        wall = time.perf_counter() - t0
        rec = dict(label=args.label, mode=args.mode, tokens=args.tokens, conc=c, wall_s=round(wall, 1),
                   errs=[r["err"] for r in res if r["err"]],
                   ttft=[round(r["ttft"], 1) if r["ttft"] else None for r in res],
                   decode_tps=[round(r["decode_tps"], 1) if r["decode_tps"] else None for r in res],
                   prompt_tokens=[r["prompt_tokens"] for r in res])
        if args.mode == "decode":
            rec["agg_decode_tps"] = round(sum(r["completion_tokens"] for r in res) / wall, 1)
        else:
            rec["agg_prefill_tps"] = round(sum(r["prompt_tokens"] for r in res) / wall, 1)
            rec["codes_ok"] = [all(cd in r["text"] for cd in d[1]) for r, d in zip(res, docs)]
        if args.mode == "warm":
            with cf.ThreadPoolExecutor(c) as ex:
                res2 = list(ex.map(lambda j: stream(base, args.model, j[0], j[1]), jobs))
            rec["replay_ttft"] = [round(r["ttft"], 1) if r["ttft"] else None for r in res2]
            rec["replay_cached"] = [r["cached"] for r in res2]
            rec["replay_codes_ok"] = [all(cd in r["text"] for cd in d[1]) for r, d in zip(res2, docs)]
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
