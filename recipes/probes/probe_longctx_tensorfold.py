#!/usr/bin/env python3
"""Long-context probe for TensorFold's GLM-5.3-Flash endpoint.

Same document/marker-retrieval methodology as probe_longctx.py (this project's
vLLM long-context probe), adapted because TensorFold's OpenAI-compatible server
has no /v1/tokenize endpoint: token counts are computed with a local HF
tokenizer loaded from the already-cached checkpoint snapshot instead of a
server round-trip. Everything else (document generator, planted codes,
retrieval question, TTFT/finish-reason/codes-retrieved checks) is identical.

Usage:
  probe_longctx_tensorfold.py --tokenizer-dir <snapshot dir> [--base-url http://10.7.0.87:8080]
                              [--tokens 32000] [--model GLM-5.3-Flash-MLX-4bit-MTP]
Exit 0 = all checks passed.
"""

import argparse
import json
import random
import sys
import time
import urllib.request

FAILURES = []

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
    return (f"Archival note {n}: the committee reviewed {s} with particular attention to "
            f"{a}, cross-referencing field measurements against the registry copies and "
            f"recording dissenting opinions in the annex for later ratification.")


def build_document(target_tokens, tok_count, seed=1234):
    rng = random.Random(seed)
    target_chars = int(target_tokens * 5.7)
    paras = []
    chars = 0
    while chars < target_chars:
        para = " ".join(make_sentence(rng) for _ in range(rng.randint(4, 8)))
        paras.append(para)
        chars += len(para) + 1
    text = "\n".join(paras)

    while tok_count(text) < target_tokens - 200:
        for _ in range(len(paras)):
            para = " ".join(make_sentence(rng) for _ in range(rng.randint(4, 8)))
            paras.append(para)
        text = "\n".join(paras)

    lo, hi = 0, len(paras)
    while lo < hi:
        mid = (lo + hi) // 2
        if tok_count("\n".join(paras[:mid + 1])) < target_tokens - 200:
            lo = mid + 1
        else:
            hi = mid
    text = "\n".join(paras[:lo + 1])
    n = tok_count(text)
    while n > target_tokens - 100:
        text = text[:int(len(text) * (target_tokens - 150) / n)]
        n = tok_count(text)
    return text, n


def plant_codes(text):
    codes = [f"CODE-{i}-{''.join(random.Random(42 + i).choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(4))}"
             for i in range(1, 5)]
    lines = text.split("\n")
    marks = [int(len(lines) * f) for f in (0.25, 0.50, 0.75, 0.97)]
    for code, mark in zip(codes, marks):
        lines[mark] = (lines[mark] + f" CLASSIFIED VERIFICATION MARKER: {code}.")
    return "\n".join(lines), codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8080")
    ap.add_argument("--model", default="GLM-5.3-Flash-MLX-4bit-MTP")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--tokens", type=int, default=32000, help="target prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=3600, help="request timeout seconds")
    ap.add_argument("--draft-false", action="store_true",
                     help="send with \"draft\": false (serial decode, for exactness comparison)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    def tok_count(t):
        return len(tok(t, add_special_tokens=False).input_ids)

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    print(f"building ~{args.tokens}-token document (seed={seed}, local-tokenizer count)...", flush=True)
    t0 = time.perf_counter()
    text, n_tok = build_document(args.tokens, tok_count, seed=seed)
    text, codes = plant_codes(text)
    print(f"document ready: ~{n_tok} tokens before markers, {len(text)} chars, "
          f"{time.perf_counter() - t0:.0f}s build time", flush=True)
    print(f"planted codes: {codes}", flush=True)

    question = ("The document above contains four CLASSIFIED VERIFICATION MARKERS, "
                "each with a code of the form CODE-<digit>-<4 letters>. "
                "List all four codes in the order they appear, nothing else.")
    payload = {"model": args.model, "max_tokens": args.max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": text + "\n\n" + question}]}
    if args.draft_false:
        payload["draft"] = False

    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    print(f"streaming long-context request (timeout {args.timeout}s, draft={not args.draft_false})...", flush=True)
    t0 = time.perf_counter()
    first = None
    pieces = []
    usage = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    usage = ev["usage"]
                chs = ev.get("choices") or []
                if chs:
                    finish = chs[0].get("finish_reason") or finish
                    piece = (chs[0].get("delta") or {}).get("content") or ""
                    if piece and first is None:
                        first = time.perf_counter() - t0
                        print(f"first token after {first:.1f}s", flush=True)
                    pieces.append(piece)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] request-failed — {e!r}", flush=True)
        sys.exit(1)
    total = time.perf_counter() - t0
    answer = "".join(pieces)

    def check(name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        if not ok:
            FAILURES.append(name)

    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    check("prompt-tokens-near-target",
          abs(prompt_tokens - args.tokens) <= max(500, args.tokens * 0.02),
          f"prompt_tokens={prompt_tokens}, target={args.tokens}")
    check("ttft-reasonable", first is not None and first < 1800,
          f"ttft={first:.1f}s" if first else "no tokens")
    check("finish-stop", finish == "stop", f"finish_reason={finish}")
    found = [c for c in codes if c in answer]
    check("codes-retrieved", len(found) == len(codes),
          f"{len(found)}/{len(codes)} found; answer={answer.strip()[:200]!r}")
    completion_tokens = (usage or {}).get("completion_tokens")
    decode_time = total - (first or 0)
    decode_tps = (completion_tokens - 1) / decode_time if completion_tokens and completion_tokens > 1 and decode_time > 0 else None
    prefill_tps = (prompt_tokens / first) if first and prompt_tokens else None
    print(f"       total={total:.1f}s  ttft={first:.2f}s  prefill~{prefill_tps:.0f} tok/s  "
          f"decode~{decode_tps:.2f} tok/s  completion_tokens={completion_tokens}", flush=True)
    print(f"       answer={answer.strip()!r}", flush=True)

    print(f"\n{'LONG-CONTEXT CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
