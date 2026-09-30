#!/usr/bin/env python3
"""Microbenchmark: FlashInfer TRTLLM sparse-MLA (what production runs, FLASHINFER_MLA_SPARSE_SM120)
vs the Triton fp8_ds_mla kernel adapted from mmastrac/glm-5.3-flash-4x-gx10, at GLM-5.3-Flash TP=2
per-rank shapes (32 heads, 512 latent, top-k 2048, packed 656-byte fp8_ds_mla KV).

Runs on one GPU, no distributed init. Production must be stopped first (unified memory).
Usage (inside the v22 image): python bench_sparse_mla.py [--quick]
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/w")
from gb10_sparse_mla_fp8ds import ROW_BYTES, sparse_mla_fp8ds  # noqa: E402

H, LATENT, ROPE_PAD, TOPK, PAGE = 32, 512, 64, 2048, 64
SCALE = 256 ** -0.5


def make_cache(n_rows, dev):
    g = torch.Generator(device=dev).manual_seed(0)
    cache = torch.zeros(n_rows, ROW_BYTES, dtype=torch.uint8, device=dev)
    nope = (torch.randn(n_rows, LATENT, device=dev, generator=g) * 4.0).clamp(-400, 400).to(torch.float8_e4m3fn)
    cache[:, :LATENT] = nope.view(torch.uint8)
    scales = (torch.rand(n_rows, 4, device=dev, generator=g) * 0.015 + 0.005).contiguous()
    cache[:, LATENT:LATENT + 16] = scales.view(torch.uint8).view(n_rows, 16)
    return cache


def make_inputs(T, n_rows, dev, partial=False):
    g = torch.Generator(device=dev).manual_seed(1)
    q = torch.randn(T, H, LATENT + ROPE_PAD, device=dev, generator=g, dtype=torch.bfloat16)
    q[:, :, LATENT:] = 0
    idx = torch.stack([torch.randperm(n_rows, device=dev, generator=g)[:TOPK] for _ in range(min(T, 64))])
    idx = idx.repeat((T + idx.shape[0] - 1) // idx.shape[0], 1)[:T].to(torch.int32).contiguous()
    lens = torch.full((T,), TOPK, dtype=torch.int32, device=dev)
    if partial:
        lens = torch.randint(1, TOPK + 1, (T,), dtype=torch.int32, device=dev, generator=g)
        for t in range(T):
            idx[t, lens[t]:] = -1
    return q, idx, lens


def ref(q, cache, idx, lens):
    T = q.shape[0]
    out = torch.zeros(T, H, LATENT, dtype=torch.float32, device=q.device)
    for t in range(T):
        n = int(lens[t])
        rows = idx[t, :n].long()
        raw = cache[rows]
        k = raw[:, :LATENT].contiguous().view(torch.float8_e4m3fn).float()
        sc = raw[:, LATENT:LATENT + 16].contiguous().view(torch.float32)
        k = (k.view(n, 4, 128) * sc[:, :, None]).view(n, LATENT)
        s = (q[t, :, :LATENT].float() @ k.T) * SCALE
        out[t] = torch.softmax(s, -1) @ k
    return out


def flashinfer_call(q, cache, idx, lens, ws):
    from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla
    T = q.shape[0]
    idx_safe = idx.clone()
    idx_safe[idx_safe < 0] = 0
    out = q.new_empty((T, H, LATENT))
    r = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1), kv_cache=cache.view(-1, 1, PAGE, ROW_BYTES),
        workspace_buffer=ws, qk_nope_head_dim=256, kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE_PAD, block_tables=idx_safe.unsqueeze(1), seq_lens=lens.clamp(min=1),
        max_seq_len=TOPK, out=out.unsqueeze(1), bmm1_scale=SCALE, bmm2_scale=1.0,
        sparse_mla_top_k=TOPK, kv_scale_format="arbitrary_fp32")
    return r.squeeze(1)


def timeit(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters


CONFIGS = [dict(block_n=bn, num_stages=ns, num_warps=nw) for bn in (16, 32) for ns in (1, 2) for nw in (4, 8)]


def tri_call(cfg, q, kv, idx, lens):
    return lambda: sparse_mla_fp8ds(q, kv, idx, lens, SCALE, **cfg)


def best_cfg(q, kv, idx, lens):
    """Fastest config that fits in shared memory for this shape."""
    best = None
    for cfg in CONFIGS:
        try:
            fn = tri_call(cfg, q, kv, idx, lens)
            fn()
            t = timeit(fn, warm=1, iters=3)
        except Exception as e:  # noqa: BLE001
            print(f"   cfg {cfg}: {type(e).__name__}", flush=True)
            continue
        print(f"   cfg {cfg}: {t:.3f} ms", flush=True)
        if best is None or t < best[0]:
            best = (t, cfg)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rows", type=int, default=3584 * 36)
    args = ap.parse_args()
    dev = torch.device("cuda")
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _get_workspace_buffer
    ws = _get_workspace_buffer(dev)
    print(f"gpu={torch.cuda.get_device_name()} free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB", flush=True)
    cache = make_cache(args.rows, dev)
    kv_flat = cache

    print("\n== correctness (T=96, partial lengths) ==", flush=True)
    q, idx, lens = make_inputs(96, args.rows, dev, partial=True)
    r = ref(q, cache, idx, lens)
    cfg0 = best_cfg(q, kv_flat, idx.clamp(min=0), lens)
    print(f"correctness cfg: {cfg0}", flush=True)
    tri = sparse_mla_fp8ds(q, kv_flat, idx.clamp(min=0), lens, SCALE, **cfg0).float()
    print(f"triton vs ref: max abs {(tri - r).abs().max():.4e}  rel {((tri - r).norm() / r.norm()):.3e}", flush=True)
    try:
        fi = flashinfer_call(q, cache, idx, lens, ws).float()
        print(f"flashinfer vs ref: max abs {(fi - r).abs().max():.4e}  rel {((fi - r).norm() / r.norm()):.3e}", flush=True)
        print(f"triton vs flashinfer: rel {((tri - fi).norm() / fi.norm()):.3e}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"flashinfer call failed: {type(e).__name__}: {e}", flush=True)
        fi = None

    print("\n== flashinfer decode-size probe ==", flush=True)
    for T in (1, 3, 16, 48, 64, 65, 96):
        q2, idx2, lens2 = make_inputs(T, args.rows, dev)
        try:
            flashinfer_call(q2, cache, idx2, lens2, ws)
            print(f"   T={T}: ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   T={T}: {type(e).__name__}: {str(e)[:90]}", flush=True)

    print("\n== timing (ms per layer-call; 11 MLA layers in the model) ==", flush=True)
    shapes = [("decode T=3", 3), ("decode T=48", 48), ("prefill T=1024", 1024), ("prefill T=7168", 7168)]
    if args.quick:
        shapes = shapes[1:2] + shapes[3:]
    for name, T in shapes:
        q, idx, lens = make_inputs(T, args.rows, dev)
        idxc = idx.clamp(min=0)
        flops = 2.0 * T * H * TOPK * (LATENT + ROPE_PAD) + 2.0 * T * H * TOPK * LATENT
        cfg = best_cfg(q, kv_flat, idxc, lens)
        t_tri = timeit(tri_call(cfg, q, kv_flat, idxc, lens))
        row = f"{name:16s} triton {t_tri:8.3f} ms ({flops/t_tri/1e9:6.1f} TFLOPS) cfg={cfg}"
        if fi is not None:
            try:
                t_fi = timeit(lambda: flashinfer_call(q, cache, idx, lens, ws))
            except Exception as e:  # noqa: BLE001
                print(row + f" | flashinfer n/a ({str(e)[:50]})", flush=True)
                continue
            row += f" | flashinfer {t_fi:8.3f} ms ({flops/t_fi/1e9:6.1f} TFLOPS) | speedup {t_fi/t_tri:.2f}x"
        print(row, flush=True)


if __name__ == "__main__":
    main()
