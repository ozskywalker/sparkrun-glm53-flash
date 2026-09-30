#!/usr/bin/env python3
"""Decode-size sparse-MLA timing inside CUDA graphs (how decode actually runs): Triton fp8_ds_mla vs FlashInfer."""
import sys

import torch

sys.path.insert(0, "/w")
import bench_sparse_mla as b  # noqa: E402
from gb10_sparse_mla_fp8ds import sparse_mla_fp8ds  # noqa: E402


def graph_time(fn, reps=50):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (10 * reps)


dev = torch.device("cuda")
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _get_workspace_buffer  # noqa: E402
ws = _get_workspace_buffer(dev)
rows = 3584 * 36
cache = b.make_cache(rows, dev)
for T in (1, 3, 6, 16, 32, 48):
    q, idx, lens = b.make_inputs(T, rows, dev)
    idxc = idx.clamp(min=0)
    best = None
    for cfg in b.CONFIGS:
        try:
            t = graph_time(lambda: sparse_mla_fp8ds(q, cache, idxc, lens, b.SCALE, **cfg))
        except Exception:  # noqa: BLE001
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    tf = graph_time(lambda: b.flashinfer_call(q, cache, idx, lens, ws))
    print(f"T={T:3d} graph: triton {best[0]*1e3:7.1f} us {best[1]} | flashinfer {tf*1e3:7.1f} us | ratio {tf/best[0]:.2f}x", flush=True)
