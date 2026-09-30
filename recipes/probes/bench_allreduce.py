#!/usr/bin/env python3
"""Two-node NCCL all-reduce latency at GLM-5.3-Flash decode/prefill activation sizes (hidden 4096, bf16).

Run one process per node with RANK/WORLD_SIZE=2/MASTER_ADDR/MASTER_PORT set and production's NCCL env.
Reports per-op latency eager and inside a CUDA graph (which is how decode actually runs).
"""
import os
import time

import torch
import torch.distributed as dist

HIDDEN = 4096


def main():
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    rank = dist.get_rank()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    rows = [1, 3, 16, 48, 128, 512, 1024, 7168]
    if rank == 0:
        print(f"{'rows':>6} {'bytes':>10} {'eager us':>10} {'graph us':>10}", flush=True)
    for r in rows:
        x = torch.ones(r, HIDDEN, dtype=torch.bfloat16, device=dev)
        nbytes = x.numel() * 2
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        n = 200 if nbytes < 4_000_000 else 20
        t0 = time.perf_counter()
        for _ in range(n):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        eager = (time.perf_counter() - t0) / n * 1e6
        k = 50 if nbytes < 4_000_000 else 5
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            for _ in range(3):
                dist.all_reduce(x)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(k):
                dist.all_reduce(x)
        g.replay()
        torch.cuda.synchronize()
        m = 20
        t0 = time.perf_counter()
        for _ in range(m):
            g.replay()
        torch.cuda.synchronize()
        graph = (time.perf_counter() - t0) / (m * k) * 1e6
        if rank == 0:
            print(f"{r:>6} {nbytes:>10} {eager:>10.1f} {graph:>10.1f}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
