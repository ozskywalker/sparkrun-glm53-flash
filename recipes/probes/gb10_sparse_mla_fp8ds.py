"""Sparse NoPE-MLA attention in Triton over the packed fp8_ds_mla KV layout.

Adapted from mmastrac/glm-5.3-flash-4x-gx10 experimental/fixes/gb10_sparse_mla.py,
which assumes a plain [S, 512] FP8 latent with one tensor-wide scale. This fork's
KV cache is the 656-byte packed layout: 512 fp8 latent bytes, four fp32 scales (one
per 128-channel tile), then a 128-byte rope pad (zeros for GLM's NoPE MLA). K and V
are the same latent row, dequantized per tile.

One program per query token (all heads' rows of Q in registers); short batches split
each token's top-k rows across SPLITS programs (flash-decoding) and combine after.
"""
import torch
import triton
import triton.language as tl

ROW_BYTES = 656
LATENT = 512
TILE = 128


@triton.jit
def _load_kv(kv_ptr, rows, live, dims, ROW: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr, DT: tl.constexpr):
    base = kv_ptr + rows * ROW
    raw = tl.load(base[:, None] + dims[None, :], mask=live[:, None], other=0)
    f = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    sptr = (base[:, None] + D + (dims[None, :] // TILE) * 4).to(tl.pointer_type(tl.float32))
    sc = tl.load(sptr, mask=live[:, None], other=0.0)
    return (f * sc).to(DT)


@triton.jit
def _split(q_ptr, kv_ptr, idx_ptr, len_ptr, acc_ptr, ml_ptr, qk_scale, sq_t, sq_h,
           H: tl.constexpr, D: tl.constexpr, ROW: tl.constexpr, TILE: tl.constexpr, TOPK: tl.constexpr,
           CHUNK: tl.constexpr, SPLITS: tl.constexpr, BLOCK_N: tl.constexpr, HP: tl.constexpr, DT: tl.constexpr):
    t = tl.program_id(0)
    sp = tl.program_id(1)
    heads = tl.arange(0, HP)
    hm = heads < H
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + t * sq_t + heads[:, None] * sq_h + dims[None, :], mask=hm[:, None], other=0.0).to(DT)
    n_valid = tl.load(len_ptr + t)
    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for start in range(sp * CHUNK, sp * CHUNK + CHUNK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        live = offs < n_valid
        rows = tl.load(idx_ptr + t * TOPK + offs, mask=live, other=0).to(tl.int64)
        kv = _load_kv(kv_ptr, rows, live, dims, ROW, D, TILE, DT)
        s = tl.dot(q, tl.trans(kv)) * qk_scale
        s = tl.where(live[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
        p = tl.where(m_new[:, None] == float("-inf"), 0.0, tl.exp2(s - m_new[:, None]))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(DT), kv)
        m_i = m_new
    b = (t * SPLITS + sp) * H
    tl.store(acc_ptr + (b + heads[:, None]) * D + dims[None, :], acc, mask=hm[:, None])
    tl.store(ml_ptr + (b + heads) * 2, m_i, mask=hm)
    tl.store(ml_ptr + (b + heads) * 2 + 1, l_i, mask=hm)


@triton.jit
def _combine(acc_ptr, ml_ptr, o_ptr, H: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    dims = tl.arange(0, D)
    sps = tl.arange(0, SPLITS)
    b = (t * SPLITS + sps) * H + h
    m = tl.load(ml_ptr + b * 2)
    l = tl.load(ml_ptr + b * 2 + 1)
    m_max = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - m_max))
    l_sum = tl.sum(l * w, 0)
    acc = tl.load(acc_ptr + b[:, None] * D + dims[None, :])
    out = tl.sum(acc * w[:, None], 0)
    out = tl.where(l_sum > 0, out / l_sum, 0.0)
    tl.store(o_ptr + (t * H + h) * D + dims, out.to(o_ptr.dtype.element_ty))


@triton.jit
def _fwd(q_ptr, kv_ptr, idx_ptr, len_ptr, o_ptr, qk_scale, sq_t, sq_h,
         H: tl.constexpr, D: tl.constexpr, ROW: tl.constexpr, TILE: tl.constexpr, TOPK: tl.constexpr,
         BLOCK_N: tl.constexpr, HP: tl.constexpr, DT: tl.constexpr):
    t = tl.program_id(0)
    heads = tl.arange(0, HP)
    hm = heads < H
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + t * sq_t + heads[:, None] * sq_h + dims[None, :], mask=hm[:, None], other=0.0).to(DT)
    n_valid = tl.load(len_ptr + t)
    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for start in range(0, TOPK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        live = offs < n_valid
        rows = tl.load(idx_ptr + t * TOPK + offs, mask=live, other=0).to(tl.int64)
        kv = _load_kv(kv_ptr, rows, live, dims, ROW, D, TILE, DT)
        s = tl.dot(q, tl.trans(kv)) * qk_scale
        s = tl.where(live[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
        p = tl.where(m_new[:, None] == float("-inf"), 0.0, tl.exp2(s - m_new[:, None]))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(DT), kv)
        m_i = m_new
    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    tl.store(o_ptr + t * H * D + heads[:, None] * D + dims[None, :], out.to(o_ptr.dtype.element_ty), mask=hm[:, None])


_SPLIT_TARGET = 96


def sparse_mla_fp8ds(q, kv_bytes, idx, lens, sm_scale, block_n=32, num_warps=4, num_stages=2, splits=None,
                     dt=tl.bfloat16):
    """q [T, H, >=512] (first 512 dims used; last dim contiguous), kv_bytes uint8 [S, 656] (flat token rows),
    idx [T, TOPK] int32 flat row ids with the valid ones first, lens [T] int32.
    Returns [T, H, 512] in q.dtype."""
    T, H, _ = q.shape
    D = LATENT
    hp = max(16, triton.next_power_of_2(H))
    assert q.stride(2) == 1 and kv_bytes.is_contiguous() and idx.stride(1) == 1 and kv_bytes.dtype == torch.uint8
    out = torch.empty(T, H, D, dtype=q.dtype, device=q.device)
    qk_scale = sm_scale * 1.4426950408889634
    topk = idx.shape[1]
    if splits is None:
        splits = 1
        while T * splits < _SPLIT_TARGET and splits < 16 and topk % (splits * 2 * block_n) == 0:
            splits *= 2
    if splits > 1:
        acc = torch.empty(T, splits, H, D, dtype=torch.float32, device=q.device)
        ml = torch.empty(T, splits, H, 2, dtype=torch.float32, device=q.device)
        _split[(T, splits)](q, kv_bytes, idx, lens, acc, ml, qk_scale, q.stride(0), q.stride(1),
                            H=H, D=D, ROW=ROW_BYTES, TILE=TILE, TOPK=topk, CHUNK=topk // splits, SPLITS=splits,
                            BLOCK_N=block_n, HP=hp, DT=dt, num_warps=num_warps, num_stages=num_stages)
        _combine[(T, H)](acc, ml, out, H=H, D=D, SPLITS=splits, num_warps=4)
        return out
    _fwd[(T,)](q, kv_bytes, idx, lens, out, qk_scale, q.stride(0), q.stride(1),
               H=H, D=D, ROW=ROW_BYTES, TILE=TILE, TOPK=topk, BLOCK_N=block_n, HP=hp, DT=dt,
               num_warps=num_warps, num_stages=num_stages)
    return out
