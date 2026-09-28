#!/usr/bin/env python3
"""Fix the K-pool tail-seed kernel's dense-packing stride assumption.

``_kpool_tail_seed_kernel`` (the PREFILL path that seeds the paged tail
cache with each request's trailing, not-yet-a-full-pool raw K + gate score)
computes its write offset as::

    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM

This assumes every physical block occupies exactly ``2 * KPOOL * HEAD_DIM``
elements contiguously -- i.e. the tensor's block-dimension stride equals the
*logical*, unpadded per-block size (``KpoolTailSpec.unpadded_page_size_bytes``,
2048 B on this checkpoint: ``2 * index_kpool(4) * index_head_dim(128) *
sizeof(bf16)``).

That assumption is false whenever ``KpoolTailSpec.page_size_padded`` is set
-- which it is on this deployment: the tail spec is padded to co-own the
indexer's page size (``[glm53-kv-capacity-log] group 1: KpoolTailSpec ...
page_size=118,272 B`` in this fork's own boot logs, 57.75x the logical
2048 B). Confirmed directly in vLLM's own KV-cache-tensor construction code
(``vllm/v1/worker/gpu/attn_utils.py``, the ``page_size_padded`` branch of
``_reshape_attention_kv_cache``): "the only stride that must change is the
block stride: every other (contiguous) stride already steps within the
unpadded region of a page, so no further adjustment is needed." The real
tensor's ``.stride(0)`` is ``page_size_bytes // dtype_size`` (59,136
elements here), not ``2 * KPOOL`` (1024). Every other stride (the [2, KPOOL,
HEAD_DIM] intra-block layout) is unaffected -- only the ``blk *`` multiplier
is wrong.

Net effect: for any block index ``blk >= 1`` -- under real concurrent
serving, essentially every request except whichever one happens to land in
block 0 -- the kernel writes ~57x too close to the tensor's start, landing
inside a completely different (and completely unrelated) region of the
shared cache. Silent corruption, no crash: exactly the class of bug this
fork spent a week chasing in the b12x MoE integration, found here via an
independent project (cbertucci33/vllm-v29-glm53flash-exl3-dgx, commit
111425cfc, "Fix GLM hybrid prefix-cache corruption") hitting the identical
bug in the identical kernel on 2026-09-18.

The companion DECODE-side kernel (``_kpool_decode_update_batched_kernel``,
same file) is NOT affected -- it already receives the real stride as an
explicit ``TAIL_BLOCK_ELEMS`` constexpr, computed at its call site via
``tail_kv_cache.stride(0)`` (see ``kpool_decode_update_and_maybe_write_
cache_batched``'s launcher). This patch makes the prefill-seed kernel match
that already-correct pattern, using the identical parameter name for
consistency within the same file.

Mechanism: this fork's own build-time regex/string-replace patch pattern
(see ``patch_kpool_tail_slotmap.py`` for the template). Fail-closed,
idempotent, preflights both pinned anchors before writing either.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_KPOOL_COMPRESS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/"
        "nvidia/ops/kpool_compress.py",
    )
)
MARK = (
    "    # [glm53-kpool-tail-seed-stride] blk multiplies the REAL per-block\n"
)

KERNEL_ANCHOR = '''@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy token ``i``'s raw K + gate into its request's tail block.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // KPOOL  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // KPOOL == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)
'''

KERNEL_PATCHED = '''@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy token ``i``'s raw K + gate into its request's tail block.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // KPOOL  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // KPOOL == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    # [glm53-kpool-tail-seed-stride] blk multiplies the REAL per-block
    # element stride (TAIL_BLOCK_ELEMS, from tail_kv_cache.stride(0) at the
    # call site) -- NOT the logical 2*KPOOL*HEAD_DIM dense-packing size,
    # which is only correct when the tail cache's block dimension is
    # unpadded. This tensor is padded on this deployment (co-owns the
    # indexer's page size -- see this patch's module docstring). The
    # intra-block offset (t % KPOOL) * HEAD_DIM is unaffected: only the
    # block dimension's stride is ever padded (vllm/v1/worker/gpu/
    # attn_utils.py's page_size_padded strided-view branch changes stride(0)
    # only). Matches the already-correct pattern the decode-side sibling
    # kernel (_kpool_decode_update_batched_kernel, TAIL_BLOCK_ELEMS) already
    # uses in this same file.
    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)
'''

LAUNCHER_ANCHOR = '''def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel)."""
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
'''

LAUNCHER_PATCHED = '''def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel)."""
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    # [glm53-kpool-tail-seed-stride] The block dimension's stride may be
    # padded past the logical 2*kpool*head_dim (see the kernel's own
    # comment and this patch's module docstring) -- assert the ONLY other
    # assumption the kernel still makes (that padding never touches the
    # intra-block layout) still holds, rather than silently trusting it
    # forever.
    assert tail_kv_cache.stride(1) == kpool * head_dim, (
        "kpool tail cache K/score-half stride drifted: expected "
        f"{kpool * head_dim}, got {tail_kv_cache.stride(1)} -- the "
        "attn_utils.py page_size_padded strided-view contract (only the "
        "block dim's stride changes) no longer holds; re-derive this patch"
    )
    assert tail_kv_cache.stride(2) == head_dim, (
        "kpool tail cache pos-within-pool stride drifted: expected "
        f"{head_dim}, got {tail_kv_cache.stride(2)}"
    )
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
'''


def tail_write_offset(
    t: int, *, kpool: int, head_dim: int, tail_block_elems: int
) -> int:
    """CPU replica of the PATCHED kernel's ``base`` (fixed) offset math."""
    if kpool < 1 or head_dim < 1 or tail_block_elems < 1:
        raise ValueError("kpool, head_dim, tail_block_elems must be >= 1")
    blk = t // kpool
    return blk * tail_block_elems + (t % kpool) * head_dim


def tail_write_offset_buggy(t: int, *, kpool: int, head_dim: int) -> int:
    """CPU replica of the ORIGINAL (buggy) dense-packing offset math."""
    if kpool < 1 or head_dim < 1:
        raise ValueError("kpool, head_dim must be >= 1")
    blk = t // kpool
    return (blk * 2 * kpool + t % kpool) * head_dim


def verified_state(text: str) -> bool:
    return (
        text.count(KERNEL_ANCHOR) == 0
        and text.count(LAUNCHER_ANCHOR) == 0
        and text.count(KERNEL_PATCHED) == 1
        and text.count(LAUNCHER_PATCHED) == 1
        and text.count(MARK) == 1
        and "base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM" in text
        and "TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0)," in text
    )


def prepare(source: str) -> tuple[str, str]:
    mark_count = source.count(MARK)
    if mark_count:
        if mark_count != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent kpool tail-seed stride patch "
                f"(marker={mark_count})"
            )
        return source, "already present"
    if verified_state(source):
        return source, "already patched"

    n_kernel = source.count(KERNEL_ANCHOR)
    if n_kernel != 1:
        raise ValueError(
            "pinned _kpool_tail_seed_kernel anchor drifted "
            f"(occurrences={n_kernel})"
        )
    n_launcher = source.count(LAUNCHER_ANCHOR)
    if n_launcher != 1:
        raise ValueError(
            "pinned kpool_seed_tail_cache anchor drifted "
            f"(occurrences={n_launcher})"
        )

    patched = source.replace(KERNEL_ANCHOR, KERNEL_PATCHED, 1)
    patched = patched.replace(LAUNCHER_ANCHOR, LAUNCHER_PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("kpool tail-seed stride post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kpool-tail-seed.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob("kpool_compress*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"kpool tail-seed stride preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: kpool tail-seed stride {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
