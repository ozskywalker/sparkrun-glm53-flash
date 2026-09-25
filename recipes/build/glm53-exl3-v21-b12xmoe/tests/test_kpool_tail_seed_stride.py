#!/usr/bin/env python3
"""Regression tests for the K-pool tail-seed kernel's padded-stride fix."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_kpool_tail_seed_stride.py",
        ROOT / "overlay" / "patch_kpool_tail_seed_stride.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_kpool_tail_seed_stride import (  # noqa: E402
    KERNEL_ANCHOR,
    LAUNCHER_ANCHOR,
    MARK,
    prepare,
    tail_write_offset,
    tail_write_offset_buggy,
    verified_state,
)

INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
    "ops/kpool_compress.py"
)

# Minimal self-contained fixture: enough surrounding structure for the patch
# to apply and the result to compile, without needing triton installed.
PINNED_FIXTURE = (
    "import torch\n"
    "import triton\n"
    "import triton.language as tl\n\n"
    "INDEX_HEAD_DIM = 128\n\n\n"
    + KERNEL_ANCHOR
    + "\n\n"
    + LAUNCHER_ANCHOR
)


def _run_patch(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_KPOOL_COMPRESS_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_offset_math_diverges_for_blk_ge_1() -> None:
    """The bug this fixes: buggy vs fixed offsets agree only at blk=0."""
    kpool, head_dim = 4, 128
    real_tail_block_elems = 59_136  # this deployment's real page_size_bytes(118272) // 2

    # blk == 0 (t in [0, kpool)): both formulas agree -- this is exactly why
    # the bug was invisible on any single-request/low-concurrency probe that
    # only ever happened to land in the first block.
    for t in range(kpool):
        buggy = tail_write_offset_buggy(t, kpool=kpool, head_dim=head_dim)
        fixed = tail_write_offset(
            t, kpool=kpool, head_dim=head_dim, tail_block_elems=real_tail_block_elems
        )
        assert buggy == fixed == (t % kpool) * head_dim

    # blk >= 1: real production concurrency territory. The buggy formula is
    # dramatically undershooting where the real (padded) block lives.
    for blk in (1, 2, 5, 100):
        t = blk * kpool  # first token of this block
        buggy = tail_write_offset_buggy(t, kpool=kpool, head_dim=head_dim)
        fixed = tail_write_offset(
            t, kpool=kpool, head_dim=head_dim, tail_block_elems=real_tail_block_elems
        )
        assert buggy == blk * 2 * kpool * head_dim  # the old, wrong, dense math
        assert fixed == blk * real_tail_block_elems  # the correct, strided math
        assert fixed != buggy
        # This deployment's real numbers: 57.75x apart at blk=1.
        if blk == 1:
            assert fixed == 59_136
            assert buggy == 1024
            assert fixed / buggy == real_tail_block_elems / 1024


def test_offset_math_matches_logical_shape_when_unpadded() -> None:
    """When there's no padding (tail_block_elems == 2*kpool*head_dim), the
    fixed formula must degenerate back to the original dense math exactly --
    the fix must be a strict generalization, not a behavior change for any
    deployment where the tail cache happens to be unpadded."""
    kpool, head_dim = 4, 128
    dense_elems = 2 * kpool * head_dim
    for t in range(0, 40):
        buggy = tail_write_offset_buggy(t, kpool=kpool, head_dim=head_dim)
        fixed = tail_write_offset(
            t, kpool=kpool, head_dim=head_dim, tail_block_elems=dense_elems
        )
        assert buggy == fixed


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(PINNED_FIXTURE)
        first = _run_patch(target)
        assert first.returncode == 0, first.stderr
        text = target.read_text()
        assert verified_state(text)
        assert MARK in text
        assert "base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM" in text
        assert "TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0)," in text
        assert "already present" not in first.stdout
        compile(text, str(target), "exec")

        second = _run_patch(target)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        assert second.stdout.count("already present") == 1
        again, action = prepare(text)
        assert action == "already present"
        assert again == text


def test_fail_closed() -> None:
    drifted = PINNED_FIXTURE.replace(
        "base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM",
        "base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM  # comment drift",
        1,
    )
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(drifted)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert target.read_text() == drifted

    partial = PINNED_FIXTURE.replace(KERNEL_ANCHOR, MARK + KERNEL_ANCHOR, 1)
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(partial)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_KPOOL_COMPRESS_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(src.read_text())
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr
        assert verified_state(target.read_text())


def test_live_kernel_writes_at_strided_offset_if_gpu() -> None:
    """When a GPU is actually available, launch the PATCHED kernel for real
    against a synthetic padded-stride tensor (blk >= 1) and confirm the
    write lands exactly where the strided-view math says it must -- not
    just that the offset formula is right in isolation, but that the real
    Triton kernel, compiled and run, agrees."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    if os.environ.get("EXL3_SELFCHECK_GPU", "1") == "0":
        return

    src = Path(os.environ.get("GLM53_KPOOL_COMPRESS_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(src.read_text())
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "glm53_kpool_compress_patched_live", src
    )
    assert spec is not None and spec.loader is not None
    # Re-import from the REAL installed (already-patched-at-build-time)
    # module rather than the temp copy above -- exercises exactly what
    # production actually loads.
    import vllm.models.glm5next.nvidia.ops.kpool_compress as live_mod  # noqa: E402

    kpool, head_dim = 4, 128
    num_blocks = 3
    tail_block_elems = 4096  # deliberately padded: real logical size is 1024
    device = "cuda"

    # Backing storage sized so block 2's padded region is addressable.
    backing = torch.zeros(
        num_blocks * tail_block_elems, dtype=torch.bfloat16, device=device
    )
    tail_kv_cache = torch.as_strided(
        backing,
        size=(num_blocks, 2, kpool, head_dim),
        stride=(tail_block_elems, kpool * head_dim, head_dim, 1),
    )

    n = kpool  # one full trailing tail for block 1 (blk=1 -> not block 0)
    t_base = 1 * kpool  # blk = 1
    tslot = torch.arange(t_base, t_base + n, dtype=torch.int64, device=device)
    key = torch.full((n, head_dim), 7.0, dtype=torch.bfloat16, device=device)
    score = torch.full((n, head_dim), -3.0, dtype=torch.bfloat16, device=device)

    live_mod.kpool_seed_tail_cache(tail_kv_cache, key, score, tslot, kpool, head_dim)

    # Every token here is exactly a full trailing pool (ahead-slot check
    # never finds a same-block neighbor since tslot is this call's only
    # data) -- t % kpool sweeps 0..kpool-1, so this covers the full block.
    written = tail_kv_cache[1, 0].float()  # block 1's K half
    assert torch.allclose(written, torch.full_like(written, 7.0)), (
        "kernel did not write block 1's K half at the padded stride offset "
        f"(TAIL_BLOCK_ELEMS={tail_block_elems}) -- got {written}"
    )
    written_score = tail_kv_cache[1, 1].float()
    assert torch.allclose(written_score, torch.full_like(written_score, -3.0))
    # Block 0 must be untouched -- this is exactly the corruption the old
    # dense-formula bug would have caused (block 1's writes landing inside
    # block 0's region instead).
    assert torch.count_nonzero(tail_kv_cache[0].float()) == 0


def test_recipe_wiring_if_present() -> None:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.is_file():
        return
    image = dockerfile.read_text()
    assert "COPY overlay/patch_kpool_tail_seed_stride.py" in image
    assert "COPY tests/test_kpool_tail_seed_stride.py" in image
    assert "RUN python3 /opt/glm53/patch_kpool_tail_seed_stride.py" in image
    assert "python3 /opt/glm53/test_kpool_tail_seed_stride.py" in image
    # Must run AFTER the slotmap clamp patch (both touch the same tail
    # cache path) and after the b12x MoE dependency install, to match how
    # this recipe's other kpool-adjacent patches are ordered.
    assert image.index(
        "RUN python3 /opt/glm53/patch_kpool_tail_slotmap.py"
    ) < image.index("RUN python3 /opt/glm53/patch_kpool_tail_seed_stride.py")


def main() -> int:
    test_offset_math_diverges_for_blk_ge_1()
    test_offset_math_matches_logical_shape_when_unpadded()
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_live_kernel_writes_at_strided_offset_if_gpu()
    test_recipe_wiring_if_present()
    print("kpool tail-seed stride patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
