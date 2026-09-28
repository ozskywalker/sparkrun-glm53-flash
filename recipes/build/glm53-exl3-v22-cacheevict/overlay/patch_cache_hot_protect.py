#!/usr/bin/env python3
"""Bookkeeping for GLM53_CACHE_HOT_PROTECT: remember prefix-cache hits.

Backport of Reederey87/glm53-flash-exl3-2x-dgx-spark PR #82 (2026-09-27
upstream-check round), adapted for this fork's build-time-patch/runtime-
flag convention (see patch_cache_tail_evict.py's docstring for why). This
one needed no logic change beyond that: the actual on/off decision already
lives inside cache_tail_evict.select_blocks's hot_protect_enabled() check,
which reads the env var every call. mark_reused/clear_reused bookkeeping
below always runs (cheap, and inert unless GLM53_CACHE_HOT_PROTECT=1 makes
select_blocks consult it) -- only the outer "skip editing the file if the
flag isn't set right now" gate was removed, since every other patch in
this fork's overlay applies unconditionally at image-build time.

Requires patch_cache_tail_evict.py to have already run (the tail-first
marker must be present in block_pool.py) -- hot-protect is one more
ranking key layered on top of depth-first eviction, not a standalone mode.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-cache-hot-protect]"
TAIL = "[glm53-cache-tail-evict]"
POOL = Path(
    os.environ.get(
        "GLM53_BLOCK_POOL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py",
    )
)
MANAGER = Path(
    os.environ.get(
        "GLM53_KV_MANAGER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/"
        "single_type_kv_cache_manager.py",
    )
)
UTILS = Path(
    os.environ.get(
        "GLM53_KV_UTILS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py",
    )
)

HIT_OLD = (
    "        # Touch the computed blocks to make sure they won't be evicted.\n"
    "        if self.enable_caching:\n"
    "            self.block_pool.touch(new_computed_blocks)\n"
)
HIT_NEW = (
    "        # Touch the computed blocks to make sure they won't be evicted.\n"
    "        if self.enable_caching:\n"
    "            self.block_pool.touch(new_computed_blocks)\n"
    "            # [glm53-cache-hot-protect] a hit is a reuse; one-shot blocks stay cold\n"
    "            import sys as _glm53_sys\n"
    "            if \"/opt/glm53\" not in _glm53_sys.path:\n"
    "                _glm53_sys.path.insert(0, \"/opt/glm53\")\n"
    "            from cache_tail_evict import mark_reused as _glm53_mark_reused\n"
    "            for _glm53_block in new_computed_blocks:\n"
    "                if _glm53_block.block_hash is not None:\n"
    "                    _glm53_mark_reused(_glm53_block.block_id)\n"
)
RESET_OLD = (
    "    def reset_hash(self):\n"
    '        """Reset the block hash when the block is evicted."""\n'
    "        self._block_hash = None\n"
    "        self._block_hash_num_tokens = None\n"
)
RESET_NEW = (
    "    def reset_hash(self):\n"
    '        """Reset the block hash when the block is evicted."""\n'
    "        # [glm53-cache-hot-protect] the block is about to hold new tokens\n"
    "        import sys as _glm53_sys\n"
    "        if \"/opt/glm53\" not in _glm53_sys.path:\n"
    "            _glm53_sys.path.insert(0, \"/opt/glm53\")\n"
    "        from cache_tail_evict import clear_reused as _glm53_clear_reused\n"
    "        _glm53_clear_reused(self.block_id)\n"
    "        self._block_hash = None\n"
    "        self._block_hash_num_tokens = None\n"
)


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".glm53-tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> int:
    if not POOL.is_file() or TAIL not in POOL.read_text():
        print(
            f"{MARK} needs the tail-first marker in {POOL} -- run "
            "patch_cache_tail_evict.py first",
            file=sys.stderr,
            flush=True,
        )
        return 1
    manager = MANAGER.read_text()
    utils = UTILS.read_text()
    if MARK in manager and MARK in utils:
        print(f"{MARK} already present in {MANAGER} and {UTILS}", flush=True)
        return 0
    if MARK in manager or MARK in utils:
        print(
            f"{MARK} partial install: manager={MARK in manager} utils={MARK in utils}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if manager.count(HIT_OLD) != 1 or utils.count(RESET_OLD) != 1:
        print(
            f"{MARK} anchor drift: hit={manager.count(HIT_OLD)} "
            f"reset={utils.count(RESET_OLD)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    new_manager = manager.replace(HIT_OLD, HIT_NEW, 1)
    new_utils = utils.replace(RESET_OLD, RESET_NEW, 1)
    if MARK not in new_manager or MARK not in new_utils:
        print(f"{MARK} replacement did not apply cleanly", file=sys.stderr, flush=True)
        return 1
    _write(MANAGER, new_manager)
    _write(UTILS, new_utils)
    print(
        f"{MARK} applied to {MANAGER} and {UTILS} "
        "(runtime-gated on GLM53_CACHE_HOT_PROTECT)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
