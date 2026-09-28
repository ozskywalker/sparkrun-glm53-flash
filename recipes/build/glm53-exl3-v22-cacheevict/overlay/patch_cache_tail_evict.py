#!/usr/bin/env python3
"""When GLM53_CACHE_TAIL_EVICT=1, evict the deepest cached block first.

Backport of Reederey87/glm53-flash-exl3-2x-dgx-spark PR #81 (2026-09-27
upstream-check round), adapted for this fork's build-time-patch/runtime-
flag convention (see patch_kv_capacity_log.py): every other overlay patch
in this fork applies unconditionally at image-build time and reads its
env var inside the installed code at real request time, rather than
gating whether the file gets edited at all -- upstream's own script gates
the file edit itself, which would never fire under a plain `docker build`
with no env var set. Ported the mechanism (depth-first eviction among
cached blocks, unhashed blocks still go first), not the gating style.

Flag off (anything but the literal "1", checked every call): falls back
to the stock ``popleft_n`` pop, byte-identical to unpatched vLLM. Flag on:
routes through ``cache_tail_evict.select_blocks`` instead. Rollback is the
flag at 0 -- no rebuild needed, since the check happens at call time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-cache-tail-evict]"
POOL = Path(
    os.environ.get(
        "GLM53_BLOCK_POOL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py",
    )
)

POP_OLD = "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
POP_NEW = (
    "        ret: list[KVCacheBlock] = "
    "self._glm53_tail_first_blocks(num_blocks)  # [glm53-cache-tail-evict]\n"
)
DEF_OLD = "    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:\n"
DEF_NEW = '''    def _glm53_tail_first_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """[glm53-cache-tail-evict] Deepest cached block first when
        GLM53_CACHE_TAIL_EVICT=1 (checked every call, not just at build
        time); stock LRU pop otherwise. See cache_tail_evict.py."""
        import os as _glm53_os

        if _glm53_os.environ.get("GLM53_CACHE_TAIL_EVICT", "0").strip() != "1":
            return self.free_block_queue.popleft_n(num_blocks)
        import sys as _glm53_sys

        if "/opt/glm53" not in _glm53_sys.path:
            _glm53_sys.path.insert(0, "/opt/glm53")
        from cache_tail_evict import select_blocks as _glm53_select_blocks

        return _glm53_select_blocks(self.free_block_queue, num_blocks)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
'''


def main() -> int:
    text = POOL.read_text()
    if MARK in text:
        print(f"{MARK} already present in {POOL}", flush=True)
        return 0
    if text.count(POP_OLD) != 1 or text.count(DEF_OLD) != 1:
        print(
            f"{MARK} anchor drift in {POOL}: "
            f"pop={text.count(POP_OLD)} def={text.count(DEF_OLD)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    updated = text.replace(DEF_OLD, DEF_NEW, 1).replace(POP_OLD, POP_NEW, 1)
    if updated.count(MARK) < 2 or POP_OLD in updated:
        print(f"{MARK} replacement did not apply cleanly", file=sys.stderr, flush=True)
        return 1
    tmp = POOL.with_suffix(".py.glm53-tmp")
    tmp.write_text(updated)
    os.replace(tmp, POOL)
    print(f"{MARK} applied to {POOL} (runtime-gated on GLM53_CACHE_TAIL_EVICT)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
