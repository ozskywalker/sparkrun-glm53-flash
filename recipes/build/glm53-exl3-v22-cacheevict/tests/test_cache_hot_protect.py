"""Installer for the reused-prefix mark. The rank tests live next to tail-first.

Adapted from Reederey87/glm53-flash-exl3-2x-dgx-spark PR #82's test suite:
this fork's patch_cache_hot_protect.py always applies (given the tail-first
precondition) rather than gating the file edit on GLM53_CACHE_HOT_PROTECT --
that flag was already runtime-checked correctly inside
cache_tail_evict.hot_protect_enabled(), so no logic changed here, only the
installer's own now-removed flag gate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Repo checkout: tests/ and overlay/ are siblings. Built image: this test and
# everything it needs land flat in /opt/glm53/ (see the Dockerfile's COPY).
_SIBLING = HERE.parent / "overlay"
OVERLAY = _SIBLING if _SIBLING.is_dir() else HERE
INSTALLER = OVERLAY / "patch_cache_hot_protect.py"

MANAGER = """\
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
"""
UTILS = '''\
    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None
        self._block_hash_num_tokens = None
'''


def _env(pool: Path, manager: Path, utils: Path) -> dict:
    import os

    env = os.environ.copy()
    env["GLM53_BLOCK_POOL_PY"] = str(pool)
    env["GLM53_KV_MANAGER_PY"] = str(manager)
    env["GLM53_KV_UTILS_PY"] = str(utils)
    return env


def _run(tmp: Path, pool_text: str) -> subprocess.CompletedProcess[str]:
    pool = tmp / "block_pool.py"
    manager = tmp / "manager.py"
    utils = tmp / "utils.py"
    pool.write_text(pool_text)
    manager.write_text(MANAGER)
    utils.write_text(UTILS)
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=_env(pool, manager, utils),
        text=True,
        capture_output=True,
        check=False,
    )


def test_requires_the_tail_marker(tmp_path: Path):
    result = _run(tmp_path, "no marker here\n")
    assert result.returncode == 1
    assert (tmp_path / "manager.py").read_text() == MANAGER
    assert (tmp_path / "utils.py").read_text() == UTILS


def test_patches_once(tmp_path: Path):
    result = _run(tmp_path, "# [glm53-cache-tail-evict]\n")
    assert result.returncode == 0, result.stderr
    manager = (tmp_path / "manager.py").read_text()
    utils = (tmp_path / "utils.py").read_text()
    assert manager.count("[glm53-cache-hot-protect]") == 1
    assert "mark_reused" in manager
    assert utils.count("[glm53-cache-hot-protect]") == 1
    assert "clear_reused" in utils
    again = subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=_env(tmp_path / "block_pool.py", tmp_path / "manager.py", tmp_path / "utils.py"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert again.returncode == 0, again.stderr
    assert (tmp_path / "manager.py").read_text() == manager
    assert (tmp_path / "utils.py").read_text() == utils


def test_drift_writes_nothing(tmp_path: Path):
    pool = tmp_path / "block_pool.py"
    manager = tmp_path / "manager.py"
    utils = tmp_path / "utils.py"
    pool.write_text("# [glm53-cache-tail-evict]\n")
    manager.write_text(MANAGER.replace("touch(new_computed_blocks)", "touch(blocks)"))
    utils.write_text(UTILS)
    before_m = manager.read_text()
    before_u = utils.read_text()
    result = subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=_env(pool, manager, utils),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert manager.read_text() == before_m
    assert utils.read_text() == before_u
