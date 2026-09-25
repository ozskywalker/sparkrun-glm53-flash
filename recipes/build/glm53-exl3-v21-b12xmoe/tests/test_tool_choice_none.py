#!/usr/bin/env python3
"""Apply overlay/patch_tool_choice_none.py to a copy of vLLM's glm47_moe.py.

Matches this project's house pattern for anchor-patch tests (see
test_default_max_new_tokens.py, test_apc_no_store.py): copy the real
installed vLLM target into a temp dir, apply, assert idempotence and
--status, and confirm a drifted/foreign anchor fails closed without
touching the file. Run inside a container with vllm installed (or point
GLM53_GLM47_MOE_PARSER_PY_SRC elsewhere) -- there is no GPU at Docker-build
time, so this checks the patch mechanism, not live decode behavior; the
`bad_words`-masks-the-opener claim itself was validated by upstream on real
hardware (PR #215's own live-window record) before this backport.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_tool_choice_none.py",
        HERE.parent / "overlay" / "patch_tool_choice_none.py",
    )
    if p.is_file()
)

DEFAULT_SRC = Path(
    os.environ.get(
        "GLM53_GLM47_MOE_PARSER_PY_SRC",
        "/usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py",
    )
)

MARK = "# [glm53-tool-choice-none]"


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")
    if not DEFAULT_SRC.is_file():
        raise SystemExit(
            f"missing {DEFAULT_SRC} -- run inside a container with vllm "
            "installed, or set GLM53_GLM47_MOE_PARSER_PY_SRC"
        )

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "glm47_moe.py"
        shutil.copyfile(DEFAULT_SRC, target)
        pristine = target.read_text(encoding="utf-8")

        # applies cleanly, parses, carries the marker exactly once
        subprocess.check_call([sys.executable, str(PATCH), str(target)])
        patched = target.read_text(encoding="utf-8")
        assert patched.count(MARK) == 1, "marker not exactly once after apply"
        ast.parse(patched, str(target))
        assert "def adjust_request(" in patched
        assert "request.bad_words.append(TOOL_CALL_START)" in patched
        # the original _handle_tool_end body must still be present, untouched
        assert "self._tool_slots[idx].name = self._tool_slots[idx].name.strip()" in patched

        # idempotent: second apply is a no-op, byte-identical
        subprocess.check_call([sys.executable, str(PATCH), str(target)])
        assert target.read_text(encoding="utf-8") == patched, "not idempotent"

        # --status reports applied
        status = subprocess.run(
            [sys.executable, str(PATCH), "--status", str(target)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "APPLIED" in status.stdout and "NOT APPLIED" not in status.stdout

        # a file carrying the marker but with a hand-edited (drifted) region
        # is refused, not silently "skipped" -- restore pristine, apply, then
        # corrupt just the marked block and re-run.
        drifted = patched.replace(
            "request.bad_words.append(TOOL_CALL_START)",
            "request.bad_words.append(TOOL_CALL_START)  # tampered",
        )
        target.write_text(drifted, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(PATCH), str(target)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "drifted region should fail closed"
        assert target.read_text(encoding="utf-8") == drifted, "drift handling touched the file"

        # missing anchor (foreign/edited file) fails closed, no partial write
        target.write_text(pristine.replace("_handle_tool_end", "_handle_tool_end_renamed"), encoding="utf-8")
        before = target.read_text(encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(PATCH), str(target)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "missing anchor should fail closed"
        assert target.read_text(encoding="utf-8") == before, "missing-anchor path touched the file"

        # file permissions survive the atomic write (patch_apc_no_store.py's
        # 2026-09-16 production incident: mkstemp+replace silently drops mode)
        target.write_text(pristine, encoding="utf-8")
        target.chmod(0o644)
        subprocess.check_call([sys.executable, str(PATCH), str(target)])
        assert oct(target.stat().st_mode)[-3:] == "644", "atomic write changed file mode"

    print("tool-choice-none patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
