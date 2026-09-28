#!/usr/bin/env python3
"""Honor ``tool_choice:"none"`` at decode time (glm47 parser hook).

Backport of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks PR #215 (2026-09-17
upstream-check round, field-validated on their own 2x-Spark deployment).
Anchor verified byte-for-byte against this image's installed vLLM before
being written here.

Problem: with tools declared, ``tool_choice:"none"`` keeps the tool block in
the prompt (this deployment relies on that shared-prefix retention -- see
``overlay/patch_apc_no_store.py``'s own reasoning for why the prompt is not
trimmed). The API layer already strips ``tool_calls`` from the *response*
three ways, but nothing stops the model from *generating* ``<tool_call>...``
in the first place -- so the answer step has no text, and a strict client
collector sees a complete tool call with no answer and fails closed.

Mechanism: ``Glm47MoeParser`` does not currently override ``adjust_request``
(only ``_emit_name_delta``, ``_handle_tool_end``, ``is_reasoning_end`` are
overridden -- verified against this image). This adds that override,
chaining through the real base implementation first
(``vllm/parser/engine/parser_engine.py``'s ``ParserEngine.adjust_request``,
the class this parser directly extends -- it sets
``request.skip_special_tokens = False``, it is not a no-op; verified by
reading it, not assumed). For a chat request with
``tool_choice == "none"`` and ``tools`` set, it appends the parser's own
``TOOL_CALL_START`` (``"<tool_call>"``) to ``request.bad_words``. The v1
sampler masks that opener at every decode position, including every
spec-decode draft position on the verify path -- our shipped MTP-2
speculator's drafted tokens are still subject to ``bad_words`` rejection
like any other candidate, so a drafted opener is simply rejected, not
silently accepted. Prompt bytes and usage accounting are unchanged; a
client-supplied ``bad_words`` list is preserved (this only appends).

This fix is only live because ``--reasoning-parser glm45`` is set in this
fork's launch command (confirmed in every ``recipes/*-vllm.yaml``) -- that
flag is what makes ``adjust_request`` run for every chat request here. If a
future recipe ever drops ``--reasoning-parser``, this patch goes inert
(silently -- there is no launch-time gate for that, matching upstream's own
scope; not worth a bespoke preflight for a flag this fork has never shipped
without).

Not covered (unchanged from upstream, not attempted here either): beam
search + none; the Responses API's none+tools path; opener spellings beyond
this tokenizer's own encoding of ``<tool_call>``.

Fail closed if the anchor drifts. Same atomic-write technique as
``overlay/patch_apc_no_store.py`` (temp file + ``os.replace``, explicit
``os.chmod`` to the original mode first) -- a bare ``mkstemp``+``replace``
silently drops file permissions and crashed this fleet in production once
already (2026-09-16); never repeat that pattern.
"""
from __future__ import annotations

import ast
import os
import stat
import sys
import tempfile
from pathlib import Path

MARK = "# [glm53-tool-choice-none]"

P = Path(
    os.environ.get(
        "GLM53_GLM47_MOE_PARSER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py",
    )
)

OLD = """    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)
"""

NEW = (
    """    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        """
    + MARK
    + """ tools stay in the prompt (shared-prefix
        # retention); keep the model from *opening* a call at decode time
        # instead, by masking the opener token like any other bad word.
        if (
            isinstance(request, ChatCompletionRequest)
            and getattr(request, "tool_choice", None) == "none"
            and getattr(request, "tools", None)
        ):
            request.bad_words.append(TOOL_CALL_START)
        return request

"""
    + OLD
)


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status): applied|skipped|missing:<reason>."""
    if MARK in src:
        if src.count(NEW) != 1:
            return src, "missing:drifted-region"
        return src, "skipped"
    if src.count(OLD) != 1:
        return src, "missing:handle-tool-end-anchor"
    if src.count('TOOL_CALL_START = "<tool_call>"') != 1:
        return src, "missing:tool-call-start-const"
    if "ChatCompletionRequest" not in src or "ResponsesRequest" not in src:
        return src, "missing:request-type-imports"
    updated = src.replace(OLD, NEW, 1)
    ast.parse(updated, str(P))
    return updated, "applied"


def atomic_write(path: Path, text: str) -> None:
    """temp file + os.replace, preserving the target's original mode.

    mkstemp creates its temp file 0600 regardless of the target's real
    permissions; os.replace is a rename, so that mode would otherwise stick.
    A non-root runtime user losing read access to a just-patched vLLM file
    crashed this fleet at first request once already -- always re-apply the
    original mode before replacing.
    """
    orig_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".glm53", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, orig_mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    status_only = len(argv) > 1 and argv[1] == "--status"
    target = Path(argv[2 if status_only else 1]) if len(argv) > (2 if status_only else 1) else P
    if status_only:
        applied = target.is_file() and MARK in target.read_text(encoding="utf-8")
        print("tool-choice-none               :", "APPLIED" if applied else "NOT APPLIED")
        return 0
    if not target.is_file():
        print(f"[glm53-tool-choice-none] missing {target}", file=sys.stderr)
        return 1
    text = target.read_text(encoding="utf-8")
    updated, status = apply_text(text)
    if status not in ("applied", "skipped"):
        print(f"[glm53-tool-choice-none] {status}: {target}", file=sys.stderr)
        return 1
    if status == "applied":
        atomic_write(target, updated)
    print(f"[glm53-tool-choice-none] {status}: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
