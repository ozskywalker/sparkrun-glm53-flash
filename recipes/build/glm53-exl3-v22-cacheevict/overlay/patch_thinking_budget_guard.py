#!/usr/bin/env python3
"""Fix `_requires_logits_processing`'s missing thinking-budget check.

`Sampler.apply_sampling_params` (Model Runner V2's GPU sampler --
confirmed this deployment actually runs V2, not V1: this fork's own real
boot log's `model_runner.py` line numbers for "Loading model from
scratch"/"Estimated CUDA graph memory"/"Graph capturing finished" match
`vllm/v1/worker/gpu/model_runner.py` exactly, nowhere near the V1
candidate's line numbers) starts with::

    if not self._requires_logits_processing(idx_mapping_np):
        return logits

and later, unconditionally further down, calls::

    self.thinking_budget_state.apply(...)

`_requires_logits_processing` checks logit_bias / penalties / bad_words /
non-default temperature / min_p / top_k / top_p, but never checks
`thinking_budget_state`/`use_thinking_budget` at all. Net effect: a
request with `sampling_params.thinking_token_budget` set, but otherwise
default sampling (temperature exactly 0.0 or 1.0, nothing else
non-default -- the overwhelmingly common case for real API callers),
returns early and never reaches `thinking_budget_state.apply(...)`. The
forced reasoning-end marker never gets inserted once the budget is
exceeded -- silently, no error, no log line.

Confirmed reachable in this deployment specifically: `--reasoning-parser
glm45` feeds real non-empty `reasoning_start_token_ids`/
`reasoning_end_token_ids`/`natural_reasoning_end_token_ids` into
`ThinkingBudgetState.__init__`, so `self.enabled = True` there (not
short-circuited off by an empty reasoning config -- see
`thinking_budget.py`, whose `__init__` only sets
`self.use_thinking_budget` inside the `if not self.enabled: return`
early-return's complement, i.e. it does NOT exist as an attribute at all
when `enabled` is False -- any fix must check `.enabled` before touching
`.use_thinking_budget`, not just add a bare `np.any(...)` call).

Independently confirmed as a real, previously-shipped fix by
mmastrac/glm-5.3-flash-4x-gx10 (commit `a30d4b24f`) hitting the identical
bug in the identical guard on their own deployment.

Mechanism: this fork's own build-time regex/string-replace patch pattern
(see `patch_kpool_tail_seed_stride.py` for the template). Fail-closed,
idempotent, preflights the pinned anchor before writing.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_SAMPLER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/"
        "sampler.py",
    )
)
MARK = (
    "        # [glm53-thinking-budget-guard] thinking_budget_state.apply() is\n"
)

ANCHOR = '''    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return True
        if np.any(self.penalties_state.use_penalty[idx_mapping_np]):
            return True
        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return True

        states = self.sampling_states
        temperatures = states.temperature.np[idx_mapping_np]
        if np.any((temperatures != 0.0) & (temperatures != 1.0)):
            return True
        if np.any(states.min_p.np[idx_mapping_np] != 0.0):
            return True
        if np.any(states.top_k.np[idx_mapping_np] != states.vocab_size):
            return True
        return bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))
'''

PATCHED = '''    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return True
        if np.any(self.penalties_state.use_penalty[idx_mapping_np]):
            return True
        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return True
        # [glm53-thinking-budget-guard] thinking_budget_state.apply() is
        # called unconditionally later in apply_sampling_params, but this
        # guard never checked for it -- a request with thinking_token_budget
        # set and otherwise-default sampling (temp 0.0/1.0, nothing else
        # non-default) returned early above and silently never got its
        # budget enforced. use_thinking_budget only exists as an attribute
        # when .enabled is True (see ThinkingBudgetState.__init__), so check
        # .enabled first rather than assume the attribute exists.
        if self.thinking_budget_state.enabled and np.any(
            self.thinking_budget_state.use_thinking_budget[idx_mapping_np]
        ):
            return True

        states = self.sampling_states
        temperatures = states.temperature.np[idx_mapping_np]
        if np.any((temperatures != 0.0) & (temperatures != 1.0)):
            return True
        if np.any(states.min_p.np[idx_mapping_np] != 0.0):
            return True
        if np.any(states.top_k.np[idx_mapping_np] != states.vocab_size):
            return True
        return bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))
'''


def verified_state(text: str) -> bool:
    return (
        text.count(ANCHOR) == 0
        and text.count(PATCHED) == 1
        and text.count(MARK) == 1
        and "if self.thinking_budget_state.enabled and np.any(" in text
    )


def prepare(source: str) -> tuple[str, str]:
    mark_count = source.count(MARK)
    if mark_count:
        if mark_count != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent thinking-budget guard patch "
                f"(marker={mark_count})"
            )
        return source, "already present"
    if verified_state(source):
        return source, "already patched"

    n = source.count(ANCHOR)
    if n != 1:
        raise ValueError(
            f"pinned _requires_logits_processing anchor drifted (occurrences={n})"
        )

    patched = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("thinking-budget guard post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-thinking-budget.tmp")
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
    for pyc in cache.glob("sampler*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"thinking-budget guard preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: thinking-budget guard {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
