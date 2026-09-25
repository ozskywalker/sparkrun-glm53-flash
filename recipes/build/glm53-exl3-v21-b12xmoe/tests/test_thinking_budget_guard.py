#!/usr/bin/env python3
"""Regression tests for the thinking-budget sampler-guard fix."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_thinking_budget_guard.py",
        ROOT / "overlay" / "patch_thinking_budget_guard.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_thinking_budget_guard import (  # noqa: E402
    ANCHOR,
    MARK,
    PATCHED,
    prepare,
    verified_state,
)

INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/sampler.py"
)

# Minimal self-contained fixture: just enough surrounding structure for the
# patch to find its anchor and the result to compile.
PINNED_FIXTURE = "import numpy as np\n\n\nclass Sampler:\n" + ANCHOR


def _run_patch(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_SAMPLER_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


class _FakeArrayState:
    """Duck-typed stand-in for the small per-field state objects the real
    guard reaches into (LogitBiasState, PenaltiesState, BadWordsState,
    SamplingStates, ThinkingBudgetState) -- only the attributes the guard
    actually reads."""


def _requires_logits_processing_fn(source_text: str):
    """Compile ANCHOR or PATCHED into a real function object and return it,
    bound to nothing -- call as fn(fake_self, idx_mapping_np)."""
    ns: dict = {"np": __import__("numpy")}
    exec(compile("class _Probe:\n" + source_text, "<probe>", "exec"), ns)
    return ns["_Probe"]._requires_logits_processing


def _make_fake_self(*, use_thinking_budget: bool, thinking_enabled: bool = True):
    import numpy as np

    n = 4
    self_ = types.SimpleNamespace()
    self_.logit_bias_state = _FakeArrayState()
    self_.logit_bias_state.use_logit_bias = np.zeros(n, dtype=bool)
    self_.penalties_state = _FakeArrayState()
    self_.penalties_state.use_penalty = np.zeros(n, dtype=bool)
    self_.bad_words_state = _FakeArrayState()
    self_.bad_words_state.num_bad_words = _FakeArrayState()
    self_.bad_words_state.num_bad_words.np = np.zeros(n, dtype=int)

    states = _FakeArrayState()
    vocab_size = 1000
    states.vocab_size = vocab_size
    states.temperature = _FakeArrayState()
    states.temperature.np = np.zeros(n, dtype=float)  # all greedy (default)
    states.min_p = _FakeArrayState()
    states.min_p.np = np.zeros(n, dtype=float)
    states.top_k = _FakeArrayState()
    states.top_k.np = np.full(n, vocab_size, dtype=int)
    states.top_p = _FakeArrayState()
    states.top_p.np = np.ones(n, dtype=float)
    self_.sampling_states = states

    tbs = _FakeArrayState()
    tbs.enabled = thinking_enabled
    if thinking_enabled:
        tbs.use_thinking_budget = np.zeros(n, dtype=bool)
        tbs.use_thinking_budget[0] = use_thinking_budget
    self_.thinking_budget_state = tbs
    return self_


def test_unpatched_guard_misses_thinking_budget_only_request() -> None:
    """RED: proves the bug. A request with ONLY thinking_token_budget set
    (everything else at its default value) must be exactly the case that
    silently skips budget enforcement pre-fix."""
    import numpy as np

    fn = _requires_logits_processing_fn(ANCHOR)
    fake_self = _make_fake_self(use_thinking_budget=True)
    idx_mapping_np = np.array([0, 1, 2, 3])
    assert fn(fake_self, idx_mapping_np) is False, (
        "expected the UNPATCHED guard to (buggily) return False here -- if "
        "it now returns True, the anchor text may have drifted from what "
        "this test assumes"
    )


def test_patched_guard_catches_thinking_budget_only_request() -> None:
    """GREEN: same exact scenario as the RED test above, but against the
    fixed guard -- must now return True."""
    import numpy as np

    fn = _requires_logits_processing_fn(PATCHED)
    fake_self = _make_fake_self(use_thinking_budget=True)
    idx_mapping_np = np.array([0, 1, 2, 3])
    assert fn(fake_self, idx_mapping_np) is True


def test_patched_guard_unchanged_when_thinking_budget_not_set() -> None:
    """The fix must not change behavior for requests that don't use a
    thinking budget at all -- still False when nothing else is non-default."""
    import numpy as np

    fn = _requires_logits_processing_fn(PATCHED)
    fake_self = _make_fake_self(use_thinking_budget=False)
    idx_mapping_np = np.array([0, 1, 2, 3])
    assert fn(fake_self, idx_mapping_np) is False


def test_patched_guard_handles_thinking_budget_state_disabled() -> None:
    """ThinkingBudgetState.use_thinking_budget only exists as an attribute
    when .enabled is True (see thinking_budget.py's __init__) -- the fix
    must check .enabled first or this raises AttributeError instead of
    returning False."""
    import numpy as np

    fn = _requires_logits_processing_fn(PATCHED)
    fake_self = _make_fake_self(use_thinking_budget=False, thinking_enabled=False)
    idx_mapping_np = np.array([0, 1, 2, 3])
    assert fn(fake_self, idx_mapping_np) is False


def test_patched_guard_still_catches_existing_conditions() -> None:
    """The fix must not disturb any of the guard's pre-existing checks --
    non-default temperature alone must still trigger it."""
    import numpy as np

    fn = _requires_logits_processing_fn(PATCHED)
    fake_self = _make_fake_self(use_thinking_budget=False)
    fake_self.sampling_states.temperature.np[0] = 0.7
    idx_mapping_np = np.array([0, 1, 2, 3])
    assert fn(fake_self, idx_mapping_np) is True


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "sampler.py"
        target.write_text(PINNED_FIXTURE)
        first = _run_patch(target)
        assert first.returncode == 0, first.stderr
        text = target.read_text()
        assert verified_state(text)
        assert MARK in text
        assert "if self.thinking_budget_state.enabled and np.any(" in text
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
        "return bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))",
        "return bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))  # drift",
        1,
    )
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "sampler.py"
        target.write_text(drifted)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert target.read_text() == drifted

    partial = PINNED_FIXTURE.replace(ANCHOR, MARK + ANCHOR, 1)
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "sampler.py"
        target.write_text(partial)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_SAMPLER_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "sampler.py"
        target.write_text(src.read_text())
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr
        assert verified_state(target.read_text())


def test_recipe_wiring_if_present() -> None:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.is_file():
        return
    image = dockerfile.read_text()
    assert "COPY overlay/patch_thinking_budget_guard.py" in image
    assert "COPY tests/test_thinking_budget_guard.py" in image
    assert "RUN python3 /opt/glm53/patch_thinking_budget_guard.py" in image
    assert "python3 /opt/glm53/test_thinking_budget_guard.py" in image


def main() -> int:
    test_unpatched_guard_misses_thinking_budget_only_request()
    test_patched_guard_catches_thinking_budget_only_request()
    test_patched_guard_unchanged_when_thinking_budget_not_set()
    test_patched_guard_handles_thinking_budget_state_disabled()
    test_patched_guard_still_catches_existing_conditions()
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("thinking-budget guard patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
