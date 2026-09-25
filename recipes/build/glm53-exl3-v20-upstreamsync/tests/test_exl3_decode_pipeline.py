#!/usr/bin/env python3
"""CPU-only checks for overlay/patch_exl3_decode_pipeline.py (PR #217 backport).

Two things get checked without a GPU or a real exllamav3 checkout:

1. The native-source patcher's anchor-based text transformation, against a
   synthetic tree that reproduces just the five exact anchor strings the
   real patch matches (copied from the patch's own source, not re-derived --
   this catches accidental edits to those literals, not exllamav3 drift;
   the Dockerfile's own build-time `hasattr(exllamav3_ext,
   'glm53_fast_moe_version')` + `glm53_fast_moe_version() == 1` assertion is
   what proves it against the real pinned commit's real source).
2. `overlay/exl3.py`'s `exl3_moe_fast_requested()` -- the pure-Python
   GLM53_EXL3_MOE_FAST=0/1 validation. `exl3.py` itself imports real
   `torch`/`vllm` at module level and its classes self-register with
   vLLM's quantization registry on import (`register_quantization_config`)
   -- loading the whole file a second time via importlib, after
   test_exl3_overlay.py has already imported the real installed copy,
   risks a duplicate-registration collision that has nothing to do with
   what this test actually wants to check. So this extracts just the one
   function's AST and execs it in an isolated namespace instead of
   importing the module -- no torch/vllm dependency, no registry
   side-effect, still a real test of the actual shipped source text (not
   a hand-copied reimplementation that could silently drift from it).

Same style as tests/test_apc_no_store.py and tests/test_kv_capacity_log.py:
copy/import into a tmpdir, exercise apply + fail-closed + idempotency,
assert on markers and exceptions rather than on GPU output.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_exl3_decode_pipeline.py",
        HERE.parent / "overlay" / "patch_exl3_decode_pipeline.py",
    )
    if p.is_file()
)
EXL3_PY = next(
    p
    for p in (
        HERE.parent / "overlay" / "exl3.py",  # source tree (this repo)
        Path(
            os.environ.get(
                "GLM53_EXL3_PY",
                "/usr/local/lib/python3.12/dist-packages/vllm/model_executor"
                "/layers/quantization/exl3.py",
            )
        ),  # installed image (COPY overlay/exl3.py -> vllm's own module path)
    )
    if p.is_file()
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("patch_exl3_decode_pipeline", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The five literal anchors patch_exl3_decode_pipeline.py's `once()` calls
# match against, copied verbatim from that file so this test breaks the
# instant either side drifts without the other.
_KERNEL_SRC = """template<int t_bits, int MOE_TILESIZE_N>
__global__ __launch_bounds__(NUM_THREADS)
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    if (0) {
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
    }
    gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
}
"""

_HOST_SRC = """#include <set>
#include "exl3_moe.h"

fp_exl3_moe_kernel dispatch(int K, int N_off, int device) {
    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * K + N_off];
    return kernel;
}
"""

_BINDINGS_SRC = """#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exl3_moe", &exl3_moe, "exl3_moe");
}
"""


def _make_tree(root: Path) -> None:
    quant = root / "quant"
    (quant / "comp_units").mkdir(parents=True)
    (quant / "exl3_moe_kernel.cuh").write_text(_KERNEL_SRC)
    (quant / "exl3_moe.cu").write_text(_HOST_SRC)
    (root / "bindings.cpp").write_text(_BINDINGS_SRC)


def check_apply_creates_expected_files() -> None:
    mod = _load_patch_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_tree(root)
        mod.patch(str(root))
        fast_kernel = root / "quant" / "glm53_exl3_moe_fast_kernel.cuh"
        wrapper = root / "quant" / "comp_units" / "glm53_exl3_moe_fast.cu"
        check(
            "apply-creates-fast-kernel-file",
            fast_kernel.is_file() and "glm53_exl3_moe_fast_kernel" in fast_kernel.read_text(),
        )
        check(
            "apply-creates-comp-unit",
            wrapper.is_file() and "MOE_FRAG_STAGES 1" in wrapper.read_text(),
        )
        host = (root / "quant" / "exl3_moe.cu").read_text()
        check("apply-adds-flag-name-to-host", "GLM53_EXL3_MOE_FAST" in host)
        check(
            "apply-adds-dispatch-guard",
            "glm53_fast_moe_enabled(device)" in host and "K == 4 && N_off == 1" in host,
        )
        bindings = (root / "bindings.cpp").read_text()
        check("apply-adds-version-binding", "glm53_fast_moe_version" in bindings)
        check(
            "apply-does-not-touch-stock-symbol",
            'm.def("exl3_moe", &exl3_moe, "exl3_moe");' in bindings,
        )


def check_reapply_refused() -> None:
    mod = _load_patch_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_tree(root)
        mod.patch(str(root))
        try:
            mod.patch(str(root))
            check("reapply-refused", False, "second patch() call did not raise")
        except RuntimeError as exc:
            check("reapply-refused", "already present" in str(exc), str(exc))


def check_missing_anchor_fails_closed() -> None:
    mod = _load_patch_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_tree(root)
        # Corrupt the one anchor the redundant-Hadamard hunk depends on.
        kernel_path = root / "quant" / "exl3_moe_kernel.cuh"
        kernel_path.write_text(kernel_path.read_text().replace("had_hf_r_128_inner", "had_hf_drifted"))
        try:
            mod.patch(str(root))
            check("missing-anchor-fails-closed", False, "patch() did not raise on drifted anchor")
        except RuntimeError as exc:
            check("missing-anchor-fails-closed", "anchor" in str(exc).lower(), str(exc))


def _extract_function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name!r} not found in exl3.py")


def check_fast_requested_validation() -> None:
    src = EXL3_PY.read_text()
    func_src = _extract_function(src, "exl3_moe_fast_requested")
    ns: dict = {"os": os}
    exec(func_src, ns)
    fast_requested = ns["exl3_moe_fast_requested"]

    old_env = dict(os.environ)
    try:
        os.environ.pop("GLM53_EXL3_MOE_FAST", None)
        check("fast-default-off", fast_requested() is False)

        os.environ["GLM53_EXL3_MOE_FAST"] = "1"
        check("fast-explicit-on", fast_requested() is True)

        os.environ["GLM53_EXL3_MOE_FAST"] = "0"
        check("fast-explicit-off", fast_requested() is False)

        os.environ["GLM53_EXL3_MOE_FAST"] = "yes"
        try:
            fast_requested()
            check("fast-rejects-non-0-1", False, "did not raise on GLM53_EXL3_MOE_FAST=yes")
        except RuntimeError as exc:
            check("fast-rejects-non-0-1", "0 or 1" in str(exc), str(exc))
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def main() -> int:
    check_apply_creates_expected_files()
    check_reapply_refused()
    check_missing_anchor_fails_closed()
    check_fast_requested_validation()
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed: {FAILURES}", file=sys.stderr)
        return 1
    print("\nALL EXL3 DECODE-PIPELINE PATCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
