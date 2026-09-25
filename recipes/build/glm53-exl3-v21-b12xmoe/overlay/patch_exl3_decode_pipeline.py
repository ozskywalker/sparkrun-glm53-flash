#!/usr/bin/env python3
"""Add an opt-in SM121 K4/N256 thin-decode MoE pipeline; stock kernels untouched.

Backport of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks PR #217 (2026-09-18
upstream-recheck round; the thin-only split of PR #182 the maintainer
actually merged, after excluding a KDA FP8 large-M dispatch half whose own
numerical study found reproducible shifts at the geometry it engages --
none of that excluded half is ported here, by design). Ported against the
exact same pinned exllamav3 commit this fork already builds
(`c5d9c657966ffeeaa9353f0cc899f18629da4a13`, confirmed identical to what
upstream's patch docstring cites -- no drift to reconcile).

PR #217's own numerical study verdict is formally **INCONCLUSIVE**, not a
pass (predeclared T3a control gate failed on stock-vs-stock itself, an
instrument-calibration property of their material, not evidence against
the candidate -- see the PR body for the full study). The descriptive
serving A/B on their current-main head measured +7.71% to +14.59% median
decode tok/s in the range-resolved cells, two cells withheld by their own
tool's overlap rule. This backport ports the mechanism and its fail-closed
gating only; it does not re-run or re-validate that study against this
fork's checkpoint (288 experts, TP=2 across 2 hosts) or workload. Treat as
available-but-off, not benchmarked-and-recommended, until this fleet runs
its own trial.

What it does (additive only; stock `exl3_moe` kernels are byte-unchanged):
  * copies `quant/exl3_moe_kernel.cuh` to
    `quant/glm53_exl3_moe_fast_kernel.cuh` with a `bool shared_input`
    template parameter and the `glm53_exl3_moe_fast_kernel` entry point
    (1 fragment stage / 8 shared-memory stages, vs. stock 3/3);
  * when `shared_input` is true, the redundant gate/up input Hadamard is
    skipped and the up GEMM reads the already-transformed gate buffer;
  * host dispatch in `quant/exl3_moe.cu` selects the fast kernel only when
    K == 4, N_off == 1 (hidden % 256 == intermediate % 256 == 0),
    `GLM53_EXL3_MOE_FAST=1`, and the device is SM121 (12.1) -- checked via
    `cudaDeviceGetAttribute`, not assumed;
  * `shared_input` is true only when the gate/up SUH pointer tables are the
    identical allocation, aliased in `overlay/exl3.py` (this fork's own
    file, patched separately -- see its own diff/comment) after a
    load-time all-expert `torch.equal` proof, only in fast mode.

Usage (in the image build, same layer as the aarch64 stub + fat-kernel
patches, all three operating on the same checked-out exllamav3 source tree
before it's compiled):
    python3 patch_exl3_decode_pipeline.py /tmp/exllamav3/exllamav3/exllamav3_ext

Fails loud (RuntimeError, not silent) if any of its five native-source
anchors don't match exactly once, or if the patch's own marker
(`GLM53_EXL3_MOE_FAST` string) is already present in the target -- matching
this repo's `overlay/patch_exl3_fat_kernel.py` convention for native-source
patches (this one predates the atomic-write/`--status` convention used by
the pure-Python overlay patches, since it runs once against a throwaway
build-time checkout, not a long-lived installed file).
"""

from pathlib import Path
import sys


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one native source anchor: {old!r}')
    return text.replace(old, new, 1)


def patch(root):
    root = Path(root)
    quant = root / 'quant'
    host_path = quant / 'exl3_moe.cu'
    host = host_path.read_text()
    if 'GLM53_EXL3_MOE_FAST' in host:
        raise RuntimeError('Decode pipeline patch already present; use a clean source tree')
    kernel = (quant / 'exl3_moe_kernel.cuh').read_text()
    kernel = once(kernel, 'template<int t_bits, int MOE_TILESIZE_N>',
                  'template<int t_bits, int MOE_TILESIZE_N, bool shared_input>')
    kernel = once(kernel, 'void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)',
                  'void glm53_exl3_moe_fast_kernel(EXL3_MOE_KERNEL_ARGS)')
    redundant = '''                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );'''
    kernel = once(kernel, redundant,
                  '                if constexpr (!shared_input) {\n' + redundant + '\n                }')
    kernel = once(kernel,
                  'gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);',
                  'gemm_up(shared_input ? temp_state_g : temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);')
    wrapper = '''#include "exl3_moe_instances.cuh"
#undef MOE_FRAG_STAGES
#define MOE_FRAG_STAGES 1
#undef MOE_SH_STAGES
#define MOE_SH_STAGES 8
#include "../glm53_exl3_moe_fast_kernel.cuh"
fp_exl3_moe_kernel glm53_exl3_moe_fast_k4_n256(bool shared_input) {
    return shared_input ? glm53_exl3_moe_fast_kernel<4, 256, true>
                        : glm53_exl3_moe_fast_kernel<4, 256, false>;
}
'''
    helper = '''
#include <cstdlib>
#include <cstring>

fp_exl3_moe_kernel glm53_exl3_moe_fast_k4_n256(bool shared_input);

static bool glm53_fast_moe_enabled(int device) {
    static const bool requested = [] {
        const char* value = std::getenv("GLM53_EXL3_MOE_FAST");
        TORCH_CHECK(!value || !std::strcmp(value, "0") || !std::strcmp(value, "1"),
                    "GLM53_EXL3_MOE_FAST must be 0 or 1");
        return value && !std::strcmp(value, "1");
    }();
    if (!requested) return false;
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "Unexpected device index");
    static thread_local int supported[MAX_DEVICES] = {};
    if (!supported[device]) {
        int major = 0, minor = 0;
        cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        supported[device] = major == 12 && minor == 1 ? 1 : -1;
    }
    return supported[device] == 1;
}
'''
    host = once(host, '#include <set>', '#include <set>\n' + helper)
    anchor = '    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * K + N_off];'
    host = once(host, anchor, anchor + '''
    if (K == 4 && N_off == 1 && glm53_fast_moe_enabled(device)) {
        kernel = glm53_exl3_moe_fast_k4_n256(
            gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr());
    }
''')
    bindings_path = root / 'bindings.cpp'
    bindings = once(bindings_path.read_text(),
                    '    m.def("exl3_moe", &exl3_moe, "exl3_moe");',
                    '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'
                    '    m.def("glm53_fast_moe_version", []() { return 1; });')
    # Validate every anchor before writing any source file.
    (quant / 'glm53_exl3_moe_fast_kernel.cuh').write_text(kernel)
    (quant / 'comp_units/glm53_exl3_moe_fast.cu').write_text(wrapper)
    host_path.write_text(host)
    bindings_path.write_text(bindings)
    print(f'Installed opt-in SM121 K4/N256 decode pipeline into {root}')


if __name__ == '__main__':
    patch(sys.argv[1])
