# SPDX-License-Identifier: MIT AND Apache-2.0
#
# This file is Enntity/sparkglm's overlay/exl3.py, taken wholesale (not this
# project's own overlay/exl3.py from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-
# Sparks @ c190db1) to graft in their grouped-prefill fat-expert dispatch
# (EXL3_GROUPED_PREFILL_K4). Provenance for the grouped-prefill and
# cooperative-decode lineages: docs/PROVENANCE.md in that repository --
# briefly, K4 trellis/Hadamard/MMA from turboderp-org/exllamav3
# @c5d9c657966ffeeaa9353f0cc899f18629da4a13, the M64 cp.async fat-GEMM
# pipeline from Reederey87/glm53-flash-exl3-2x-dgx-spark@0c03250, and the
# grouped-prefill task planner/phase-wide scheduling as sparkglm's own
# original work.
#
# Scope decision for this project (grouped-prefill only, not decode-coop):
# EXL3_DECODE_COOP_K4 code paths are left in this file unmodified but are
# INERT here -- exl3_decode_moe.cu/.cuh (the separate file those paths call
# into) is deliberately NOT compiled into this build's extension, and this
# file's own dispatch already guards every decode-coop call site with
# `hasattr(exllamav3_ext, "exl3_decode_moe_k4")` (confirmed by reading this
# file directly), so a missing symbol degrades to "not available," not a
# crash. EXL3_DECODE_COOP_K4 must never be set in this project's recipes.
"""EXL3/MCG trellis quantization for GLM-5.3-Flash routed experts.

Checkpoint ABI (brandonmusic/GLM-5.3-Flash-tr3-4bpw):
  quant_method=exl3, codebook=mcg, scope=glm53_routed_experts_only
  per expert matrix: trellis (int16) + suh/svh (fp16) + mcg (int32 marker)

Non-routed tensors stay native (UnquantizedLinearMethod). Experts never
expand to a persistent BF16 weight; LinearEXL3 / exllamav3_ext runs the
trellis GEMM. TP=2 shards gate/up column-wise and down row-wise; the MoE
runner all-reduces the combined output.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

EXLLAMAV3_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
EXLLAMAV3_VERSION = "0.0.43"
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")
SWIGLU_LIMIT_DEFAULT = 10.0
# Default fused-kernel temp rows/expert. 1024 covers MNBT=1024 in one launch
# but measured slower than 128+fallback (P2b). Override with EXL3_TEMP_ROWS_FUSED.
TEMP_ROWS_FUSED = 128
MOE_ACT_SILU = 0
# Shared fused scratch: decode is sequential across layers.
_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_DECODE_COOP_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_GROUPED_PREFILL_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_COUNT_CACHE: dict[tuple, tuple[torch.Tensor, torch.cuda.Stream]] = {}
# E3 grouped fat-expert scratch (v11-e3 addition): fat-row activation
# buffers, grown once. Deliberately a DIFFERENT cache from
# _GROUPED_PREFILL_SCRATCH_CACHE above (sparkglm's kernel, a different
# tier) -- the two must never share a cache key or one tier's scratch growth
# could silently invalidate the other's.
_FAT_GROUPED_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_GROUPED_BYTES: dict[tuple, int] = {}
_FAT_BUCKET_EDGES = (16, 32, 64, 128, 256, 512, 1024, 2048)
_FAT_STATS: dict[str, Any] = {
    "layers": 0,
    "fat_layers": 0,
    "fat_experts": 0,
    "max_rows": 0,
    "sum_max_rows": 0,
    "hist": [0] * (len(_FAT_BUCKET_EDGES) + 1),
}
# "grouped" (E3, v11-e3 addition) sits above "kernel" (sparkglm's
# EXL3_GROUPED_PREFILL_K4, already living inside the "kernel" tier below) --
# a real fallback chain: E3 ineligible/disabled falls through to "kernel",
# which still tries sparkglm's kernel before E1/sorted/legacy.
_FAT_TIERS = ("grouped", "kernel", "batched", "sorted", "legacy")


def _initialize_tiny_dummy_exl3(layer: torch.nn.Module) -> None:
    """Make vLLM's dummy loader produce a valid, deterministic EXL3 payload.

    The generic dummy loader intentionally leaves integer parameters untouched.
    That is correct for most quantizers, but EXL3's trellis and MCG marker are
    integer ABI, so an uninitialized dummy model either fails the marker check or
    feeds undefined trellis words to CUDA. This mode is deliberately gated by an
    explicit environment variable and must never rewrite a real checkpoint.
    """
    if os.environ.get("SPARKGLM_TINY_DUMMY", "0") != "1":
        return

    with torch.no_grad():
        # Give each expert and projection a distinct valid trellis pattern so
        # routing/expert-ID mistakes alter output tokens instead of hiding behind
        # identical dummy experts. Small scales keep the useless model finite.
        # v21-b12xmoe: w13_* is projection-major [2, E, ...] (b12x
        # trellis_t256_proj layout, see create_weights) -- experts read off
        # shape[1], not shape[0].
        experts = int(layer.w13_trellis.shape[1])
        device = layer.w13_trellis.device
        expert = torch.arange(experts, dtype=torch.int32, device=device)
        w13_code = (expert[:, None] * 257 + torch.tensor(
            [17, 113], dtype=torch.int32, device=device
        )[None, :]) % 32767
        w2_code = (expert * 521 + 29) % 32767
        layer.w13_trellis.copy_(
            w13_code.t().contiguous().to(torch.int16).reshape(2, experts, 1, 1, 1)
        )
        layer.w2_trellis.copy_(
            w2_code.to(torch.int16).reshape(experts, 1, 1, 1)
        )
        scale = (1.0 + expert.to(torch.float32) / max(experts, 1)) / 128.0
        # Gate/up SUH remains shared, matching the production checkpoint and
        # keeping the paired fat-expert path eligible. Broadcast targets the
        # expert axis (now axis 1, not axis 0) -- see create_weights.
        layer.w13_suh.copy_(scale[None, :, None])
        layer.w13_svh.copy_((scale * 0.75)[None, :, None])
        layer.w2_suh.copy_((scale * 0.5)[:, None])
        layer.w2_svh.copy_((scale * 0.625)[:, None])
        layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)

    logger.warning_once(
        "SPARKGLM_TINY_DUMMY=1: initialized synthetic EXL3 trellis weights; "
        "outputs are intentionally meaningless"
    )
# Machine-checkable E2/E3 (fat-expert) diagnostics. Load-time fields mirror
# the most recent weight load; counters are monotonic per process and are the
# ground truth for what actually ran. The key set is contract — extend it only
# together with a bump of EXL3_FAT_DIAG_SCHEMA.
# Schema 3 (v11-e3 addition) adds the E3 grouped tier: sym_fat_moe,
# grouped_calls, grouped_scratch_bytes, grouped_eligible (schema 2 was this
# fork's own sparkglm-graft bump -- sym_fat_gemm_pair/sym_fat_swiglu_had/
# fat_pair_enabled/fat_fused_activation_enabled below -- not upstream's E3
# schema 2, which used the same number for a different key set; renumbered
# to 3 here to keep the "schema N -> exact key set" contract meaningful).
EXL3_FAT_DIAG_SCHEMA = 3
EXL3_FAT_DIAG_KEYS = (
    "schema",
    "configured_tier",
    "effective_tier",
    "tier_reason",
    "shared_suh",
    "shared_suh_layers",
    "moe_layers_loaded",
    "sym_exl3_moe",
    "sym_fat_gemm",
    "sym_fat_gemm_scatter",
    "sym_fat_gemm_pair",
    "sym_fat_swiglu_had",
    "fat_pair_enabled",
    "fat_fused_activation_enabled",
    "sym_fat_moe",
    "grouped_calls",
    "grouped_scratch_bytes",
    "grouped_eligible",
    "cap_major",
    "cap_minor",
    "cap_ok",
    "tp_rank",
    "tp_size",
    "fused_temps_allocs",
    "fused_temps_bytes",
    "fat_scratch_allocs",
    "fat_scratch_bytes",
    "fat_scratch_peak_bytes",
    "prefill_layer_calls",
    "thin_calls",
    "row_tile_calls",
    "fallback_calls",
    "fallback_reasons",
    "fat_expert_runs",
    "direct_calls",
    "pair_calls",
    "fused_activation_calls",
    "scatter_calls",
    "fat_stat_layers",
    "fat_layers",
    "fat_expert_slots",
    "max_rows",
)
_EXL3_FAT_DIAG: dict[str, Any] = {
    "schema": EXL3_FAT_DIAG_SCHEMA,
    "configured_tier": "legacy",
    "effective_tier": "legacy",
    "tier_reason": "unresolved",
    "shared_suh": False,
    "shared_suh_layers": 0,
    "moe_layers_loaded": 0,
    "sym_exl3_moe": False,
    "sym_fat_gemm": False,
    "sym_fat_gemm_scatter": False,
    "sym_fat_gemm_pair": False,
    "sym_fat_swiglu_had": False,
    "fat_pair_enabled": False,
    "fat_fused_activation_enabled": False,
    "sym_fat_moe": False,
    "grouped_calls": 0,
    "grouped_scratch_bytes": 0,
    "grouped_eligible": False,
    "cap_major": -1,
    "cap_minor": -1,
    "cap_ok": False,
    "tp_rank": -1,
    "tp_size": 1,
    "fused_temps_allocs": 0,
    "fused_temps_bytes": 0,
    "fat_scratch_allocs": 0,
    "fat_scratch_bytes": 0,
    "fat_scratch_peak_bytes": 0,
    "prefill_layer_calls": 0,
    "thin_calls": 0,
    "row_tile_calls": 0,
    "fallback_calls": {tier: 0 for tier in _FAT_TIERS},
    "fallback_reasons": {},
    "fat_expert_runs": 0,
    "direct_calls": 0,
    "pair_calls": 0,
    "fused_activation_calls": 0,
    "scatter_calls": 0,
}
_FAT_SCRATCH_BYTES: dict[tuple, int] = {}
_exl3_fat_tier_logged = False
_exl3_decode_coop_logged = False

# --- SwiGLU-clamp shadow diagnostic (v21-b12xmoe, 2026-09-18) ---
# Answers a real open question from the b12x fused-trellis MoE validation:
# b12x's full_rotation=True mode (needed for its numerics) cannot replicate
# the +-SWIGLU_LIMIT_DEFAULT clamp this fork's existing kernels apply to
# gate/up before SiLU. Does this checkpoint's real traffic ever actually
# approach that clamp?
#
# This can NOT be answered by instrumenting the three Python-visible clamp
# call sites (apply_exl3_python_loop, apply_exl3_sorted_fat,
# apply_exl3_batched_fat) -- checked directly, and it turns out none of
# them are what production actually runs. Production's real path
# (EXL3_FAT_GROUPED=1, tier=grouped, confirmed via _exl3_fat_effective_tier
# repeatedly this session) dispatches fat experts through
# apply_exl3_grouped_fat, whose entire gate/up GEMM *and* SwiGLU activation
# (clamp included) happen inside one opaque compiled kernel call
# (ext.exl3_fat_moe_gateup) -- there is no Python-visible pre-clamp tensor
# on that path. The non-fat "thin" path (_exl3_moe_launch, the majority of
# tokens under normal routing) is equally opaque. Both are compiled CUDA;
# neither exposes intermediate values without native code changes, which
# is out of scope here.
#
# So this measures it a different way: a lightweight SHADOW computation,
# gated behind GLM53_EXL3_CLAMP_DIAG=1, that runs alongside whichever real
# path actually serves the request (fused/grouped/b12x/loop -- doesn't
# matter which) and independently recomputes just gate=h@W_gate,
# up=h@W_up for the same routed tokens (skipping SiLU, down-projection,
# and output assembly -- the real result is never touched). Mathematically
# this reproduces the SAME pre-clamp values the real path's kernel computes
# internally, since it's the identical GEMM against the identical weights,
# just via a Python-visible LinearEXL3 call. NOT free: this roughly
# doubles the gate/up GEMM cost for every routed token while enabled --
# a deliberate data-collection-window tool, never meant to run 24/7 in
# real production traffic. Default off adds zero overhead (the sampler is
# never invoked).
EXL3_CLAMP_DIAG_LOG_EVERY = 200
_EXL3_CLAMP_DIAG: dict[str, Any] = {
    "enabled": False,
    "calls": 0,
    "total_values": 0,
    "max_abs_gate": 0.0,
    "max_abs_up": 0.0,
    "exceed_gate_5": 0,
    "exceed_gate_8": 0,
    "exceed_gate_10": 0,
    "exceed_gate_12": 0,
    "exceed_up_5": 0,
    "exceed_up_8": 0,
    "exceed_up_10": 0,
    "exceed_up_12": 0,
}


def decode_coop_enabled() -> bool:
    """Experimental exact-shape cooperative K4 decode kernel."""
    return os.environ.get("EXL3_DECODE_COOP_K4", "0") != "0"


def decode_coop_max_tokens() -> int:
    """Largest captured target batch supported by the cooperative scratch."""
    raw = os.environ.get("EXL3_DECODE_COOP_MAX_TOKENS", "16").strip()
    return max(1, min(32, int(raw)))


def grouped_prefill_enabled() -> bool:
    """Experimental GPU-resident K4 fat-expert prefill for the GLM TP2 shape."""
    return os.environ.get("EXL3_GROUPED_PREFILL_K4", "0") != "0"


def fused_moe_row_tile_enabled() -> bool:
    """GPU row tiles instead of LinearEXL3 fallback. Prefill-only; decode stays one launch.

    Measured slower than the 128-row fallback at MNBT=1024 (8 full-grid launches).
    Default off; keep for MNBT > temp rows if a later bump still overflows.
    """
    return os.environ.get("EXL3_MOE_ROW_TILE", "0") != "0"


def temp_rows_fused() -> int:
    raw = os.environ.get("EXL3_TEMP_ROWS_FUSED", "").strip()
    if not raw:
        return int(TEMP_ROWS_FUSED)
    return max(1, int(raw))

def sorted_fat_fallback_enabled() -> bool:
    """Use the existing expert-sorted buffers for oversized prefill experts."""
    return os.environ.get("EXL3_FAT_SORTED", "0") != "0"

def batched_fat_fallback_enabled() -> bool:
    """Enable E1 batched fat experts; implies expert-sorted routing."""
    return os.environ.get("EXL3_FAT_BATCHED", "0") != "0"

def fat_kernel_enabled() -> bool:
    """Enable the E2 fat kernel; implies E1 batching and sorted routing."""
    return os.environ.get("EXL3_FAT_KERNEL", "0") != "0"


def fat_tile_m() -> int:
    """Row tile for the direct E2 GEMMs; only compiled variants are valid."""
    value = int(os.environ.get("EXL3_FAT_TILE_M", "64"))
    if value not in (64, 128):
        raise ValueError("EXL3_FAT_TILE_M must be 64 or 128")
    return value


def fat_pair_enabled() -> bool:
    """Read gate/up trellises directly instead of packing them per expert."""
    return os.environ.get("EXL3_FAT_PAIR", "1") != "0"


def fat_fused_activation_enabled() -> bool:
    """Fuse the fat SwiGLU epilogue and down-projection input Hadamard."""
    return os.environ.get("EXL3_FAT_FUSED_ACT", "1") != "0"


def grouped_fat_enabled() -> bool:
    """Enable the E3 grouped fat-expert kernels (v11-e3, default OFF).

    One gather + one gate/up + one down launch cover every fat expert of a
    layer from device-side segment tables: no per-expert launches and no
    host synchronization on the routing counts. Needs the exl3_fat_moe
    kernels (exllamav3_ext built with exl3_fat_moe.cu). EXL3_FAT_GROUPED=0
    (default) leaves the E2/sparkglm-kernel path and its cap untouched.
    """
    return os.environ.get("EXL3_FAT_GROUPED", "0") != "0"


def fat_expert_log_enabled() -> bool:
    """Per-step routing histogram. It costs one host sync per MoE layer under
    the grouped (E3) tier, which has none of its own otherwise, so it
    defaults off there."""
    default = "0" if grouped_fat_enabled() else "1"
    return os.environ.get("EXL3_FAT_EXPERT_LOG", default) != "0"

def configured_fat_tier() -> str:
    """Highest fat tier the env requests: grouped > kernel > batched > sorted > legacy."""
    if grouped_fat_enabled():
        return "grouped"
    if fat_kernel_enabled():
        return "kernel"
    if batched_fat_fallback_enabled():
        return "batched"
    if sorted_fat_fallback_enabled():
        return "sorted"
    return "legacy"


EXL3_FAT_MOE_SYMBOLS = (
    "exl3_fat_moe_gather",
    "exl3_fat_moe_gateup",
    "exl3_fat_moe_down",
    "exl3_fat_moe_tile_rows_gateup",
    "exl3_fat_moe_tile_rows_down",
)
# Grouped (E3) kernels: 16 B vector atomics (sm_90+); the image builds sm_121a.
EXL3_FAT_MOE_MIN_CAPABILITY = (9, 0)
_FAT_MOE_EXT_CACHE: list = []


def load_fat_moe_ext():
    """Module carrying the E3 kernels, or None. Resolved once.

    Unlike upstream's own two-source resolution (exllamav3_ext or the
    additive exl3_fat_moe_ext layered-image module), this build only ever
    does the full-rebuild path -- exl3_fat_moe_ext is not built here -- so
    this only checks exllamav3_ext, but keeps the same function name/shape
    so exl3_fat_moe_symbols()/grouped dispatch below don't need to know that.
    """
    if _FAT_MOE_EXT_CACHE:
        return _FAT_MOE_EXT_CACHE[0]
    found = None
    try:
        ext = load_exllamav3_ext()
        if all(hasattr(ext, name) for name in EXL3_FAT_MOE_SYMBOLS):
            found = ext
    except Exception:
        found = None
    _FAT_MOE_EXT_CACHE.append(found)
    return found


def exl3_fat_moe_symbols() -> bool:
    """True when every E3 grouped kernel entry point is importable."""
    return load_fat_moe_ext() is not None


def grouped_fat_eligibility(layer: torch.nn.Module) -> tuple[bool, str]:
    """Load-time checkpoint/device eligibility for the K4/MCG grouped (E3) kernels.

    The grouped pointer-table interface does not carry K/mcg/mul1 like the E2
    GEMM interface, so those invariants are checked here, once per layer,
    before the tier is resolved: 4-bit trellis (64 int16 words per tile),
    MCG codebook without mul1, shared gate/up SUH, tile-friendly dimensions
    (hidden % 256 for the 2x128 down tile and the gather's 128-wide blocks,
    intermediate % 128 for the gate/up tile), a device capability with 16 B
    vector atomics, and every packed tensor on one CUDA device.
    """
    bits = int(getattr(layer, "_exl3_bits", 0) or 0)
    if bits != 4:
        return False, f"bits_{bits}"
    k_words = int(getattr(layer, "_exl3_k_words", 0) or 0)
    if k_words != 64:
        return False, f"k_words_{k_words}"
    inners = getattr(layer, "_exl3_inners", None) or []
    if not inners:
        return False, "no_inners"
    for pack in inners:
        for which in ("gate", "up", "down"):
            inner = pack[which]
            if int(getattr(inner, "K", 0)) != 4:
                return False, f"K_{getattr(inner, 'K', '?')}"
            if not bool(getattr(inner, "mcg", False)):
                return False, "not_mcg"
            if bool(getattr(inner, "mul1", False)):
                return False, "mul1"
    if not bool(getattr(layer, "_exl3_shared_w13_suh", False)):
        return False, "shared_suh_absent"
    hidden = int(getattr(layer, "_exl3_hidden_size", 0) or 0)
    inter = int(getattr(layer, "_exl3_intermediate_local", 0) or 0)
    if hidden <= 0 or hidden % 256:
        return False, f"hidden_{hidden}"
    if inter <= 0 or inter % 128:
        return False, f"intermediate_{inter}"
    device = layer.w13_trellis.device
    if device.type != "cuda":
        return False, f"device_{device.type}"
    for name in ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"):
        if getattr(layer, name).device != device:
            return False, f"device_mismatch_{name}"
    cap = exl3_device_capability()
    if cap < EXL3_FAT_MOE_MIN_CAPABILITY:
        return False, f"capability_{cap[0]}.{cap[1]}"
    return True, "eligible"


def exl3_fat_symbols() -> tuple[bool, bool, bool]:
    """(exl3_moe, exl3_fat_gemm, exl3_fat_gemm_scatter) availability."""
    try:
        ext = load_exllamav3_ext()
    except Exception:
        return False, False, False
    return (
        hasattr(ext, "exl3_moe"),
        hasattr(ext, "exl3_fat_gemm"),
        hasattr(ext, "exl3_fat_gemm_scatter"),
    )


def exl3_fat_feature_symbols() -> tuple[bool, bool]:
    """(paired gate/up, fused activation/Hadamard) symbol availability."""
    try:
        ext = load_exllamav3_ext()
    except Exception:
        return False, False
    return (
        hasattr(ext, "exl3_fat_gemm_pair"),
        hasattr(ext, "exl3_fat_swiglu_had"),
    )


def exl3_fat_m64_symbols() -> tuple[bool, bool, bool]:
    """Direct, paired, and scatter symbols for the 64-row E2 variant."""
    try:
        ext = load_exllamav3_ext()
    except Exception:
        return False, False, False
    return (
        hasattr(ext, "exl3_fat_gemm_m64"),
        hasattr(ext, "exl3_fat_gemm_pair_m64"),
        hasattr(ext, "exl3_fat_gemm_scatter_m64"),
    )


def exl3_device_capability() -> tuple[int, int]:
    """CUDA capability of the current device; (-1, -1) without a GPU."""
    if not torch.cuda.is_available():
        return -1, -1
    try:
        return tuple(int(v) for v in torch.cuda.get_device_capability())
    except Exception:
        return -1, -1


def _exl3_tp_rank_size() -> tuple[int, int]:
    """TP identity so each rank's diag line is attributable; -1 outside vLLM."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        return -1, 1


def resolve_exl3_fat_tier(
    shared_suh: bool,
    symbols: tuple[bool, bool, bool] | None = None,
    feature_symbols: tuple[bool, bool] | None = None,
    m64_symbols: tuple[bool, bool, bool] | None = None,
    fat_moe_symbols: bool | None = None,
    grouped_eligible: tuple[bool, str] | None = None,
) -> tuple[str, str]:
    """Map the configured fat tier onto what this image + checkpoint can run.

    Order matters: the checkpoint cap comes first. E1 batched and the E2
    kernel run gate+up as one stacked GEMM behind a single input Hadamard
    (gate.suh), so a checkpoint without shared SUH caps the tier at sorted —
    a legitimate lower tier that needs no fat symbols, whatever the image.
    Only when the kernel would actually run does a missing symbol become an
    image/flag mismatch, failing closed here at load instead of mid-prefill.

    Grouped (E3): a request without the E3 symbols fails closed. A
    checkpoint or device the grouped kernels cannot run (K4/MCG, dimensions,
    capability) deliberately falls back to the E2/sparkglm kernel tier with
    the reason recorded, and that fallback is itself subject to the E2
    symbol check below.
    """
    configured = configured_fat_tier()
    if configured == "legacy":
        return configured, "none_requested"
    if not shared_suh and configured in ("grouped", "kernel", "batched"):
        return "sorted", "shared_suh_absent"
    reason_prefix = ""
    if configured == "grouped":
        if fat_moe_symbols is None:
            fat_moe_symbols = exl3_fat_moe_symbols()
        if not fat_moe_symbols:
            raise RuntimeError(
                "EXL3_FAT_GROUPED=1 requires exllamav3_ext."
                + "/".join(EXL3_FAT_MOE_SYMBOLS)
                + "; this image was built without exl3_fat_moe.cu — set "
                "EXL3_FAT_GROUPED=0 or serve the v11-e3 image"
            )
        if grouped_eligible is None:
            grouped_eligible = (True, "eligible")
        if grouped_eligible[0]:
            return configured, "grouped_ok"
        # Ineligible for the grouped (E3) kernels: fall back to the
        # E2/sparkglm kernel tier deliberately, not further down the ladder.
        configured = "kernel"
        reason_prefix = f"grouped_ineligible_{grouped_eligible[1]}:"
    if symbols is None:
        symbols = exl3_fat_symbols()
    if configured == "kernel":
        missing = [
            name
            for name, present in zip(
                ("exl3_fat_gemm", "exl3_fat_gemm_scatter"), symbols[1:]
            )
            if not present
        ]
        if missing:
            raise RuntimeError(
                "EXL3_FAT_KERNEL=1 requires exllamav3_ext."
                + "/".join(missing)
                + "; this image was built without the fat kernel — unset "
                "EXL3_FAT_KERNEL or serve an E2 image"
            )
        if feature_symbols is None:
            feature_symbols = exl3_fat_feature_symbols()
        missing_features = [
            name
            for name, enabled, present in zip(
                ("exl3_fat_gemm_pair", "exl3_fat_swiglu_had"),
                (fat_pair_enabled(), fat_fused_activation_enabled()),
                feature_symbols,
            )
            if enabled and not present
        ]
        if missing_features:
            raise RuntimeError(
                "enabled EXL3 fat features require exllamav3_ext."
                + "/".join(missing_features)
                + "; disable the matching EXL3_FAT_* flag or serve the "
                "matching extension image"
            )
        if fat_tile_m() == 64:
            if m64_symbols is None:
                m64_symbols = exl3_fat_m64_symbols()
            missing_m64 = [
                name
                for name, present in zip(
                    (
                        "exl3_fat_gemm_m64",
                        "exl3_fat_gemm_pair_m64",
                        "exl3_fat_gemm_scatter_m64",
                    ),
                    m64_symbols,
                )
                if not present
            ]
            if missing_m64:
                raise RuntimeError(
                    "EXL3_FAT_TILE_M=64 requires exllamav3_ext."
                    + "/".join(missing_m64)
                    + "; use EXL3_FAT_TILE_M=128 or serve the matching image"
                )
    return configured, f"{reason_prefix}{configured}_ok"


def exl3_fat_diag() -> dict[str, Any]:
    """Snapshot of the E2 diagnostics; the key set is EXL3_FAT_DIAG_KEYS."""
    diag = dict(_EXL3_FAT_DIAG)
    diag["fallback_calls"] = dict(_EXL3_FAT_DIAG["fallback_calls"])
    diag["fallback_reasons"] = dict(_EXL3_FAT_DIAG["fallback_reasons"])
    diag.update(
        fat_stat_layers=_FAT_STATS["layers"],
        fat_layers=_FAT_STATS["fat_layers"],
        fat_expert_slots=_FAT_STATS["fat_experts"],
        max_rows=_FAT_STATS["max_rows"],
    )
    return diag


def _exl3_fat_diag_line() -> str:
    d = exl3_fat_diag()
    parts = [
        f"schema={d['schema']}",
        f"configured_tier={d['configured_tier']}",
        f"effective_tier={d['effective_tier']}",
        f"tier_reason={d['tier_reason']}",
        f"shared_suh={int(d['shared_suh'])}",
        f"shared_suh_layers={d['shared_suh_layers']}/{d['moe_layers_loaded']}",
        f"sym_exl3_moe={int(d['sym_exl3_moe'])}",
        f"sym_fat_gemm={int(d['sym_fat_gemm'])}",
        f"sym_fat_gemm_scatter={int(d['sym_fat_gemm_scatter'])}",
        f"sym_fat_gemm_pair={int(d['sym_fat_gemm_pair'])}",
        f"sym_fat_swiglu_had={int(d['sym_fat_swiglu_had'])}",
        f"fat_pair_enabled={int(d['fat_pair_enabled'])}",
        f"fat_fused_activation_enabled={int(d['fat_fused_activation_enabled'])}",
        f"sym_fat_moe={int(d['sym_fat_moe'])}",
        f"grouped_eligible={int(d['grouped_eligible'])}",
        f"grouped_calls={d['grouped_calls']}",
        f"grouped_scratch_bytes={d['grouped_scratch_bytes']}",
        f"fat_tile_m={fat_tile_m()}",
        f"cap={d['cap_major']}.{d['cap_minor']}",
        f"cap_ok={int(d['cap_ok'])}",
        f"tp_rank={d['tp_rank']} tp_size={d['tp_size']}",
        f"prefill_layer_calls={d['prefill_layer_calls']}",
        f"thin_calls={d['thin_calls']}",
        f"row_tile_calls={d['row_tile_calls']}",
        "fallback_calls="
        + ",".join(f"{t}={d['fallback_calls'][t]}" for t in _FAT_TIERS),
        "fallback_reasons="
        + (
            ",".join(f"{r}={n}" for r, n in sorted(d["fallback_reasons"].items()))
            or "none"
        ),
        f"fat_expert_runs={d['fat_expert_runs']}",
        f"direct_calls={d['direct_calls']}",
        f"pair_calls={d['pair_calls']}",
        f"fused_activation_calls={d['fused_activation_calls']}",
        f"scatter_calls={d['scatter_calls']}",
        f"fat_layers={d['fat_layers']}",
        f"fat_expert_slots={d['fat_expert_slots']}",
        f"max_rows={d['max_rows']}",
        f"fused_temps_allocs={d['fused_temps_allocs']}",
        f"fused_temps_bytes={d['fused_temps_bytes']}",
        f"fat_scratch_allocs={d['fat_scratch_allocs']}",
        f"fat_scratch_bytes={d['fat_scratch_bytes']}",
        f"fat_scratch_peak_bytes={d['fat_scratch_peak_bytes']}",
    ]
    return " ".join(parts)


def _record_exl3_fat_reason(reason: str) -> None:
    reasons = _EXL3_FAT_DIAG["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _record_exl3_fat_tier(layer: torch.nn.Module, tier: str, reason: str) -> None:
    _EXL3_FAT_DIAG["fallback_calls"][tier] += 1
    _record_exl3_fat_reason(reason)
    layer._exl3_last_fat_fallback = tier
    layer._exl3_last_fat_reason = reason


def _record_exl3_fat_resolution(layer: torch.nn.Module) -> None:
    """Resolve the E2 tier once per MoE layer at weight load and log once.

    Per-layer truth lands on layer._exl3_fat_effective_tier; the module state
    mirrors the most recent load. A resolution that changes between layers of
    one model is a checkpoint property worth a loud line, not a silent one.
    """
    global _exl3_fat_tier_logged
    shared_suh = bool(getattr(layer, "_exl3_shared_w13_suh", False))
    grouped_eligible = grouped_fat_eligibility(layer)
    layer._exl3_grouped_eligible = grouped_eligible
    effective_tier, tier_reason = resolve_exl3_fat_tier(
        shared_suh, grouped_eligible=grouped_eligible
    )
    layer._exl3_fat_effective_tier = effective_tier
    layer._exl3_fat_tier_reason = tier_reason

    diag = _EXL3_FAT_DIAG
    sym_moe, sym_gemm, sym_scatter = exl3_fat_symbols()
    sym_pair, sym_fused_activation = exl3_fat_feature_symbols()
    cap_major, cap_minor = exl3_device_capability()
    diag["moe_layers_loaded"] += 1
    if shared_suh:
        diag["shared_suh_layers"] += 1
    diag["shared_suh"] = diag["shared_suh_layers"] == diag["moe_layers_loaded"]
    diag["configured_tier"] = configured_fat_tier()
    diag["sym_exl3_moe"] = sym_moe
    diag["sym_fat_gemm"] = sym_gemm
    diag["sym_fat_gemm_scatter"] = sym_scatter
    diag["sym_fat_gemm_pair"] = sym_pair
    diag["sym_fat_swiglu_had"] = sym_fused_activation
    diag["fat_pair_enabled"] = fat_pair_enabled()
    diag["fat_fused_activation_enabled"] = fat_fused_activation_enabled()
    diag["sym_fat_moe"] = exl3_fat_moe_symbols()
    diag["grouped_eligible"] = bool(grouped_eligible[0])
    diag["cap_major"] = cap_major
    diag["cap_minor"] = cap_minor
    # LinearEXL3 (and the fat GEMM built on it) needs >= Ampere; GB10 is SM121.
    diag["cap_ok"] = (cap_major, cap_minor) >= (8, 0)
    diag["tp_rank"], diag["tp_size"] = _exl3_tp_rank_size()

    if diag["tier_reason"] == "unresolved":
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    elif diag["effective_tier"] != effective_tier:
        logger.warning(
            "exl3 e2 diag tier changed %s -> %s (%s): %s",
            diag["effective_tier"],
            effective_tier,
            tier_reason,
            _exl3_fat_diag_line(),
        )
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    if not _exl3_fat_tier_logged:
        _exl3_fat_tier_logged = True
        if (
            diag["effective_tier"] != diag["configured_tier"]
            and diag["configured_tier"] != "legacy"
        ):
            logger.warning("exl3 e2 diag degraded %s", _exl3_fat_diag_line())
        else:
            logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())


def reset_exl3_fat_diag_counters() -> None:
    """Zero the E2 runtime counters; load-time fields and live bytes stay.

    Scratch current/peak restart from the resident cache so a windowed read
    (e.g. exactly one cold request) still reports honest byte counts.
    """
    diag = _EXL3_FAT_DIAG
    for key in (
        "prefill_layer_calls",
        "thin_calls",
        "row_tile_calls",
        "fat_expert_runs",
        "direct_calls",
        "pair_calls",
        "fused_activation_calls",
        "scatter_calls",
        "fat_scratch_allocs",
        "grouped_calls",
    ):
        diag[key] = 0
    diag["fallback_calls"] = {tier: 0 for tier in _FAT_TIERS}
    diag["fallback_reasons"] = {}
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]
    diag["grouped_scratch_bytes"] = sum(_FAT_GROUPED_BYTES.values())


def reset_exl3_fat_expert_stats() -> None:
    _FAT_STATS["layers"] = 0
    _FAT_STATS["fat_layers"] = 0
    _FAT_STATS["fat_experts"] = 0
    _FAT_STATS["max_rows"] = 0
    _FAT_STATS["sum_max_rows"] = 0
    _FAT_STATS["hist"] = [0] * (len(_FAT_BUCKET_EDGES) + 1)


def _fat_bucket(n: int) -> int:
    for i, edge in enumerate(_FAT_BUCKET_EDGES):
        if n <= edge:
            return i
    return len(_FAT_BUCKET_EDGES)


def record_exl3_fat_expert_stats(
    counts: torch.Tensor,
    *,
    max_rows: int | None = None,
    counts_host: list[int] | None = None,
) -> dict[str, Any]:
    """Prefill-only routing stats. Reuse an existing host copy when available."""
    if counts_host is None:
        if max_rows is None:
            max_rows = int(counts.max().item())
        n_fat = int((counts > temp_rows_fused()).sum().item())
    else:
        if max_rows is None:
            max_rows = max(counts_host, default=0)
        cap = temp_rows_fused()
        n_fat = sum(n > cap for n in counts_host)
    st = _FAT_STATS
    st["layers"] += 1
    st["sum_max_rows"] += max_rows
    st["hist"][_fat_bucket(max_rows)] += 1
    if max_rows > st["max_rows"]:
        st["max_rows"] = max_rows
    if n_fat:
        st["fat_layers"] += 1
        st["fat_experts"] += n_fat
    # 42 routed-MoE layers per engine step (MoE from layer 3 of 45).
    if st["layers"] % 42 == 0:
        avg = st["sum_max_rows"] / st["layers"]
        le128 = sum(st["hist"][:4])
        gt128 = sum(st["hist"][4:])
        logger.info(
            "exl3 fat-expert P0: layers=%d fat_layers=%d (%.1f%%) fat_expert_slots=%d "
            "max_rows=%d avg_max_rows=%.1f hist_le128=%d hist_gt128=%d hist=%s",
            st["layers"],
            st["fat_layers"],
            100.0 * st["fat_layers"] / st["layers"],
            st["fat_experts"],
            st["max_rows"],
            avg,
            le128,
            gt128,
            st["hist"],
        )
        logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())
    return {
        "max_rows": max_rows,
        "n_fat": n_fat,
        "layers": st["layers"],
        "fat_layers": st["fat_layers"],
    }


def _narrow_tp(tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk).contiguous()


def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def _install_exllamav3_namespace() -> None:
    """Load LinearEXL3 without running exllamav3/__init__.py (FlashAttention)."""
    if "exllamav3.modules.quant.exl3" in sys.modules:
        return
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 package is not installed in this image")
    package_root = Path(list(spec.submodule_search_locations)[0])

    # Stub only packages whose __init__.py pulls FlashAttention / serving extras.
    # Leave .ext, .util, and .modules.quant as real modules so LinearEXL3 loads.
    for name, path in (
        ("exllamav3", package_root),
        ("exllamav3.modules", package_root / "modules"),
        ("exllamav3.model", package_root / "model"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module

    if "exllamav3.model.config" not in sys.modules:
        config = types.ModuleType("exllamav3.model.config")
        config.__file__ = str(package_root / "model/config.py")
        config.__package__ = "exllamav3.model"
        config.Config = type("Config", (), {})
        sys.modules[config.__name__] = config


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand."""
    cls = load_linear_exl3_cls()
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        mcg=mcg.contiguous(),
        out_dtype=out_dtype,
        transformers_fix=True,
    )


def execute_exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Real EXL3 expert GEMM entry (LinearEXL3 / exllamav3_ext)."""
    inner = make_linear_exl3(trellis, suh, svh, mcg, out_dtype=torch.float16)
    return inner.forward(x.contiguous().half(), {}, out_dtype=out_dtype)


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


def _exl3_moe_accepts_num_active(fn) -> bool:
    try:
        import inspect

        if "num_active" in inspect.signature(fn).parameters:
            return True
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    return "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30


def pin_exl3_expert_map(
    layer: torch.nn.Module, device: torch.device
) -> torch.Tensor | None:
    """Move expert_map onto `device` once. CUDA graph capture forbids a CPU→GPU copy."""
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        layer.expert_map = emap.to(device=device, dtype=torch.long)
    return layer.expert_map


def map_topk_to_local(
    ids: torch.Tensor,
    n_local: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """ids (T, K) global expert ids → local ids, invalid/non-local → n_local sentinel.

    `expert_map` must already live on `ids.device` (see pin_exl3_expert_map).
    """
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply (CUDA graphs forbid the copy)"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique-expert LinearEXL3 loop. `only_experts` is local ids (fat-expert fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    unique = torch.unique(ids)
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def _exl3_clamp_diag_sample(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
) -> None:
    """Shadow-recompute gate/up (no SiLU, no down, no output) for every
    routed token, purely to sample this checkpoint's real pre-clamp
    activation distribution. See the _EXL3_CLAMP_DIAG comment for why this
    can't be read off the real serving path directly. Never touches `weights`
    beyond routing -- output is discarded, real serving is unaffected."""
    del weights
    diag = _EXL3_CLAMP_DIAG
    unique = torch.unique(ids)
    max_gate = 0.0
    max_up = 0.0
    total = 0
    exceed_gate = {5: 0, 8: 0, 10: 0, 12: 0}
    exceed_up = {5: 0, 8: 0, 10: 0, 12: 0}
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        token_idx, _ = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        gate_abs = gate.abs()
        up_abs = up.abs()
        max_gate = max(max_gate, float(gate_abs.max().item())) if gate_abs.numel() else max_gate
        max_up = max(max_up, float(up_abs.max().item())) if up_abs.numel() else max_up
        total += gate_abs.numel() + up_abs.numel()
        for threshold in (5, 8, 10, 12):
            exceed_gate[threshold] += int((gate_abs > threshold).sum().item())
            exceed_up[threshold] += int((up_abs > threshold).sum().item())
    diag["calls"] += 1
    diag["total_values"] += total
    diag["max_abs_gate"] = max(diag["max_abs_gate"], max_gate)
    diag["max_abs_up"] = max(diag["max_abs_up"], max_up)
    for threshold in (5, 8, 10, 12):
        diag[f"exceed_gate_{threshold}"] += exceed_gate[threshold]
        diag[f"exceed_up_{threshold}"] += exceed_up[threshold]
    if diag["calls"] % EXL3_CLAMP_DIAG_LOG_EVERY == 0:
        logger.info("exl3 clamp diag %s", _exl3_clamp_diag_line())


def _exl3_clamp_diag_line() -> str:
    d = _EXL3_CLAMP_DIAG
    return " ".join(
        [
            f"calls={d['calls']}",
            f"total_values={d['total_values']}",
            f"max_abs_gate={d['max_abs_gate']:.4f}",
            f"max_abs_up={d['max_abs_up']:.4f}",
            "exceed_gate="
            + ",".join(f">{t}:{d[f'exceed_gate_{t}']}" for t in (5, 8, 10, 12)),
            "exceed_up="
            + ",".join(f">{t}:{d[f'exceed_up_{t}']}" for t in (5, 8, 10, 12)),
        ]
    )


def reset_exl3_clamp_diag_counters() -> None:
    diag = _EXL3_CLAMP_DIAG
    diag["calls"] = 0
    diag["total_values"] = 0
    diag["max_abs_gate"] = 0.0
    diag["max_abs_up"] = 0.0
    for threshold in (5, 8, 10, 12):
        diag[f"exceed_gate_{threshold}"] = 0
        diag[f"exceed_up_{threshold}"] = 0


def apply_exl3_sorted_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run oversized experts from contiguous slices of the existing sort.

    One `counts.tolist()` in the caller replaces the legacy per-expert
    `unique.tolist()`, expert-map `.item()`, and `(ids == expert).nonzero()`
    synchronizations. Slices are views; only the LinearEXL3 inputs and outputs
    allocate, as they do in the legacy fallback.
    """
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1
        token_idx = token_sorted[start:offset]
        h = xh.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h, {}, out_dtype=torch.float32)
        up = pack["up"].forward(h, {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(
            act.contiguous().half(), {}, out_dtype=torch.float32
        )
        scale = weight_sorted[start:offset].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def _grouped_prefill_scratch(
    device: torch.device,
    rows: int,
    hidden: int,
    intermediate: int,
    experts: int,
) -> dict[str, torch.Tensor]:
    """Layer-serial scratch for the GPU-resident grouped fat-expert pipeline."""
    capacity = 1 << (max(256, rows) - 1).bit_length()
    task_capacity = (capacity + 63) // 64 + experts
    key = (str(device), hidden, intermediate, experts)
    scratch = _GROUPED_PREFILL_SCRATCH_CACHE.get(key)
    if scratch is not None and int(scratch["had_input"].shape[0]) >= rows:
        return scratch
    scratch = {
        "had_input": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "gate_up": torch.empty(
            (capacity, 2 * intermediate), dtype=torch.float32, device=device
        ),
        "had_down": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "tasks": torch.empty(
            (task_capacity, 4), dtype=torch.int32, device=device
        ),
        "task_count": torch.empty(1, dtype=torch.int32, device=device),
    }
    _GROUPED_PREFILL_SCRATCH_CACHE[key] = scratch
    logger.info_once(
        "exl3 grouped prefill scratch: rows=%d bytes=%d",
        capacity,
        sum(t.numel() * t.element_size() for t in scratch.values()),
    )
    return scratch


def _fat_scratch(
    device: torch.device,
    rows: int,
    gate: Any,
) -> dict[str, torch.Tensor]:
    """Return shared prefill scratch, growing once if the configured chunk grows."""
    hidden = int(gate.in_features)
    intermediate = int(gate.out_features)
    configured = int(
        os.environ.get(
            "EXL3_FAT_SCRATCH_ROWS",
            os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"),
        )
        or 0
    )
    needed = max(256, rows, configured)
    capacity = 1 << (needed - 1).bit_length()
    key = (
        str(device),
        hidden,
        intermediate,
        int(gate.K),
        int(gate.trellis.shape[-1]),
    )
    scratch = _FAT_SCRATCH_CACHE.get(key)
    if scratch is not None and int(scratch["h"].shape[0]) >= rows:
        return scratch

    in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
    scratch = {
        "packed13": torch.empty(
            (in_tiles, 2 * out_tiles, k_words),
            dtype=torch.int16,
            device=device,
        ),
        "svh13": torch.empty(
            2 * intermediate, dtype=torch.float16, device=device
        ),
        "w13": torch.empty(
            (hidden, 2 * intermediate), dtype=torch.float16, device=device
        ),
        "w2": torch.empty(
            (intermediate, hidden), dtype=torch.float16, device=device
        ),
        "h": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "h13": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "gate_up": torch.empty(
            (capacity, 2 * intermediate), dtype=torch.float32, device=device
        ),
        "act": torch.empty(
            (capacity, intermediate), dtype=torch.float32, device=device
        ),
        "act_h": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "h2": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "down": torch.empty(
            (capacity, hidden), dtype=torch.float32, device=device
        ),
    }
    _FAT_SCRATCH_CACHE[key] = scratch
    _FAT_SCRATCH_BYTES[key] = sum(
        t.numel() * t.element_size() for t in scratch.values()
    )
    diag = _EXL3_FAT_DIAG
    diag["fat_scratch_allocs"] += 1
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    if diag["fat_scratch_bytes"] > diag["fat_scratch_peak_bytes"]:
        diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]
    return scratch


def _stage_counts_to_host(
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.cuda.Stream]:
    """Copy routing counts on a side stream before launching thin experts."""
    key = (str(counts.device), int(counts.numel()))
    cached = _FAT_COUNT_CACHE.get(key)
    if cached is None:
        host = torch.empty(
            int(counts.numel()), dtype=counts.dtype, device="cpu", pin_memory=True
        )
        stream = torch.cuda.Stream(device=counts.device)
        cached = (host, stream)
        _FAT_COUNT_CACHE[key] = cached
    host, stream = cached
    current = torch.cuda.current_stream(counts.device)
    with torch.cuda.stream(stream):
        stream.wait_stream(current)
        host.copy_(counts, non_blocking=True)
    return host, stream


def apply_exl3_batched_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
    use_kernel: bool = False,
) -> torch.Tensor:
    """Run fat experts with persistent buffers and optional direct trellis GEMM."""
    ext = load_exllamav3_ext()
    tile_m = fat_tile_m()
    if tile_m == 64:
        fat_gemm = ext.exl3_fat_gemm_m64
        fat_gemm_pair = ext.exl3_fat_gemm_pair_m64
        fat_gemm_scatter = ext.exl3_fat_gemm_scatter_m64
    else:
        fat_gemm = ext.exl3_fat_gemm
        fat_gemm_pair = ext.exl3_fat_gemm_pair
        fat_gemm_scatter = ext.exl3_fat_gemm_scatter
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1

        token_idx = token_sorted[start:offset]
        gate = inners[e]["gate"]
        up = inners[e]["up"]
        down = inners[e]["down"]
        scratch = _fat_scratch(xh.device, n_rows, gate)
        intermediate = int(gate.out_features)

        h = scratch["h"][:n_rows]
        h13 = scratch["h13"][:n_rows]
        torch.index_select(xh, 0, token_idx, out=h)
        ext.had_r_128(h, h13, gate.suh, None, 1.0)

        gate_up = scratch["gate_up"][:n_rows]
        if use_kernel:
            if not hasattr(ext, "exl3_fat_gemm"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires exllamav3_ext.exl3_fat_gemm"
                )
            if fat_pair_enabled():
                if not hasattr(ext, "exl3_fat_gemm_pair"):
                    raise RuntimeError(
                        "EXL3_FAT_PAIR=1 requires "
                        "exllamav3_ext.exl3_fat_gemm_pair"
                    )
                fat_gemm_pair(
                    h13,
                    gate.trellis,
                    up.trellis,
                    gate_up,
                    gate.svh,
                    up.svh,
                    gate.K,
                    gate.mcg,
                    gate.mul1,
                )
                _EXL3_FAT_DIAG["pair_calls"] += 1
            else:
                packed13 = scratch["packed13"]
                out_tiles = int(gate.trellis.shape[1])
                packed13[:, :out_tiles].copy_(gate.trellis)
                packed13[:, out_tiles:].copy_(up.trellis)
                svh13 = scratch["svh13"]
                svh13[:intermediate].copy_(gate.svh)
                svh13[intermediate:].copy_(up.svh)
                fat_gemm(
                    h13,
                    packed13,
                    gate_up,
                    svh13,
                    gate.K,
                    gate.mcg,
                    gate.mul1,
                )
            _EXL3_FAT_DIAG["direct_calls"] += 1
        else:
            packed13 = scratch["packed13"]
            out_tiles = int(gate.trellis.shape[1])
            packed13[:, :out_tiles].copy_(gate.trellis)
            packed13[:, out_tiles:].copy_(up.trellis)
            svh13 = scratch["svh13"]
            svh13[:intermediate].copy_(gate.svh)
            svh13[intermediate:].copy_(up.svh)
            w13 = scratch["w13"]
            ext.reconstruct(w13, packed13, gate.K, gate.mcg, gate.mul1)
            ext.hgemm(h13, w13, gate_up)
            ext.had_r_128(gate_up, gate_up, None, svh13, 1.0)

        h2 = scratch["h2"][:n_rows]
        if use_kernel and fat_fused_activation_enabled():
            if not hasattr(ext, "exl3_fat_swiglu_had"):
                raise RuntimeError(
                    "EXL3_FAT_FUSED_ACT=1 requires "
                    "exllamav3_ext.exl3_fat_swiglu_had"
                )
            ext.exl3_fat_swiglu_had(gate_up, h2, down.suh, float(limit))
            _EXL3_FAT_DIAG["fused_activation_calls"] += 1
        else:
            gate_out = gate_up[:, :intermediate]
            up_out = gate_up[:, intermediate:]
            gate_out.clamp_(max=limit)
            up_out.clamp_(min=-limit, max=limit)
            act = scratch["act"][:n_rows]
            torch.sigmoid(gate_out, out=act)
            act.mul_(gate_out).mul_(up_out)
            act_h = scratch["act_h"][:n_rows]
            act_h.copy_(act)
            ext.had_r_128(act_h, h2, down.suh, None, 1.0)
        if use_kernel:
            if not hasattr(ext, "exl3_fat_gemm_scatter"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires "
                    "exllamav3_ext.exl3_fat_gemm_scatter"
                )
            fat_gemm_scatter(
                h2,
                down.trellis,
                out,
                down.svh,
                token_idx,
                weight_sorted[start:offset],
                down.K,
                down.mcg,
                down.mul1,
            )
            _EXL3_FAT_DIAG["scatter_calls"] += 1
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, down.K, down.mcg, down.mul1)
            down_out = scratch["down"][:n_rows]
            ext.hgemm(h2, w2, down_out)
            ext.had_r_128(down_out, down_out, None, down.svh, 1.0)
            down_out.mul_(weight_sorted[start:offset].unsqueeze(-1))
            out.index_add_(0, token_idx, down_out)
    return out


def _grouped_scratch(
    device: torch.device, rows: int, hidden: int, intermediate: int
) -> dict[str, torch.Tensor]:
    """Fat-row activation buffers for the E3 grouped tier, grown once.

    Capacity covers every routed slot of the configured prefill chunk
    (MAX_NUM_BATCHED_TOKENS x top-k, EXL3_FAT_GROUPED_TOPK, default 8) so
    steady-state prefill never reallocates; a larger request grows it once
    more (outside CUDA graph capture, where growth would be illegal).
    Persistent: h13 [rows, hidden] fp16 + h2 [rows, intermediate] fp16.
    Cache key/storage (_FAT_GROUPED_CACHE) is deliberately separate from
    sparkglm's _GROUPED_PREFILL_SCRATCH_CACHE (a different tier).
    """
    configured = int(
        os.environ.get(
            "EXL3_FAT_SCRATCH_ROWS",
            os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"),
        )
        or 0
    )
    topk = int(os.environ.get("EXL3_FAT_GROUPED_TOPK", "8") or 8)
    needed = max(256, rows)
    key = (str(device), hidden, intermediate)
    scratch = _FAT_GROUPED_CACHE.get(key)
    if scratch is not None and int(scratch["h13"].shape[0]) >= needed:
        return scratch
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "EXL3 grouped scratch growth during CUDA graph capture; warm the "
            f"largest shape first (need {needed} rows)"
        )
    capacity = max(needed, configured * topk)
    scratch = {
        "h13": torch.empty((capacity, hidden), dtype=torch.float16, device=device),
        "h2": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
    }
    _FAT_GROUPED_CACHE[key] = scratch
    _FAT_GROUPED_BYTES[key] = sum(
        t.numel() * t.element_size() for t in scratch.values()
    )
    _EXL3_FAT_DIAG["grouped_scratch_bytes"] = sum(_FAT_GROUPED_BYTES.values())
    return scratch


def _excl_cumsum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inclusive = torch.cumsum(x, 0)
    return inclusive - x, inclusive


def build_grouped_fat_tables(
    counts: torch.Tensor,
    cap: int,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    rows_cap: int,
    tile_rows: int,
) -> dict[str, torch.Tensor]:
    """Device-side row/segment tables for the E3 grouped fat kernels.

    Fat experts (count > cap) are laid out back to back in expert order in a
    fat-row buffer; each kernel CTA owns one `tile_rows` slice of one expert.
    Everything is computed with device ops on capacity-sized tensors and the
    kernels read the live `num_rows` / `num_segs`, so no host sync happens
    and the layer stays CUDA-graph capturable. `counts` excludes the
    invalid/nonlocal sentinel bucket, which the sort places after every
    real expert, so sentinel routes never enter a fat segment.
    """
    n_exp = int(counts.numel())
    device = counts.device
    fat_rows = torch.where(counts > cap, counts, torch.zeros_like(counts))
    row_off, row_cum = _excl_cumsum(fat_rows)
    sorted_off, _ = _excl_cumsum(counts)
    tiles = (fat_rows + (tile_rows - 1)) // tile_rows
    tile_off, tile_cum = _excl_cumsum(tiles)
    num_segs = tile_cum[-1:].to(torch.int32)
    num_rows = row_cum[-1:].to(torch.int32)
    max_segs = (rows_cap + tile_rows - 1) // tile_rows + n_exp
    seg = torch.arange(max_segs, device=device)
    e = torch.searchsorted(tile_cum, seg, right=True).clamp_(max=n_exp - 1)
    local_tile = seg - tile_off[e]
    seg_row0 = row_off[e] + local_tile * tile_rows
    seg_rows = torch.clamp(fat_rows[e] - local_tile * tile_rows, min=0, max=tile_rows)
    r = torch.arange(rows_cap, device=device)
    re = torch.searchsorted(row_cum, r, right=True).clamp_(max=n_exp - 1)
    src = (sorted_off[re] + (r - row_off[re])).clamp_(max=rows_cap - 1)
    return {
        "seg_expert": e.to(torch.int32),
        "seg_row0": seg_row0.to(torch.int32),
        "seg_rows": seg_rows.to(torch.int32),
        "num_segs": num_segs,
        "num_rows": num_rows,
        "row_expert": re.to(torch.int32),
        "row_token": token_sorted.index_select(0, src),
        "row_weight": weight_sorted.index_select(0, src),
    }


def apply_exl3_grouped_fat(
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    layer: torch.nn.Module,
    cap: int,
    limit: float,
) -> None:
    """E3: every fat expert of the layer in three launches, no host sync."""
    ext = load_fat_moe_ext()
    if ext is None:
        raise RuntimeError("EXL3 grouped tier selected but the E3 kernels are not loaded")
    ptrs = layer._exl3_ptrs
    device = ptrs["gate_trellis"].device
    if xh.device != device or out.device != device or counts.device != device:
        raise RuntimeError(
            f"EXL3 grouped tier: activations on {xh.device}, experts on {device}"
        )
    if not (xh.is_contiguous() and out.is_contiguous() and out.dtype == torch.float32):
        raise RuntimeError("EXL3 grouped tier needs contiguous fp16 input / fp32 output")
    hidden = int(xh.shape[1])
    intermediate = int(layer._exl3_intermediate_local)
    rows_cap = int(token_sorted.numel())
    scratch = _grouped_scratch(device, rows_cap, hidden, intermediate)
    h13 = scratch["h13"][:rows_cap]
    h2 = scratch["h2"][:rows_cap]
    tile_gu = int(ext.exl3_fat_moe_tile_rows_gateup())
    tile_dn = int(ext.exl3_fat_moe_tile_rows_down())
    token_sorted = token_sorted.contiguous()
    weight_sorted = weight_sorted.contiguous()
    tg = build_grouped_fat_tables(
        counts, cap, token_sorted, weight_sorted, rows_cap, tile_gu
    )
    td = (
        tg
        if tile_dn == tile_gu
        else build_grouped_fat_tables(
            counts, cap, token_sorted, weight_sorted, rows_cap, tile_dn
        )
    )
    ext.exl3_fat_moe_gather(
        xh, tg["row_token"], tg["row_expert"], ptrs["gate_suh"], h13, tg["num_rows"]
    )
    ext.exl3_fat_moe_gateup(
        h13,
        ptrs["gate_trellis"],
        ptrs["up_trellis"],
        ptrs["gate_svh"],
        ptrs["up_svh"],
        ptrs["down_suh"],
        h2,
        tg["seg_expert"],
        tg["seg_row0"],
        tg["seg_rows"],
        tg["num_segs"],
        float(limit),
    )
    ext.exl3_fat_moe_down(
        h2,
        ptrs["down_trellis"],
        ptrs["down_svh"],
        out,
        td["row_token"],
        td["row_weight"],
        td["seg_expert"],
        td["seg_row0"],
        td["seg_rows"],
        td["num_segs"],
    )
    _EXL3_FAT_DIAG["grouped_calls"] += 1


def exl3_moe_fast_requested() -> bool:
    """Opt-in SM121 K4/N256 thin-decode dispatch (default off).

    Mirrors the native dispatcher's validation: anything other than 0/1
    raises at load instead of surfacing as a native TORCH_CHECK on the
    first decode call.
    """
    raw = os.environ.get("GLM53_EXL3_MOE_FAST", "0")
    if raw not in ("0", "1"):
        raise RuntimeError("GLM53_EXL3_MOE_FAST must be 0 or 1")
    return raw == "1"


def exl3_clamp_diag_enabled() -> bool:
    """SwiGLU-clamp shadow diagnostic (default off, see the comment above
    _EXL3_CLAMP_DIAG). Roughly doubles gate/up GEMM cost while on -- a
    data-collection tool, not a production default."""
    raw = os.environ.get("GLM53_EXL3_CLAMP_DIAG", "0")
    if raw not in ("0", "1"):
        raise RuntimeError("GLM53_EXL3_CLAMP_DIAG must be 0 or 1")
    return raw == "1"


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    import exllamav3_ext

    # Fail closed: an explicitly requested fast thin-decode path must never
    # silently run the stock kernel on an image built without it.
    fast = exl3_moe_fast_requested()
    if fast:
        if not hasattr(exllamav3_ext, "glm53_fast_moe_version"):
            raise RuntimeError(
                "GLM53_EXL3_MOE_FAST=1 requires the native decode-pipeline "
                "image (exllamav3_ext.glm53_fast_moe_version); this image "
                "was built without overlay/patch_exl3_decode_pipeline.py"
            )
        if exllamav3_ext.glm53_fast_moe_version() != 1:
            raise RuntimeError("Unsupported native EXL3 decode-pipeline version")

    device = layer.w13_trellis.device
    n_exp = len(inners)
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    # Gate/up SUH equality was verified across every expert at load time
    # (layer._exl3_shared_w13_suh, torch.equal on the packed tensors),
    # before weights were released. Aliasing the pointer tables is what lets
    # the native fast path prove the reuse predicate by pointer identity
    # (gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr()) and skip the
    # redundant up-input Hadamard. Only the fast path needs it, so FAST=0
    # leaves the tables exactly as the stock path builds them; FAST=1 with an
    # unequal checkpoint keeps both tables and takes the independent-transform
    # fast kernel.
    if fast and bool(getattr(layer, "_exl3_shared_w13_suh", False)):
        layer._exl3_ptrs["up_suh"] = layer._exl3_ptrs["gate_suh"]
    idx = int(device.index) if device.index is not None else 0
    concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
    if concurrency < 1:
        concurrency = 1
    rows = temp_rows_fused()
    key = (str(device), hidden, intermediate, concurrency, rows)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
        _EXL3_FAT_DIAG["fused_temps_allocs"] += 1
    # Layers share one cache entry, so assign (never accumulate) the bytes.
    _EXL3_FAT_DIAG["fused_temps_bytes"] = sum(
        t.numel() * t.element_size() for t in temps
    )
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)

    # All MoE layers execute serially, so a single persistent arena per device
    # and shape is sufficient. Keeping allocation out of apply is required for
    # CUDA graph capture and makes the candidate's memory cost explicit.
    layer._exl3_decode_coop_scratch = None
    if (
        decode_coop_enabled()
        and int(layer._exl3_bits) == 4
        and hidden == 4096
        and intermediate == 1024
        and hasattr(exllamav3_ext, "exl3_decode_moe_k4")
    ):
        max_tokens = decode_coop_max_tokens()
        max_routes = max_tokens * 8
        coop_key = (str(device), hidden, intermediate, max_routes)
        scratch = _DECODE_COOP_SCRATCH_CACHE.get(coop_key)
        if scratch is None:
            scratch = {
                "had_gate": torch.empty(
                    (max_routes, hidden), dtype=torch.float16, device=device
                ),
                "had_up": torch.empty(
                    (max_routes, hidden), dtype=torch.float16, device=device
                ),
                "gate": torch.empty(
                    (max_routes, intermediate), dtype=torch.float16, device=device
                ),
                "up": torch.empty(
                    (max_routes, intermediate), dtype=torch.float16, device=device
                ),
                "had_down": torch.empty(
                    (max_routes, intermediate), dtype=torch.float16, device=device
                ),
                "down": torch.empty(
                    (max_routes, hidden), dtype=torch.float16, device=device
                ),
            }
            _DECODE_COOP_SCRATCH_CACHE[coop_key] = scratch
        layer._exl3_decode_coop_scratch = scratch

        global _exl3_decode_coop_logged
        if not _exl3_decode_coop_logged:
            scratch_mib = sum(
                tensor.numel() * tensor.element_size()
                for tensor in scratch.values()
            ) / (1024 * 1024)
            logger.warning(
                "EXL3 cooperative decode candidate enabled: K4 H4096 I1024 "
                "max_tokens=%d scratch=%.2f MiB",
                max_tokens,
                scratch_mib,
            )
            _exl3_decode_coop_logged = True


# ----------------------------------------------------------------------------
# v21-b12xmoe: b12x fused-trellis MoE dispatch (opt-in, GLM53_EXL3_B12X_MOE=1).
#
# Zero-copy by construction: w13_trellis/w13_suh/w13_svh/w13_mcg are already
# allocated projection-major ([2,E,...], see create_weights above) to match
# b12x's own trellis_t256_proj storage exactly, so
# prepare_trellis256_moe_weights() never has to reshape/copy the trellis
# tensors -- confirmed at load time below via a data_ptr() equality assert,
# not just assumed. Measured on real layer-10 checkpoint weights: new
# steady-state residency ~71 MiB/rank total (the unavoidable
# intermediate_rotations concat), vs. the 47.25 GiB/rank a post-load
# black-box-adapter version of this same idea cost (see
# b12x_moe_adapter_infeasible_2026_09_18 in this project's memory).
#
# KNOWN LIMITATION, not worked around: b12x's full_rotation=True mode (the
# only mode that accepts EXL3's trellis_t256 prepared weights) REJECTS a
# SwiGLU clamp outright -- "intermediate_rotation requires unclamped gated
# silu or situ (no swiglu limit/oai)" is a real b12x kernel.py ValueError,
# not a guess. This fork's own apply_exl3_python_loop / apply_exl3_fused_moe
# both clamp gate/up to +-SWIGLU_LIMIT_DEFAULT (10.0) before SiLU; the b12x
# path here does not and cannot apply that same clamp. Parity testing
# (v21-b12xmoe's tests/test_exl3_overlay.py) confirms agreement across
# realistic activation ranges where the clamp never triggers, but this is a
# genuine, uncorrected semantic difference for out-of-range activations, not
# a solved problem. `limit` is accepted for call-site signature symmetry
# with the other two apply_* functions and is otherwise unused here.
_B12X_PIN_VERSION = "1.3.0"
_B12X_PREPARE_EXPECTED_PARAMS = {
    "w13", "w2", "hidden_size", "intermediate_size", "num_experts",
    "activation", "fc1_tile_n", "fc2_tile_n", "device", "seed",
    "params_dtype", "w13_layout", "trellis_bits", "dummy_scale", "codebook",
    "gate_suh", "up_suh", "intermediate_rotations", "down_svh",
    "tile_config", "workspace",
}


def exl3_b12x_moe_requested() -> bool:
    """Opt-in b12x fused-trellis MoE dispatch (default off).

    Mirrors exl3_moe_fast_requested's validation: anything other than 0/1
    raises at load instead of surfacing as a confusing failure on the first
    decode call.
    """
    raw = os.environ.get("GLM53_EXL3_B12X_MOE", "0")
    if raw not in ("0", "1"):
        raise RuntimeError("GLM53_EXL3_B12X_MOE must be 0 or 1")
    return raw == "1"


def _verify_b12x_pin():
    """Fail closed on a b12x version/signature drift, not silently.

    prepare_trellis256_moe_weights lives in b12x's PRIVATE
    b12x.moe._shared.kernels.w4a16.prepare module (not the public
    b12x.moe.fused_moe.api, which is for b12x's own atoms/rate format, not
    EXL3's trellis_t256 -- b12x's own fused_moe/trellis.py imports the same
    private path for the same reason). A version bump could silently change
    this function's parameter names/semantics; assert the pin and the
    expected signature explicitly rather than discovering a mismatch as a
    confusing runtime error mid-decode.
    """
    import inspect

    import b12x

    installed = getattr(b12x, "__version__", None)
    if installed is not None and installed != _B12X_PIN_VERSION:
        raise RuntimeError(
            f"GLM53_EXL3_B12X_MOE=1 requires b12x=={_B12X_PIN_VERSION}, "
            f"found {installed!r} -- re-verify prepare_trellis256_moe_weights's "
            "signature and the w13_layout='trellis_t256_proj' contract "
            "against the new version before bumping this pin"
        )
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

    actual_params = set(inspect.signature(prepare_trellis256_moe_weights).parameters)
    if not _B12X_PREPARE_EXPECTED_PARAMS.issubset(actual_params):
        missing = _B12X_PREPARE_EXPECTED_PARAMS - actual_params
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1: b12x.moe._shared.kernels.w4a16.prepare."
            f"prepare_trellis256_moe_weights is missing expected parameter(s) "
            f"{sorted(missing)} -- signature drift, re-verify against b12x "
            f"source before proceeding (installed version: {installed!r})"
        )
    return prepare_trellis256_moe_weights


def build_b12x_prepared_state(layer: torch.nn.Module) -> None:
    """Build layer._b12x_prepared once after load. Fail closed on any mismatch.

    Requires the projection-major w13_* layout from create_weights above.
    Verifies the central zero-copy thesis with a real data_ptr() equality
    assert, not just trusting b12x's "no bytes copied" docstring claim.
    """
    prepare_trellis256_moe_weights = _verify_b12x_pin()

    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)
    device = layer.w13_trellis.device
    num_experts = int(layer.w13_trellis.shape[1])  # [2, E, ...] -- projection-major

    gate_svh = layer.w13_svh[0].contiguous()
    up_svh = layer.w13_svh[1].contiguous()
    down_suh = layer.w2_suh.contiguous()
    # Order is [gate_svh, up_svh, down_suh], per-expert-interleaved (dim=1,
    # giving [E, 3*intermediate]) -- confirmed against b12x source
    # (prepare.py's own rotation-order comment) and empirically: a 6-way
    # permutation sweep in this recipe's tests found this ordering matches
    # apply_exl3_python_loop's reference output (cos>=0.999999) while moving
    # gate_svh out of position 0 collapses agreement to cos~0.64-0.79. (Note:
    # swapping positions 1/2 -- up_svh vs down_suh -- was NOT independently
    # discriminating on this checkpoint's real data at the standard tolerance
    # because down_suh's real values are naturally ~65x smaller in magnitude
    # than up_svh's; a follow-up adversarial test with down_suh artificially
    # scaled to up_svh's magnitude still showed only a small, non-bit-identical
    # difference between the two orders (maxabs ~0.007 on outputs ~0.1-0.3),
    # confirming the position is genuinely low-sensitivity for this model
    # rather than a test bug -- the ordering used here is the one the b12x
    # source states is correct, not merely the one that happened to pass.)
    intermediate_rotations = torch.cat([gate_svh, up_svh, down_suh], dim=1).contiguous()

    prepared = prepare_trellis256_moe_weights(
        layer.w13_trellis, layer.w2_trellis,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=num_experts,
        activation="silu",
        fc1_tile_n=256,
        fc2_tile_n=256,
        device=device,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=int(layer._exl3_bits),
        codebook="mcg",
        gate_suh=layer.w13_suh[0].contiguous(),
        up_suh=layer.w13_suh[1].contiguous(),
        intermediate_rotations=intermediate_rotations,
        down_svh=layer.w2_svh.contiguous(),
    )
    if prepared.w13.data_ptr() != layer.w13_trellis.data_ptr():
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1: b12x's prepared w13 is not a zero-copy "
            "view of this layer's w13_trellis -- the central premise of this "
            "integration (projection-major allocation matching b12x's "
            "trellis_t256_proj layout) does not hold on this b12x build; "
            "refusing to proceed rather than silently eat the multi-GiB copy "
            "this design exists to avoid"
        )
    if prepared.w2.data_ptr() != layer.w2_trellis.data_ptr():
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1: b12x's prepared w2 is not a zero-copy "
            "view of this layer's w2_trellis"
        )
    layer._b12x_prepared = prepared
    layer._b12x_num_experts = num_experts

    # Build (or reuse, if another layer already built one for the same
    # shape) the fixed-capacity scratch arena NOW, at load time -- long
    # before any CUDA graph capture, for either the main model's own
    # captures or the MTP speculator's. See _b12x_arena_for's docstring for
    # why this must never be built lazily on first apply-time use.
    topk, m_max = _b12x_topk_and_m_max_from_vllm_config()
    layer._b12x_topk = topk
    _b12x_arena_for(prepared, device, hidden, intermediate, num_experts, topk, m_max)


def _b12x_topk_and_m_max_from_vllm_config() -> tuple[int, int]:
    """(num_experts_per_tok, max_num_batched_tokens) from the live vLLM config.

    Both are fixed, load-time-known constants (a model architecture constant
    and a scheduler config value), not request-varying -- so unlike
    _glm53_layer_types elsewhere in this file, there is no safe default to
    degrade to on failure. An arena sized from a wrong/guessed value here is
    a silent-corruption risk (an undersized buffer), not a crash, so this
    raises loudly rather than swallowing the exception.
    """
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()
    topk = getattr(cfg.model_config.hf_text_config, "num_experts_per_tok", None)
    m_max = getattr(cfg.scheduler_config, "max_num_batched_tokens", None)
    if topk is None or m_max is None:
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1: could not read num_experts_per_tok "
            f"({topk!r}) and/or max_num_batched_tokens ({m_max!r}) from the "
            "live vLLM config -- refusing to guess a b12x scratch-arena size "
            "from a default, since an undersized arena is a silent memory-"
            "corruption risk, not a crash"
        )
    return int(topk), int(m_max)


@dataclass
class _B12XArena:
    """One fixed-capacity scratch arena, shared across every MoE layer with
    matching (device, hidden, intermediate, num_experts, topk) -- including
    both the main model's routed-expert layers and the MTP speculator's
    (Glm5NextMTP builds its MoE layer from the same Glm5NextMoE class, so it
    shares this key in practice). Mirrors _FUSED_TEMP_CACHE's existing
    pattern: sized once for the worst case, at load time, so no torch.empty()
    ever happens again on the apply path -- provably cannot be a CUDA-graph-
    capture-time allocation because there is exactly one allocation, ever,
    made outside of any capture context (build_b12x_prepared_state runs
    during process_weights_after_loading, unconditionally before any graph
    capture for either the main model or the speculator).
    """

    m_max: int
    topk: int
    num_experts: int
    hidden: int
    intermediate: int
    sms: int
    coupled_hadamard: bool
    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    output: torch.Tensor
    fc1_c_tmp: torch.Tensor
    fc2_c_tmp: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    rotation_a_gate: torch.Tensor
    rotation_a_up: torch.Tensor


_B12X_ARENA_CACHE: dict[tuple, _B12XArena] = {}


def _b12x_build_arena(prepared, device: torch.device, hidden: int, intermediate: int,
                       num_experts: int, topk: int, m_max: int) -> _B12XArena:
    """Size one arena for the worst case across every m in [1, m_max].

    Uses b12x's own plan_w4a16_buffers as the authoritative sizing source --
    an exhaustive sweep over every m, not a hand-derived formula or a
    monotonicity assumption (block_size_m's bucket selection and the
    sms-based min() cap in packed_gemm_scratch_elements make several of
    these fields non-obviously monotonic in m; measuring every value is
    cheap -- pure host-side Python arithmetic, no GPU ops -- and removes any
    risk of an off-by-one that under-sizes a buffer, which would be a silent
    corruption risk rather than a crash).
    """
    from b12x.moe._shared.kernels.w4a16.host import plan_w4a16_buffers

    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    max_route_slots = 0
    max_route_blocks = 0
    max_fc1_c_tmp = 0
    max_fc2_c_tmp = 0
    max_cache13 = 0
    max_cache2 = 0
    max_rotation = 0
    for m in range(1, int(m_max) + 1):
        plan = plan_w4a16_buffers(
            prepared, m=m, topk=topk, route_num_experts=num_experts, sms=sms,
            full_rotation=True, block_size_m=None,
        )
        max_route_slots = max(max_route_slots, plan.route_slots)
        max_route_blocks = max(max_route_blocks, plan.route_blocks)
        max_fc1_c_tmp = max(max_fc1_c_tmp, plan.fc1_c_tmp_elements)
        max_fc2_c_tmp = max(max_fc2_c_tmp, plan.fc2_c_tmp_elements)
        max_cache13 = max(max_cache13, plan.intermediate_cache13_elements)
        max_cache2 = max(max_cache2, plan.intermediate_cache2_elements)
        max_rotation = max(max_rotation, plan.rotation_a_elements)

    coupled_hadamard = bool(getattr(prepared, "coupled_hadamard", False))
    max_output_elements = int(m_max) * int(hidden)

    rotation_a_gate = torch.empty((max(max_rotation, 1),), dtype=torch.float16, device=device)
    rotation_a_up = (
        rotation_a_gate
        if coupled_hadamard
        else torch.empty((max(max_rotation, 1),), dtype=torch.float16, device=device)
    )

    return _B12XArena(
        m_max=int(m_max),
        topk=int(topk),
        num_experts=int(num_experts),
        hidden=int(hidden),
        intermediate=int(intermediate),
        sms=sms,
        coupled_hadamard=coupled_hadamard,
        intermediate_cache13=torch.empty(
            (max(max_cache13, 1),), dtype=torch.float16, device=device
        ),
        intermediate_cache2=torch.empty(
            (max(max_cache2, 1),), dtype=torch.float16, device=device
        ),
        output=torch.empty((max(max_output_elements, 1),), dtype=torch.float32, device=device),
        fc1_c_tmp=torch.empty((max(max_fc1_c_tmp, 1),), dtype=torch.float32, device=device),
        fc2_c_tmp=torch.empty((max(max_fc2_c_tmp, 1),), dtype=torch.float32, device=device),
        packed_route_indices=torch.empty(
            (max(max_route_slots, 1),), dtype=torch.int32, device=device
        ),
        block_expert_ids=torch.empty(
            (max(max_route_blocks, 1),), dtype=torch.int32, device=device
        ),
        packed_route_count=torch.empty((1,), dtype=torch.int32, device=device),
        expert_offsets=torch.empty((num_experts + 1,), dtype=torch.int32, device=device),
        expert_counts=torch.empty((num_experts,), dtype=torch.int32, device=device),
        rotation_a_gate=rotation_a_gate,
        rotation_a_up=rotation_a_up,
    )


def _b12x_arena_for(prepared, device: torch.device, hidden: int, intermediate: int,
                     num_experts: int, topk: int, m_max: int | None = None) -> _B12XArena:
    """Get (building once if needed) the shared arena for this shape.

    Called eagerly from build_b12x_prepared_state at load time (m_max
    provided) so the cache is always warm by the time apply_exl3_b12x_moe
    runs. The is_current_stream_capturing() branch below is defense in
    depth, not the primary safety mechanism -- it should never fire given
    the eager-build-at-load-time design, but if some future code path
    reaches this cold during capture anyway, fail loudly rather than
    silently allocate (reintroducing the exact bug this arena exists to
    close) or silently fall back to a different numerical path (which would
    permanently and invisibly downgrade that graph's MoE tier forever, the
    same blind-spot class this fork already got burned by once this week).
    """
    key = (str(device), hidden, intermediate, num_experts, topk)
    arena = _B12X_ARENA_CACHE.get(key)
    if arena is not None:
        return arena
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1: b12x scratch arena was never built before "
            "CUDA graph capture reached this layer shape "
            f"{key} -- build_b12x_prepared_state() must build it during "
            "process_weights_after_loading, well before any capture. "
            "Refusing to allocate scratch memory mid-capture."
        )
    if m_max is None:
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1: no b12x scratch arena exists yet for "
            f"layer shape {key} and no m_max was provided to build one -- "
            "this should only ever be reached from build_b12x_prepared_state, "
            "not from a bare apply-time lookup"
        )
    arena = _b12x_build_arena(prepared, device, hidden, intermediate, num_experts, topk, m_max)
    _B12X_ARENA_CACHE[key] = arena
    return arena


def apply_exl3_b12x_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor:
    """b12x fused-trellis MoE apply. See the module comment above this section
    for the zero-copy design and the known SwiGLU-clamp limitation. `inners`
    and `expert_map` are accepted for call-site symmetry with
    apply_exl3_fused_moe/apply_exl3_python_loop but unused: b12x dispatches
    from layer._b12x_prepared directly, and this integration does not yet
    support TP expert-parallelism remapping (`expert_map`) -- see
    grouped_fat_eligibility-style checks for the pattern to extend this if
    that's ever needed.
    """
    del inners, limit
    from b12x.moe._shared.kernels.w4a16.host import plan_w4a16_buffers
    from b12x.moe._shared.kernels.w4a16.route_pack import pack_topk_routes_by_expert
    from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe

    prepared = getattr(layer, "_b12x_prepared", None)
    if prepared is None:
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1 but layer._b12x_prepared was never built "
            "-- build_b12x_prepared_state() must run in "
            "process_weights_after_loading before this is called"
        )
    if expert_map is not None:
        raise RuntimeError(
            "GLM53_EXL3_B12X_MOE=1 does not support expert-parallel "
            "expert_map remapping yet"
        )
    device = x2d.device
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)
    num_experts = int(layer._b12x_num_experts)
    m = int(ids.shape[0])
    topk = int(ids.shape[-1])
    expected_topk = int(layer._b12x_topk)
    if topk != expected_topk:
        raise RuntimeError(
            f"GLM53_EXL3_B12X_MOE=1: this call's topk ({topk}) does not match "
            f"the model-constant topk ({expected_topk}) the scratch arena was "
            "sized for at load time -- the arena's worst-case sizing assumed "
            "a fixed topk, so proceeding could silently overflow a buffer"
        )

    # Fixed-capacity arena, built once at load time (build_b12x_prepared_state)
    # -- no torch.empty() happens on this path, ever, so this is safe to call
    # from inside CUDA graph capture (both the main model's and the MTP
    # speculator's), unlike the m-keyed lazy-allocation design this replaced.
    arena = _b12x_arena_for(prepared, device, hidden, intermediate, num_experts, topk)
    if m > arena.m_max:
        raise RuntimeError(
            f"GLM53_EXL3_B12X_MOE=1: this call's m ({m}) exceeds the scratch "
            f"arena's swept worst case (m_max={arena.m_max}, taken from "
            "max_num_batched_tokens at load time) -- refusing to silently "
            "allocate a bigger buffer, which would reintroduce the exact "
            "CUDA-graph-capture-time-allocation bug this arena exists to "
            "prevent; re-check max_num_batched_tokens against real traffic"
        )

    a_input = x2d.contiguous().half()
    topk_ids = ids.reshape(m, topk).to(torch.int32).contiguous()
    topk_weights = weights.reshape(m, topk).to(torch.float32).contiguous()

    plan = plan_w4a16_buffers(
        prepared, m=m, topk=topk, route_num_experts=num_experts, sms=arena.sms,
        full_rotation=True, block_size_m=None,
    )
    block_size_m = plan.block_size_m

    intermediate_cache13 = arena.intermediate_cache13.narrow(
        0, 0, plan.intermediate_cache13_elements
    )
    intermediate_cache2 = arena.intermediate_cache2.narrow(
        0, 0, plan.intermediate_cache2_elements
    ).view(plan.routed_rows, intermediate)
    output = arena.output.narrow(0, 0, m * hidden).view(m, hidden)
    fc1_c_tmp = arena.fc1_c_tmp.narrow(0, 0, plan.fc1_c_tmp_elements)
    fc2_c_tmp = arena.fc2_c_tmp.narrow(0, 0, plan.fc2_c_tmp_elements)
    packed_route_indices = arena.packed_route_indices.narrow(0, 0, plan.route_slots)
    block_expert_ids = arena.block_expert_ids.narrow(0, 0, plan.route_blocks)
    rotation_a_gate = arena.rotation_a_gate.narrow(0, 0, plan.rotation_a_elements).view(
        plan.routed_rows, hidden
    )
    rotation_a_up = (
        rotation_a_gate
        if arena.coupled_hadamard
        else arena.rotation_a_up.narrow(0, 0, plan.rotation_a_elements).view(
            plan.routed_rows, hidden
        )
    )

    pack_topk_routes_by_expert(
        topk_ids, block_size_m, num_experts,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=arena.packed_route_count,
        expert_offsets=arena.expert_offsets,
        expert_counts=arena.expert_counts,
    )
    out = run_w4a16_moe(
        a_input, prepared, topk_weights, topk_ids,
        activation="silu",
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=output,
        fc1_c_tmp=fc1_c_tmp,
        fc2_c_tmp=fc2_c_tmp,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=arena.packed_route_count,
        expert_offsets=arena.expert_offsets,
        expert_counts=arena.expert_counts,
        expert_map=None,
        output_expert_map=None,
        full_rotation=True,
        suh_gate_table=prepared.gate_suh,
        suh_up_table=prepared.up_suh,
        svh_table=prepared.down_svh,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        rotation_a_gate=rotation_a_gate,
        rotation_a_up=rotation_a_up,
        route_block_size_m=block_size_m,
    )
    return out
# ----------------------------------------------------------------------------


def _exl3_moe_launch(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    expert_count: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, torch.Tensor],
    k: int,
    limit: float,
    n_active_host: int | None,
) -> None:
    args = (
        xh,
        out,
        expert_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        MOE_ACT_SILU,
        k,
        k,
        k,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        True,
        False,
        True,
        False,
        True,
        False,
        float(limit),
    )
    if n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)


def _exl3_moe_row_tiles(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, torch.Tensor],
    k: int,
    limit: float,
    n_active_host: int | None,
    max_rows: int,
) -> None:
    """Launch exl3_moe once per 128-row slice of the sorted expert-token buffer.

    Prefill-only (host syncs). Decode never reaches here: tokens <= temp rows.
    Kernel skips experts with count > temp rows; tiles keep every expert inside temps.
    """
    n_exp = int(counts.shape[0])
    device = counts.device
    tile = int(temps[0].shape[1])
    n_tiles = (max_rows + tile - 1) // tile
    prefix = torch.empty(n_exp, dtype=torch.long, device=device)
    prefix[0] = 0
    if n_exp > 1:
        prefix[1:] = counts[:-1].cumsum(0)
    for t in range(n_tiles):
        row0 = t * tile
        tile_counts = (counts - row0).clamp(min=0, max=tile)
        n_tile = int(tile_counts.sum().item())
        if n_tile == 0:
            continue
        cum = tile_counts.cumsum(0)
        idx = torch.arange(n_tile, device=device)
        expert_id = torch.searchsorted(cum, idx, right=True)
        local_row = idx - (cum[expert_id] - tile_counts[expert_id])
        src = prefix[expert_id] + row0 + local_row
        tile_ec = torch.zeros(n_exp + 1, dtype=torch.long, device=device)
        tile_ec[:n_exp] = tile_counts
        _exl3_moe_launch(
            fn,
            xh,
            out,
            tile_ec,
            token_sorted.index_select(0, src),
            weight_sorted.index_select(0, src),
            temps,
            ptrs,
            k,
            limit,
            n_active_host,
        )


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor:
    """One exl3_moe launch per layer when tokens or hottest expert fit temp rows.

    Cap is temps dim1 (`EXL3_TEMP_ROWS_FUSED`, default 128). Overflow uses GPU
    row tiles if `EXL3_MOE_ROW_TILE=1`; otherwise fat experts use the highest
    enabled tier: kernel implies batched, batched implies sorted, then legacy.
    Decode (tokens ≤ cap) stays a single graph-safe launch.
    """
    import exllamav3_ext

    tokens, hidden = x2d.shape
    n_exp = len(inners)
    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    # scatter_add stays on GPU. torch.bincount can host-stage and break CUDA graphs.
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(
        0, local.long(), torch.ones(local.shape, dtype=torch.long, device=local.device)
    )
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None
    k = int(getattr(layer, "_exl3_k", 4))
    # Actual kernel cap is the allocated temp dim1 (env-selected at load).
    cap = int(temps[0].shape[1])

    # Reset per call so a "kernel" label from an earlier prefill cannot
    # masquerade through later decode/thin/row-tile calls.
    layer._exl3_last_fat_fallback = "none"
    layer._exl3_last_fat_reason = "no_fat_experts"

    if tokens <= cap:
        coop_scratch = getattr(layer, "_exl3_decode_coop_scratch", None)
        if (
            coop_scratch is not None
            and tokens <= decode_coop_max_tokens()
            and k == 4
            and hidden == 4096
            and int(temps[2].shape[2]) == 1024
        ):
            # expert_count is [count(e0), ..., count(eN-1), sentinel=0].
            # Exclusive offsets need no host read: cumsum(count)-count.
            expert_offsets = torch.cumsum(expert_count, dim=0).sub(expert_count)
            expert_sorted = local[order].to(dtype=torch.int32).contiguous()
            exllamav3_ext.exl3_decode_moe_k4(
                xh,
                out,
                expert_offsets,
                expert_sorted,
                token_sorted,
                weight_sorted,
                coop_scratch["had_gate"],
                coop_scratch["had_up"],
                coop_scratch["gate"],
                coop_scratch["up"],
                coop_scratch["had_down"],
                coop_scratch["down"],
                ptrs["gate_trellis"],
                ptrs["gate_suh"],
                ptrs["gate_svh"],
                ptrs["up_trellis"],
                ptrs["up_suh"],
                ptrs["up_svh"],
                ptrs["down_trellis"],
                ptrs["down_suh"],
                ptrs["down_svh"],
                float(limit),
            )
            layer._exl3_last_apply = "decode_coop_k4"
            return out
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        return out

    # Prefill larger than temps. E1 copies routing counts on a side stream and
    # launches thin experts immediately, overlapping the D2H synchronization.
    # Decode never reaches here (capture sizes << cap).
    _EXL3_FAT_DIAG["prefill_layer_calls"] += 1
    use_row_tiles = fused_moe_row_tile_enabled()

    # E3 (v11-e3 addition, top of the tier ladder): every fat expert of the
    # layer in three device-driven launches (gather/gateup/down), no host
    # sync. _exl3_fat_effective_tier was resolved once at load time by
    # _record_exl3_fat_resolution -> resolve_exl3_fat_tier, which already
    # falls back to "kernel" (sparkglm's tier, below) if EXL3_FAT_GROUPED=1
    # was requested but this layer/device is ineligible for E3 -- that
    # fallback is encoded in want_fat_kernel below, not here.
    if (
        grouped_fat_enabled()
        and not use_row_tiles
        and getattr(layer, "_exl3_fat_effective_tier", None) == "grouped"
    ):
        # Thin experts (count <= cap) still go through the fused kernel on
        # this same stream; E3 only ever handles count > cap rows.
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        apply_exl3_grouped_fat(
            xh, out, counts, token_sorted, weight_sorted, layer, cap, limit
        )
        _record_exl3_fat_tier(layer, "grouped", "grouped_ok")
        if fat_expert_log_enabled():
            record_exl3_fat_expert_stats(counts)
        return out

    # A grouped request that fell back (ineligible layer/device, or E3
    # symbols missing -- resolve_exl3_fat_tier already raised at load time
    # for the latter case) still forces the kernel tier on, so sparkglm's
    # exl3_grouped_prefill_k4 below remains the real fallback rather than
    # silently dropping to E2/E1/sorted.
    want_fat_kernel = fat_kernel_enabled() or grouped_fat_enabled()
    want_batched_fat = batched_fat_fallback_enabled() or want_fat_kernel
    use_sorted_fat = sorted_fat_fallback_enabled() or want_batched_fat
    use_batched_fat = (
        want_batched_fat
        and bool(getattr(layer, "_exl3_shared_w13_suh", False))
    )
    use_fat_kernel = use_batched_fat and want_fat_kernel

    # The grouped candidate leaves routing counts on device and runs all fat
    # experts through five phase-wide launches. The stock thin kernel is still
    # launched first on this stream; it skips counts above `cap`, while the
    # grouped kernels skip counts at or below it.
    use_grouped_prefill = (
        grouped_prefill_enabled()
        and use_fat_kernel
        and not use_row_tiles
        and k == 4
        and hidden == 4096
        and int(temps[2].shape[2]) == 1024
    )
    if use_grouped_prefill:
        if not hasattr(exllamav3_ext, "exl3_grouped_prefill_k4"):
            raise RuntimeError(
                "EXL3_GROUPED_PREFILL_K4=1 requires "
                "exllamav3_ext.exl3_grouped_prefill_k4"
            )
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        expert_offsets = torch.cumsum(expert_count, dim=0).sub(expert_count)
        scratch = _grouped_prefill_scratch(
            xh.device,
            int(token_sorted.numel()),
            hidden,
            int(temps[2].shape[2]),
            n_exp,
        )
        task_capacity = (int(token_sorted.numel()) + 63) // 64 + n_exp
        exllamav3_ext.exl3_grouped_prefill_k4(
            xh,
            out,
            expert_offsets,
            token_sorted,
            weight_sorted,
            scratch["had_input"],
            scratch["gate_up"],
            scratch["had_down"],
            scratch["tasks"][:task_capacity],
            scratch["task_count"],
            ptrs["gate_trellis"],
            ptrs["gate_suh"],
            ptrs["gate_svh"],
            ptrs["up_trellis"],
            ptrs["up_svh"],
            ptrs["down_trellis"],
            ptrs["down_suh"],
            ptrs["down_svh"],
            cap,
            float(limit),
        )
        _record_exl3_fat_tier(layer, "kernel", "grouped_gpu_resident")
        return out

    launched = False
    counts_host = None
    if use_batched_fat and not use_row_tiles:
        counts_cpu, count_stream = _stage_counts_to_host(counts)
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        launched = True
        count_stream.synchronize()
        counts_host = counts_cpu.tolist()
    elif use_sorted_fat:
        counts_host = counts.tolist()

    max_rows = (
        max(counts_host, default=0)
        if counts_host is not None
        else int(counts.max().item())
    )
    if fat_expert_log_enabled():
        record_exl3_fat_expert_stats(
            counts, max_rows=max_rows, counts_host=counts_host
        )
    if max_rows <= cap:
        _EXL3_FAT_DIAG["thin_calls"] += 1
        _record_exl3_fat_reason("thin_only")
        if not launched:
            _exl3_moe_launch(
                fn, xh, out, expert_count, token_sorted, weight_sorted,
                temps, ptrs, k, limit, n_active_host,
            )
        return out

    if use_row_tiles:
        _EXL3_FAT_DIAG["row_tile_calls"] += 1
        layer._exl3_last_fat_fallback = "row_tile"
        layer._exl3_last_fat_reason = "row_tile_preempts_fat"
        _record_exl3_fat_reason("row_tile_preempts_fat")
        _exl3_moe_row_tiles(
            fn, xh, out, counts, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host, max_rows,
        )
        return out

    if not launched:
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
    if use_batched_fat:
        if use_fat_kernel:
            _record_exl3_fat_tier(layer, "kernel", "kernel_ok")
        else:
            _record_exl3_fat_tier(layer, "batched", "batched_ok")
        assert counts_host is not None
        apply_exl3_batched_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
            use_kernel=use_fat_kernel,
        )
    elif use_sorted_fat:
        _record_exl3_fat_tier(
            layer,
            "sorted",
            "degraded_shared_suh" if want_batched_fat else "sorted_ok",
        )
        assert counts_host is not None
        apply_exl3_sorted_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
        )
    else:
        _record_exl3_fat_tier(layer, "legacy", "legacy_default")
        fat = (counts > cap).nonzero(as_tuple=False).view(-1)
        if fat.numel():
            apply_exl3_python_loop(
                x2d,
                ids,
                weights,
                inners,
                expert_map,
                limit,
                only_experts=set(int(i) for i in fat.tolist()),
                out=out,
            )
            _EXL3_FAT_DIAG["fat_expert_runs"] += int(fat.numel())
    return out


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float = SWIGLU_LIMIT_DEFAULT,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)
    if exl3_clamp_diag_enabled() and not torch.cuda.is_current_stream_capturing():
        # Fixed 2026-09-19: this fork's CUDA-graph warmup/capture phase runs
        # a real forward pass with synthetic dummy data through this exact
        # code path before any real request is ever served. The diagnostic's
        # torch.unique(ids) call (and everything downstream of it) requires
        # a host sync, which CUDA explicitly forbids while a stream is
        # capturing ("operation not permitted when stream is capturing" --
        # crashed both ranks on first deploy). Skipping during capture is
        # also the right behavior independent of the crash: capture uses
        # synthetic data, not real traffic, so sampling it would pollute the
        # very statistic this diagnostic exists to measure.
        _EXL3_CLAMP_DIAG["enabled"] = True
        _exl3_clamp_diag_sample(x2d, ids, weights, inners, expert_map)
    if exl3_b12x_moe_requested():
        # Fail closed: requested-but-not-built means process_weights_after_
        # loading either never ran this path or its own fail-closed check
        # already raised -- apply_exl3_b12x_moe raises with a clear message
        # rather than silently falling through to the loop/fused path below.
        out = apply_exl3_b12x_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "b12x"
        return out.to(dtype=x.dtype)
    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "fused"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for suffix in EXL3_SUFFIXES:
        if tail == suffix or tail.endswith("_" + suffix):
            return suffix
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Routed-experts-only EXL3/MCG. Dense / shared / attention stay native."""

    def __init__(
        self,
        bits: int = 4,
        codebook: str = "mcg",
        scope: str = "glm53_routed_experts_only",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.codebook = str(codebook)
        self.scope = str(scope)
        self.raw_config = dict(kwargs)
        if self.codebook != "mcg":
            raise ValueError(
                f"this overlay only implements codebook=mcg; got {self.codebook!r}"
            )
        if self.bits not in (3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 bits={self.bits}")

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # LinearEXL3 uses CUDA >= Ampere; GB10 is SM121.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skip = {
            "bits",
            "codebook",
            "scope",
            "quant_method",
            # tr3 ships a 37 MiB per-tensor ledger; keep it off the config object.
            "tensor_storage",
        }
        return cls(
            bits=int(config.get("bits", 4)),
            codebook=str(config.get("codebook", "mcg")),
            scope=str(config.get("scope", "glm53_routed_experts_only")),
            **{k: v for k, v in config.items() if k not in skip},
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self)
        if isinstance(layer, LinearBase):
            group = _glm53_dense_fp8_group(prefix)  # [glm53-dense-fp8]
            if group is not None:
                return Glm53DenseFp8Method(group, prefix)
            return UnquantizedLinearMethod()
        return None


# ----------------------------------------------------------------------------
# [glm53-dense-fp8] Optional FP8 weight-only (Marlin) path for the BF16 dense
# projections, ported from MiaAI-Lab's 2026-09-08 upstream update
# (overlay/patch_dense_fp8.py, commit eb0da1d). GLM53_DENSE_FP8=off (default)
# | comma list of groups:
#   shared  mlp.shared_experts.{gate_up_proj,down_proj}
#   dense   mlp.{gate_up_proj,down_proj} of the dense-MLP layers
#   kda     self_attn.{in_proj_qkvbfg_a,f_b_proj,g_b_proj,o_proj} of KDA layers
#   mla     self_attn.{fused_qkv_a_proj,q_b_proj,o_proj} of MLA layers (kv_b_proj
#           stays BF16: MLA reads its weight directly for the absorbed matmuls)
# Weights load as BF16 exactly as today, then process_weights_after_loading
# quantizes per output channel to FP8 e4m3 and repacks for the Marlin kernel.
# PROVISIONAL, per upstream's own commit message: "changes target numerics;
# needs a KLD panel before it can become a default." Their own proxy measure:
# KL vs stock 0.002-0.013 nats/position, argmax agreement 94-100% -- a real,
# non-zero accuracy trade-off, not validated with a full KLD panel even by
# its authors. Gated off by default here too -- SCRATCH A/B MEASUREMENT BUILD
# ONLY (recipes/build/glm53-exl3-v20-densefp8-ab), not a promotion candidate.
# Ported from v16-densefp8's overlay/exl3.py onto the v20-upstreamsync
# baseline to re-measure the tradeoff against the current kernel stack. See
# recipes/VALIDATION.md.
# ----------------------------------------------------------------------------
_GLM53_DENSE_FP8_SUFFIXES = {
    "shared": (".mlp.shared_experts.gate_up_proj", ".mlp.shared_experts.down_proj"),
    "dense": (".mlp.gate_up_proj", ".mlp.down_proj"),
    "kda": (".self_attn.in_proj_qkvbfg_a", ".self_attn.f_b_proj", ".self_attn.g_b_proj", ".self_attn.o_proj"),
    "mla": (".self_attn.fused_qkv_a_proj", ".self_attn.q_b_proj", ".self_attn.o_proj"),
}


def _glm53_dense_fp8_groups() -> set[str]:
    raw = os.environ.get("GLM53_DENSE_FP8", "off").strip().lower()
    if raw in ("", "off", "0", "no", "none"):
        return set()
    if raw in ("all", "on", "1"):
        return {"shared", "dense", "kda", "mla"}
    groups = {g.strip() for g in raw.split(",") if g.strip()}
    unknown = groups - set(_GLM53_DENSE_FP8_SUFFIXES)
    if unknown:
        raise ValueError(f"GLM53_DENSE_FP8: unknown group(s) {sorted(unknown)}")
    return groups


def _glm53_layer_types() -> list[str] | None:
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config().model_config.hf_text_config
        lt = getattr(cfg, "layer_types", None)
        return list(lt) if lt else None
    except Exception:  # noqa: BLE001
        return None


def _glm53_dense_fp8_group(prefix: str, groups: set[str] | None = None, layer_types: list[str] | None = None) -> str | None:
    """Group name if `prefix` (vLLM module path) is an allow-listed dense projection."""
    groups = _glm53_dense_fp8_groups() if groups is None else groups
    if not groups:
        return None
    if ".mtp" in prefix or "visual" in prefix or "draft" in prefix:
        return None
    m = re.search(r"\.layers\.(\d+)\.", prefix)
    layer_idx = int(m.group(1)) if m else None
    for group in ("shared", "dense", "kda", "mla"):
        if group not in groups:
            continue
        if not any(prefix.endswith(s) for s in _GLM53_DENSE_FP8_SUFFIXES[group]):
            continue
        if group == "dense" and ".shared_experts." in prefix:
            continue
        if group in ("kda", "mla"):
            lt = _glm53_layer_types() if layer_types is None else layer_types
            if lt is None or layer_idx is None or layer_idx >= len(lt):
                continue
            is_kda = lt[layer_idx] == "linear_attention"
            if (group == "kda") != is_kda:
                continue
        return group
    return None


class Glm53DenseFp8Method(UnquantizedLinearMethod):
    """BF16 weight at load time; per-output-channel FP8 e4m3 + Marlin at apply."""

    def __init__(self, group: str, prefix: str = "") -> None:
        super().__init__()
        self.group = group
        self.prefix = prefix
        self.ready = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )

        w = layer.weight.data
        if w.dtype != torch.bfloat16 and w.dtype != torch.float16:
            raise RuntimeError(f"[glm53-dense-fp8] expected a BF16/FP16 weight, got {w.dtype} for {self.group}")
        n, k = w.shape
        assert n == layer.output_size_per_partition and k == layer.input_size_per_partition, (w.shape, layer)
        wf = w.float()
        scales = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0  # [N] per output channel
        fp8 = (wf / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        del wf
        layer.orig_dtype = w.dtype
        layer.weight = torch.nn.Parameter(fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales.to(layer.orig_dtype), requires_grad=False)
        layer.weight_block_size = None
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)
        layer.glm53_fp8_n, layer.glm53_fp8_k = n, k
        self.ready = True
        print(
            f"[glm53-dense-fp8] quantized group={self.group} prefix={self.prefix!r} "
            f"shape=({n},{k}) scale_range=[{scales.min().item():.6g}, {scales.max().item():.6g}]",
            flush=True,
        )

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not self.ready:
            return super().apply(layer, x, bias)
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.glm53_fp8_n,
            size_k=layer.glm53_fp8_k,
            bias=bias,
        )


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed MCG trellis experts: create/load packed tensors, LinearEXL3 apply."""

    def __init__(self, moe, quant_config: Exl3Config) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.bits = quant_config.bits
        self._logged = False

    def get_fused_moe_quant_config(self, layer: "RoutedExperts") -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}"
            )
        k_words = self.bits * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_size_per_partition // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        # w13_* : stacked [{gate=0, up=1}, expert, ...] -- PROJECTION-MAJOR,
        # not expert-major. v21-b12xmoe change: b12x's
        # prepare_trellis256_moe_weights(w13_layout="trellis_t256_proj")
        # requires exactly this [2, E, ...] backing (see
        # b12x/moe/_shared/kernels/w4a16/prepare.py's docstring) so the
        # existing per-expert incremental weight_loader (_load_exl3) writes
        # b12x's desired contiguous layout directly -- zero extra copy at
        # fused-MoE prepare time. The stock expert_params_mapping still
        # hits these names by string ("experts.w13_" + suffix); it never
        # reads the parameter's shape, only its registered name (verified
        # against vllm/model_executor/layers/fused_moe/layer.py: num_experts
        # flows top-down as a constructor arg, never inferred from a
        # parameter's shape).
        w13_trellis = Parameter(
            torch.empty(
                2, num_experts, in_tiles, out_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w13_suh = Parameter(
            torch.empty(2, num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w13_svh = Parameter(
            torch.empty(
                2, num_experts, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w13_mcg = Parameter(
            torch.empty(2, num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w2_suh = Parameter(
            torch.empty(
                num_experts, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w2_svh = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w2_mcg = Parameter(
            torch.empty(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )

        packed = {
            "w13_trellis": w13_trellis,
            "w13_suh": w13_suh,
            "w13_svh": w13_svh,
            "w13_mcg": w13_mcg,
            "w2_trellis": w2_trellis,
            "w2_suh": w2_suh,
            "w2_svh": w2_svh,
            "w2_mcg": w2_mcg,
        }
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("EXL3 create_weights must not allocate dense expert weights")

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        layer = param
        # param is the Parameter; expert_id is already physical. Map to local
        # via the owning module if present on the weight_loader closure... we
        # look up from param's __dict__ after register. RoutedExperts.weight_loader
        # maps global→local; glm5next calls *our* loader, so map here.
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        suffix = _suffix_from_mapped_name(weight_name)
        loaded = loaded_weight.detach().contiguous()

        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size)
            # v21-b12xmoe: w13_* is [2, E, ...] (projection-major), not
            # [E, 2, ...] -- see create_weights.
            dest = param.data[shard_idx, expert_id]
        elif shard_id == "w2":
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(
                f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                f"expert={expert_id}: dest {tuple(dest.shape)} != "
                f"loaded {tuple(sharded.shape)}"
            )
        dest.copy_(sharded)
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "w13_trellis"):
            return
        _initialize_tiny_dummy_exl3(layer)
        # Bind owner for any late loads; stitch LinearEXL3 handles.
        for name in (
            "w13_trellis",
            "w13_suh",
            "w13_svh",
            "w13_mcg",
            "w2_trellis",
            "w2_suh",
            "w2_svh",
            "w2_mcg",
        ):
            getattr(layer, name)._exl3_owner = layer

        mcg13 = layer.w13_mcg.reshape(-1)
        mcg2 = layer.w2_mcg.reshape(-1)
        if not torch.all(mcg13 == MCG_MARKER_SIGNED_INT32) or not torch.all(
            mcg2 == MCG_MARKER_SIGNED_INT32
        ):
            raise RuntimeError(
                "EXL3 mcg marker is not the MCG int32 0xCBAC1FED / "
                f"{MCG_MARKER_SIGNED_INT32}; packed ABI mismatch"
            )

        # v21-b12xmoe: w13_* is [2, E, ...] (projection-major) -- experts
        # read off shape[1], and gate/up selection is [0]/[1] on axis 0.
        n_exp = int(layer.w13_trellis.shape[1])
        layer._exl3_shared_w13_suh = bool(
            torch.equal(layer.w13_suh[0], layer.w13_suh[1])
        )
        inners: list[dict[str, Any]] = []
        for e in range(n_exp):
            gate = make_linear_exl3(
                layer.w13_trellis[0, e],
                layer.w13_suh[0, e],
                layer.w13_svh[0, e],
                layer.w13_mcg[0, e],
            )
            up = make_linear_exl3(
                layer.w13_trellis[1, e],
                layer.w13_suh[1, e],
                layer.w13_svh[1, e],
                layer.w13_mcg[1, e],
            )
            down = make_linear_exl3(
                layer.w2_trellis[e],
                layer.w2_suh[e],
                layer.w2_svh[e],
                layer.w2_mcg[e],
            )
            inners.append({"gate": gate, "up": up, "down": down})
        layer._exl3_inners = inners
        # Tier resolution needs the LinearEXL3 handles (K/mcg/mul1) for the
        # grouped (E3) eligibility check, so it runs once the inners exist
        # (v11-e3: moved from before this loop, where inners was still empty).
        _record_exl3_fat_resolution(layer)
        fused_ok = False
        fused_err = None
        if fused_moe_enabled():
            try:
                import exllamav3_ext

                if hasattr(exllamav3_ext, "exl3_moe"):
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                else:
                    fused_err = "exllamav3_ext.exl3_moe missing"
            except Exception as exc:
                fused_err = repr(exc)
                layer._exl3_ptrs = None
        if exl3_moe_fast_requested() and not fused_ok:
            # Fail closed: an explicitly requested fast thin-decode path must
            # never silently run the stock kernel or the Python loop. The
            # version gate inside build_exl3_fused_state raises through the
            # same path; this also covers fused disabled / exl3_moe missing.
            raise RuntimeError(
                "GLM53_EXL3_MOE_FAST=1 requires the fused exl3_moe path on an "
                "image built with overlay/patch_exl3_decode_pipeline.py; "
                f"load-time setup failed: {fused_err or 'EXL3_FUSED_MOE=0'}"
            )
        if exl3_b12x_moe_requested():
            # Fail closed: build_b12x_prepared_state itself raises on a
            # missing/mismatched b12x install or a broken zero-copy
            # invariant -- never caught here, so a requested-but-broken b12x
            # path surfaces at load time, not as a confusing failure mid-decode.
            build_b12x_prepared_state(layer)
        if not self._logged:
            if fused_ok:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=exl3_moe concurrency=%s "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    getattr(layer, "_exl3_fused_concurrency", "?"),
                )
            else:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=python_loop (%s) "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    fused_err or "EXL3_FUSED_MOE=0",
                )
            self._logged = True

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        limit = getattr(self.moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT
        return apply_exl3_experts(
            x, topk_ids, topk_weights, layer, limit=float(limit)
        )
