# Copyright (c) 2026 BAAI. All rights reserved.
"""C500 BF16 tiling overrides for the existing FlagGems MetaX NT kernel."""

import functools
import importlib
import logging

import torch
import triton

logger = logging.getLogger(__name__)

# (M, N, K) -> (BLOCK_M, BLOCK_N, BLOCK_K). Other shapes retain autotuning.
_C500_NT_TILES = {
    (2048, 2560, 2048): (128, 128, 64),
    (2048, 2048, 2048): (128, 128, 64),
    (2048, 12288, 2048): (128, 128, 64),
    (4096, 12288, 2048): (128, 128, 64),
    (4096, 2560, 2048): (128, 128, 64),
    (4096, 2048, 2048): (128, 128, 128),
}


def _make_nt_dispatch(original, entries, device):
    seen = set()
    @functools.wraps(original)
    def general_mm_nt(a, b, c, M, N, K):
        entry = entries.get((M, N, K))
        if (
            entry is None
            or a.device != device
            or b.device != device
            or c.device != device
            or not (a.dtype == b.dtype == c.dtype == torch.bfloat16)
            or a.stride() != (K, 1)
            or b.stride() != (1, K)
            or c.stride() != (N, 1)
        ):
            return original(a, b, c, M, N, K)

        if (M, N, K) not in seen:
            logger.info("C500_NT_AB_HIT shape=%s tile=%s", (M, N, K), _C500_NT_TILES[(M, N, K)])
            seen.add((M, N, K))
        # Each shape has a separate LibEntry: launch options such as warps and
        # stages must not collide in the original entry's dispatch cache.
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        with torch.cuda.device(device):
            entry[grid](a, b, c, M, N, K, GROUP_M=8)
        return c

    general_mm_nt._fl_c500_nt_devices = getattr(
        original, "_fl_c500_nt_devices", frozenset()
    ) | {device}
    return general_mm_nt


def apply_c500_nt_gemm(device):
    """Install once per worker/device, before model compilation and capture.

    Only the NT helper is wrapped; mm/mm_out retain their original GEMV,
    Split-K and layout selection. Fixed single-candidate tuners reuse the
    installed kernel and its heuristics without consulting old best configs.
    """
    device = torch.device(device)
    if torch.cuda.get_device_name(device) != "MetaX C500":
        return False

    from flag_gems.utils.libentry import LibEntry, LibTuner, libtuner
    import flag_gems

    # FlagGems loads vendor ops as _metax.ops.mm. Importing the full package
    # path can create a second module that is not used by registered aten ops.
    module_name = flag_gems.mm.__module__
    if module_name not in ("_metax.ops.mm", "flag_gems.runtime.backend._metax.ops.mm"):
        logger.warning("C500 NT tiling: active mm is not MetaX; using defaults")
        return False
    mm = importlib.import_module(module_name)
    original = mm.general_mm_nt
    if device in getattr(original, "_fl_c500_nt_devices", ()):
        return True
    kernel = mm.mm_kernel_nt
    if not isinstance(kernel, LibEntry) or not isinstance(kernel.fn, LibTuner):
        logger.warning("C500 NT tiling: unsupported FlagGems wrapper; using defaults")
        return False

    entries = {}
    for shape, (bm, bn, bk) in _C500_NT_TILES.items():
        config = triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
             "pipeline": "cpasync", "scenario": ""},
            num_warps=4, num_stages=4, num_ctas=1,
        )
        # Keep all original heuristics/JIT code. A single candidate bypasses
        # best-config lookup and benchmark selection in LibTuner.run.
        entries[shape] = LibEntry(
            libtuner(configs=[config], key=["M", "N", "K"])(kernel.fn.fn)
        )

    mm.general_mm_nt = _make_nt_dispatch(original, entries, device)
    logger.info("Enabled %d C500 BF16 NT tiling overrides on %s", len(entries), device)
    return True
