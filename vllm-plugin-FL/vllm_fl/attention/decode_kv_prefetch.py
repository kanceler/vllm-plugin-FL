# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page-aligned BF16 K/V look-ahead for the guarded BI-V150 #013 head."""

import hashlib
import inspect
import linecache
import logging
import os

_LOOP = "    for j in range(loop_lo, loop_hi):\n"
_LOAD_BEGIN = "        physical_block_idx = tl.load(\n"
_LOAD_END = "        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)\n"
_EPILOGUE = "    # ---- Epilogue "
_HEAD = "def fl_decode_index_attention("
_PREFETCH_HEAD = "fl_decode_kv_prefetch_attention"
_LOAD_HASH = "052173c8cc06c9243e90e264a641d0c073357e38f358f8a723ac09d321fe91fe"


def _loads(page, mask, prefix=""):
    return (
        f"{prefix}K_load = tl.load(key_cache_ptr + {page} * stride_k_cache_0 "
        f"+ kv_head_idx * stride_k_cache_2 + offs_d[:, None] * stride_k_cache_3 + "
        f"offs_t[None, :] * stride_k_cache_1, mask=dim_mask[:, None] & {mask}[None, :], other=0.0)",
        f"{prefix}V_load = tl.load(value_cache_ptr + {page} * stride_v_cache_0 "
        f"+ kv_head_idx * stride_v_cache_2 + offs_d[None, :] * stride_v_cache_3 + "
        f"offs_t[:, None] * stride_v_cache_1, mask=dim_mask[None, :] & {mask}[:, None], other=0.0)",
    )


def _block(lines, indent):
    return "".join(" " * indent + line + "\n" for line in lines)


def specialize_source(source):
    """Replace only the page-aligned loads; preserve the math and epilogue."""
    if any(
        source.count(marker) != 1
        for marker in (_HEAD, _LOOP, _LOAD_BEGIN, _LOAD_END, _EPILOGUE)
    ):
        raise ValueError("unsupported BI-V150 standalone Decode loop source")
    loop, begin, end, epilogue = (
        source.index(marker) for marker in (_LOOP, _LOAD_BEGIN, _LOAD_END, _EPILOGUE)
    )
    if not loop < begin < end < epilogue:
        raise ValueError("unsupported BI-V150 Decode loop order")
    if hashlib.sha256(source[begin:end].encode()).hexdigest() != _LOAD_HASH:
        raise ValueError("unsupported BI-V150 Decode KV load body")

    first = (
        "page_id = tl.load(block_tables_ptr + block_table_offset + loop_lo, "
        "mask=loop_lo < loop_hi, other=0).to(tl.int64)",
        "first_mask = (loop_lo < loop_hi) & (loop_lo * TILE_SIZE + offs_t < max_seq_prefix_len)",
        *_loads("page_id", "first_mask"),
    )
    next_page = (
        "next_j = j + 1",
        "next_page_id = tl.load(block_tables_ptr + block_table_offset + next_j, "
        "mask=next_j < loop_hi, other=0).to(tl.int64)",
        "next_mask = (next_j < loop_hi) & (next_j * TILE_SIZE + offs_t < max_seq_prefix_len)",
        *_loads("next_page_id", "next_mask", "next_"),
    )
    carry = ("page_id = next_page_id", "K_load = next_K_load", "V_load = next_V_load")
    return (
        source[:loop]
        + _block(first, 4)
        + source[loop:begin]
        + _block(next_page, 8)
        + source[end:epilogue]
        + _block(carry, 8)
        + "\n"
        + source[epilogue:]
    ).replace(_HEAD, f"def {_PREFETCH_HEAD}(", 1)


class _PrefetchHead:
    def __init__(self, original, prefetch):
        self.original = original
        self.prefetch = prefetch

    def __getitem__(self, grid):
        head = self.prefetch if grid[0] >= 16 else self.original
        return head[grid]


def _build_prefetch_head(head):
    function = head.fn
    source = specialize_source(inspect.getsource(function))
    filename = (
        "<vllm_fl.decode_kv_prefetch."
        + hashlib.sha256(source.encode()).hexdigest()
        + ">"
    )
    # The Triton decorator reads the generated source via inspect.getsource.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(function.__globals__)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace[_PREFETCH_HEAD]


def install_decode_kv_prefetch(cache_dtype):
    """Attach an opt-in head behind the existing BF16 Decode dispatch guard."""
    if os.environ.get("VLLM_FL_BIV150_DECODE_KV_PREFETCH", "0").lower() not in (
        "1",
        "true",
    ):
        return False
    if cache_dtype not in ("auto", "bfloat16") or any(
        os.environ.get(flag, "0").lower() in ("1", "true")
        for flag in (
            "VLLM_FL_K_STATIC_FOLD",
            "VLLM_FL_INT8_KV_REPAIR",
            "VLLM_FL_K_TOP4",
            "VLLM_FL_K_TOP4_JOINT",
            "VLLM_FL_V_ONLY",
        )
    ):
        return False

    from vllm.v1.attention.backends import triton_attn

    from vllm_fl.attention.decode_index_policy import DecodeIndexKernel

    launcher = getattr(triton_attn, "_fl_tuned_launcher", None)
    if launcher is None:
        return False
    dispatch = launcher.__globals__.get("kernel_unified_attention")
    if not isinstance(dispatch, DecodeIndexKernel):
        return False
    original = dispatch.specialized
    if isinstance(original, _PrefetchHead):
        return True

    logger = logging.getLogger(__name__)
    try:
        prefetch = _build_prefetch_head(original)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BI-V150 Decode KV prefetch not installed: %s", exc)
        return False
    dispatch.specialized = _PrefetchHead(original, prefetch)
    logger.info("BI-V150 Decode KV prefetch installed (batch >= 16)")
    return True
