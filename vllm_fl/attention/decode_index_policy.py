# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guarded single-query index/grid specialization; enabled by default on BI-V150."""

import hashlib
import inspect
import linecache
import os
from functools import wraps

from vllm.logger import init_logger

logger = init_logger(__name__)
_installed = False
_ORIGINAL_LOOKUP = """    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )"""
_DIRECT_LOOKUP = """    # Qualified dense single-query prefix, followed only by graph padding.
    seq_idx = q_block_global_idx
    q_block_local_idx = 0
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_batch_in_all_start_index
    seq_len = tl.load(seq_lens_ptr + seq_idx)"""


def dense_single_query_prefix(starts):
    """CPU list only: one or more 1s, optionally followed by padding 0s."""
    if len(starts) < 2 or starts[0] != 0:
        return False
    padding = False
    valid = False
    for left, right in zip(starts, starts[1:]):
        delta = right - left
        if delta == 0:
            padding = True
        elif delta == 1 and not padding:
            valid = True
        else:
            return False
    return valid


def qualify_metadata(metadata, common):
    """Reuse CPU metadata once per step; never read a GPU scalar per layer."""
    if metadata.causal is not True or metadata.use_cascade:
        return None
    if metadata.mm_prefix_range_tensor is not None:
        return None
    mixed = getattr(metadata, "_fl_mixed_plan", None)
    if mixed is not None:
        # #011 has already verified the prefix contains only one-token rows.
        return mixed.decode_starts
    if metadata.max_query_len != 1:
        return None
    cpu = getattr(common, "query_start_loc_cpu", None)
    if cpu is None or cpu.device.type != "cpu":
        return None
    starts = cpu[: common.num_reqs + 1].tolist()
    return metadata.query_start_loc if dense_single_query_prefix(starts) else None


def build_specialized_head(upstream):
    """Make one narrowly transformed JIT function; no installed file writes."""
    original = inspect.getsource(upstream.kernel_unified_attention.fn)
    if original.count(_ORIGINAL_LOOKUP) != 1:
        raise ValueError("unsupported unified Attention lookup source")
    source = original.replace(_ORIGINAL_LOOKUP, _DIRECT_LOOKUP, 1).replace(
        "def kernel_unified_attention(", "def fl_decode_index_attention(", 1
    )
    filename = (
        "<vllm_fl.decode_index." + hashlib.sha256(source.encode()).hexdigest() + ">"
    )
    # Triton uses inspect.getsource. An in-memory source entry keeps its hash
    # reproducible without writing/replacing vLLM or a generated .py file.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(upstream))
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace["fl_decode_index_attention"]


def active_metadata():
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    return context.attn_metadata


class DecodeIndexKernel:
    """Keep the original kernel for every unsupported launch/metadata shape."""

    def __init__(self, original, specialized, metadata_provider=active_metadata):
        self.original = original
        self.specialized = specialized
        self.metadata_provider = metadata_provider
        self.force_original = False  # Diagnostic A/B only, never an env hot switch.

    def eligible(self, kwargs):
        if self.force_original:
            return False
        import torch

        expected = {
            "IS_3D": True,
            "num_query_heads": 16,
            "num_queries_per_kv": 8,
            "HEAD_SIZE": 128,
            "HEAD_SIZE_PADDED": 128,
            "BLOCK_SIZE": 16,
            "BLOCK_M": 16,
            "BLOCK_Q": 2,
            "TILE_SIZE": 16,
            "NUM_SEGMENTS_PER_SEQ": 16,
            "num_warps": 2,
            "num_stages": 1,
            "KV_QUANT_MODE": 0,
            "USE_CAUSAL": True,
        }
        if any(kwargs.get(key) != value for key, value in expected.items()):
            return False
        if any(
            kwargs.get(key, False)
            for key in (
                "USE_ALIBI_SLOPES",
                "USE_ALIBI_SQRT",
                "USE_QQ_BIAS",
                "USE_SOFTCAP",
                "USE_SINKS",
                "USE_PER_SEQ_CAUSAL",
                "USE_MM_PREFIX",
                "USE_TD",
                "USE_FP8",
            )
        ):
            return False
        if kwargs.get("SLIDING_WINDOW", 0) > 0 or kwargs.get("CHUNK_LOOKBACK", -1) >= 0:
            return False
        q = kwargs["query_ptr"]
        if not (q.device.type == "cuda" and q.dtype == torch.bfloat16):
            return False
        if not 1 <= kwargs["num_seqs"] <= 64 or q.shape[0] != kwargs["num_seqs"]:
            return False
        if any(
            kwargs[name].dtype != torch.bfloat16
            for name in ("key_cache_ptr", "value_cache_ptr", "output_ptr")
        ):
            return False
        try:
            metadata = self.metadata_provider()
        except (AssertionError, RuntimeError):
            return False
        if not isinstance(metadata, dict):
            return False
        target = kwargs["query_start_len_ptr"]
        return any(
            getattr(meta, "_fl_decode_index_starts", None) is target
            for meta in metadata.values()
        )

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if not args and self.eligible(kwargs):
                compact = (kwargs["num_seqs"], grid[1], grid[2])
                return self.specialized[compact](**kwargs)
            return self.original[grid](*args, **kwargs)

        return launch


def install_decode_index_policy(backend):
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_BIV150_DECODE_DIRECT_INDEX", "1").lower() not in (
        "1",
        "true",
    ):
        return False
    from vllm_fl.attention.mixed_policy import _is_validated_device

    if not _is_validated_device():
        return False
    try:
        from vllm.v1.attention.ops import triton_unified_attention as upstream

        launcher = backend.unified_attention
        while "_fl_prefill_tiles" not in launcher.__globals__:
            launcher = launcher.__wrapped__
        specialized = build_specialized_head(upstream)
        original = launcher.__globals__["kernel_unified_attention"]
        if isinstance(original, DecodeIndexKernel):
            _installed = True
            return True
        builder = backend.TritonAttentionMetadataBuilder
        original_build = builder.build
    except Exception as exc:  # noqa: BLE001
        logger.warning("Decode direct-index policy unchanged: %s", exc)
        return False

    @wraps(original_build)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        try:
            target = qualify_metadata(metadata, common_attn_metadata)
        except (AttributeError, TypeError, IndexError):
            target = None
        metadata._fl_decode_index_starts = target
        return metadata

    builder.build = build
    launcher.__globals__["kernel_unified_attention"] = DecodeIndexKernel(
        original, specialized
    )
    _installed = True
    logger.info("BI-V150 Decode direct-index policy installed")
    return True
