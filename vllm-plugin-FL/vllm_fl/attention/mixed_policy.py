# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep one-token rows on the decode launch path inside mixed batches.

The upstream unified launcher selects one configuration for the whole batch.
A prefill row consequently makes all decode rows use its large, unsplit 2D
kernel. Split a leading run of decode rows into a separate launch of the same
kernel. Build rebased prefix sums once per step, using the existing CPU copy
to find the split; no layer performs a device-to-host read.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os

from vllm.logger import init_logger

logger = init_logger(__name__)
_active_plan = ContextVar("fl_mixed_attention_plan", default=None)
_installed = False
_reported = False


def _is_validated_device() -> bool:
    """Keep this measured policy on Iluvatar BI-V150; fail closed elsewhere."""
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "vendor_name", None) != "iluvatar":
            return False

        import torch

        # FlagOSPlatform.get_device_name() returns a logical device type, so
        # query the actual current GPU through torch instead.
        return "bi-v150" in torch.cuda.get_device_name().lower().split()
    except Exception as exc:
        logger.debug("FL mixed attention split: cannot identify BI-V150: %s", exc)
        return False


@dataclass(frozen=True)
class MixedPlan:
    num_decodes: int
    decode_starts: object
    prefill_starts: object
    source_starts: object


def make_plan(metadata, common):
    """Return a conservative split, or None for unsupported/unprofitable cases."""
    if metadata.max_query_len <= 1 or metadata.use_cascade:
        return None
    if metadata.causal is not True or metadata.mm_prefix_range_tensor is not None:
        return None
    cpu_starts = getattr(common, "query_start_loc_cpu", None)
    if cpu_starts is None or cpu_starts.device.type != "cpu":
        return None
    starts = cpu_starts[: common.num_reqs + 1].tolist()
    if not starts or starts[0] != 0:
        return None
    decodes = 0
    for left, right in zip(starts, starts[1:]):
        if right - left != 1:
            break
        decodes += 1
    # A few rows do not amortize two extra kernel launches. Also require the
    # upstream scratch capacity and 3D threshold to cover the entire prefix.
    # On the initial paged/uneven GPU matrix, splitting 4/8 decode rows
    # regressed due to extra launches; 16+ rows amortized that overhead.
    if not (16 <= decodes < common.num_reqs):
        return None
    if decodes > metadata.seq_threshold_3D:
        return None
    if decodes > metadata.softmax_segm_output.shape[0]:
        return None
    return MixedPlan(
        num_decodes=decodes,
        decode_starts=metadata.query_start_loc[: decodes + 1],
        prefill_starts=metadata.query_start_loc[decodes:] - decodes,
        source_starts=metadata.query_start_loc,
    )


def dispatch_mixed(original, plan, kwargs):
    """Launch phase-specific views; return the original result on the fallback."""
    global _reported
    if plan is None or kwargs.get("cu_seqlens_q") is not plan.source_starts:
        return original(**kwargs)
    if kwargs.get("max_seqlen_q", 1) <= 1 or kwargs.get("causal") is not True:
        return original(**kwargs)
    # Specialized quantization, bias, and descriptor layouts retain their
    # upstream path. Supporting them requires independent numerical validation.
    if (
        kwargs.get("qq_bias") is not None
        or kwargs.get("mm_prefix_range") is not None
        or kwargs.get("output_scale") is not None
        or kwargs.get("use_td", False)
        or int(kwargs.get("kv_quant_mode", 0)) != 0
    ):
        return original(**kwargs)
    d = plan.num_decodes
    decode = dict(kwargs)
    prefill = dict(kwargs)
    for name in ("q", "out"):
        decode[name] = kwargs[name][:d]
        prefill[name] = kwargs[name][d:]
    for name in ("seqused_k", "block_table"):
        decode[name] = kwargs[name][:d]
        prefill[name] = kwargs[name][d:]
    # BF16 inputs use no Q descale. A non-scalar K/V descale, if present,
    # carries a sequence axis and must follow the same split.
    if kwargs.get("q_descale") is not None:
        return original(**kwargs)
    for name in ("k_descale", "v_descale"):
        scale = kwargs.get(name)
        if scale is not None and scale.ndim > 0 and scale.numel() > 1:
            decode[name] = scale[:d]
            prefill[name] = scale[d:]
    decode["cu_seqlens_q"] = plan.decode_starts
    decode["max_seqlen_q"] = 1
    prefill["cu_seqlens_q"] = plan.prefill_starts
    if not _reported:
        print(
            "FL mixed attention split active: decode_rows=%d, total_tokens=%d; "
            "same Triton kernels with phase-specific launch geometry."
            % (d, kwargs["q"].shape[0]),
            flush=True,
        )
        _reported = True
    original(**decode)
    return original(**prefill)


def install_mixed_policy(backend) -> bool:
    """Install once after the existing tile policy; preserve backend selection."""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_MIXED_ATTN_SPLIT", "1").lower() in ("0", "false", "no"):
        return False
    if not _is_validated_device():
        return False
    try:
        builder = backend.TritonAttentionMetadataBuilder
        implementation = backend.TritonAttentionImpl
        original_build = builder.build
        original_forward = implementation.forward
        original_attention = backend.unified_attention
    except AttributeError:
        logger.warning("FL mixed attention split: unsupported upstream API; skipped.")
        return False

    @wraps(original_build)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        metadata._fl_mixed_plan = make_plan(metadata, common_attn_metadata)
        return metadata

    @wraps(original_forward)
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        plan = getattr(attn_metadata, "_fl_mixed_plan", None)
        if plan is None:
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        token = _active_plan.set(plan)
        try:
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        finally:
            _active_plan.reset(token)

    @wraps(original_attention)
    def attention(*args, **kwargs):
        if args:
            return original_attention(*args, **kwargs)
        return dispatch_mixed(original_attention, _active_plan.get(), kwargs)

    builder.build = build
    implementation.forward = forward
    backend.unified_attention = attention
    _installed = True
    print(
        "FL mixed attention split installed (existing backend and kernels retained).",
        flush=True,
    )
    return True
