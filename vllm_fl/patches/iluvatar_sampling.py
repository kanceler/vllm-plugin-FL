# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default-enabled BI-V150 row parallelism for the vLLM top-k/top-p kernel.

Keep the upstream selection algorithm, floating point operations and RNG path.
Only replace its persistent one-program-per-SM grid with one program per row.
"""

import logging
import os
from functools import cache, wraps

logger = logging.getLogger(__name__)
_buffers = {}


@cache
def _validated_device(index):
    import torch

    from vllm.platforms import current_platform

    try:
        return (
            getattr(current_platform, "vendor_name", None) == "iluvatar"
            and "bi-v150" in torch.cuda.get_device_name(index).lower().split()
        )
    except Exception:
        return False


def apply_row_parallel(logits, k, p, mask_value=float("-inf")):
    """Validated shape only; caller handles fallback outside this contract."""
    import torch

    from vllm.v1.sample.ops import topk_topp_triton as upstream

    batch, vocab = logits.shape
    stream = torch.cuda.current_stream(logits.device).cuda_stream
    key = (logits.device, stream)
    buffer = _buffers.get(key)
    if buffer is None:
        # Bounded by the supported B<=64, V=130560 contract. Separate streams
        # must not overwrite another launch's pivot-selection scratch.
        buffer = logits.new_empty((64, vocab))
        _buffers[key] = buffer
    tables = upstream._TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        tables = (
            logits.new_tensor(upstream._NORMAL_CDF_TO_SIGMA_TABLE),
            logits.new_tensor(upstream._PERCENTILE_TO_STD_TABLE),
        )
        upstream._TRITON_TABLE_CACHE[logits.device] = tables
    normal_cdf, percentile_std = tables
    upstream._topk_topp_kernel[(batch,)](
        logits,
        logits.stride(0),
        buffer,
        percentile_std,
        normal_cdf,
        k.to(torch.int32) if k is not None else logits,
        p.to(torch.float32) if p is not None else logits,
        BATCH_SIZE=batch,
        VOCAB_SIZE=vocab,
        MASK_VALUE=mask_value,
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=k is not None,
        TOPP_ENABLED=p is not None,
    )
    return logits


def install_row_parallel_sampling():
    """Install once at backend import; an explicit 0 disables this policy."""
    if os.environ.get("VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING", "1").lower() not in (
        "1",
        "true",
    ):
        return False
    import torch

    from vllm.v1.sample.ops import (
        topk_topp_sampler as sampler,
        topk_topp_triton as upstream,
    )

    original = sampler.apply_top_k_top_p_triton
    if getattr(original, "_fl_biv150_row_parallel", False):
        return True
    expected = [
        "LOGITS",
        "LOGITS_STRIDE_0",
        "BUFFER",
        "PERCENTILE_TO_STD_TABLE",
        "NORMAL_CDF_TO_SIGMA_TABLE",
        "K",
        "P",
        "BATCH_SIZE",
        "VOCAB_SIZE",
        "MASK_VALUE",
        "BLOCK_SIZE",
        "BLOCK_SIZE_TRUNC",
        "TOPK_ENABLED",
        "TOPP_ENABLED",
    ]
    if list(upstream._topk_topp_kernel.arg_names) != expected:
        logger.warning("BI-V150 sampling: unsupported upstream kernel ABI; unchanged")
        return False

    @wraps(original)
    def dispatch(logits, k, p, mask_value=float("-inf")):
        if (
            logits.ndim == 2
            and logits.device.type == "cuda"
            and logits.dtype == torch.float32
            and logits.stride(1) == 1
            and 17 <= logits.shape[0] <= 64
            and logits.shape[1] == 130560
            and (k is not None or p is not None)
            and all(
                value is None
                or (
                    value.ndim == 1
                    and value.shape[0] == logits.shape[0]
                    and value.device == logits.device
                    and value.is_contiguous()
                )
                for value in (k, p)
            )
            and _validated_device(logits.device.index)
        ):
            return apply_row_parallel(logits, k, p, mask_value)
        return original(logits, k, p, mask_value)

    dispatch._fl_biv150_row_parallel = True
    sampler.apply_top_k_top_p_triton = dispatch
    logger.info("BI-V150 row-parallel sampling installed")
    return True
