# SPDX-License-Identifier: Apache-2.0
"""Opaque model wrapper for FlagGems scale-after-dot weight_int8pack_mm."""

from typing import Optional

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def supports_int8pack(x, q, scale, bias=None):
    # weight_int8pack_mm has no bias epilogue. Keep biased layers on the
    # existing correctness-preserving fallback rather than adding another
    # BF16 rounding point after the GEMM output.
    return (
        x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and x.ndim >= 2
        and x.is_contiguous()
        and x.numel() > 0
        and q.device == x.device
        and q.dtype == torch.int8
        and q.ndim == 2
        and q.is_contiguous()
        and min(q.shape) > 0
        and x.shape[-1] == q.shape[1]
        and scale.device == q.device
        and scale.dtype == torch.float32
        and scale.shape == (q.shape[0],)
        and scale.is_contiguous()
        and bias is None
    )


def _int8pack_linear_impl(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not supports_int8pack(x, q, scale, bias):
        raise ValueError("unsupported scale-after-dot W8A16 input/layout")
    import flag_gems

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    output = flag_gems.weight_int8pack_mm(x_2d, q, scale)
    return output.reshape(*original_shape[:-1], q.shape[0])


def _int8pack_linear_fake(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], q.shape[0]))


direct_register_custom_op(
    op_name="fl_w8a16_down_int8pack",
    op_func=_int8pack_linear_impl,
    fake_impl=_int8pack_linear_fake,
)


def int8pack_linear(x, q, scale, bias=None):
    """Keep the runtime-M dispatch opaque to the compiled model graph."""
    return torch.ops.vllm.fl_w8a16_down_int8pack(x, q, scale, bias)
