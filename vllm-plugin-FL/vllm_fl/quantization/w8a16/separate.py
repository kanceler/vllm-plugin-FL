# SPDX-License-Identifier: Apache-2.0
"""Complete transient conversion + Iluvatar BF16 MM + scale-after-dot call."""
import importlib
from typing import Optional

import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op


def _validate(x, q, scale, bias):
    if (x.dtype != torch.bfloat16 or x.ndim < 2 or x.device.type != "cuda"
            or q.dtype != torch.int8 or q.ndim != 2 or not q.is_contiguous()
            or min(q.shape) == 0 or x.shape[-1] != q.shape[1]
            or q.device != x.device or scale.device != q.device
            or scale.dtype != torch.float32 or scale.shape != (q.shape[0],)
            or not scale.is_contiguous()):
        raise ValueError("Separate requires BF16 activation, contiguous INT8 [N,K], FP32 [N] scale")
    if bias is not None and (bias.shape != (q.shape[0],) or bias.device != x.device
                             or bias.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("expected BF16/FP32 [N] bias on activation device")


def _separate_impl(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                   bias: Optional[torch.Tensor] = None, projection_id: int = 0) -> torch.Tensor:
    _validate(x, q, scale, bias)
    if x.numel() == 0:
        return x.new_empty((*x.shape[:-1], q.shape[0]))
    from .policy import dispatch
    with torch.cuda.device(x.device):
        activation = x.reshape(-1, x.shape[-1]).contiguous()
        if activation.data_ptr() % 16:
            activation = activation.clone()
        result = dispatch(activation, q, scale, projection_id)
        if bias is not None:
            result = (result.float() + bias.float()).to(torch.bfloat16)
        return result.reshape(*x.shape[:-1], q.shape[0])


def _separate_fake(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                   bias: Optional[torch.Tensor] = None, projection_id: int = 0) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], q.shape[0]))


direct_register_custom_op(op_name="fl_tianshu_w8a16_separate",
                          op_func=_separate_impl, fake_impl=_separate_fake)


def separate_linear(x, q, scale, bias=None, projection_id=0):
    return torch.ops.vllm.fl_tianshu_w8a16_separate(x, q, scale, bias, projection_id)
