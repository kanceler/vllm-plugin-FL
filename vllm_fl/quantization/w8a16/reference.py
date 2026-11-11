# SPDX-License-Identifier: Apache-2.0
"""Explicit numerical contract for per-local-output-channel W8A16."""

import torch


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("expected BF16 weight [N,K]")
    if min(weight.shape) == 0:
        raise ValueError("weight dimensions must be nonempty")
    w = weight.float()
    if not torch.isfinite(w).all().item():
        raise ValueError("nonfinite weight cannot be quantized")
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127)
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def dequantize_reference(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.dtype != torch.int8:
        raise ValueError("expected INT8 weight [N,K]")
    if scale.shape != (q.shape[0],) or scale.dtype != torch.float32:
        raise ValueError("expected FP32 scale [N]")
    if scale.device != q.device:
        raise ValueError("weight and scale must share device")
    return (q.float() * scale[:, None]).to(torch.bfloat16)
