# SPDX-License-Identifier: Apache-2.0
"""Separate INT8/FP32 -> BF16 dequantization; no persistent BF16 cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_kernel(Q, S, W, K: tl.constexpr, SIZE: tl.constexpr,
                    BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < SIZE
    q = tl.load(Q + idx, mask, other=0).to(tl.float32)
    scale = tl.load(S + idx // K, mask, other=0)
    tl.store(W + idx, (q * scale).to(tl.bfloat16), mask)


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.dtype != torch.int8 or not q.is_contiguous():
        raise ValueError("expected contiguous INT8 [N,K]")
    if (scale.shape != (q.shape[0],) or scale.dtype != torch.float32
            or not scale.is_contiguous() or scale.device != q.device):
        raise ValueError("expected contiguous FP32 [N] scale on weight device")
    if q.device.type != "cuda" or min(q.shape) == 0:
        raise ValueError("dequantization requires nonempty CUDA-alike weight")
    with torch.cuda.device(q.device):
        output = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
        _dequant_kernel[(triton.cdiv(q.numel(), 1024),)](
            q, scale, output, q.shape[1], q.numel(), BLOCK=1024)
    return output
