# SPDX-License-Identifier: Apache-2.0
"""INT8 storage with tile-local dequantization rounded before BF16 dot."""

import torch
import triton
import triton.language as tl
from typing import Optional

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fused_kernel(X, Q, S, B, Y, M: tl.constexpr, N: tl.constexpr,
                  K: tl.constexpr, SX: tl.constexpr, HAS_BIAS: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Group adjacent M tiles to reuse weight tiles in L2.
    pid = tl.program_id(0)
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    group = pid // (8 * gn)
    size = tl.minimum(gm - group * 8, 8)
    pm = group * 8 + pid % size
    pn = (pid % (8 * gn)) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    scale = tl.load(S + rn, rn < N, other=0)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + rk
        a = tl.load(X + rm[:, None] * SX + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K), other=0)
        q = tl.load(Q + rn[None, :] * K + kk[:, None],
                    (rn[None, :] < N) & (kk[:, None] < K), other=0)
        # This cast is part of the numerical contract. Moving scale after
        # dot would remove the per-weight BF16 rounding and change results.
        b = (q.to(tl.float32) * scale[None, :]).to(tl.bfloat16)
        acc += tl.dot(a, b)
    if HAS_BIAS:
        acc += tl.load(B + rn, rn < N, other=0)[None, :].to(tl.float32)
    tl.store(Y + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             (rm[:, None] < M) & (rn[None, :] < N))


def supports_fused(x, q, scale, bias=None):
    return (x.device.type == 'cuda' and x.dtype == torch.bfloat16
            and x.ndim >= 2 and x.is_contiguous() and x.numel() > 0
            and q.device == x.device and q.dtype == torch.int8
            and q.ndim == 2 and q.is_contiguous() and min(q.shape) > 0
            and x.shape[-1] == q.shape[1]
            and scale.device == q.device and scale.dtype == torch.float32
            and scale.shape == (q.shape[0],) and scale.is_contiguous()
            and (bias is None or (bias.device == x.device
                 and bias.dtype == torch.bfloat16
                 and bias.shape == (q.shape[0],) and bias.is_contiguous())))


def _fused_linear_impl(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not supports_fused(x, q, scale, bias):
        raise ValueError('unsupported fused W8A16 input/layout')
    n, k = q.shape
    m = x.numel() // k
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    bm, bn, bk = (16, 64, 64) if m <= 32 else (32, 64, 64)
    with torch.cuda.device(x.device):
        _fused_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            x, q, scale, bias if bias is not None else output, output,
            m, n, k, k, bias is not None, bm, bn, bk,
            num_warps=4, num_stages=2, enable_fp_fusion=False)
    return output


def _fused_linear_fake(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Propagate symbolic leading dimensions without launching or specializing M.
    return x.new_empty((*x.shape[:-1], q.shape[0]))


direct_register_custom_op(
    op_name="fl_w8a16_down_fused",
    op_func=_fused_linear_impl,
    fake_impl=_fused_linear_fake,
)


def fused_linear(x, q, scale, bias=None):
    # Keep the M-dependent tile selection and constexpr launch outside Dynamo.
    # The op remains in the compiled model graph and supports CUDA graph capture.
    return torch.ops.vllm.fl_w8a16_down_fused(x, q, scale, bias)
