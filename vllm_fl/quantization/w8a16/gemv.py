"""Experimental weight-only GEMV; no persistent BF16 weights or K workspace."""
import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(A, Q, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                RN: tl.constexpr, RM: tl.constexpr, RK: tl.constexpr):
    ni = tl.program_id(0) * RN + tl.arange(0, RN)
    mi = tl.program_id(1) * RM
    ki = tl.arange(0, RK)
    # Keep Q in registers and reuse it across a small group of activation rows.
    q = tl.load(Q + ni[:, None] * K + ki[None, :],
                (ni[:, None] < N) & (ki[None, :] < K), 0).to(tl.float32)
    scale = tl.load(S + ni, ni < N, 0)
    for r in tl.static_range(RM):
        a = tl.load(A + (mi+r)*K + ki, (mi+r < M) & (ki < K), 0).to(tl.float32)
        value = tl.sum(q * a[None, :], 1) * scale
        tl.store(Y + (mi+r)*N + ni, value, (mi+r < M) & (ni < N))


def gemv(a, q, scale, config):
    m, k = a.shape
    n = q.shape[0]
    rn, rm, warps = config
    y = torch.empty((m, n), device=a.device, dtype=a.dtype)
    gemv_kernel[(triton.cdiv(n, rn), triton.cdiv(m, rm))](
        a, q, scale, y, m, n, k, rn, rm, triton.next_power_of_2(k), num_warps=warps)
    return y
