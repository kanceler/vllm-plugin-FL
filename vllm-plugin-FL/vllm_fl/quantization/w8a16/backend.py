"""Iluvatar Down W8A16 complete-call interfaces; initial configs are not tuned."""
import importlib
import torch
import triton
import triton.language as tl


@triton.jit
def convert(Q, W, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(W + i, tl.load(Q + i, i < SIZE, 0).to(tl.bfloat16), i < SIZE)


@triton.jit
def convert_kn(Q, W, N: tl.constexpr, K: tl.constexpr,
               TN: tl.constexpr, TK: tl.constexpr):
    ni = tl.program_id(0) * TN + tl.arange(0, TN)
    ki = tl.program_id(1) * TK + tl.arange(0, TK)
    value = tl.load(Q + ni[:, None] * K + ki[None, :],
                    (ni[:, None] < N) & (ki[None, :] < K), 0)
    tl.store(W + ki[None, :] * N + ni[:, None], value.to(tl.bfloat16),
             (ni[:, None] < N) & (ki[None, :] < K))


@triton.jit
def partial(A, Q, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr):
    # Contiguous K partitions and FP32 workspace avoid atomic accumulation.
    pid, z = tl.program_id(0), tl.program_id(1)
    mi = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    ni = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    steps = tl.cdiv(K, BK * SK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(steps):
        ki = (z * steps + step) * BK + tl.arange(0, BK)
        a = tl.load(A + mi[:, None] * K + ki[None, :],
                    (mi[:, None] < M) & (ki[None, :] < K), 0)
        w = tl.load(Q + ni[None, :] * K + ki[:, None],
                    (ni[None, :] < N) & (ki[:, None] < K), 0).to(tl.bfloat16)
        acc += tl.dot(a, w, out_dtype=tl.float32)
    tl.store(P + z * M * N + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def reduce_scale(P, S, Y, SIZE: tl.constexpr, N: tl.constexpr,
                 SK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), tl.float32)
    for z in range(SK):
        value += tl.load(P + z * SIZE + i, i < SIZE, 0)
    scale = tl.load(S + i % N, i < SIZE, 0)
    tl.store(Y + i, value * scale, i < SIZE)


@triton.jit
def dense(A, Q, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    mi = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    ni = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        ki = step * BK + tl.arange(0, BK)
        a = tl.load(A + mi[:, None] * K + ki[None, :],
                    (mi[:, None] < M) & (ki[None, :] < K), 0)
        w = tl.load(Q + ni[None, :] * K + ki[:, None],
                    (ni[None, :] < N) & (ki[:, None] < K), 0).to(tl.bfloat16)
        acc += tl.dot(a, w, out_dtype=tl.float32)
    scale = tl.load(S + ni, ni < N, 0)
    tl.store(Y + mi[:, None] * N + ni[None, :], acc * scale[None, :],
             (mi[:, None] < M) & (ni[None, :] < N))


def run(a, q, scale, algorithm='auto', config=None, convert_config=(1024, 8), layout='nk', group_m=0):
    """All allocation/conversion/reduction stays inside this call.

    Auto is a conservative initial policy, not a measured performance boundary.
    Config is (BM, BN, BK, warps, stages[, split_k]).
    """
    if (a.ndim != 2 or q.ndim != 2 or a.dtype != torch.bfloat16
            or q.dtype != torch.int8 or scale.dtype != torch.float32
            or not a.is_contiguous() or not q.is_contiguous()
            or not scale.is_contiguous() or a.shape[1] != q.shape[1]
            or scale.shape != (q.shape[0],) or a.device != q.device
            or scale.device != q.device or a.device.type != 'cuda'):
        raise ValueError('expected contiguous BF16 A, INT8 Q and FP32 per-channel scale')
    m, k = a.shape
    n = q.shape[0]
    if min(n, k) == 0:
        raise ValueError('empty weight is unsupported')
    y = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m == 0:
        return y
    if algorithm == 'auto':
        algorithm = 'separate' if (n, k) == (2048, 6144) and m >= 128 else 'dense'
    cfg = config or (32, 64, 64, 8, 1)
    bm, bn, bk, warps, stages = cfg[:5]
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    if algorithm == 'fused_native':
        return scale_mm(a, q.T, scale, cfg, group_m, out=y)
    if algorithm == 'dense':
        dense[grid](a, q, scale, y, m, n, k, bm, bn, bk,
                    num_warps=warps, num_stages=stages)
    elif algorithm == 'splitk':
        sk = cfg[5] if len(cfg) == 6 else 4
        if sk < 1:
            raise ValueError('split_k must be positive')
        workspace = torch.empty((sk, m, n), device=a.device, dtype=torch.float32)
        partial[(grid[0], sk)](a, q, workspace, m, n, k, bm, bn, bk, sk,
                              num_warps=warps, num_stages=stages)
        reduce_scale[(triton.cdiv(m*n, 256),)](workspace, scale, y, m*n, n, sk, 256,
                                             num_warps=8)
    elif algorithm in ('separate', 'separate_rounded'):
        provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
        if layout == 'nk':
            converted = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
            block, cw = convert_config
            convert[(triton.cdiv(q.numel(), block),)](q, converted, q.numel(), block, num_warps=cw)
            weight = converted.T
        elif layout == 'kn':
            converted = torch.empty((k, n), device=q.device, dtype=torch.bfloat16)
            tn, tk, cw = convert_config
            convert_kn[(triton.cdiv(n, tn), triton.cdiv(k, tk))](
                q, converted, n, k, tn, tk, num_warps=cw)
            weight = converted
        else:
            raise ValueError(f'unknown conversion layout: {layout}')
        if group_m:
            if algorithm != 'separate':
                raise ValueError('grouped experiment requires direct scale')
            return scale_mm(a, weight, scale, cfg, group_m, out=y)
        # Direct uses a separate variant of the native MM body, not MetaX MM.
        kernel = provider.mm_kernel_scale if algorithm == 'separate' else provider.mm_kernel
        while not isinstance(kernel, triton.runtime.JITFunction):
            kernel = kernel.fn
        args = (a, weight, y)
        if algorithm == 'separate':
            args += (scale,)
        kernel[(grid[0], 1)](*args, m, n, k, k, 1, *weight.stride(), n, 1,
            acc_dtype=tl.float32, input_precision=None, fp8_fast_accum=True,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLIT_K=1,
            EVEN_K=k % bk == 0, AB_DTYPE=tl.bfloat16, UPGRADE=False,
            UPGRADE_A_OFFS=m*k > 2**31-1, UPGRADE_B_OFFS=n*k > 2**31-1,
            UPGRADE_C_OFFS=m*n > 2**31-1, num_warps=warps, num_stages=stages)
        if algorithm == 'separate_rounded':
            y = (y.float() * scale).to(torch.bfloat16)
    else:
        raise ValueError(f'unknown algorithm: {algorithm}')
    return y


def scale_mm(a, weight, scale, config, group_m=0, out=None, return_kernel=False):
    """Reuse Iluvatar MM body for BF16 or INT8-loaded B; no weight staging.

    INT8 B requires divisible K because the original remainder branch does not
    cast its final masked tile. Explicit experiments reject that case.
    """
    m, k = a.shape
    n = weight.shape[1]
    bm, bn, bk, warps, stages = config[:5]
    if (a.dtype != torch.bfloat16 or weight.dtype not in (torch.int8, torch.bfloat16)
            or weight.shape[0] != k or scale.dtype != torch.float32
            or scale.shape != (n,) or a.device != weight.device
            or a.device != scale.device or not a.is_contiguous()
            or group_m < 0):
        raise ValueError('unsupported scale MM inputs')
    if weight.dtype == torch.int8 and k % bk:
        raise ValueError('INT8 native body requires K divisible by BLOCK_K')
    provider = importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    kernel = provider.mm_kernel_scale_grouped if group_m else provider.mm_kernel_scale
    y = out if out is not None else torch.empty((m, n), device=a.device, dtype=a.dtype)
    compiled = kernel[(triton.cdiv(m, bm)*triton.cdiv(n, bn), 1)](
        a, weight, y, scale, m, n, k, *a.stride(), *weight.stride(), *y.stride(),
        acc_dtype=tl.float32, input_precision=None, fp8_fast_accum=True,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m or 8, SPLIT_K=1,
        EVEN_K=k % bk == 0, AB_DTYPE=tl.bfloat16, UPGRADE=False,
        UPGRADE_A_OFFS=m*k > 2**31-1, UPGRADE_B_OFFS=n*k > 2**31-1,
        UPGRADE_C_OFFS=m*n > 2**31-1, num_warps=warps, num_stages=stages)
    return (y, compiled) if return_kernel else y
