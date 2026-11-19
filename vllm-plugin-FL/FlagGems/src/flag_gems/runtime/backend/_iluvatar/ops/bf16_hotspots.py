"""Opt-in BF16 fixed configurations for verified Iluvatar NT shapes."""

import importlib

import torch
from flag_gems.runtime import torch_device_fn

import triton
import triton.language as tl
from flag_gems.utils import triton_lang_extension as tle

@triton.jit
def hotspot_mm(
    A,
    B,
    C,
    Scales,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    acc_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FAST_STORE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    # matrix multiplication
    if UPGRADE:
        pid = tle.program_id(0)
        pid_z = tle.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    # grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # # re-order program ID for better L2 performance
    # width = GROUP_M * grid_n
    # group_id = pid // width
    # group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    # pid_m = group_id * GROUP_M + (pid % group_size)
    # pid_n = (pid % width) // (group_size)
    # Group adjacent M tiles to expose reuse of each B tile across CTAs.
    if GROUP_M > 0:
        grid_m = tl.cdiv(M, BLOCK_M)
        width = GROUP_M * grid_n
        group_id = pid // width
        first_m = group_id * GROUP_M
        group_size = tl.minimum(grid_m - first_m, GROUP_M)
        pid_m = first_m + (pid % width) % group_size
        pid_n = (pid % width) // group_size
    else:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ram = (tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rbn = (tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)).to(tl.int64)
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for k in range(0, loop_num):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

    tl.static_assert(SPLIT_K == 1)
    if APPLY_SCALE:
        scale = tl.load(Scales + rbn).to(tl.float32)
        acc = acc * scale[None, :]
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if FAST_STORE and EVEN_M:
        if EVEN_N:
            tl.store(C, acc)
        else:
            tl.store(C, acc, mask=(rn < N)[None, :])
    else:
        tl.store(C, acc, mask=mask)


HOTSPOTS = {
    (2048, 2048, 6144): 0,
    (2048, 12288, 2048): 4,
    (2048, 2560, 2048): 0,
    (2048, 2048, 2048): 0,
    (4096, 12288, 2048): 4,
    (64, 2048, 2048): 0,
    (64, 2560, 2048): 0,
}

_provider = importlib.import_module(".mm", __package__)
_original_launch = _provider._launch_mm
mm_kernel = _provider.mm_kernel


def _launch_hotspot(a, b, c, M, N, K):
    group = HOTSPOTS.get((M,N,K))
    if (group is not None and a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.stride() == (K,1) and b.stride() == (1,K) and c.stride() == (N,1)):
        if M == 64 or K == 6144:
            bm, bn, bk = (64, 64, 128) if M == 64 else (256, 256, 32)
            kernel = mm_kernel
            while not isinstance(kernel, triton.runtime.JITFunction):
                kernel = kernel.fn
            with torch_device_fn.device(a.device):
                kernel[(M//bm*(N//bn),1)](a,b,c,M,N,K,*a.stride(),*b.stride(),*c.stride(),
                    acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,
                    BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=8,SPLIT_K=1,EVEN_K=True,
                    AB_DTYPE=tl.bfloat16,UPGRADE=False,UPGRADE_A_OFFS=False,
                    UPGRADE_B_OFFS=False,UPGRADE_C_OFFS=False,num_warps=16,num_stages=2)
            return c
        with torch_device_fn.device(a.device):
            hotspot_mm[(M//256*(N//256),1)](a,b,c,c,M,N,K,*a.stride(),*b.stride(),*c.stride(),
                acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,
                BLOCK_M=256,BLOCK_N=256,BLOCK_K=32,GROUP_M=group,SPLIT_K=1,EVEN_K=True,
                AB_DTYPE=tl.bfloat16,UPGRADE=False,UPGRADE_A_OFFS=False,UPGRADE_B_OFFS=False,
                UPGRADE_C_OFFS=False,num_warps=16,num_stages=2,FAST_STORE=True,
                EVEN_M=True,EVEN_N=True,APPLY_SCALE=False)
        return c
    return _original_launch(a, b, c, M, N, K)


def install():
    _provider._launch_mm = _launch_hotspot
