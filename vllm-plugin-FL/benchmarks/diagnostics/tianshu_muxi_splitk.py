# SPDX-License-Identifier: Apache-2.0
"""MetaX Split-K arithmetic reused unchanged; Iluvatar launch adapter only.
Source SHA256: 826ee3b73b01f4cc329286f9e268c87a87f7b069464afe7b121e682f1a9abf03
"""
import torch
import triton
import triton.language as tl

@triton.jit
def weight_int8pack_mm_splitk_partial(
    A,
    B,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_n = N // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_iters_per_split = K // (BLOCK_K * SPLIT_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(
        pid_k * k_iters_per_split, (pid_k + 1) * k_iters_per_split
    ):
        offsets_k = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offsets_m[:, None] < M, other=0.0)
        b = tl.load(
            B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        )
        acc += tl.dot(
            a,
            b.to(tl.bfloat16),
            out_dtype=tl.float32,
            allow_tf32=False,
        )

    partial_ptrs = (
        P
        + pid_k * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :]
    )
    if EVEN_M:
        tl.store(partial_ptrs, acc)
    else:
        tl.store(partial_ptrs, acc, mask=offsets_m[:, None] < M)

@triton.jit
def weight_int8pack_mm_splitk_reduce_scale(
    P,
    Scales,
    C,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split_id in range(0, SPLIT_K):
        acc += tl.load(
            P + split_id * M * N + offsets, mask=mask, other=0.0
        )
    scales = tl.load(Scales + offsets % N, mask=mask, other=0.0).to(tl.float32)
    tl.store(C + offsets, (acc * scales).to(tl.bfloat16), mask=mask)

@triton.jit
def weight_int8pack_mm_splitk_reduce_scale_vector(
    P, Scales, C, M, N, SPLIT_K: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # Offline variant: load split planes together and reduce across that axis.
    # This changes FP32 addition order; callers must validate numerical error.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    splits = tl.arange(0, triton.next_power_of_2(SPLIT_K))
    values = tl.load(P + splits[:, None] * M * N + offsets[None, :],
                     mask=(splits[:, None] < SPLIT_K) &
                          (offsets[None, :] < M * N), other=0.0)
    acc = tl.sum(values, axis=0)
    scale = tl.load(Scales + offsets % N, mask=offsets < M * N, other=0.0)
    tl.store(C + offsets, (acc * scale).to(tl.bfloat16), mask=offsets < M * N)

def run(a, q, scales, config, reduction=('serial', 256)):
    m, k = a.shape
    n = q.shape[0]
    bm, bn, bk, warps, stages, sk = config
    variant, block = reduction
    if (a.dtype != torch.bfloat16 or q.dtype != torch.int8 or scales.dtype != torch.float32
            or not a.is_contiguous() or not q.is_contiguous() or not scales.is_contiguous()
            or q.shape[1] != k or scales.shape != (n,) or n % bn or k % (bk*sk)
            or sk < 2 or variant not in ('serial','vector') or block not in (128,256,512)):
        raise ValueError('unsupported Split-K shape, layout, dtype or config')
    partials = torch.empty((sk,m,n), device=a.device, dtype=torch.float32)
    y = torch.empty((m,n), device=a.device, dtype=a.dtype)
    # Explicit JIT launch avoids LibEntry launch-option cache collisions.
    # CoreX does not accept MetaX pipeline="basic"; use its native compiler.
    weight_int8pack_mm_splitk_partial[(triton.cdiv(m,bm)*triton.cdiv(n,bn),sk)](
        a,q,partials,m,n,k,*a.stride(),q.stride(1),q.stride(0),
        SPLIT_K=sk,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,EVEN_M=m%bm==0,
        num_warps=warps,num_stages=stages)
    reducer = (weight_int8pack_mm_splitk_reduce_scale_vector if variant=='vector'
               else weight_int8pack_mm_splitk_reduce_scale)
    reducer[(triton.cdiv(m*n,block),)](partials,scales,y,m,n,
        SPLIT_K=sk,BLOCK_SIZE=block,num_warps=4)
    return y
