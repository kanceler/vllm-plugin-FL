"""Experimental Down Split-K weight lookahead kernels."""
import triton
import triton.language as tl

@triton.jit
def down_partial_prefetch(
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

    first = pid_k * k_iters_per_split
    kk0 = (first) * BLOCK_K + tl.arange(0, BLOCK_K)
    a0 = tl.load(A + offsets_m[:,None]*stride_am + kk0[None,:]*stride_ak, mask=offsets_m[:,None]<M, other=0)
    b0 = tl.load(B + kk0[:,None]*stride_bk + offsets_n[None,:]*stride_bn)
    for k_iter in range(first + 1, first + k_iters_per_split):
        kk1 = (k_iter) * BLOCK_K + tl.arange(0, BLOCK_K)
        a1 = tl.load(A + offsets_m[:,None]*stride_am + kk1[None,:]*stride_ak, mask=offsets_m[:,None]<M, other=0)
        b1 = tl.load(B + kk1[:,None]*stride_bk + offsets_n[None,:]*stride_bn)
        acc += tl.dot(a0, b0.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32=False)
        a0 = a1
        b0 = b1
    acc += tl.dot(a0, b0.to(tl.bfloat16), out_dtype=tl.float32, allow_tf32=False)

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
def down_partial_prefetch_nt(
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

    first = pid_k * k_iters_per_split
    kk0 = (first) * BLOCK_K + tl.arange(0, BLOCK_K)
    a0 = tl.load(A + offsets_m[:,None]*stride_am + kk0[None,:]*stride_ak, mask=offsets_m[:,None]<M, other=0)
    b0 = tl.load(B + offsets_n[:,None]*stride_bn + kk0[None,:]*stride_bk)
    for k_iter in range(first + 1, first + k_iters_per_split):
        kk1 = (k_iter) * BLOCK_K + tl.arange(0, BLOCK_K)
        a1 = tl.load(A + offsets_m[:,None]*stride_am + kk1[None,:]*stride_ak, mask=offsets_m[:,None]<M, other=0)
        b1 = tl.load(B + offsets_n[:,None]*stride_bn + kk1[None,:]*stride_bk)
        acc += tl.dot(a0, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)
        a0 = a1
        b0 = b1
    acc += tl.dot(a0, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)

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

