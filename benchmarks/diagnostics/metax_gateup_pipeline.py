"""Experimental weight lookahead depths and conversion placement."""
import triton
import triton.language as tl

@triton.jit
def gateup_depth1_raw(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Carry INT8 loads across iterations; all conversion remains in this kernel.
    kk = 0 * BLOCK_K + offsets_k
    a0 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
    b0 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0)
    for k_start in range(1 * BLOCK_K, K, BLOCK_K):
        kk = k_start + offsets_k
        a1 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
        b1 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0)
        acc += tl.dot(a0, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)
        a0 = a1
        b0 = b1
    acc += tl.dot(a0, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])


@triton.jit
def gateup_depth2_converted(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Carry converted BF16 weights across iterations; all conversion remains in this kernel.
    kk = 0 * BLOCK_K + offsets_k
    a0 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
    b0 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0).to(tl.bfloat16)
    kk = 1 * BLOCK_K + offsets_k
    a1 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
    b1 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0).to(tl.bfloat16)
    for k_start in range(2 * BLOCK_K, K, BLOCK_K):
        kk = k_start + offsets_k
        a2 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
        b2 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0).to(tl.bfloat16)
        acc += tl.dot(a0, tl.trans(b0), out_dtype=tl.float32, allow_tf32=False)
        a0 = a1
        b0 = b1
        a1 = a2
        b1 = b2
    acc += tl.dot(a0, tl.trans(b0), out_dtype=tl.float32, allow_tf32=False)
    acc += tl.dot(a1, tl.trans(b1), out_dtype=tl.float32, allow_tf32=False)

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])


@triton.jit
def gateup_depth2_raw(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Carry INT8 loads across iterations; all conversion remains in this kernel.
    kk = 0 * BLOCK_K + offsets_k
    a0 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
    b0 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0)
    kk = 1 * BLOCK_K + offsets_k
    a1 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
    b1 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0)
    for k_start in range(2 * BLOCK_K, K, BLOCK_K):
        kk = k_start + offsets_k
        a2 = tl.load(A + offsets_m[:, None]*stride_am + kk[None, :]*stride_ak, mask=(offsets_m[:, None]<M)&(kk[None, :]<K), other=0)
        b2 = tl.load(B + offsets_n[:, None]*stride_bn + kk[None, :]*stride_bk, mask=(offsets_n[:, None]<N)&(kk[None, :]<K), other=0)
        acc += tl.dot(a0, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)
        a0 = a1
        b0 = b1
        a1 = a2
        b1 = b2
    acc += tl.dot(a0, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)
    acc += tl.dot(a1, tl.trans(b1.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])


@triton.jit
def gateup_weightonly1_converted(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Keep A in the regular loop so weight lookahead does not force A to be loop-carried.
    kk = 0 * BLOCK_K + offsets_k
    b0 = tl.load(B + offsets_n[:,None]*stride_bn + kk[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(kk[None,:]<K), other=0).to(tl.bfloat16)
    for k_start in range(0, K, BLOCK_K):
        next_k = k_start + 1 * BLOCK_K + offsets_k
        bn = tl.load(B + offsets_n[:,None]*stride_bn + next_k[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(next_k[None,:]<K), other=0).to(tl.bfloat16)
        kk = k_start + offsets_k
        a = tl.load(A + offsets_m[:,None]*stride_am + kk[None,:]*stride_ak, mask=(offsets_m[:,None]<M)&(kk[None,:]<K), other=0)
        acc += tl.dot(a, tl.trans(b0), out_dtype=tl.float32, allow_tf32=False)
        b0 = bn

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])


@triton.jit
def gateup_weightonly1_raw(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Keep A in the regular loop so weight lookahead does not force A to be loop-carried.
    kk = 0 * BLOCK_K + offsets_k
    b0 = tl.load(B + offsets_n[:,None]*stride_bn + kk[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(kk[None,:]<K), other=0)
    for k_start in range(0, K, BLOCK_K):
        next_k = k_start + 1 * BLOCK_K + offsets_k
        bn = tl.load(B + offsets_n[:,None]*stride_bn + next_k[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(next_k[None,:]<K), other=0)
        kk = k_start + offsets_k
        a = tl.load(A + offsets_m[:,None]*stride_am + kk[None,:]*stride_ak, mask=(offsets_m[:,None]<M)&(kk[None,:]<K), other=0)
        acc += tl.dot(a, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)
        b0 = bn

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])


@triton.jit
def gateup_weightonly2_converted(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Keep A in the regular loop so weight lookahead does not force A to be loop-carried.
    kk = 0 * BLOCK_K + offsets_k
    b0 = tl.load(B + offsets_n[:,None]*stride_bn + kk[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(kk[None,:]<K), other=0).to(tl.bfloat16)
    kk = 1 * BLOCK_K + offsets_k
    b1 = tl.load(B + offsets_n[:,None]*stride_bn + kk[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(kk[None,:]<K), other=0).to(tl.bfloat16)
    for k_start in range(0, K, BLOCK_K):
        next_k = k_start + 2 * BLOCK_K + offsets_k
        bn = tl.load(B + offsets_n[:,None]*stride_bn + next_k[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(next_k[None,:]<K), other=0).to(tl.bfloat16)
        kk = k_start + offsets_k
        a = tl.load(A + offsets_m[:,None]*stride_am + kk[None,:]*stride_ak, mask=(offsets_m[:,None]<M)&(kk[None,:]<K), other=0)
        acc += tl.dot(a, tl.trans(b0), out_dtype=tl.float32, allow_tf32=False)
        b0 = b1
        b1 = bn

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])


@triton.jit
def gateup_weightonly2_raw(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Keep A in the regular loop so weight lookahead does not force A to be loop-carried.
    kk = 0 * BLOCK_K + offsets_k
    b0 = tl.load(B + offsets_n[:,None]*stride_bn + kk[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(kk[None,:]<K), other=0)
    kk = 1 * BLOCK_K + offsets_k
    b1 = tl.load(B + offsets_n[:,None]*stride_bn + kk[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(kk[None,:]<K), other=0)
    for k_start in range(0, K, BLOCK_K):
        next_k = k_start + 2 * BLOCK_K + offsets_k
        bn = tl.load(B + offsets_n[:,None]*stride_bn + next_k[None,:]*stride_bk, mask=(offsets_n[:,None]<N)&(next_k[None,:]<K), other=0)
        kk = k_start + offsets_k
        a = tl.load(A + offsets_m[:,None]*stride_am + kk[None,:]*stride_ak, mask=(offsets_m[:,None]<M)&(kk[None,:]<K), other=0)
        acc += tl.dot(a, tl.trans(b0.to(tl.bfloat16)), out_dtype=tl.float32, allow_tf32=False)
        b0 = b1
        b1 = bn

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])

