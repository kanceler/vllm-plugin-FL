"""Experimental lookahead kernel; production dispatch unchanged."""
import triton
import triton.language as tl

@triton.jit
def gateup_prefetch(
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

    # Explicit lookahead carries converted B across loop iterations.
    ap = A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    bp = B + offsets_n[:, None] * stride_bn + offsets_k[None, :] * stride_bk
    a0 = tl.load(ap, mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K), other=0)
    b0 = tl.load(bp, mask=(offsets_n[:, None] < N) & (offsets_k[None, :] < K), other=0).to(tl.bfloat16)
    for k_start in range(BLOCK_K, K, BLOCK_K):
        kk = k_start + offsets_k
        ap1 = A + offsets_m[:, None] * stride_am + kk[None, :] * stride_ak
        bp1 = B + offsets_n[:, None] * stride_bn + kk[None, :] * stride_bk
        a1 = tl.load(ap1, mask=(offsets_m[:, None] < M) & (kk[None, :] < K), other=0)
        b1 = tl.load(bp1, mask=(offsets_n[:, None] < N) & (kk[None, :] < K), other=0).to(tl.bfloat16)
        acc += tl.dot(a0, tl.trans(b0), out_dtype=tl.float32, allow_tf32=False)
        a0 = a1
        b0 = b1
    acc += tl.dot(a0, tl.trans(b0), out_dtype=tl.float32, allow_tf32=False)

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])

