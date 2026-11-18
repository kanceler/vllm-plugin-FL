"""Offline Down partial candidates; no production dispatch changes."""
import triton
import triton.language as tl


@triton.jit
def down_partial_acc(
    A, B, P, M, N: tl.constexpr, K: tl.constexpr,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, NT_LOAD: tl.constexpr, DOT_ACC: tl.constexpr,
):
    pid, split = tl.program_id(0), tl.program_id(1)
    rm = (pid // (N // BLOCK_N)) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = (pid % (N // BLOCK_N)) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    iters = K // (BLOCK_K * SPLIT_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for i in range(split * iters, (split + 1) * iters):
        kk = i * BLOCK_K + rk
        a = tl.load(A + rm[:, None] * K + kk[None, :],
                    mask=rm[:, None] < M, other=0.)
        if NT_LOAD:
            b = tl.load(B + rn[:, None] * K + kk[None, :])
            weight = tl.trans(b.to(tl.bfloat16))
        else:
            b = tl.load(B + rn[None, :] * K + kk[:, None])
            weight = b.to(tl.bfloat16)
        # Compare an explicit loop-carried dot accumulator with dot then add.
        if DOT_ACC:
            acc = tl.dot(a, weight, acc, out_dtype=tl.float32, allow_tf32=False)
        else:
            acc += tl.dot(a, weight, out_dtype=tl.float32, allow_tf32=False)
    tl.store(P + split * M * N + rm[:, None] * N + rn[None, :], acc,
             mask=rm[:, None] < M)
