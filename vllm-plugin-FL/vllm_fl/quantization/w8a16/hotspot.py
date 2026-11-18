"""Version-bound MM variant derived from the pinned Iluvatar scale provider.
GROUP_M=0 preserves row-major scheduling; FAST_STORE only removes proven masks.
No runtime dispatch changes. Native control uses the same body without scaling.
"""
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


def run(a, q, scale, choice):
    """Complete Separate; persistent weights remain INT8 NK, no BF16 cache."""
    import torch
    from .backend import convert, convert_kn
    m,k=a.shape
    n=q.shape[0]
    cv=choice['convert']
    if choice['layout']=='nk':
        converted=torch.empty(q.shape,device=q.device,dtype=torch.bfloat16)
        block,warps=cv
        convert[(triton.cdiv(q.numel(),block),)](q,converted,q.numel(),block,num_warps=warps)
        weight=converted.T
    elif choice['layout']=='kn':
        converted=torch.empty((k,n),device=q.device,dtype=torch.bfloat16)
        tn,tk,warps=cv
        convert_kn[(triton.cdiv(n,tn),triton.cdiv(k,tk))](q,converted,n,k,tn,tk,num_warps=warps)
        weight=converted
    else:
        raise ValueError('unknown hotspot conversion layout')
    y=torch.empty((m,n),device=a.device,dtype=a.dtype)
    bm,bn,bk,warps,stages=choice['config']
    hotspot_mm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),1)](
        a,weight,y,scale,m,n,k,*a.stride(),*weight.stride(),*y.stride(),
        acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,
        BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,GROUP_M=choice['group'],SPLIT_K=1,
        EVEN_K=k%bk==0,FAST_STORE=choice['fast'],EVEN_M=m%bm==0,EVEN_N=n%bn==0,
        APPLY_SCALE=True,AB_DTYPE=tl.bfloat16,UPGRADE=False,
        UPGRADE_A_OFFS=m*k>2**31-1,UPGRADE_B_OFFS=n*k>2**31-1,
        UPGRADE_C_OFFS=m*n>2**31-1,num_warps=warps,num_stages=stages)
    return y
