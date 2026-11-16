"""Static post-RoPE permutation + scale folded into Q.

Keeps the frozen 276B separate-metadata cache allocation for a controlled
comparison. Layer parameters are supplied by the static service calibration artifact.
"""
from functools import lru_cache
import torch
from vllm.triton_utils import triton, tl
from vllm_fl.attention import k_top4_joint_kernel as layout
from vllm_fl.attention import k_static_fold_launch as frozen


@triton.jit
def _q(Q,O,P,S,qt:tl.constexpr,qh:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,128)
    ch=tl.load(P+(h//8)*128+d)
    s=tl.load(S+(h//8)*128+d)
    q=tl.load(Q+t*qt+h*qh+ch).to(tl.float32)
    tl.store(O+t*16*128+h*128+d,q*tl.where(d<4,1.,s))


@triton.jit
def _write(K,V,KC,VC,VS,SLOTS,P,SC,kt:tl.constexpr,kh:tl.constexpr,
           vt:tl.constexpr,vh:tl.constexpr,kb:tl.constexpr,vb:tl.constexpr,
           sb:tl.constexpr,S:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)
    slot=tl.load(SLOTS+t).to(tl.int64)
    if slot<0:return
    page=slot//S;loc=slot%S;d=tl.arange(0,128)
    ch=tl.load(P+h*128+d);scale=tl.load(SC+h*128+d)
    x=tl.load(K+t*kt+h*kh+ch).to(tl.float32)
    z=x/scale
    body=tl.where(d<4,0.,tl.clamp(tl.where(z>=0,z+.5,z-.5),-128.,127.)).to(tl.int8)
    tl.store(KC+page*kb+loc*256+h*128+d,body)
    meta=KC+page*kb+S*256+loc*32+h*16
    for i in tl.static_range(4):
        original=tl.load(P+h*128+i)
        value=tl.load(K+t*kt+h*kh+original)
        tl.store((meta+4+i*2).to(tl.pointer_type(tl.bfloat16)),value)
    y=tl.load(V+t*vt+h*vh+d).to(tl.float32)
    sv=tl.maximum(tl.max(tl.abs(y),0)/127.,1e-6);zy=y/sv
    tl.store(VC+page*vb+loc*256+h*128+d,tl.clamp(tl.where(zy>=0,zy+.5,zy-.5),-128.,127.).to(tl.int8))
    tl.store(VS+page*sb+loc*2+h,sv)


def process_q(q,out,perm,scale):
    return _q[(q.shape[0],16)](q,out,perm,scale,q.stride(0),q.stride(1),num_warps=1)


def write(k,v,raw,slots,perm,scale):
    kc,vc,vs=layout.views(raw,'separate_meta')
    return _write[(slots.numel(),2)](k,v,kc,vc,vs,slots,perm,scale,
        k.stride(0),k.stride(1),v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),raw.shape[1],num_warps=1)


@lru_cache(None)
def kernel():
    old=layout.kernel('separate_meta','chain');src=old.src
    start=src.index('        ks = ');end=src.index('        K = K_real.to(Q.dtype)',start)
    replacement='''        K_real = K_load.to(Q.dtype)
        packed_v0 = tl.load((key_cache_ptr + meta_offset + 4).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
        packed_v1 = tl.load((key_cache_ptr + meta_offset + 8).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
        for protected in tl.static_range(4):
            word = packed_v0 if protected < 2 else packed_v1
            bits = ((word >> ((protected % 2) * 16)) & 65535).to(tl.uint16)
            val = bits.to(tl.bfloat16, bitcast=True)
            K_real = tl.where(offs_d[:, None] == protected, val[None, :], K_real)
'''
    jit=triton.jit(old.fn);jit._unsafe_update_src(src[:start]+replacement+src[end:]);return jit


@lru_cache(None)
def direct_kernel():
    from .static_kv_compat import specialize_source
    original = kernel()
    source = specialize_source(original.src)
    specialized = triton.jit(original.fn)
    specialized._unsafe_update_src(source)
    return specialized


def launcher(config, direct=False):
    fn = frozen.launcher(config)
    head = kernel()
    if direct:
        from vllm.v1.kv_cache_interface import KVQuantMode
        from .static_kv_compat import StaticDecodeKernel
        head = StaticDecodeKernel(head, direct_kernel(), bf16=torch.bfloat16,
                                 int8=torch.int8, kv_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)
    fn.__globals__['kernel_unified_attention'] = head
    return fn


def parameters(fixed,scale):
    """Host-side immutable tables, built before capture, in permuted order."""
    orders=[ids+[i for i in range(128) if i not in ids] for ids in fixed]
    assert len(orders)==2 and all(sorted(x)==list(range(128)) for x in orders)
    perm=torch.tensor(orders,device=scale.device,dtype=torch.int32)
    return perm,scale.gather(1,perm.long()).contiguous()
