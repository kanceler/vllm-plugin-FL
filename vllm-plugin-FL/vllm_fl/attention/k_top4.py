"""Diagnostic K INT8/top4 + V INT8 cache. D=128, per-token/head FP32 scales."""
import types
import torch
from vllm.triton_utils import triton, tl


def views(raw):
    b,s,h,w=raw.shape
    if raw.dtype!=torch.uint8 or w!=276 or raw.stride(-1)!=1:
        raise ValueError('Expected uint8 cache [pages, slots, heads, 276]')
    stride=raw.stride(0);off=raw.storage_offset()
    if stride%4 or off%4:
        raise ValueError('Unaligned cache')
    k=torch.as_strided(raw.view(torch.int8),(b,s,h,144),(stride,h*144,144,1),off)
    v=torch.as_strided(raw.view(torch.int8),(b,s,h,128),(stride,h*128,128,1),off+s*h*144)
    vs=torch.as_strided(raw.view(torch.float32),(b,s,h),(stride//4,h,1),off//4+s*h*68)
    return k,v,vs


@triton.jit
def _write(K,V,KC,VC,VS,SLOTS,kt:tl.constexpr,kh:tl.constexpr,vt:tl.constexpr,vh:tl.constexpr,
           kb:tl.constexpr,vb:tl.constexpr,sb:tl.constexpr,H:tl.constexpr,S:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)
    slot=tl.load(SLOTS+t).to(tl.int64)
    if slot<0:return
    p=slot//S;loc=slot%S;d=tl.arange(0,128)
    x=tl.load(K+t*kt+h*kh+d).to(tl.float32)
    y=tl.load(V+t*vt+h*vh+d).to(tl.float32)
    a=tl.abs(x); base=KC+p*kb+loc*H*144+h*144
    # Stable tie policy: lower channel wins. Selected channels are zeroed in
    # the integer body; saved BF16 originals replace them inside attention.
    for i in tl.static_range(4):
        maximum=tl.max(a,0)
        index=tl.min(tl.where(a==maximum,d,2147483647),0)
        saved=tl.sum(tl.where(d==index,x,0.),0)
        tl.store((base+128+i).to(tl.pointer_type(tl.uint8)),index)
        tl.store((base+132+2*i).to(tl.pointer_type(tl.bfloat16)),saved)
        a=tl.where(d==index,-1.,a)
    scale=tl.maximum(tl.max(a,0)/127.,1e-6)
    z=tl.where(a<0,0.,x)/scale
    qi=tl.clamp(tl.where(z>=0,z+.5,z-.5),-128.,127.).to(tl.int8)
    tl.store(base+d,qi)
    tl.store((base+140).to(tl.pointer_type(tl.float32)),scale)
    sv=tl.maximum(tl.max(tl.abs(y),0)/127.,1e-6)
    vy=y/sv
    tl.store(VC+p*vb+loc*H*128+h*128+d,
             tl.clamp(tl.where(vy>=0,vy+.5,vy-.5),-128.,127.).to(tl.int8))
    tl.store(VS+p*sb+loc*H+h,sv)


def write(k,v,raw,slots):
    if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or k.shape!=v.shape or k.shape[-1]!=128:
        raise ValueError('BF16 matching D128 K/V required')
    if k.stride(-1)!=1 or v.stride(-1)!=1 or slots.numel()!=k.shape[0]:
        raise ValueError('Unsupported strides or slot count')
    kc,vc,vs=views(raw)
    _write[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),
        v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=4)


_launcher=None


def launcher():
    global _launcher
    if _launcher is not None:return _launcher
    from vllm_fl.attention.v_only import launcher as base_launcher
    base=base_launcher();ns=dict(base.__globals__);old=ns['kernel_unified_attention']
    needle='K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)'
    if old.src.count(needle)!=1:raise RuntimeError('Unsupported K load source')
    replacement='''meta_offset = (physical_block_idx * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + (seq_offset % BLOCK_SIZE) * stride_k_cache_1)
        ks = tl.load((key_cache_ptr + meta_offset + 140).to(tl.pointer_type(tl.float32)), mask=tile_mask, other=1.)
        K_real = K_load.to(tl.float32) * ks[None, :]
        for protected in tl.static_range(4):
            idx = tl.load((key_cache_ptr + meta_offset + 128 + protected).to(tl.pointer_type(tl.uint8)), mask=tile_mask, other=255)
            val = tl.load((key_cache_ptr + meta_offset + 132 + 2 * protected).to(tl.pointer_type(tl.bfloat16)), mask=tile_mask, other=0.)
            K_real = tl.where(offs_d[:, None] == idx[None, :], val[None, :].to(tl.float32), K_real)
        K = K_real.to(Q.dtype)'''
    fixed=triton.jit(old.fn);fixed._unsafe_update_src(old.src.replace(needle,replacement))
    ns['kernel_unified_attention']=fixed
    _launcher=types.FunctionType(base.__code__,ns,base.__name__,base.__defaults__,base.__closure__)
    _launcher.__kwdefaults__=base.__kwdefaults__
    return _launcher
