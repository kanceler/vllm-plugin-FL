"""Diagnostic K group32 INT8 + per-head INT8 V; no top4 or service registration."""
from functools import lru_cache
import torch
from vllm.triton_utils import triton, tl
from vllm_fl.attention import k_top4, v_only
import k_top4_frozen as frozen

views = k_top4.views  # Same 144-byte K record; last 16 bytes are four FP32 scales.


@triton.jit
def _write(K,V,KC,VC,VS,SLOTS,kt:tl.constexpr,kh:tl.constexpr,vt:tl.constexpr,vh:tl.constexpr,
           kb:tl.constexpr,vb:tl.constexpr,sb:tl.constexpr,H:tl.constexpr,S:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)
    slot=tl.load(SLOTS+t).to(tl.int64)
    if slot<0:return
    p=slot//S;loc=slot%S;d=tl.arange(0,128)
    x=tl.load(K+t*kt+h*kh+d).to(tl.float32).reshape(4,32)
    scales=tl.maximum(tl.max(tl.abs(x),1)/127.,1e-6)
    z=x/scales[:,None]
    qi=tl.clamp(tl.where(z>=0,z+.5,z-.5),-128.,127.).to(tl.int8)
    base=KC+p*kb+loc*H*144+h*144
    tl.store(base+d,qi.reshape(128))
    tl.store((base+128).to(tl.pointer_type(tl.float32))+tl.arange(0,4),scales)
    y=tl.load(V+t*vt+h*vh+d).to(tl.float32)
    sv=tl.maximum(tl.max(tl.abs(y),0)/127.,1e-6);zv=y/sv
    tl.store(VC+p*vb+loc*H*128+h*128+d,
             tl.clamp(tl.where(zv>=0,zv+.5,zv-.5),-128.,127.).to(tl.int8))
    tl.store(VS+p*sb+loc*H+h,sv)


def write(k,v,raw,slots,warps=1):
    if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or k.shape!=v.shape or k.shape[-1]!=128:
        raise ValueError('Matching BF16 D128 K/V required')
    if k.stride(-1)!=1 or v.stride(-1)!=1 or slots.numel()!=k.shape[0]:
        raise ValueError('Unsupported strides/slots')
    kc,vc,vs=views(raw)
    return _write[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),
        v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=warps)


@lru_cache(None)
def kernel():
    old=v_only.launcher().__globals__['kernel_unified_attention']
    needle='K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)'
    if old.src.count(needle)!=1:raise RuntimeError('Unsupported K source')
    replacement='''meta_offset = (physical_block_idx * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + (seq_offset % BLOCK_SIZE) * stride_k_cache_1)
        group_scales = tl.load((key_cache_ptr + meta_offset[None, :] + 128
            + (offs_d[:, None] // 32) * 4).to(tl.pointer_type(tl.float32)),
            mask=(offs_d[:, None] < HEAD_SIZE) & tile_mask[None, :], other=1.)
        K = (K_load.to(tl.float32) * group_scales).to(Q.dtype)'''
    jit=triton.jit(old.fn);jit._unsafe_update_src(old.src.replace(needle,replacement))
    return jit


def launcher(config):
    fn=frozen.launcher(config,'v_only')
    fn.__globals__['kernel_unified_attention']=kernel()
    return fn
