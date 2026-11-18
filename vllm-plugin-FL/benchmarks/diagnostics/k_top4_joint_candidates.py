"""Joint page/metadata experiments, retaining dynamic top4 quantization."""
from functools import lru_cache
import torch
from vllm.triton_utils import triton
import k_top4_frozen as frozen
import k_top4_layout_candidates as prior


def views(raw,head_major=False):
    if not head_major:return prior.views(raw,'separate_meta')
    b,s,h,w=raw.shape
    if raw.dtype!=torch.uint8 or w!=276 or raw.stride(-1)!=1:raise ValueError('Expected raw 276-byte cache')
    stride=raw.stride(0);off=raw.storage_offset()
    if stride%4 or off%4:raise ValueError('Unaligned cache')
    k=torch.as_strided(raw.view(torch.int8),(b,s,h,128),(stride,128,s*128,1),off)
    v=torch.as_strided(raw.view(torch.int8),(b,s,h,128),(stride,h*128,128,1),off+s*h*144)
    vs=torch.as_strided(raw.view(torch.float32),(b,s,h),(stride//4,h,1),off//4+s*h*68)
    return k,v,vs


def metadata(raw,head_major=False):
    b,s,h,_=raw.shape;stride=raw.stride(0);off=raw.storage_offset()+s*h*128
    return torch.as_strided(raw,(b,s,h,16),(stride,16 if head_major else h*16,s*16 if head_major else 16,1),off)


@lru_cache(None)
def writer(head_major=False):
    if not head_major:return prior.writer('separate_meta')
    old=prior.writer('separate_meta');src=old.src
    assert 'base=KC+p*kb+loc*H*128+h*128' in src
    src=src.replace('base=KC+p*kb+loc*H*128+h*128','base=KC+p*kb+h*S*128+loc*128')
    src=src.replace('meta=KC+p*kb+S*H*128+loc*H*16+h*16','meta=KC+p*kb+S*H*128+h*S*16+loc*16')
    jit=triton.jit(old.fn);jit._unsafe_update_src(src);return jit


def write(k,v,raw,slots,head_major=False,warps=1):
    if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or k.shape!=v.shape or k.shape[-1]!=128 or k.stride(-1)!=1 or v.stride(-1)!=1 or k.shape[0]!=slots.numel():raise ValueError('Invalid K/V contract')
    kc,vc,vs=views(raw,head_major)
    return writer(head_major)[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=warps)


@lru_cache(None)
def kernel(head_major=False,vector_meta=False):
    old=prior.kernel('separate_meta','chain');src=old.src
    if head_major:
        needle='kv_head_idx * 16 + (seq_offset % BLOCK_SIZE) * (num_query_heads // num_queries_per_kv) * 16'
        assert src.count(needle)==1
        src=src.replace(needle,'kv_head_idx * BLOCK_SIZE * 16 + (seq_offset % BLOCK_SIZE) * 16')
    if vector_meta:
        a=src.index('        ks = tl.load(');b=src.index('        for protected in tl.static_range(4):',a)
        src=src[:a]+'''        meta_words = tl.load((key_cache_ptr + meta_offset[None, :]
            + tl.arange(0, 4)[:, None] * 4).to(tl.pointer_type(tl.uint32)),
            mask=tile_mask[None, :], other=0)
        even_words, odd_words = tl.split(tl.reshape(tl.trans(meta_words), (TILE_SIZE, 2, 2)))
        packed_idx, packed_v1 = tl.split(even_words)
        packed_v0, scale_bits = tl.split(odd_words)
        ks = scale_bits.to(tl.float32, bitcast=True)
        K_real = (K_load.to(tl.float32) * ks[None, :]).to(Q.dtype)
'''+src[b:]
    jit=triton.jit(old.fn);jit._unsafe_update_src(src);return jit


def launcher(config,head_major=False,vector_meta=False):
    fn=frozen.launcher(config)
    fn.__globals__['kernel_unified_attention']=kernel(head_major,vector_meta)
    return fn
