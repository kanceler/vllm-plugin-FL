"""Numerically equivalent dynamic-top4 layout/restoration experiments only."""
from functools import lru_cache
import torch
from vllm.triton_utils import triton
from vllm_fl.attention import k_top4
import k_top4_frozen as frozen


def width(layout):
    return {'record144':276,'record160':292,'separate_meta':276}[layout]


def views(raw,layout):
    b,s,h,w=raw.shape
    if raw.dtype!=torch.uint8 or w!=width(layout) or raw.stride(-1)!=1:
        raise ValueError('Unexpected experimental cache contract')
    stride=raw.stride(0);off=raw.storage_offset()
    if stride%4 or off%4:raise ValueError('Unaligned cache')
    record=160 if layout=='record160' else 144
    body=128 if layout=='separate_meta' else record
    k=torch.as_strided(raw.view(torch.int8),(b,s,h,128),(stride,h*body,body,1),off)
    v=torch.as_strided(raw.view(torch.int8),(b,s,h,128),(stride,h*128,128,1),off+s*h*record)
    vs=torch.as_strided(raw.view(torch.float32),(b,s,h),(stride//4,h,1),off//4+s*h*(record+128)//4)
    return k,v,vs


@lru_cache(None)
def writer(layout):
    if layout=='record144':return frozen.writer_kernel()
    old=frozen.writer_kernel();src=old.src
    needle='a=tl.abs(x); base=KC+p*kb+loc*H*144+h*144'
    if src.count(needle)!=1:raise RuntimeError('Unsupported writer source')
    if layout=='record160':
        src=src.replace(needle,'a=tl.abs(x); base=KC+p*kb+loc*H*160+h*160')
    elif layout=='separate_meta':
        src=src.replace(needle,'''a=tl.abs(x); base=KC+p*kb+loc*H*128+h*128
    meta=KC+p*kb+S*H*128+loc*H*16+h*16''')
        for a,b in [('base+128+i','meta+i'),('base+132+2*i','meta+4+2*i'),('base+140','meta+12')]:src=src.replace(a,b)
    else:raise ValueError(layout)
    jit=triton.jit(old.fn);jit._unsafe_update_src(src);return jit


def write(k,v,raw,slots,layout):
    if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or k.shape!=v.shape or k.shape[-1]!=128 or k.stride(-1)!=1 or v.stride(-1)!=1 or slots.numel()!=k.shape[0]:
        raise ValueError('Matching BF16 contiguous head dimension and token slots required')
    kc,vc,vs=views(raw,layout)
    return writer(layout)[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=1)


@lru_cache(None)
def kernel(layout,restore):
    old=frozen.packed_kernel();src=old.src
    if layout=='separate_meta':
        start=src.index('        meta_offset = ');end=src.index('        ks = ',start)
        src=src[:start]+'''        meta_offset = (physical_block_idx * stride_k_cache_0
            + BLOCK_SIZE * (num_query_heads // num_queries_per_kv) * 128
            + kv_head_idx * 16 + (seq_offset % BLOCK_SIZE) * (num_query_heads // num_queries_per_kv) * 16)
'''+src[end:]
        for a,b in [('meta_offset + 140','meta_offset + 12'),('meta_offset + 128','meta_offset'),('meta_offset + 132','meta_offset + 4'),('meta_offset + 136','meta_offset + 8')]:src=src.replace(a,b)
    if restore=='or_tree':
        start=src.index('        for protected in tl.static_range(4):');end=src.index('        K = K_real.to(Q.dtype)',start)
        # The quantized body stores exactly zero at every protected dimension.
        # Build independent masked BF16 bit patterns before merging with body.
        body=''
        for i in range(4):
            body+=f'''        idx{i} = (packed_idx >> {i*8}) & 255
        bits{i} = (({'packed_v0' if i<2 else 'packed_v1'} >> {(i%2)*16}) & 65535).to(tl.uint16)
        part{i} = tl.where(offs_d[:, None] == idx{i}[None, :], bits{i}[None, :], 0).to(tl.uint16)
'''
        body+='''        protected_bits = (part0 | part1) | (part2 | part3)
        K_real = (K_real.to(tl.uint16, bitcast=True) | protected_bits).to(tl.bfloat16, bitcast=True)
'''
        src=src[:start]+body+src[end:]
    elif restore!='chain':raise ValueError(restore)
    jit=triton.jit(old.fn);jit._unsafe_update_src(src);return jit


def launcher(config,layout,restore):
    fn=frozen.launcher(config)
    fn.__globals__['kernel_unified_attention']=kernel(layout,restore)
    return fn
