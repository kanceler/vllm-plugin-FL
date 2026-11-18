"""Explicit diagnostic candidate v1. No service registration or runtime tuning.

Callers select a manifest record or baseline_config and own segment workspace.
Configuration is launcher-local, so Prefill history cannot change its tile.
"""
import hashlib
import json
import types
from functools import lru_cache
from pathlib import Path
import torch
from vllm.triton_utils import triton
from vllm_fl.attention import k_top4, v_only, config_model


def clone(fn):
    result=types.FunctionType(fn.__code__,dict(fn.__globals__),fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__
    return result


def baseline_config(qlen, segmented=False):
    """Canonical original parameters, independent of global Prefill state."""
    from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size
    if qlen>1:
        bm,warps,_,tile=config_model.derive_prefill_config(128,8,qlen)
        tile=tile or _get_tile_size(128,0,2,True)
    else:
        bm,warps,_=config_model.derive_decode_tiles(128,8)
        tile=_get_tile_size(128,0,2,not segmented)
    return dict(block_m=bm,tile=tile,warps=warps,stages=1,segments=16 if segmented else 1)


@lru_cache(None)
def packed_kernel():
    old=k_top4.launcher().__globals__['kernel_unified_attention']
    src=old.src
    needle='K_real = K_load.to(tl.float32) * ks[None, :]'
    assert src.count(needle)==1
    src=src.replace(needle,'K_real = (K_load.to(tl.float32) * ks[None, :]).to(Q.dtype)')
    start=src.index('        for protected in tl.static_range(4):')
    stop=src.index('        K = K_real.to(Q.dtype)',start)
    body='''        packed_idx = tl.load((key_cache_ptr + meta_offset + 128).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0xffffffff)
        packed_v0 = tl.load((key_cache_ptr + meta_offset + 132).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
        packed_v1 = tl.load((key_cache_ptr + meta_offset + 136).to(tl.pointer_type(tl.uint32)), mask=tile_mask, other=0)
        for protected in tl.static_range(4):
            idx = (packed_idx >> (protected * 8)) & 255
            word = packed_v0 if protected < 2 else packed_v1
            bits = ((word >> ((protected % 2) * 16)) & 65535).to(tl.uint16)
            val = bits.to(tl.bfloat16, bitcast=True)
            K_real = tl.where(offs_d[:, None] == idx[None, :], val[None, :], K_real)
'''
    jit=triton.jit(old.fn);jit._unsafe_update_src(src[:start]+body+src[stop:])
    return jit


def launcher(config, family='k'):
    """Explicit fixed configuration; caller supplies correct segment buffers."""
    if family not in ('k','v_only'):raise ValueError(family)
    fn=clone(k_top4.launcher() if family=='k' else v_only.launcher())
    if family=='k':fn.__globals__['kernel_unified_attention']=packed_kernel()
    bm,warps,stages,tile=(config[k] for k in ('block_m','warps','stages','tile'))
    fn.__globals__['_fl_prefill_tiles']=lambda **kw:(bm,warps,stages)
    fn.__globals__['_get_tile_size']=lambda *a,**kw:tile
    return fn


@lru_cache(None)
def writer_kernel():
    old=k_top4._write
    needle='saved=tl.sum(tl.where(d==index,x,0.),0)'
    assert old.src.count(needle)==1
    jit=triton.jit(old.fn)
    jit._unsafe_update_src(old.src.replace(needle,'saved=tl.load(K+t*kt+h*kh+index).to(tl.float32)'))
    return jit


def write(k,v,raw,slots):
    if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or k.shape!=v.shape or k.shape[-1]!=128:
        raise ValueError('Matching BF16 D128 K/V required')
    if k.stride(-1)!=1 or v.stride(-1)!=1 or slots.numel()!=k.shape[0]:
        raise ValueError('Unsupported strides or slot count')
    kc,vc,vs=k_top4.views(raw)
    return writer_kernel()[(slots.numel(),k.shape[1])](k,v,kc,vc,vs,slots,k.stride(0),k.stride(1),
        v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),vs.stride(0),k.shape[1],raw.shape[1],num_warps=1)


def identity():
    modules=(k_top4,v_only,config_model)
    return dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,triton=triton.__version__,
        files={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules})


def load_manifest(path):
    manifest=json.loads(Path(path).read_text())
    if manifest['identity']!=identity():
        raise ValueError('Frozen manifest environment/source mismatch; use baseline_config, not stale records')
    return manifest
