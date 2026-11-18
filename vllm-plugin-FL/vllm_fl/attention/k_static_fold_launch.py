"""Explicit static-service launch configuration, no online tuning."""
import types
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


def launcher(config, family='k'):
    """Explicit fixed configuration; caller supplies correct segment buffers."""
    if family not in ('k','v_only'):raise ValueError(family)
    fn=clone(k_top4.launcher() if family=='k' else v_only.launcher())
    
    bm,warps,stages,tile=(config[k] for k in ('block_m','warps','stages','tile'))
    fn.__globals__['_fl_prefill_tiles']=lambda **kw:(bm,warps,stages)
    fn.__globals__['_get_tile_size']=lambda *a,**kw:tile
    return fn


