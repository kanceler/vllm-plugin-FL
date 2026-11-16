"""Q transform fusion experiments, byte-compatible with static candidate v1."""
from functools import lru_cache
import types
from vllm.triton_utils import tl
from vllm.triton_utils import triton
import k_static_protect_fold_candidate as prior
import k_top4_layout_candidates as layout


def _write(Q,QO,K,V,KC,VC,VS,SLOTS,P,SC,qt:tl.constexpr,qh:tl.constexpr,kt:tl.constexpr,kh:tl.constexpr,
           vt:tl.constexpr,vh:tl.constexpr,kb:tl.constexpr,vb:tl.constexpr,sb:tl.constexpr,S:tl.constexpr):
    return


@lru_cache(None)
def writer_kernel(parallel_heads=False):
    old=prior._write;src=old.src
    assert src.count('def _write(K,V,')==1
    src=src.replace('def _write(K,V,','def _write(Q,QO,K,V,')
    src=src.replace('kt:tl.constexpr,kh:tl.constexpr,','qt:tl.constexpr,qh:tl.constexpr,kt:tl.constexpr,kh:tl.constexpr,')
    needle='    slot=tl.load(SLOTS+t).to(tl.int64)'
    # Transform Q even for a padded negative KV slot. Each KV head owns eight
    # Q heads; incremental write tokens and Q tokens have identical ordering.
    body='''    qd=tl.arange(0,128);heads=tl.arange(0,8)
    qch=tl.load(P+h*128+qd);qs=tl.load(SC+h*128+qd)
    qx=tl.load(Q+t*qt+(h*8+heads[:,None])*qh+qch[None,:]).to(tl.float32)
    tl.store(QO+t*16*128+(h*8+heads[:,None])*128+qd[None,:],qx*tl.where(qd[None,:]<4,1.,qs[None,:]))
'''
    if parallel_heads:
        body="""    qd=tl.arange(0,128);kvh=h//8
    qch=tl.load(P+kvh*128+qd);qs=tl.load(SC+kvh*128+qd)
    qx=tl.load(Q+t*qt+h*qh+qch).to(tl.float32)
    tl.store(QO+t*16*128+h*128+qd,qx*tl.where(qd<4,1.,qs))
    if h%8!=0:return
    h=h//8
"""
    assert src.count(needle)==1
    jit=triton.jit(_write);jit._unsafe_update_src(src.replace(needle,body+needle));return jit


def write_q(q,qo,k,v,raw,slots,perm,scale,warps=1,parallel_heads=False):
    if q.shape[0]!=slots.numel() or k.shape[0]!=slots.numel():
        raise ValueError('Fused Q/write requires matching incremental token order/count')
    kc,vc,vs=layout.views(raw,'separate_meta')
    return writer_kernel(parallel_heads)[(slots.numel(),16 if parallel_heads else 2)](q,qo,k,v,kc,vc,vs,slots,perm,scale,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),
        kc.stride(0),vc.stride(0),vs.stride(0),raw.shape[1],num_warps=warps)


@lru_cache(None)
def attention_kernel():
    import k_static_fold_fusion_signature as signature
    old=prior.kernel();src=old.src
    assert src.count('\n    query_ptr,')==1
    src=src.replace('\n    query_ptr,','\n    query_ptr,\n    static_perm_ptr,\n    static_scale_ptr,')
    needle='    query_offset = ('
    assert src.count(needle)==1
    src=src.replace(needle,'''    tl.static_assert(not USE_TD_QO, "Static Q fusion requires ordinary Q loads")
    static_q_channels = tl.load(static_perm_ptr + kv_head_idx * 128 + offs_d, mask=offs_d < 128, other=0)
    static_q_scales = tl.load(static_scale_ptr + kv_head_idx * 128 + offs_d, mask=offs_d < 128, other=1.)
'''+needle)
    a=src.index(needle);b=src.index('    dim_mask = ',a)
    block=src[a:b];assert block.count('offs_d[None, :]')==1
    src=src[:a]+block.replace('offs_d[None, :]','static_q_channels[None, :]')+src[b:]
    needle='    block_table_offset = seq_idx * block_table_stride'
    assert src.count(needle)==1
    src=src.replace(needle,'''    Q = (Q.to(tl.float32) * tl.where(offs_d[None, :] < 4, 1., static_q_scales[None, :])).to(Q.dtype)

'''+needle)
    stub=signature.kernel_unified_attention
    fn=types.FunctionType(stub.__code__,dict(old.fn.__globals__),stub.__name__,stub.__defaults__,stub.__closure__)
    fn.__annotations__=stub.__annotations__;fn.__kwdefaults__=stub.__kwdefaults__
    jit=triton.jit(fn);jit._unsafe_update_src(src);return jit


class BoundAttention:
    def __init__(self,perm,scale):
        self.perm=perm;self.scale=scale

    def __getitem__(self,grid):
        launch=attention_kernel()[grid]
        def run(*args,**kwargs):
            return launch(*args,static_perm_ptr=self.perm,static_scale_ptr=self.scale,**kwargs)
        return run


def attention_launcher(config,perm,scale):
    fn=prior.launcher(config)
    fn.__globals__['kernel_unified_attention']=BoundAttention(perm,scale)
    return fn
