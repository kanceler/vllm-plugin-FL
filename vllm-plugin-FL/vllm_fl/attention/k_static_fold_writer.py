"""Joint Q transform and static K/V writer; frozen diagnostic arithmetic."""
from functools import lru_cache
import types
from vllm.triton_utils import tl
from vllm.triton_utils import triton
from vllm_fl.attention import k_static_fold_kernel as prior
from vllm_fl.attention import k_top4_joint_kernel as layout


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


