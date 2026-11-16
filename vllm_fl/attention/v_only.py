"""Experimental BF16 K / per-token-head INT8 V paged attention.

Each page stores all BF16 keys, then INT8 values, then FP32 V scales.
No persistent BF16 value copy or extra cache allocation is used.
"""
import os
import types
import torch
from vllm.triton_utils import triton, tl
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend, TritonAttentionImpl,
)
from vllm.v1.kv_cache_interface import KVQuantMode


def enabled():
    return os.environ.get("VLLM_FL_V_ONLY", "0") == "1"


def views(raw):
    b, s, h, width = raw.shape
    if raw.dtype != torch.uint8 or width != 388 or raw.stride(-1) != 1:
        raise ValueError("V-only requires uint8 pages with head_dim=128")
    stride = raw.stride(0)
    off = raw.storage_offset()
    if stride % 4 or off % 4:
        raise ValueError("V-only page alignment must be a multiple of four bytes")
    k = torch.as_strided(raw.view(torch.bfloat16), (b,s,h,128),
                         (stride//2,h*128,128,1), off//2)
    v = torch.as_strided(raw.view(torch.int8), (b,s,h,128),
                         (stride,h*128,128,1), off+s*h*256)
    scales = torch.as_strided(raw.view(torch.float32), (b,s,h),
                              (stride//4,h,1), off//4+s*h*96)
    return k,v,scales


@triton.jit
def _write(K,V,KC,VC,SC,SLOTS, kt:tl.constexpr, kh:tl.constexpr,
           vt:tl.constexpr,vh:tl.constexpr,kb:tl.constexpr,vb:tl.constexpr,
           sb:tl.constexpr,H:tl.constexpr,S:tl.constexpr):
    tok=tl.program_id(0); head=tl.program_id(1)
    slot=tl.load(SLOTS+tok).to(tl.int64)
    if slot < 0:
        return
    page=slot//S; local=slot%S
    d=tl.arange(0,128)
    kval=tl.load(K+tok*kt+head*kh+d)
    value=tl.load(V+tok*vt+head*vh+d).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(value))/127.,1e-6)
    scaled=value/scale
    quant=tl.clamp(tl.where(scaled>=0,scaled+.5,scaled-.5),-128.,127.)
    tl.store(KC+page*kb+local*H*128+head*128+d,kval)
    tl.store(VC+page*vb+local*H*128+head*128+d,quant)
    tl.store(SC+page*sb+local*H+head,scale)


def write(k,v,raw,slots):
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("V-only accepts BF16 K/V only")
    if k.shape != v.shape or k.shape[-1] != 128 or k.stride(-1)!=1 or v.stride(-1)!=1:
        raise ValueError("Unsupported V-only input shape/stride")
    kc,vc,sc=views(raw)
    _write[(slots.numel(),k.shape[1])](k,v,kc,vc,sc,slots,k.stride(0),k.stride(1),
        v.stride(0),v.stride(1),kc.stride(0),vc.stride(0),sc.stride(0),
        k.shape[1],raw.shape[1],num_warps=4)


_launcher=None


def launcher():
    global _launcher
    if _launcher is not None:
        return _launcher
    from vllm_fl.attention.tile_policy import install_tile_policy
    from vllm.v1.attention.backends import triton_attn
    install_tile_policy()
    base=getattr(triton_attn.unified_attention,"_fl_int8_repaired",None)
    if base is None:
        raise RuntimeError("V-only requires VLLM_FL_INT8_KV_REPAIR=1")
    ns=dict(base.__globals__)
    old=ns["kernel_unified_attention"]
    needle="""k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )"""
    if old.src.count(needle)!=1:
        raise RuntimeError("Unsupported vLLM K-scale source for V-only")
    fixed=triton.jit(old.fn)
    source = old.src.replace(needle,
        "k_token_head_scales = tl.full([TILE_SIZE], 1.0, tl.float32)")
    cast = "V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)"
    prob = "P_v = (P * v_token_head_scales[None, :]).to(V.dtype)"
    if source.count(cast) != 1 or source.count(prob) != 1:
        raise RuntimeError("Unsupported V-only scale-on-V source")
    # Dequantize V in FP32 before BF16 dot; keep probabilities unscaled.
    source = source.replace(cast, "V = V_load.to(tl.float32)")
    source = source.replace(prob,
        "P_v = P.to(Q.dtype)\n            V = (V * v_token_head_scales[:, None]).to(Q.dtype)")
    fixed._unsafe_update_src(source)
    ns["kernel_unified_attention"]=fixed
    derive = ns["_fl_prefill_tiles"]

    def vonly_tiles(max_seqlen_q, num_queries_per_kv, head_size):
        bm, warps, stages = derive(max_seqlen_q, num_queries_per_kv, head_size)
        # V scale intermediates can push final's two-stage Prefill above the
        # device's shared-memory limit. Keep its BM/tile/warps and shallow the
        # pipeline only for V-only Prefill; Decode uses the final policy.
        if max_seqlen_q > 1:
            stages = 1
        return bm, warps, stages

    ns["_fl_prefill_tiles"] = vonly_tiles
    _launcher=types.FunctionType(base.__code__,ns,base.__name__,base.__defaults__,base.__closure__)
    _launcher.__kwdefaults__=base.__kwdefaults__
    return _launcher


class VOnlyBackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return VOnlyImpl

    @staticmethod
    def get_kv_cache_shape(num_blocks,block_size,num_kv_heads,head_size,cache_dtype_str="auto"):
        if head_size!=128 or block_size%16:
            raise ValueError("V-only supports head_dim128 and block multiples of16")
        return (num_blocks,block_size,num_kv_heads,388)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            raise NotImplementedError("V-only does not support layer interleaving")
        return (0,1,2,3)


class VOnlyImpl(TritonAttentionImpl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.head_size!=128 or self.kv_cache_dtype not in ("auto","bfloat16"):
            raise ValueError("V-only requires head_dim128 and ordinary BF16 cache config")
        self._vonly_launch=launcher()
        print("[FL] V-only implementation initialized: K=BF16 V=INT8 scale=FP32",flush=True)

    def do_kv_cache_update(self,layer,key,value,kv_cache,slot_mapping):
        write(key,value,kv_cache,slot_mapping)

    def fused_rope_kvcache_supported(self):
        return False

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("V-only output quantization is unsupported")
        if attn_metadata.use_cascade:
            raise NotImplementedError("V-only cascade attention is unsupported")
        kc,vc,sc=views(kv_cache)
        m=attn_metadata; n=m.num_actual_tokens
        self._vonly_launch(q=query[:n],k=kc,v=vc,out=output[:n],
            cu_seqlens_q=m.query_start_loc,max_seqlen_q=m.max_query_len,
            seqused_k=m.seq_lens,max_seqlen_k=m.max_seq_len,softmax_scale=self.scale,
            causal=m.causal,window_size=self.sliding_window,block_table=m.block_table,
            softcap=self.logits_soft_cap,q_descale=None,k_descale=None,v_descale=None,
            seq_threshold_3D=m.seq_threshold_3D,num_par_softmax_segments=m.num_par_softmax_segments,
            softmax_segm_output=m.softmax_segm_output,softmax_segm_max=m.softmax_segm_max,
            softmax_segm_expsum=m.softmax_segm_expsum,alibi_slopes=self.alibi_slopes,
            sinks=self.sinks,mm_prefix_range=m.mm_prefix_range_tensor,
            use_alibi_sqrt=self.use_alibi_sqrt,chunk_lookback=self.chunk_lookback,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,k_scale_cache=sc,v_scale_cache=sc,
            use_td=False)
        return output
