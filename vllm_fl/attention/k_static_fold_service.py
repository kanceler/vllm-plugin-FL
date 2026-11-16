"""Static protected K + V INT8 service with joint Q/K/V update inside forward."""
from functools import lru_cache
import re
import torch
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl, TritonAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm_fl.attention.k_top4_joint_service import KTop4JointBackend
from vllm_fl.attention import k_static_fold_kernel as kernel
from vllm_fl.attention import k_static_fold_writer as writer
from vllm_fl.attention.k_static_fold_policy import calibration,choose,verify


@lru_cache(None)
def launch(items):return kernel.launcher(dict(items))


class KStaticFoldMetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self,kv_cache_spec,layer_names,vllm_config,device):
        super().__init__(kv_cache_spec,layer_names,vllm_config,device)
        if (self.num_heads_q,self.num_heads_kv,self.headdim,self.block_size)!=(16,2,128,32):
            raise ValueError('Static K service requires TP1/16Q/2KV/D128/page32')
        self.seq_threshold_3D=80;self.num_par_softmax_segments=32
        shape=(80,16,32)
        self.softmax_segm_output=torch.empty((*shape,128),dtype=torch.float32,device=device)
        self.softmax_segm_max=torch.empty(shape,dtype=torch.float32,device=device)
        self.softmax_segm_expsum=torch.empty(shape,dtype=torch.float32,device=device)
        # Shared across layers in this builder: model attention calls execute
        # sequentially on the worker stream. No per-step or per-layer allocation.
        capacity=vllm_config.scheduler_config.max_num_batched_tokens
        self.static_q_workspace=torch.empty((capacity,16,128),dtype=torch.bfloat16,device=device)
        print(f'[FL] Static K workspace initialized: page32 threshold80 segments32 Qcapacity={capacity}',flush=True)

    def build(self,*args,**kwargs):
        m=super().build(*args,**kwargs)
        m.static_q_workspace=self.static_q_workspace
        return m


class KStaticFoldBackend(KTop4JointBackend):
    # Native vLLM capability: passes Q/K/V into forward and omits separate update.
    forward_includes_kv_cache_update=True

    @staticmethod
    def get_impl_cls():return KStaticFoldImpl
    @staticmethod
    def get_builder_cls():return KStaticFoldMetadataBuilder


class KStaticFoldImpl(TritonAttentionImpl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if (self.num_heads,self.num_kv_heads,self.head_size)!=(16,2,128) or self.kv_cache_dtype not in ('auto','bfloat16'):
            raise ValueError('Static K service requires ordinary BF16 16Q/2KV/D128')
        verify()
        self._layer_index=None;self._perm=None;self._scale=None
        self._cache_tensor=None;self._cache_views=None;self._cache_key=None

    def _parameters(self,layer,device):
        name=layer.layer_name
        match=re.search(r'(?:^|\.)layers\.(\d+)\.',name)
        if match is None:raise ValueError(f'Cannot identify calibrated attention layer: {name}')
        index=int(match.group(1))
        if self._layer_index is not None:
            if self._layer_index!=index:raise ValueError('Static implementation reused for a different layer')
            return
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Static layer parameters must initialize before capture')
        row=calibration()['layers'][index]
        self._perm=torch.tensor(row['permutation'],dtype=torch.int32,device=device)
        self._scale=torch.tensor(row['scale'],dtype=torch.float32,device=device)
        self._layer_index=index
        print(f'[FL] Static K layer initialized: layer={index} fixed4+scale_fold_Q V_INT8 fused_write page32',flush=True)

    def cache_views(self,raw):
        key=(tuple(raw.shape),tuple(raw.stride()),raw.storage_offset(),raw.dtype,raw.device)
        if raw is not self._cache_tensor or key!=self._cache_key:
            if raw.dtype!=torch.uint8 or raw.shape[1:]!=(32,2,276) or not raw.is_contiguous():
                raise ValueError('Static K cache must use page32/276B contiguous uint8')
            self._cache_tensor=raw;self._cache_key=key;self._cache_views=kernel.layout.views(raw,'separate_meta')
        return self._cache_views

    def do_kv_cache_update(self,*args,**kwargs):
        raise RuntimeError('Static backend owns KV update inside Attention.forward')

    def fused_rope_kvcache_supported(self):return False

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        self._parameters(layer,query.device)
        if attn_metadata is None:return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:raise NotImplementedError('Static output quantization unsupported')
        m=attn_metadata;n=m.num_actual_tokens
        if m.use_cascade:raise NotImplementedError('Static cascade Attention unsupported')
        if key is None or value is None:raise ValueError('Static joint update requires actual K/V')
        if query.dtype!=torch.bfloat16 or key.dtype!=torch.bfloat16 or value.dtype!=torch.bfloat16:
            raise ValueError('Static joint update requires BF16 Q/K/V')
        q=query[:n];k=key[:n];v=value[:n];slots=m.slot_mapping[:n]
        if min(query.shape[0],key.shape[0],value.shape[0],m.slot_mapping.numel())<n:
            raise ValueError('Static Q/K/V/slot token count mismatch')
        qbuf=m.static_q_workspace
        if qbuf.shape[0]<n:raise ValueError('Static Q workspace capacity exceeded')
        qo=qbuf[:n];batch=m.query_start_loc.numel()-1
        items,mode,ww,_=choose(batch,m.max_query_len);cfg=dict(items)
        if mode not in ('q_in_write','q_in_write_heads'):raise ValueError('Unsupported frozen static mode')
        kc,vc,vs=self.cache_views(kv_cache)
        if q.shape[1:]!=(16,128) or k.shape[1:]!=(2,128) or v.shape!=k.shape:
            raise ValueError('Static Q/K/V head shape mismatch')
        writer.write_q(q,qo,k,v,kv_cache,slots,self._perm,self._scale,ww,mode=='q_in_write_heads')
        a=dict(q=qo,k=kc,v=vc,out=output[:n],cu_seqlens_q=m.query_start_loc,
            max_seqlen_q=m.max_query_len,seqused_k=m.seq_lens,max_seqlen_k=m.max_seq_len,
            softmax_scale=self.scale,causal=m.causal,window_size=self.sliding_window,
            block_table=m.block_table,softcap=self.logits_soft_cap,q_descale=None,k_descale=None,v_descale=None,
            alibi_slopes=self.alibi_slopes,sinks=self.sinks,mm_prefix_range=m.mm_prefix_range_tensor,
            use_alibi_sqrt=self.use_alibi_sqrt,chunk_lookback=self.chunk_lookback,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,k_scale_cache=vs,v_scale_cache=vs,use_td=False,
            seq_threshold_3D=80)
        if m.max_query_len==1 and batch<=80:
            ns=cfg['segments'];a['num_par_softmax_segments']=ns
            for name,tail in (('softmax_segm_output',(16,ns,128)),('softmax_segm_max',(16,ns)),('softmax_segm_expsum',(16,ns))):
                count=batch
                for dim in tail:count*=dim
                buf=getattr(m,name)
                if not buf.is_contiguous() or buf.numel()<count:raise ValueError('Static segment workspace insufficient')
                a[name]=buf.view(-1)[:count].view(batch,*tail)
        launch(items)(**a)
        return output
