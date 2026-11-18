"""Fixed page32 dynamic-top4 service: full incremental write/read integration."""
from functools import lru_cache
import torch
from vllm import envs
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl, TritonAttentionMetadataBuilder
from vllm_fl.attention import v_only
from vllm_fl.attention.k_top4_joint_policy import policy, choose, verify_bindings
from vllm_fl.attention import k_top4_joint_kernel as candidate


@lru_cache(None)
def launch(items):return candidate.launcher(dict(items),'separate_meta','chain')


def is_segmented(args):
    return (args['max_seqlen_q']==1 and not envs.VLLM_BATCH_INVARIANT
        and args.get('seq_threshold_3D') is not None
        and args['cu_seqlens_q'].numel()-1<=args['seq_threshold_3D']
        and all(args.get(k) is not None for k in ('num_par_softmax_segments',
            'softmax_segm_output','softmax_segm_max','softmax_segm_expsum')))


def dispatch(**args):
    batch=args['cu_seqlens_q'].numel()-1;seg=is_segmented(args)
    items,_,_=choose(batch,args['max_seqlen_q'],seg);cfg=dict(items)
    if seg:
        segments=cfg['segments']
        for key,tail in (('softmax_segm_output',(16,segments,128)),
                         ('softmax_segm_max',(16,segments)),('softmax_segm_expsum',(16,segments))):
            size=batch
            for dim in tail:size*=dim
            buf=args[key]
            if not buf.is_contiguous() or buf.numel()<size:
                raise ValueError('Joint segment workspace capacity/contiguity mismatch')
            args[key]=buf.view(-1)[:size].view(batch,*tail)
        args['num_par_softmax_segments']=segments
    return launch(items)(**args)


class KTop4JointMetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self,kv_cache_spec,layer_names,vllm_config,device):
        super().__init__(kv_cache_spec,layer_names,vllm_config,device)
        if (self.num_heads_q,self.num_heads_kv,self.headdim,self.block_size)!=(16,2,128,32):
            raise ValueError('Joint service requires TP1, 16Q/2KV, D128/page32')
        self.num_par_softmax_segments=32
        shape=(self.seq_threshold_3D,16,32)
        self.softmax_segm_output=torch.empty((*shape,128),dtype=torch.float32,device=device)
        self.softmax_segm_max=torch.empty(shape,dtype=torch.float32,device=device)
        self.softmax_segm_expsum=torch.empty(shape,dtype=torch.float32,device=device)
        print(f'[FL] K joint workspace initialized: threshold={self.seq_threshold_3D} capacity_segments=32 page=32',flush=True)


class KTop4JointBackend(v_only.VOnlyBackend):
    @staticmethod
    def get_supported_kernel_block_sizes():return [32]
    @classmethod
    def supports_block_size(cls,block_size):return block_size is None or block_size==32
    @staticmethod
    def get_impl_cls():return KTop4JointImpl
    @staticmethod
    def get_builder_cls():return KTop4JointMetadataBuilder
    @staticmethod
    def get_kv_cache_shape(num_blocks,block_size,num_kv_heads,head_size,cache_dtype_str='auto'):
        if (block_size,num_kv_heads,head_size)!=(32,2,128):raise ValueError('Joint page32/2KV/D128 required')
        return (num_blocks,32,2,276)


class KTop4JointImpl(TritonAttentionImpl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if (self.num_heads,self.num_kv_heads,self.head_size)!=(16,2,128) or self.kv_cache_dtype not in ('auto','bfloat16'):
            raise ValueError('Joint service requires 16Q/2KV/D128 BF16 model')
        verify_bindings(policy())
        self._cache_tensor=None;self._cache_key=None;self._cache_views=None
        print('[FL] K joint v1 initialized: dynamic top4 + V INT8 scale-on-V; page32 separate_meta token_major',flush=True)

    def cache_views(self,raw):
        key=(tuple(raw.shape),tuple(raw.stride()),raw.storage_offset(),raw.dtype,raw.device)
        if raw is not self._cache_tensor or key!=self._cache_key:
            if raw.shape[1:]!=(32,2,276):raise ValueError('Joint cache layout mismatch')
            self._cache_views=candidate.views(raw,'separate_meta')
            self._cache_tensor=raw;self._cache_key=key
        return self._cache_views

    def do_kv_cache_update(self,layer,key,value,kv_cache,slot_mapping):
        # Writer1 is consistent across the frozen cross-length record selection.
        # No K/V preconversion or allocation is moved outside this operation.
        self.cache_views(kv_cache)
        return candidate.write(key,value,kv_cache,slot_mapping,'separate_meta')

    def fused_rope_kvcache_supported(self):return False

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        if attn_metadata is None:return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:raise NotImplementedError('Joint output quantization unsupported')
        if attn_metadata.use_cascade:raise NotImplementedError('Joint cascade attention unsupported')
        kc,vc,sc=self.cache_views(kv_cache);m=attn_metadata;n=m.num_actual_tokens
        dispatch(q=query[:n],k=kc,v=vc,out=output[:n],
            cu_seqlens_q=m.query_start_loc,max_seqlen_q=m.max_query_len,
            seqused_k=m.seq_lens,max_seqlen_k=m.max_seq_len,softmax_scale=self.scale,
            causal=m.causal,window_size=self.sliding_window,block_table=m.block_table,
            softcap=self.logits_soft_cap,q_descale=None,k_descale=None,v_descale=None,
            seq_threshold_3D=m.seq_threshold_3D,num_par_softmax_segments=m.num_par_softmax_segments,
            softmax_segm_output=m.softmax_segm_output,softmax_segm_max=m.softmax_segm_max,
            softmax_segm_expsum=m.softmax_segm_expsum,alibi_slopes=self.alibi_slopes,
            sinks=self.sinks,mm_prefix_range=m.mm_prefix_range_tensor,
            use_alibi_sqrt=self.use_alibi_sqrt,chunk_lookback=self.chunk_lookback,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,k_scale_cache=sc,v_scale_cache=sc,use_td=False)
        return output
