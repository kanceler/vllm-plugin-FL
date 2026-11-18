"""Opt-in model adapter for the frozen K top4 v2 + scale-on-V kernels."""
import os
from functools import lru_cache
from pathlib import Path
import torch
from vllm import envs
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder, TritonAttentionImpl
from vllm_fl.attention import k_top4, v_only
from benchmarks.diagnostics import k_top4_frozen_v2 as frozen


@lru_cache(None)
def manifest():
    path = Path(__file__).parents[2] / "benchmarks/diagnostics/configs/k_top4_integration_v1.json"
    return frozen.load_manifest(path)


@lru_cache(None)
def select_config(batch, qlen, klen, segmented):
    records = manifest()["records"]
    if qlen == 1:
        # FULL_DECODE_ONLY captures with KV length 1, then replays at growing
        # lengths. Select by captured batch/branch, never by live KV length.
        matches = [r for r in records if r["case"][0] == batch
                   and r["case"][2] == 1 and r["case"][3] == segmented]
        if matches:
            return dict(max(matches, key=lambda r: r["case"][1])["config"])
    else:
        for record in records:
            if record["case"][:4] == [batch, klen, qlen, segmented]:
                return dict(record["config"])
    return frozen.baseline_config(qlen, segmented)


@lru_cache(None)
def get_launch(items):
    return frozen.launcher(dict(items))


def dispatch(**args):
    batch = args["cu_seqlens_q"].numel() - 1
    segmented = (args["max_seqlen_q"] == 1 and not envs.VLLM_BATCH_INVARIANT
        and args.get("seq_threshold_3D") is not None
        and batch <= args["seq_threshold_3D"]
        and all(args.get(k) is not None for k in ("num_par_softmax_segments",
            "softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum")))
    qlen = args["max_seqlen_q"]
    cfg = select_config(batch, qlen, 0 if qlen == 1 else args["max_seqlen_k"], segmented)
    if segmented:
        segments = cfg["segments"]
        # Flat prefix views retain compact segment strides for the upstream
        # kernel, which computes its own offsets instead of reading strides.
        for key, tail in (("softmax_segm_output", (16, segments, 128)),
                          ("softmax_segm_max", (16, segments)),
                          ("softmax_segm_expsum", (16, segments))):
            size = batch
            for dim in tail: size *= dim
            buf = args[key]
            if not buf.is_contiguous() or buf.numel() < size:
                raise ValueError("K top4 segment workspace capacity/stride mismatch")
            args[key] = buf.view(-1)[:size].view(batch, *tail)
        args["num_par_softmax_segments"] = segments
    return get_launch(tuple(sorted(cfg.items())))(**args)


class KTop4MetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if (self.num_heads_q, self.num_heads_kv, self.headdim, self.block_size) != (16, 2, 128, 16):
            raise ValueError("K top4 service requires TP1, 16Q/2KV, D128, page16")
        self.num_par_softmax_segments = 32
        shape = (self.seq_threshold_3D, 16, 32)
        self.softmax_segm_output = torch.empty((*shape, 128), dtype=torch.float32, device=device)
        self.softmax_segm_max = torch.empty(shape, dtype=torch.float32, device=device)
        self.softmax_segm_expsum = torch.empty(shape, dtype=torch.float32, device=device)
        print(f"[FL] K top4 workspace: threshold={self.seq_threshold_3D}, capacity_segments=32", flush=True)


class KTop4Backend(v_only.VOnlyBackend):
    @staticmethod
    def get_impl_cls(): return KTop4Impl

    @staticmethod
    def get_builder_cls(): return KTop4MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        if (block_size, num_kv_heads, head_size) != (16, 2, 128):
            raise ValueError("Unsupported K top4 service cache contract")
        return (num_blocks, block_size, num_kv_heads, 276)


class KTop4Impl(v_only.VOnlyImpl):
    def __init__(self, *args, **kwargs):
        # Initialize the upstream implementation without V-only's launcher/log.
        TritonAttentionImpl.__init__(self, *args, **kwargs)
        if (self.num_heads, self.num_kv_heads, self.head_size) != (16, 2, 128):
            raise ValueError("K top4 requires 16Q/2KV/D128")
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("K top4 requires the ordinary BF16 model configuration")
        manifest()
        self._vonly_launch = dispatch
        print("[FL] K top4 v2 initialized: K=INT8+BF16 top4, V=INT8 scale-on-V, FP32 scales, writer=load_w1", flush=True)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        frozen.write(key, value, kv_cache, slot_mapping, writer="load_w1")

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("K top4 output quantization is unsupported")
        if attn_metadata.use_cascade:
            raise NotImplementedError("K top4 cascade attention is unsupported")
        kc,vc,sc=k_top4.views(kv_cache)
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
