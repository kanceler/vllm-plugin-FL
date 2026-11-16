"""Bounded V-only tile correctness and complete-call timing probe.

Diagnostic only: runs no model, changes no production policy. GPU events time
V quantization+K/V write+attention together, including all real work.
"""
import json
import os
from pathlib import Path
import sys
import traceback
import torch

os.environ['VLLM_FL_V_ONLY']='1'
os.environ['VLLM_FL_INT8_KV_REPAIR']='1'
from validate_tianshu_existing_int8_kv import run
from vllm_fl.attention.v_only import views, write, launcher
from vllm.v1.kv_cache_interface import KVQuantMode

CASES=[('old_conservative',32,4,1,32),('final',256,16,2,128),
       ('bm256_t128_s1',256,16,1,128),('bm256_t64_s2',256,16,2,64),
       ('bm256_t64_s1',256,16,1,64),('bm128_t128_s1',128,8,1,128),
       ('bm128_t64_s2',128,8,2,64)]


def set_config(bm,warps,stages,tile):
    ns=launcher().__globals__
    # Applied only to this isolated probe's launch function.
    ns['_fl_prefill_tiles']=lambda max_seqlen_q,num_queries_per_kv,head_size: (bm,warps,stages)
    ns['_get_tile_size']=lambda h,w,e,is_prefill: tile if is_prefill else 16


def measure(qlen,seqlen):
    block=16; heads=2; pages=(seqlen+block-1)//block
    cache=torch.zeros((pages,block,heads,388),device='cuda',dtype=torch.uint8)
    k=torch.randn((seqlen,heads,128),device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k);q=torch.randn((qlen,16,128),device='cuda',dtype=torch.bfloat16)
    slots=torch.arange(seqlen,device='cuda',dtype=torch.int64)
    kc,vc,vs=views(cache)
    out=torch.empty_like(q)
    args=dict(q=q,k=kc,v=vc,out=out,cu_seqlens_q=torch.tensor([0,qlen],device='cuda',dtype=torch.int32),
        max_seqlen_q=qlen,seqused_k=torch.tensor([seqlen],device='cuda',dtype=torch.int32),
        max_seqlen_k=seqlen,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
        block_table=torch.arange(pages,device='cuda',dtype=torch.int32).view(1,-1),
        softcap=0.,q_descale=None,k_descale=None,v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,k_scale_cache=vs,v_scale_cache=vs,use_td=False)
    def complete():
        write(k,v,cache,slots)
        launcher()(**args)
    for _ in range(3):complete()
    torch.cuda.synchronize()
    times=[]
    for _ in range(7):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();complete();end.record();end.synchronize()
        times.append(start.elapsed_time(end))
    return times


if __name__=='__main__':
    torch.manual_seed(20261113)
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    with (root/'raw.jsonl').open('x') as out:
        for case in CASES:
            name,bm,warps,stages,tile=case
            record=dict(name=name,bm=bm,warps=warps,stages=stages,tile=tile)
            try:
                set_config(bm,warps,stages,tile)
                record['correctness']=run(1,128,128,False,False)
                record['milliseconds_128']=measure(128,128)
                record['milliseconds_2048']=measure(2048,2048)
                record['status']='pass'
            except Exception as exc:
                record['status']='failed';record['error']=repr(exc)
                traceback.print_exc()
            print(json.dumps(record),flush=True);out.write(json.dumps(record)+'\n');out.flush()
