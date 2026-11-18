"""Page32 static KV + #011/#013 gates, then optional full-call micro.

Paired with the original static call (no phase/direct-index metadata), same
numeric kernels/calibration. Not BF16 model accuracy or official throughput.
"""
import argparse
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar  # compatibility patches
from vllm_fl.attention.tile_policy import install_tile_policy
install_tile_policy()
from vllm_fl.attention.k_static_fold_service import KStaticFoldImpl, launch
from vllm_fl.attention.k_static_fold_policy import choose
from vllm_fl.attention import k_static_fold_kernel as kernel
from vllm_fl.attention.static_kv_compat import attach_plan


def error(x,y):
    delta=x.float()-y.float()
    return dict(max_abs=delta.abs().max().item(),
                nrmse=(delta.square().mean().sqrt()/y.float().square().mean().sqrt().clamp_min(1e-8)).item())


def check_output(x,y):
    e=error(x,y)
    assert torch.isfinite(x).all() and e['max_abs']<=.08 and e['nrmse']<=.02,e
    return e


def fixture(lengths,seqlen,seed,padding=0):
    torch.manual_seed(seed)
    b=len(lengths);pages=(seqlen+31)//32
    raw=torch.zeros((b*pages+2,32,2,276),device='cuda',dtype=torch.uint8)
    table=torch.randperm(b*pages,device='cuda').reshape(b,pages).to(torch.int32)
    pos=torch.arange(seqlen,device='cuda')
    slots=(table[:,pos//32].long()*32+pos%32).flatten()
    slots=torch.cat((slots,torch.tensor([-1],device='cuda')))
    # Non-contiguous head source; last negative slot must not change unused page.
    k=torch.randn((b*seqlen+1,4,128),device='cuda',dtype=torch.bfloat16)[:,::2]*.1
    v=torch.randn((b*seqlen+1,2,128),device='cuda',dtype=torch.bfloat16)*.1
    sc=torch.full((2,128),.0625,device='cuda')
    perm,scale=kernel.parameters([[0,3,17,65],[0,9,37,93]],sc)
    kernel.write(k,v,raw,slots,perm,scale)
    torch.cuda.synchronize();assert torch.count_nonzero(raw[-1])==0
    seq=[seqlen-i%7 for i in range(b)]
    ids=torch.cat([torch.arange(i*seqlen+seq[i]-q,i*seqlen+seq[i],device='cuda') for i,q in enumerate(lengths)])
    wk,wv,ws=k[ids],v[ids],slots[ids]
    q=torch.randn((sum(lengths),16,128),device='cuda',dtype=torch.bfloat16)*.1
    starts=[0]
    for n in lengths+[0]*padding:starts.append(starts[-1]+n)
    starts_gpu=torch.tensor(starts,device='cuda',dtype=torch.int32)
    if padding:
        table=torch.cat((table,torch.zeros((padding,pages),device='cuda',dtype=table.dtype)))
    def make_meta():
        return SimpleNamespace(num_actual_tokens=sum(lengths),max_query_len=max(lengths),
            max_seq_len=seqlen,query_start_loc=starts_gpu,
            seq_lens=torch.tensor(seq+[0]*padding,device='cuda',dtype=torch.int32),
            slot_mapping=ws,block_table=table,causal=True,use_cascade=False,
            static_q_workspace=torch.empty_like(q),
            softmax_segm_output=torch.empty((80,16,32,128),device='cuda'),
            softmax_segm_max=torch.empty((80,16,32),device='cuda'),
            softmax_segm_expsum=torch.empty((80,16,32),device='cuda'),
            mm_prefix_range_tensor=None,seq_threshold_3D=80)
    control_meta,candidate_meta=make_meta(),make_meta()
    common=SimpleNamespace(num_reqs=b+padding,query_start_loc_cpu=torch.tensor(starts,dtype=torch.int32))
    attach_plan(candidate_meta,common)
    impl=KStaticFoldImpl(16,128,128**-.5,2,None,None,'auto')
    # Synthetic table tests plumbing; official model still loads all42 real tables.
    impl._layer_index=0;impl._perm=perm;impl._scale=scale
    layer=SimpleNamespace(layer_name='model.layers.0.self_attn.attn')
    control,candidate=torch.empty_like(q),torch.empty_like(q)
    caches=[raw.clone(),raw.clone()]
    def call(m,cache,out):return impl.forward(layer,q,wk,wv,cache,m,out)
    fns={'control':lambda:call(control_meta,caches[0],control),
         'candidate':lambda:call(candidate_meta,caches[1],candidate)}
    for fn in fns.values():fn()
    torch.cuda.synchronize()
    e=check_output(candidate,control)
    assert torch.equal(caches[0],caches[1]),'cache bytes differ'
    assert torch.equal(control_meta.static_q_workspace,candidate_meta.static_q_workspace),'Q transform differs'
    assert torch.count_nonzero(caches[1][-1])==0
    plan=candidate_meta._fl_static_mixed_plan
    direct_batch=plan.num_decodes if plan is not None else b+padding
    compact_expected=(plan is not None or max(lengths)==1) and direct_batch<=80
    compact_seen=False
    if compact_expected:
        items,*_=choose(direct_batch,1)
        compact_seen=launch(items,direct=True).__globals__['kernel_unified_attention'].reported
        assert compact_seen,'Qualified direct-index path did not launch'
    graphs={}
    for label,fn in fns.items():
        for _ in range(3):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        graphs[label]=g
    # Changed Q/K/V and sequence-length growth in replay, same qualified prefix ABI.
    q.mul_(.8);wk.mul_(.9);wv.mul_(.7)
    for m in (control_meta,candidate_meta):
        m.seq_lens[:b].fill_(max(lengths))
    for g in graphs.values():g.replay()
    torch.cuda.synchronize();check_output(candidate,control)
    for m in (control_meta,candidate_meta):
        m.seq_lens[:b].copy_(torch.tensor(seq,device='cuda',dtype=torch.int32))
    for g in graphs.values():g.replay()
    torch.cuda.synchronize();replay=check_output(candidate,control)
    assert torch.equal(caches[0],caches[1]),'replay cache bytes differ'
    return dict(lengths=lengths,kv_length=seqlen,padding=padding,error=e,replay_error=replay,
                mixed=plan is not None,compact=compact_seen,seed=seed),graphs


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--micro',action='store_true')
    p.add_argument('--audit',action='store_true');args=p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    # Full standard35 decode batch sizes plus Mixed, Prefill and padding boundaries.
    captures=[1,2,4]+list(range(8,257,8))
    cases=[([1]*b,257,0) for b in captures]
    cases += [([1]*b,n,0) for b,n in [(16,4096),(32,16384),(64,4096),(64,16384),(72,4096),(80,16384)]]
    cases += [([1]*16+[7,3],4096,0),([1]*32+[17,5],4096,0),([1]*64+[17],16384,0),
              ([1]*15+[7],257,0),([1]*16,257,8),([17]*4,257,0),([128],128,0),([512]*4,2048,0)]
    if args.micro:
        cases=[([1]*64,4096,0),([1]*32,16384,0),([1]*16+[17,5],4096,0),([17]*4,257,0)]
    if args.audit:
        assert args.micro, 'Audit is a complete-call micro gate'
        cases=[([1]*16+[128,128],4096,0),([1]*32+[128,128],16384,0),
               ([1]*16+[512,512],4096,0),([1]*32+[512,512],16384,0),
               ([1]*64+[512],16384,0),([1]*64+[17],16384,0)]
    rows=[];rng=random.Random(20261116)
    for i,(lengths,n,pad) in enumerate(cases):
        print('BEGIN',i,len(lengths),max(lengths),n,pad,flush=True)
        row,graphs=fixture(lengths,n,20261220+i,pad)
        if args.micro:
            timings={k:[] for k in graphs}
            for _ in range(9):
                order=list(graphs);rng.shuffle(order)
                for label in order:
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):graphs[label].replay()
                    end.record();end.synchronize();timings[label].append(start.elapsed_time(end)/10)
            row['paired_round_ms']=timings
            row['median_ms']={k:statistics.median(v) for k,v in timings.items()}
        rows.append(row);(args.output/'results.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True)
        del graphs
    (args.output/'COMPLETE').write_text('Full-call micro completed\n' if args.micro else 'All operator/byte/replay gates passed; not model accuracy\n')


if __name__=='__main__':main()
