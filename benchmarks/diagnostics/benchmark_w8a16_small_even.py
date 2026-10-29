# SPDX-License-Identifier: Apache-2.0
"""Diagnostic even-shape specialization of the existing two-step partial."""
import argparse
import importlib
from pathlib import Path
from statistics import median
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from benchmark_flaggems_w8a16 import load_helper,dump,relative,save_kernel
from benchmark_flaggems_w8a16_splitk import _partial,_reduce_scale

@triton.jit
def _partial_even(A,B,P,M,N:tl.constexpr,K:tl.constexpr,
                 SAM,SAK,SBK,SBN,S:tl.constexpr,BM:tl.constexpr,
                 BN:tl.constexpr,BK:tl.constexpr,EVEN_M:tl.constexpr):
    pid=tl.program_id(0);sk=tl.program_id(1)
    pm=pid//(N//BN);pn=pid%(N//BN)
    rows=pm*BM+tl.arange(0,BM);cols=pn*BN+tl.arange(0,BN)
    per_split=K//(BK*S)
    acc=tl.zeros((BM,BN),tl.float32)
    for it in range(sk*per_split,(sk+1)*per_split):
        kk=it*BK+tl.arange(0,BK)
        ap=A+rows[:,None]*SAM+kk[None,:]*SAK
        if EVEN_M:
            a=tl.load(ap)
        else:
            a=tl.load(ap,mask=rows[:,None]<M,other=0.0)
        b=tl.load(B+kk[:,None]*SBK+cols[None,:]*SBN)
        acc+=tl.dot(a,b.to(tl.bfloat16),out_dtype=tl.float32,allow_tf32=False)
    ptr=P+sk*M*N+rows[:,None]*N+cols[None,:]
    if EVEN_M:
        tl.store(ptr,acc)
    else:
        tl.store(ptr,acc,mask=rows[:,None]<M)


def run(x,q,s,mode,capture=False):
    m,k=x.shape;n=q.shape[0];bm=64 if m==64 else 16;sk=8
    p=torch.empty((sk,m,n),device=x.device,dtype=torch.float32)
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    use_even=(mode=='even' and n%64==0 and k%(128*sk)==0)
    with torch.cuda.device(x.device):
        grid=(triton.cdiv(m,bm)*triton.cdiv(n,64),sk)
        if use_even:
            part=_partial_even[grid](x,q,p,m,n,k,x.stride(0),x.stride(1),q.stride(1),q.stride(0),
                sk,bm,64,128,m%bm==0,num_warps=4,num_stages=2,pipeline='basic')
        else:
            part=_partial[grid](x,q,p,m,n,k,x.stride(0),x.stride(1),q.stride(1),q.stride(0),
                sk,bm,64,128,num_warps=4,num_stages=2,pipeline='basic')
        finish=_reduce_scale[(triton.cdiv(m*n,256),)](p,s,y,m,n,sk,256,num_warps=4)
    return (y,part,finish,use_even) if capture else y


def check(y,e):
    err=relative(y,e);assert err<.005,err
    torch.testing.assert_close(y,e,rtol=.02,atol=.125)
    return err

@torch.inference_mode()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--helpers-dir',type=Path,required=True);args=a.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    ref=load_helper('reference',args.helpers_dir);deq=load_helper('dequant',args.helpers_dir)
    dense=importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
    weight=importlib.import_module('flag_gems.ops.weight_int8pack_mm')
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    dump(out,'metadata.json',dict(seed=[28107,314159],m=[1,16,32,64],n=2048,k=6144,
        tile='BM16 except measured M64 uses BM64; BN64/BK128/S8/warps4/stages2/basic',
        gate='N%64==0 and K%(128*8)==0; otherwise unchanged partial',
        scope='diagnostic only; N/K constexpr even path, M mask retained when needed; same reduce and allocations',
        torch=torch.__version__,triton=triton.__version__,rounds=3,warmup=100,rep=300))
    correct=[]
    for seed in (28107,314159):
        torch.manual_seed(seed)
        for m in (1,2,3,15,16,17,31,32,33,63,64,65):
            for n,k in ((64,1024),(129,257),(17,33)):
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02;w[0]=0
                q,s=ref.quantize_weight(w);e=(F.linear(x.float(),q.float())*s).to(torch.bfloat16)
                old=run(x,q,s,'old');new,_,_,used=run(x,q,s,'even',True)
                err=check(new,e);check(old,e)
                torch.testing.assert_close(new,old,rtol=.02,atol=.125)
                assert torch.count_nonzero(new[:,0]).item()==0
                correct.append(dict(seed=seed,m=m,n=n,k=k,used_even=used,error=err))
                dump(out,'correctness.json',correct)
    torch.manual_seed(314159)
    w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)*.02;q,s=ref.quantize_weight(w)
    raw=[];summary=[]
    for m in (1,16,32,64):
        x=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        e=(F.linear(x.float(),q.float())*s).to(torch.bfloat16);errors={}
        for mode in ('old','even'):
            y,part,finish,used=run(x,q,s,mode,True);errors[mode]=check(y,e)
            save_kernel(out,f'm{m}-{mode}-partial',part,dict(m=m,used_even=used))
            save_kernel(out,f'm{m}-{mode}-reduce',finish,dict(m=m))
        paths=dict(old=lambda:run(x,q,s,'old'),even=lambda:run(x,q,s,'even'),
            bf16=lambda:dense.mm(x,w.t()),separate=lambda:dense.mm(x,deq.dequantize_weight(q,s).t()),
            original=lambda:weight.weight_int8pack_mm(x,q,s))
        for fn in paths.values():fn()
        torch.cuda.synchronize();names=list(paths);samples={n:[] for n in names}
        for rep in range(3):
            for name in names[rep:]+names[:rep]:
                ms=triton.testing.do_bench(paths[name],warmup=100,rep=300,return_mode='median')
                samples[name].append(ms);raw.append(dict(m=m,path=name,round=rep+1,ms=ms))
                dump(out,'raw.json',raw)
        row=dict(m=m,errors=errors,timings_ms={n:median(v) for n,v in samples.items()})
        summary.append(row);dump(out,'summary.json',summary);print('FINAL',row,flush=True)
if __name__=='__main__':main()
