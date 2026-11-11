# SPDX-License-Identifier: Apache-2.0
"""Bounded full-call W8A16 tuning; no production dispatch changes."""
import argparse
import importlib
from pathlib import Path
from statistics import median
import torch
import torch.nn.functional as F
import triton
import triton.testing
from benchmark_flaggems_w8a16 import load_helper, dump, relative, save_kernel
from benchmark_flaggems_w8a16_configs import launch
from benchmark_flaggems_w8a16_splitk import _partial, _reduce_scale

MS = (1,16,32,64,128,256,512,1024,2048)


def split(x,q,s,c):
    bm,bn,bk,warps,stages,sk = c
    m,k=x.shape; n=q.shape[0]
    p=torch.empty((sk,m,n),device=x.device,dtype=torch.float32)
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    with torch.cuda.device(x.device):
        _partial[(triton.cdiv(m,bm)*triton.cdiv(n,bn),sk)](
            x,q,p,m,n,k,x.stride(0),x.stride(1),q.stride(1),q.stride(0),
            sk,bm,bn,bk,num_warps=warps,num_stages=stages,pipeline='basic')
        _reduce_scale[(triton.cdiv(m*n,256),)](p,s,y,m,n,sk,256,num_warps=4)
    return y


def configs(m):
    # Limited one-factor probes around the established small-M candidate.
    if m<=16:
        return [(16,64,128,w,t,8) for w,t in ((4,2),(2,2),(4,1),(4,3))]
    if m<=64:
        return [(b,64,128,4,2,s) for b in (16,32,64) for s in (4,8)]
    # Large-M applicability check, not an exhaustive split search.
    return [(32,64,128,4,2,4)]


def check(y,e):
    err=relative(y,e)
    assert err<.005,err
    torch.testing.assert_close(y,e,rtol=.02,atol=.125)
    return err


@torch.inference_mode()
def main():
    a=argparse.ArgumentParser(); a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--helpers-dir',type=Path,required=True);args=a.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    ref=load_helper('reference',args.helpers_dir);deq=load_helper('dequant',args.helpers_dir)
    dense=importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
    weight=importlib.import_module('flag_gems.ops.weight_int8pack_mm');jit=weight.weight_int8pack_mm_kernel.jit_function
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    dump(out,'metadata.json',dict(torch=torch.__version__,triton=triton.__version__,
        flaggems_source=weight.__file__,m=MS,n=2048,k=6144,split_configs={m:configs(m) for m in MS},
        seed_screen=73,seed_final=314159,screen_warmup=50,screen_rep=150,
        final_warmup=100,final_rep=300,rounds=3,
        timing='GPU event full call including allocations and both kernels; not CPU wall time',
        scope='synthetic Down operator only; no model or production dispatch'))
    torch.manual_seed(73)
    w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)*.02;q,s=ref.quantize_weight(w)
    screen=[];chosen={}
    for m in MS:
        x=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        e=(F.linear(x.float(),q.float())*s).to(torch.bfloat16)
        candidates=[('single',c) for c in ((16,64,128,4,2),(32,64,128,4,2),(128,128,32,8,2))]
        candidates += [('split',c) for c in configs(m)]
        valid=[]
        for kind,c in candidates:
            row=dict(m=m,kind=kind,config=c)
            try:
                fn=(lambda: split(x,q,s,c)) if kind=='split' else (lambda:launch(jit,c,x,q,s))
                row['relative_l2_error']=check(fn(),e)
                # Each configuration also validates masked loads/stores and zero scales.
                xo=torch.randn(3,257,device='cuda',dtype=torch.bfloat16)
                wo=torch.randn(129,257,device='cuda',dtype=torch.bfloat16)*.02;wo[0]=0
                qo,so=ref.quantize_weight(wo);eo=(F.linear(xo.float(),qo.float())*so).to(torch.bfloat16)
                yo=split(xo,qo,so,c) if kind=='split' else launch(jit,c,xo,qo,so)
                row['masked_error']=check(yo,eo)
                assert torch.count_nonzero(yo[:,0]).item()==0
                row['ms']=triton.testing.do_bench(fn,warmup=50,rep=150,return_mode='median')
                row['status']='passed';valid.append(row)
            except Exception as exc:
                row.update(status='failed',error=repr(exc))
                if type(exc).__name__ not in ('OutOfResources','CompilationError','AssertionError'):raise
            screen.append(row);dump(out,'screening.json',screen)
        chosen[str(m)]={kind:min([r for r in valid if r['kind']==kind],key=lambda r:r['ms'])['config'] for kind in ('single','split') if any(r['kind']==kind for r in valid)}
        dump(out,'chosen.json',chosen);print('SCREEN',m,chosen[str(m)],flush=True)
    torch.manual_seed(314159)
    w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)*.02;q,s=ref.quantize_weight(w)
    torch.testing.assert_close(deq.dequantize_weight(q,s),ref.dequantize_reference(q,s),rtol=0,atol=0)
    raw=[];summary=[]
    for m in MS:
        x=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        e=(F.linear(x.float(),q.float())*s).to(torch.bfloat16)
        c=chosen[str(m)]
        paths=dict(bf16=lambda:dense.mm(x,w.t()),separate=lambda:dense.mm(x,deq.dequantize_weight(q,s).t()),
            original=lambda:weight.weight_int8pack_mm(x,q,s),
            single=lambda:launch(jit,c['single'],x,q,s))
        if 'split' in c:paths['split']=lambda:split(x,q,s,c['split'])
        errors={name:check(fn(),e) for name,fn in paths.items() if name in ('original','single','split')}
        for fn in paths.values():fn()
        torch.cuda.synchronize()
        samples={name:[] for name in paths};names=list(paths)
        for repeat in range(3):
            for name in names[repeat:]+names[:repeat]:
                ms=triton.testing.do_bench(paths[name],warmup=100,rep=300,return_mode='median')
                samples[name].append(ms);raw.append(dict(m=m,path=name,round=repeat+1,ms=ms))
                dump(out,'raw_final.json',raw)
        t={name:median(v) for name,v in samples.items()}
        summary.append(dict(m=m,configs=c,errors=errors,timings_ms=t))
        dump(out,'summary.json',summary);print('FINAL',summary[-1],flush=True)

if __name__=='__main__':main()
