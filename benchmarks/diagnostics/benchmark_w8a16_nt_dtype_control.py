# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only BF16(Q) storage control; not a quantized deployment candidate."""
from pathlib import Path
from statistics import median
import argparse
import importlib
import torch
import torch.nn.functional as F
import triton.testing
from benchmark_flaggems_w8a16 import load_helper,dump,save_kernel
from benchmark_flaggems_w8a16_large_nt import call,verify

@torch.inference_mode()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--nt-only',action='store_true')
    args=a.parse_args();out=args.output_dir;out.mkdir(exist_ok=False)
    ref=load_helper('reference',Path('.'))
    weight=importlib.import_module('flag_gems.ops.weight_int8pack_mm')
    generic=weight.weight_int8pack_mm_kernel.jit_function
    torch.manual_seed(281028);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)*.02;q,s=ref.quantize_weight(w)
    qb=q.to(torch.bfloat16)  # Exactly represented integers, converted outside timing.
    cfg=('cpasync3',3,'cpasync');raw=[];summary=[]
    dump(out,'metadata.json',dict(seed=281028,m=[256,2048],tile=[64,64,64],warps=4,stages=3,pipeline='cpasync',
        note='BF16(Q) is unscaled, exact integer values, conversion outside timing, twice weight storage; diagnostic only',rounds=3,warmup=100,rep=300))
    for m in (256,2048):
        x=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        e=(F.linear(x.float(),q.float())*s).to(torch.bfloat16)
        paths={};errors={}
        for kind in (('nt',) if args.nt_only else ('nt','generic')):
            for label,b in (('int8',q),('bf16proxy',qb)):
                name=kind+'_'+label
                y,compiled=call(kind,cfg,x,b,s,generic,True);errors[name]=verify(y,e)
                save_kernel(out,f'm{m}-{name}',compiled,dict(dtype=label,config=cfg))
                paths[name]=lambda kind=kind,b=b:call(kind,cfg,x,b,s,generic)
        for fn in paths.values():fn()
        torch.cuda.synchronize();names=list(paths)
        for rep in range(3):
            for name in names[rep:]+names[:rep]:
                ms=triton.testing.do_bench(paths[name],warmup=100,rep=300,return_mode='median')
                raw.append(dict(m=m,path=name,round=rep+1,ms=ms));dump(out,'raw.json',raw)
        summary.append(dict(m=m,errors=errors,timings_ms={n:median([r['ms'] for r in raw if r['m']==m and r['path']==n]) for n in names}))
        dump(out,'summary.json',summary);print(summary[-1],flush=True)
if __name__=='__main__':main()
