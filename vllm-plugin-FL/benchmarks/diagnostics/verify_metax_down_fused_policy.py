"""Validate the installed fused-scale policy through the real dispatch entry."""
import argparse
import importlib
import json
from pathlib import Path
import random
import statistics


def main():
    import torch
    import triton.testing
    import flag_gems
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    b=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    mm=importlib.import_module(flag_gems.mm.__module__)
    assert b._down_scale_provider_matches(mm.__file__)
    assert not b._down_scale_provider_matches('/nonexistent/provider.py')
    select=b._down_fused_scale_tile
    def emit(file,row):
        with (args.output/file).open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    torch.manual_seed(20261204)
    w=torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)
    s=w.float().abs().amax(1).clamp_min(1e-8)/127
    q=(w.float()/s[:,None]).round().clamp(-127,127).to(torch.int8)
    converted=q.to(torch.bfloat16)
    ms=[112,113,128,192,193,194,255,256,257,383,384,385,386,511,512,513,1024,2048,2049,4096,8192,16384,16385]
    for m in ms:
        a=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16)
        ref=(a.float()@q.float().T)*s
        expected=select(m)
        # Ensure the unchanged fallback has an existing provider choice.
        if expected is None and m>=113:flag_gems.mm(a,converted.t())
        hits=[]
        def record(m):
            tile=select(m);hits.append((m,tile));return tile
        b._down_fused_scale_tile=record
        try:y=b._shape_dispatch(a,q,s)
        finally:b._down_fused_scale_tile=select
        torch.cuda.synchronize()
        err=float((y.float()-ref).norm()/ref.norm());assert err<.005
        if m>=113:assert hits==[(m,expected)],hits
        stream=torch.cuda.Stream();g=torch.cuda.CUDAGraph();torch.cuda.synchronize()
        with torch.cuda.graph(g,stream=stream):z=b._shape_dispatch(a,q,s)
        g.replay();torch.cuda.synchronize()
        ge=float((z.float()-ref).norm()/ref.norm());assert ge<.005
        emit('correctness.jsonl',dict(m=m,tile=expected,l2=err,graph_l2=ge,hits=hits))
        if m not in [193,384,385,2048,2049,4096]:continue
        def current():return b._shape_dispatch(a,q,s)
        def control():
            b._down_fused_scale_tile=lambda m:(64,128,128,4,4,'cpasync')
            try:return b._shape_dispatch(a,q,s)
            finally:b._down_fused_scale_tile=select
        jobs=[('policy',current),('old_specialized',control)]
        values={n:[] for n,f in jobs}
        for repeat in range(7):
            order=list(jobs);random.Random(m+repeat).shuffle(order)
            for n,f in order:values[n].append(triton.testing.do_bench(f,warmup=20,rep=100))
        graphs=[]
        for n,f in jobs:
            stream=torch.cuda.Stream();gg=torch.cuda.CUDAGraph();torch.cuda.synchronize()
            with torch.cuda.graph(gg,stream=stream):zz=f()
            gg.replay();torch.cuda.synchronize();assert float((zz.float()-ref).norm()/ref.norm())<.005
            graphs.append((n,gg,zz))
        gv={n:[] for n,f in jobs}
        for repeat in range(5):
            order=list(graphs);random.Random(m+100+repeat).shuffle(order)
            for n,gg,zz in order:gv[n].append(triton.testing.do_bench(gg.replay,warmup=20,rep=100))
        for n,f in jobs:emit('performance.jsonl',dict(m=m,name=n,rounds_ms=values[n],median_ms=statistics.median(values[n]),graph_rounds_ms=gv[n],graph_ms=statistics.median(gv[n])))
    (args.output/'complete.txt').write_text('installed policy validation complete; no model claims')


if __name__=='__main__':main()
