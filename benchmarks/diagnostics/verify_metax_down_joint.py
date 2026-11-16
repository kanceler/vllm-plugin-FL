"""Independent full-dispatch check of the M2048 fused-path configuration winner."""
import argparse
import importlib
import json
from pathlib import Path
import random
import statistics


def main():
    import torch
    import triton
    import triton.testing
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    b = importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    old = b._cached_mm_nt_config
    configs = [('specialized_control',(64,128,128,4,4,'cpasync')),
               ('joint_candidate',(128,128,128,4,4,'cpasync'))]
    def emit(row):
        with (args.output/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    for seed in [20261123,20261124]:
        torch.manual_seed(seed)
        a = torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)
        w = torch.randn(2048,6144,device='cuda',dtype=torch.bfloat16)
        s = w.float().abs().amax(1).clamp_min(1e-8)/127
        q = (w.float()/s[:,None]).round().clamp(-127,127).to(torch.int8)
        ref = (a.float()@q.float().T)*s
        jobs = []
        for name,cfg in configs:
            bm,bn,bk,warps,stages,pipe = cfg
            config = triton.Config({'BLOCK_M':bm,'BLOCK_N':bn,'BLOCK_K':bk,'pipeline':pipe},
                                  num_warps=warps,num_stages=stages)
            hits = []
            def choose(provider,aa,weight,out,config=config,hits=hits):
                assert aa.shape==(2048,6144) and weight.shape==(2048,6144)
                hits.append(1)
                return config
            def full(choose=choose):
                # Only offline configuration selection changes. The original
                # dispatch, eligibility checks, conversion and fused call run.
                b._cached_mm_nt_config = choose
                try:return b._shape_dispatch(a,q,s)
                finally:b._cached_mm_nt_config = old
            y = full();torch.cuda.synchronize();assert len(hits)==1
            l2 = float((y.float()-ref).norm()/ref.norm());assert l2<.005
            row = {'seed':seed,'name':name,'config':cfg,'l2':l2,
                   'conversion':[4096,4],'entry':'_shape_dispatch; offline config injection'}
            jobs.append((row,full))
        times = {id(row):[] for row,fn in jobs}
        for repeat in range(9):
            order = list(jobs);random.Random(seed+repeat).shuffle(order)
            for row,fn in order:times[id(row)].append(triton.testing.do_bench(fn,warmup=25,rep=150))
        graphs = []
        for row,fn in jobs:
            row['rounds_ms'] = times[id(row)];row['median_ms'] = statistics.median(row['rounds_ms'])
            stream = torch.cuda.Stream();g = torch.cuda.CUDAGraph();torch.cuda.synchronize()
            with torch.cuda.graph(g,stream=stream):y=fn()
            g.replay();torch.cuda.synchronize()
            row['graph_l2'] = float((y.float()-ref).norm()/ref.norm());assert row['graph_l2']<.005
            graphs.append((row,g,y))
        gt = {id(row):[] for row,g,y in graphs}
        for repeat in range(7):
            order=list(graphs);random.Random(seed+100+repeat).shuffle(order)
            for row,g,y in order:gt[id(row)].append(triton.testing.do_bench(g.replay,warmup=25,rep=150))
        for row,fn in jobs:
            row['graph_rounds_ms']=gt[id(row)];row['graph_ms']=statistics.median(gt[id(row)])
            emit(row)
    (args.output/'complete.txt').write_text('independent production-interface micro complete; not integrated')


if __name__=='__main__':main()
