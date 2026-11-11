"""Validate Down conversion block size across observed and diagnostic M points."""
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
    import flag_gems
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--provider',choices=['default','wide'],required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    backend=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    from vllm_fl.dispatch.backends.vendor.metax.nt_gemm import apply_c500_nt_gemm
    if args.provider=='wide':
        assert apply_c500_nt_gemm(torch.device('cuda:0'))
    (args.output/'provider.json').write_text(json.dumps({'provider':args.provider,'registered_module':flag_gems.mm.__module__,'note':'default=no override; wide=existing four M2048 overrides only'},indent=2))
    def measure(jobs,ref,seed,label):
        valid=[]
        for name,cfg,fn in jobs:
            row={'label':label,'seed':seed,'name':name,'config':cfg}
            try:
                y=fn();torch.cuda.synchronize()
                row['l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12))
                assert row['l2']<0.005
                row['screen_ms']=triton.testing.do_bench(fn,warmup=20,rep=70)
                valid.append((row,fn))
            except Exception as exc:row['failure']=repr(exc)
            with (args.output/'screen.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(row,flush=True)
        selected=list(valid)
        selected += [x for x in valid if x[0]['name'] in ['baseline','original_bf16'] and x not in selected]
        samples={id(row):[] for row,fn in selected}
        for repeat in range(5):
            order=list(selected);random.Random(seed+repeat).shuffle(order)
            for row,fn in order:samples[id(row)].append(triton.testing.do_bench(fn,warmup=25,rep=120))
        for row,fn in selected:
            row['rounds_ms']=samples[id(row)];row['median_ms']=statistics.median(row['rounds_ms'])
            try:
                stream=torch.cuda.Stream();graph=torch.cuda.CUDAGraph();torch.cuda.synchronize()
                with torch.cuda.graph(graph,stream=stream):y=fn()
                graph.replay();torch.cuda.synchronize()
                row['graph_l2']=float((y.float()-ref).norm()/ref.norm().clamp_min(1e-12));assert row['graph_l2']<0.005
                row['graph_ms']=triton.testing.do_bench(graph.replay,warmup=25,rep=120)
            except Exception as exc:row['graph_failure']=repr(exc)
            with (args.output/'finalists.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return valid
    for m,seed in __import__('itertools').product(([2048] if args.provider=='wide' else [2048,4096,8192,16384]),[20261110,20261111]):
        torch.manual_seed(seed)
        n,k=2048,6144
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        s=w.float().abs().amax(1).clamp_min(1e-8)/127
        q=(w.float()/s[:,None]).round().clamp(-127,127).to(torch.int8)
        a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        converted=q.to(torch.bfloat16);unscaled=flag_gems.mm(a,converted.t())
        # Separate has its own BF16 intermediate rounding contract.
        ref=(unscaled.float()*s).to(torch.bfloat16).float()
        def convert(cfg):
            block,warps=cfg;out=torch.empty_like(converted)
            backend.weight_int8pack_convert.jit_function.run(q,out,q.numel(),BLOCK=block,num_warps=warps,grid=(triton.cdiv(q.numel(),block),),warmup=False)
            return out
        def separate(cc,sc):
            weight=convert(cc);y=flag_gems.mm(a,weight.t());out=torch.empty_like(y)
            block,warps=sc
            backend.weight_int8pack_output_scale.jit_function.run(y,s,out,n,out.numel(),BLOCK=block,num_warps=warps,grid=(triton.cdiv(out.numel(),block),),warmup=False)
            return out
        jobs=[]
        for block,warps in [(1024,4),(4096,4),(8192,4),(8192,8),(16384,4),(16384,8)]:
            cfg=[(block,warps),(1024,4)]
            jobs.append(('baseline' if block==1024 else 'convert_config',cfg,lambda cfg=cfg:separate(*cfg)))
        measure(jobs,ref,seed,f'separate_m{m}')
        if m==2048:
            conversion_jobs=[]
            for block,warps in [(1024,4),(4096,4),(8192,4),(8192,8),(16384,4),(16384,8)]:
                cfg=(block,warps)
                conversion_jobs.append(('baseline' if block==1024 else 'convert_config',cfg,lambda cfg=cfg:convert(cfg)))
            measure(conversion_jobs,converted.float(),seed,'convert_only')
    (args.output/'complete.txt').write_text('finished; inspect failures, both seeds and full-call timing')


if __name__=='__main__':main()
