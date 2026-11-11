"""Down Split-K lookahead and Separate block tuning; all costs remain timed."""
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
    from metax_down_pipeline import down_partial_prefetch, down_partial_prefetch_nt
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    backend=importlib.import_module('flag_gems.runtime.backend._metax.ops.weight_int8pack_mm')
    from vllm_fl.dispatch.backends.vendor.metax.nt_gemm import apply_c500_nt_gemm
    assert apply_c500_nt_gemm(torch.device('cuda:0'))
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
        selected=sorted(valid,key=lambda x:x[0]['screen_ms'])[:4]
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
    for seed in [20261110,20261111]:
        torch.manual_seed(seed)
        n,k=2048,6144
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        s=w.float().abs().amax(1).clamp_min(1e-8)/127
        q=(w.float()/s[:,None]).round().clamp(-127,127).to(torch.int8)
        for m in [56,64]:
            a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);ref=(a.float()@q.float().T)*s
            def split(kernel,cfg):
                bm,bn,bk,warps,stages,sk=cfg
                partial=torch.empty((sk,m,n),device=a.device,dtype=torch.float32)
                out=torch.empty((m,n),device=a.device,dtype=a.dtype)
                kernel.run(a,q,partial,m,n,k,*a.stride(),q.stride(1),q.stride(0),
                    SPLIT_K=sk,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,EVEN_M=m%bm==0,
                    num_warps=warps,num_stages=stages,pipeline='basic',
                    grid=(triton.cdiv(m,bm)*triton.cdiv(n,bn),sk),warmup=False)
                backend.weight_int8pack_mm_splitk_reduce_scale.jit_function.run(
                    partial,s,out,m,n,SPLIT_K=sk,BLOCK_SIZE=256,num_warps=4,
                    grid=(triton.cdiv(m*n,256),),warmup=False)
                return out
            configs=[(64,64,128,4,1,8),(64,64,128,4,2,8),(64,64,64,4,2,8),
                     (64,64,64,8,4,8),(64,64,128,4,1,16),(64,128,64,4,2,8)]
            jobs=[('baseline',configs[0],lambda:backend._configured_splitk(a,q,s,configs[0]))]
            for name,kernel in [('raw_kn',down_partial_prefetch),('raw_nt',down_partial_prefetch_nt)]:
                for cfg in configs:jobs.append((name,cfg,lambda kernel=kernel,cfg=cfg:split(kernel,cfg)))
            measure(jobs,ref,seed,f'split_m{m}')
        m=2048;a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        converted=q.to(torch.bfloat16);unscaled=flag_gems.mm(a,converted.t())
        # Separate has its own BF16 intermediate rounding contract.
        ref=(unscaled.float()*s).to(torch.bfloat16).float()
        def convert(cfg):
            block,warps=cfg;out=torch.empty_like(converted)
            backend.weight_int8pack_convert.jit_function.run(q,out,q.numel(),BLOCK=block,num_warps=warps,grid=(triton.cdiv(q.numel(),block),),warmup=False)
            return out
        def scale(cfg):
            block,warps=cfg;out=torch.empty_like(unscaled)
            backend.weight_int8pack_output_scale.jit_function.run(unscaled,s,out,n,out.numel(),BLOCK=block,num_warps=warps,grid=(triton.cdiv(out.numel(),block),),warmup=False)
            return out
        configs=[(block,warps) for block in [256,512,1024,2048,4096] for warps in [4,8]]
        conv=measure([('baseline' if cfg==(1024,4) else 'convert',cfg,lambda cfg=cfg:convert(cfg)) for cfg in configs],converted.float(),seed,'convert')
        scales=measure([('baseline' if cfg==(1024,4) else 'scale',cfg,lambda cfg=cfg:scale(cfg)) for cfg in configs],ref,seed,'scale')
        bestc=sorted(conv,key=lambda x:x[0]['screen_ms'])[:2];bests=sorted(scales,key=lambda x:x[0]['screen_ms'])[:2]
        def separate(cc,sc):
            weight=convert(cc);y=flag_gems.mm(a,weight.t());out=torch.empty_like(y)
            block,warps=sc
            backend.weight_int8pack_output_scale.jit_function.run(y,s,out,n,out.numel(),BLOCK=block,num_warps=warps,grid=(triton.cdiv(out.numel(),block),),warmup=False)
            return out
        jobs=[('baseline',[(1024,4),(1024,4)],lambda:backend._large_m_separate(a,q,s))]
        for cr,cf in bestc:
            for sr,sf in bests:
                cc=cr['config'];sc=sr['config'];jobs.append(('separate',[cc,sc],lambda cc=cc,sc=sc:separate(cc,sc)))
        measure(jobs,ref,seed,'separate_m2048')
    (args.output/'complete.txt').write_text('finished; inspect failures, both seeds and full-call timing')


if __name__=='__main__':main()
